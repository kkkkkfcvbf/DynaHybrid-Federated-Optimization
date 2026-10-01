"""Data construction and federated server/client training loop."""

import json
import os
import random
import sys
import time
from dataclasses import replace
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
from config import ExperimentConfig, get_layer_name, get_layers
from algorithms.fedavg import aggregate_state_dicts, train_one_client_fedavg
from algorithms.fedprox import train_one_client_fedprox
from algorithms.fedsophia import SophiaClientState, train_one_client_fedsophia
from algorithms.dynahybrid import train_one_client_dynahybrid
from algorithms.dynahybrid_adahessian import (
    AdaHessianClientState,
    train_one_client_dynahybrid_adahessian,
)
from algorithms.dynahybrid_lbfgs import (
    DynaHybridLBFGSClientState,
    train_one_client_dynahybrid_lbfgs,
)
from utils.scoring import StreamingLayerScoreMoments, select_dynamic_layers_by_coverage
from utils.flops import ModelFlopProfile, RoundFlops, estimate_round_flops
from utils.logger import append_metric_csv, write_metrics_csv
from utils.metrics import weighted_mean
from test import evaluate
from models import build_model


def _peak_process_rss_mb() -> float:
    """Return this process's peak resident CPU memory in MiB when available."""
    try:
        import resource

        peak_rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        # Linux reports KiB; macOS reports bytes.
        return peak_rss / (1024.0 * 1024.0) if sys.platform == "darwin" else peak_rss / 1024.0
    except (ImportError, AttributeError, OSError):
        return 0.0


def _state_dict_nbytes(state: Dict[str, torch.Tensor]) -> int:
    """Exact serialized tensor payload size, excluding protocol/container overhead."""
    return sum(t.numel() * t.element_size() for t in state.values())


def _optimizer_state_nbytes(states: List[SophiaClientState]) -> int:
    """Bytes occupied by persistent Sophia m/h tensors for all simulated clients."""
    return sum(
        tensor.numel() * tensor.element_size()
        for state in states
        for mapping in (state.m, state.h)
        for tensor in mapping.values()
    )


def _adahessian_state_nbytes(states: List[AdaHessianClientState]) -> int:
    """Bytes occupied by persistent m/h tensors for DynaHybrid-AdaHessian."""
    return sum(
        tensor.numel() * tensor.element_size()
        for state in states
        for mapping in (state.m, state.h)
        for tensor in mapping.values()
    )


def _lbfgs_estimator_state_nbytes(
    states: List[DynaHybridLBFGSClientState],
) -> int:
    """Persistent Sophia m/h and retained L-BFGS (s,y) tensors."""
    sophia_bytes = sum(
        tensor.numel() * tensor.element_size()
        for state in states
        for mapping in (state.m, state.h)
        for tensor in mapping.values()
    )
    pair_bytes = sum(
        tensor.numel() * tensor.element_size()
        for state in states
        for tensor in (*state.history.s, *state.history.y)
    )
    return sophia_bytes + pair_bytes

def get_lr_multiplier(
    cfg: ExperimentConfig,
    one_based_round: int,
    schedule: Optional[str] = None,
) -> float:
    """
    Calculate the learning-rate multiplier for the current
    communication round.
    """

    active_schedule = cfg.lr_schedule if schedule is None else schedule

    if active_schedule == "constant":
        return 1.0

    if active_schedule == "multistep":
        decay_count = sum(
            one_based_round > milestone
            for milestone in cfg.lr_milestones
        )
        return cfg.lr_decay_gamma ** decay_count

    raise ValueError(
        f"Unknown learning-rate schedule={active_schedule!r}"
    )

def make_model(cfg: ExperimentConfig) -> nn.Module:
    """Build the requested architecture without pretrained weights."""
    return build_model(cfg.model_name, cfg.num_classes)


def iid_partition_indices(num_samples: int, num_clients: int, seed: int) -> List[List[int]]:
    """Return a deterministic, exhaustive, disjoint, equal-size IID split."""
    if num_clients < 1:
        raise ValueError("num_clients must be at least 1.")
    if num_samples < num_clients:
        raise ValueError("num_samples must be at least num_clients.")
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(num_samples, generator=generator).tolist()
    chunks = [[] for _ in range(num_clients)]
    for pos, idx in enumerate(indices):
        chunks[pos % num_clients].append(idx)
    return chunks


def dataset_targets(dataset: Dataset) -> np.ndarray:
    """Read integer class labels without invoking stochastic data transforms."""
    if isinstance(dataset, Subset):
        parent_targets = dataset_targets(dataset.dataset)
        indices = np.asarray(dataset.indices, dtype=np.int64)
        return parent_targets[indices]

    targets = getattr(dataset, "targets", None)
    if targets is None:
        targets = getattr(dataset, "labels", None)
    if targets is None:
        raise TypeError(
            "Dataset must expose .targets or .labels for a Dirichlet partition."
        )
    if torch.is_tensor(targets):
        targets = targets.detach().cpu().numpy()
    labels = np.asarray(targets, dtype=np.int64).reshape(-1)
    if len(labels) != len(dataset):
        raise ValueError("Dataset label count does not match dataset length.")
    return labels


def validate_partitions(
    partitions: Sequence[Sequence[int]],
    num_samples: int,
    num_clients: int,
    min_size: int = 1,
) -> None:
    """Fail fast unless client indices form an exact valid partition."""
    if len(partitions) != num_clients:
        raise ValueError(
            f"Expected {num_clients} client partitions; found {len(partitions)}."
        )
    flattened = [int(index) for client in partitions for index in client]
    if len(flattened) != num_samples:
        raise ValueError(
            f"Partition contains {len(flattened)} assignments for {num_samples} samples."
        )
    if any(index < 0 or index >= num_samples for index in flattened):
        raise ValueError("Partition contains an out-of-range sample index.")
    if len(set(flattened)) != num_samples:
        raise ValueError("Partition contains duplicate or missing sample indices.")
    sizes = [len(client) for client in partitions]
    if min(sizes) < min_size:
        raise ValueError(
            f"Smallest client has {min(sizes)} samples; required at least {min_size}."
        )


def dirichlet_partition_indices(
    labels: Sequence[int] | np.ndarray,
    num_clients: int,
    alpha: float,
    min_size: int,
    seed: int,
    max_attempts: int = 10000,
) -> List[List[int]]:
    """Create a deterministic Dirichlet label-skew partition.

    For each class c, client proportions are independently sampled from
    Dirichlet(alpha). Every example is assigned exactly once. Sampling is
    repeated until every client contains at least ``min_size`` examples.
    """
    labels_array = np.asarray(labels, dtype=np.int64).reshape(-1)
    num_samples = len(labels_array)
    if num_clients < 1:
        raise ValueError("num_clients must be at least 1.")
    if alpha <= 0.0:
        raise ValueError("Dirichlet alpha must be positive.")
    if min_size < 1:
        raise ValueError("Dirichlet min_size must be at least 1.")
    if num_samples < num_clients * min_size:
        raise ValueError(
            "Dirichlet min_size is infeasible: "
            f"{num_samples} samples < {num_clients} clients * {min_size}."
        )

    rng = np.random.default_rng(seed)
    classes = np.unique(labels_array)
    for _ in range(max_attempts):
        client_parts: List[List[int]] = [[] for _ in range(num_clients)]
        for class_id in classes:
            class_indices = np.flatnonzero(labels_array == class_id)
            class_indices = rng.permutation(class_indices)
            proportions = rng.dirichlet(np.full(num_clients, alpha, dtype=np.float64))
            counts = rng.multinomial(len(class_indices), proportions)
            boundaries = np.cumsum(counts)[:-1]
            for client_id, part in enumerate(np.split(class_indices, boundaries)):
                client_parts[client_id].extend(int(index) for index in part)

        if min(len(part) for part in client_parts) >= min_size:
            for part in client_parts:
                rng.shuffle(part)
            validate_partitions(
                client_parts, num_samples, num_clients, min_size=min_size
            )
            return client_parts

    raise RuntimeError(
        "Unable to construct a Dirichlet partition meeting the minimum client "
        f"size after {max_attempts} attempts. Reduce --dirichlet_min_size or "
        "increase --dirichlet_alpha."
    )


def save_partition_artifacts(
    cfg: ExperimentConfig,
    partitions: Sequence[Sequence[int]],
    labels: np.ndarray,
) -> None:
    """Save exact assignments and per-client class counts for auditability."""
    os.makedirs(cfg.output_dir, exist_ok=True)
    class_ids = [int(value) for value in np.unique(labels)]
    clients = []
    arrays = {}
    for client_id, indices in enumerate(partitions):
        index_array = np.asarray(indices, dtype=np.int64)
        client_labels = labels[index_array]
        counts = {
            str(class_id): int(np.sum(client_labels == class_id))
            for class_id in class_ids
        }
        clients.append(
            {
                "client_id": client_id,
                "num_samples": int(len(index_array)),
                "class_counts": counts,
            }
        )
        arrays[f"client_{client_id}"] = index_array

    summary = {
        "dataset": cfg.dataset,
        "partition": cfg.partition,
        "seed": cfg.seed,
        "num_clients": cfg.num_clients,
        "num_samples": int(len(labels)),
        "dirichlet_alpha": cfg.dirichlet_alpha,
        "dirichlet_min_size": cfg.dirichlet_min_size,
        "clients": clients,
    }
    with open(
        os.path.join(cfg.output_dir, "partition_summary.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    np.savez_compressed(
        os.path.join(cfg.output_dir, "partition_indices.npz"), **arrays
    )


def build_dataloaders(cfg: ExperimentConfig) -> Tuple[List[DataLoader], List[int], DataLoader]:
    dataset_name = cfg.dataset.lower()
    resize = transforms.Resize((cfg.image_size, cfg.image_size))
    if dataset_name == "cifar10":
        dataset_cls = datasets.CIFAR10
        mean, std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
        train_tf = transforms.Compose([
            resize, transforms.RandomHorizontalFlip(), transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        test_tf = transforms.Compose([resize, transforms.ToTensor(), transforms.Normalize(mean, std)])
    elif dataset_name == "cifar100":
        dataset_cls = datasets.CIFAR100
        mean, std = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
        train_tf = transforms.Compose([
            resize, transforms.RandomHorizontalFlip(), transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        test_tf = transforms.Compose([resize, transforms.ToTensor(), transforms.Normalize(mean, std)])
    elif dataset_name == "mnist":
        dataset_cls = datasets.MNIST
        mnist_tf = transforms.Compose([
            resize, transforms.Grayscale(num_output_channels=3), transforms.ToTensor(),
            transforms.Normalize((0.1307,) * 3, (0.3081,) * 3),
        ])
        train_tf = test_tf = mnist_tf
    else:
        raise ValueError(f"Unsupported dataset {dataset_name!r}.")

    try:
        train_set: Dataset = dataset_cls(
            root=cfg.data_dir, train=True, download=cfg.dataset_download, transform=train_tf
        )
        test_set: Dataset = dataset_cls(
            root=cfg.data_dir, train=False, download=cfg.dataset_download, transform=test_tf
        )
    except (RuntimeError, FileNotFoundError) as exc:
        raise FileNotFoundError(
            f"{dataset_name} data not found under {cfg.data_dir!r}; run again with --download."
        ) from exc

    if cfg.train_subset is not None:
        train_set = Subset(train_set, list(range(min(cfg.train_subset, len(train_set)))))
    if cfg.test_subset is not None:
        test_set = Subset(test_set, list(range(min(cfg.test_subset, len(test_set)))))

    labels = dataset_targets(train_set)
    if cfg.partition == "iid":
        partitions = iid_partition_indices(len(train_set), cfg.num_clients, cfg.seed)
        validate_partitions(partitions, len(train_set), cfg.num_clients)
    elif cfg.partition == "dirichlet":
        partitions = dirichlet_partition_indices(
            labels=labels,
            num_clients=cfg.num_clients,
            alpha=cfg.dirichlet_alpha,
            min_size=cfg.dirichlet_min_size,
            seed=cfg.seed,
        )
    else:
        raise ValueError(f"Unsupported partition mode {cfg.partition!r}.")
    save_partition_artifacts(cfg, partitions, labels)
    client_loaders: List[DataLoader] = []
    client_sizes: List[int] = []
    for cid, idxs in enumerate(partitions):
        subset = Subset(train_set, idxs)
        client_sizes.append(len(subset))
        loader = DataLoader(
            subset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
        )
        client_loaders.append(loader)

    test_loader = DataLoader(
        test_set,
        batch_size=cfg.test_batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return client_loaders, client_sizes, test_loader



def run_federated_experiment(
    algo: str,
    cfg: ExperimentConfig,
    client_loaders: List[DataLoader],
    client_sizes: List[int],
    test_loader: DataLoader,
) -> List[Dict[str, object]]:
    device = torch.device(cfg.device)
    global_model = make_model(cfg).to(device)
    # 删除旧csv，开始新实验
    if cfg.save_csv:

        csv_path = os.path.join(
            cfg.output_dir,
            f"{algo}_metrics.csv"
        )

        if os.path.exists(csv_path):
            os.remove(csv_path)

    client_sophia_states = [SophiaClientState() for _ in range(cfg.num_clients)]
    client_adahessian_states = [
        AdaHessianClientState() for _ in range(cfg.num_clients)
    ]
    client_lbfgs_states = [
        DynaHybridLBFGSClientState() for _ in range(cfg.num_clients)
    ]
    dynamic_algorithms = {
        "dynahybrid", "dynahybrid_lbfgs", "dynahybrid_adahessian"
    }
    prev_score_moments: StreamingLayerScoreMoments | None = None
    selected_layers: List[str] = []
    selected_k: int = 0
    last_scores: Dict[str, float] = {}
    last_normalized_scores: Dict[str, float] = {}
    last_selected_coverage: float = 0.0
    last_ranked_layers: List[str] = []

    rows: List[Dict[str, object]] = []
    total_hessian_time_sec = 0.0
    total_hessian_param_elements = 0
    cumulative_flops = RoundFlops()
    cumulative_full_fedsophia_flops = RoundFlops()
    total_round_time_sec = 0.0
    total_peak_gpu_allocated_mb_rounds = 0.0
    total_peak_gpu_reserved_mb_rounds = 0.0
    hessian_round_peak_gpu_allocated_mb_sum = 0.0
    hessian_round_peak_gpu_reserved_mb_sum = 0.0
    hessian_active_round_count = 0
    cumulative_upload_bytes = 0
    cumulative_download_bytes = 0
    target_reached_round = 0
    target_reached_time_sec = 0.0
    target_reached_communication_bytes = 0
    experiment_peak_gpu_allocated_mb = 0.0
    experiment_peak_gpu_reserved_mb = 0.0
    summary_csv = os.path.join(cfg.output_dir, f"{algo}_compute_summary.csv")
    if cfg.save_csv and os.path.exists(summary_csv):
        os.remove(summary_csv)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

    all_hessian_params_per_estimate = sum(
        param.numel()
        for name, param in global_model.named_parameters()
        if get_layer_name(name, cfg.model_name, cfg.grouping_mode) is not None
    )
    flop_profile = ModelFlopProfile.from_config_and_model(cfg, global_model)

    print(f"\n========== Start experiment: {algo} ==========")
    print(f"Device: {device}; clients={cfg.num_clients}; rounds={cfg.num_rounds}; local_steps={cfg.local_steps}")
    if algo in dynamic_algorithms:
        print(f"Layer groups ({cfg.model_name}): {get_layers(cfg.model_name, cfg.grouping_mode)}")
        print(
            "DynaHybrid selection: "
            f"selection_strategy={cfg.selection_strategy}; "
            f"k_strategy={cfg.k_strategy}; "
            f"fixed_k={cfg.fixed_second_order_layers}; every "
            f"{cfg.score_interval} rounds from round {cfg.first_score_round}; "
            f"gamma={cfg.score_coverage_threshold}; "
            f"K in [{cfg.min_second_order_layers}, {cfg.max_second_order_layers}]"
        )
        if algo == "dynahybrid_lbfgs":
            print(
                "L-BFGS Hessian estimator: same-batch reversible joint-block "
                f"probe; history={cfg.lbfgs_history_size}; "
                f"Powell threshold={cfg.lbfgs_curvature_eps}; "
                f"damping={cfg.lbfgs_damping}"
            )
        if algo == "dynahybrid_adahessian":
            print(
                "AdaHessian Hessian estimator: Hutchinson diagonal; "
                f"samples={cfg.adahessian_hutchinson_samples}; "
                "update path matches DynaHybrid-Sophia except h_hat source"
            )

    for round_idx0 in range(cfg.num_rounds):
        one_based_round = round_idx0 + 1

        lr_multiplier = get_lr_multiplier(
            cfg,
            one_based_round,
            cfg.lr_schedule,
        )
        sophia_lr_multiplier = get_lr_multiplier(
            cfg,
            one_based_round,
            cfg.sophia_lr_schedule,
        )

        round_cfg = replace(
            cfg,
            fedavg_lr=cfg.fedavg_lr * lr_multiplier,
            sophia_lr=cfg.sophia_lr * sophia_lr_multiplier,
        )

        newly_selected_layers: List[str] = []
        # Selection at rounds 2, 12, 22, ... consumes only the immediately
        # preceding round's mean client gradients. No other round needs to copy
        # full layer-gradient vectors from GPU to CPU.
        next_round = one_based_round + 1
        selection_uses_scores = cfg.selection_strategy in {"dynamic", "fixed"}
        collect_scoring_grads = (
            algo in dynamic_algorithms
            and selection_uses_scores
            and (
                cfg.selection_strategy != "fixed"
                or next_round == cfg.first_score_round
            )
            and one_based_round < cfg.num_rounds
            and next_round >= cfg.first_score_round
            and (next_round - cfg.first_score_round) % cfg.score_interval == 0
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        round_start_time = time.perf_counter()
        round_hessian_time_sec = 0.0
        round_hessian_param_elements = 0

        # DynaHybrid recomputes K at rounds 2, 12, 22, ... by default, using the
        # immediately preceding round's client layer gradients. Between scoring rounds,
        # selected_layers and selected_k are kept unchanged. Round 1 stays all-SGD.
        scheduled_selection_round = (
            algo in dynamic_algorithms
            and one_based_round >= cfg.first_score_round
            and (one_based_round - cfg.first_score_round) % cfg.score_interval == 0
        )
        should_update_selection = scheduled_selection_round and not (
            cfg.selection_strategy in {"fixed", "manual"} and bool(selected_layers)
        )
        if should_update_selection:
            previous_selected_set = set(selected_layers)
            if cfg.selection_strategy == "manual":
                selected_layers = list(cfg.manual_second_order_layers)
                selected_k = len(selected_layers)
                last_scores = {}
                last_normalized_scores = {}
                last_selected_coverage = 0.0
                last_ranked_layers = list(selected_layers)
            elif cfg.selection_strategy == "random":
                all_layers = get_layers(cfg.model_name, cfg.grouping_mode)
                selected_k = (
                    cfg.fixed_second_order_layers
                    if cfg.k_strategy == "fixed"
                    else min(cfg.max_second_order_layers, max(cfg.min_second_order_layers, cfg.fixed_second_order_layers))
                )
                selected_k = min(selected_k, len(all_layers))
                rng = random.Random(cfg.seed * 1000003 + one_based_round)
                selected_layers = rng.sample(all_layers, selected_k)
                last_scores = {}
                last_normalized_scores = {}
                last_selected_coverage = 0.0
                last_ranked_layers = []
            else:
                if prev_score_moments is None:
                    raise RuntimeError(
                        "Selection round has no streaming gradient moments from the preceding round."
                    )
                last_scores = prev_score_moments.compute_scores(cfg)
                prev_score_moments = None
                if cfg.k_strategy == "fixed":
                    selection_min_k = cfg.fixed_second_order_layers
                    selection_max_k = cfg.fixed_second_order_layers
                else:
                    selection_min_k = cfg.min_second_order_layers
                    selection_max_k = cfg.max_second_order_layers
                (
                    selected_layers,
                    selected_k,
                    last_normalized_scores,
                    last_selected_coverage,
                    last_ranked_layers,
                ) = select_dynamic_layers_by_coverage(
                    scores=last_scores,
                    coverage_threshold=cfg.score_coverage_threshold,
                    min_k=selection_min_k,
                    max_k=selection_max_k,
                )
            newly_selected_layers = [
                layer for layer in selected_layers if layer not in previous_selected_set
            ]
            print(
                f"[DynaHybrid] Round {one_based_round}: "
                f"selection={cfg.selection_strategy}, K={selected_k}, "
                f"selected layers={selected_layers}, "
                f"new layers={newly_selected_layers}, "
                f"coverage={last_selected_coverage:.6f}"
            )
            print(f"[DynaHybrid] Raw scores = {last_scores}")
            print(f"[DynaHybrid] Normalized scores = {last_normalized_scores}")
            print(f"[DynaHybrid] Ranking = {last_ranked_layers}")

        client_states = []
        client_losses = []
        current_score_moments = (
            StreamingLayerScoreMoments() if collect_scoring_grads else None
        )

        # Server broadcasts global parameters by loading the same global_state for every client.
        global_state_cpu = {k: v.detach().cpu().clone() for k, v in global_model.state_dict().items()}
        model_payload_bytes = _state_dict_nbytes(global_state_cpu)
        # Logical FL traffic: server broadcasts one full model to each client.
        round_download_bytes = cfg.num_clients * model_payload_bytes

        for cid in range(cfg.num_clients):
            local_model = make_model(cfg).to(device)
            local_model.load_state_dict(global_state_cpu, strict=True)

            if algo == "fedavg":
                state, loss, layer_grads = train_one_client_fedavg(
                    local_model, client_loaders[cid], round_cfg, device,
                    collect_scoring_grads=False,
                )
            elif algo == "fedprox":
                state, loss, layer_grads = train_one_client_fedprox(
                    local_model, client_loaders[cid], round_cfg, device,
                    collect_scoring_grads=False,
                )
            elif algo == "fedsophia":
                state, loss, layer_grads, client_hessian_time_sec, client_hessian_param_elements = train_one_client_fedsophia(
                    local_model, client_loaders[cid], client_sophia_states[cid], round_cfg, device, round_idx0,
                    collect_scoring_grads=False,
                )
                round_hessian_time_sec += client_hessian_time_sec
                round_hessian_param_elements += client_hessian_param_elements
            elif algo == "dynahybrid":
                state, loss, layer_grads, client_hessian_time_sec, client_hessian_param_elements = train_one_client_dynahybrid(
                    local_model,
                    client_loaders[cid],
                    client_sophia_states[cid],
                    round_cfg,
                    device,
                    round_idx0,
                    selected_layers,
                    newly_selected_layers,
                    collect_scoring_grads=collect_scoring_grads,
                )
                round_hessian_time_sec += client_hessian_time_sec
                round_hessian_param_elements += client_hessian_param_elements
            elif algo == "dynahybrid_lbfgs":
                state, loss, layer_grads, client_hessian_time_sec, client_hessian_param_elements = train_one_client_dynahybrid_lbfgs(
                    local_model,
                    client_loaders[cid],
                    client_lbfgs_states[cid],
                    round_cfg,
                    device,
                    round_idx0,
                    selected_layers,
                    newly_selected_layers,
                    collect_scoring_grads=collect_scoring_grads,
                )
                round_hessian_time_sec += client_hessian_time_sec
                round_hessian_param_elements += client_hessian_param_elements
            elif algo == "dynahybrid_adahessian":
                state, loss, layer_grads, client_hessian_time_sec, client_hessian_param_elements = train_one_client_dynahybrid_adahessian(
                    local_model,
                    client_loaders[cid],
                    client_adahessian_states[cid],
                    round_cfg,
                    device,
                    round_idx0,
                    selected_layers,
                    newly_selected_layers,
                    collect_scoring_grads=collect_scoring_grads,
                )
                round_hessian_time_sec += client_hessian_time_sec
                round_hessian_param_elements += client_hessian_param_elements
            else:
                raise ValueError(f"Unknown algo={algo}")

            client_states.append(state)
            client_losses.append(loss)
            if current_score_moments is not None:
                # Use the same p_k=n_k/sum_j n_j weighting as model aggregation.
                # Passing n_k is sufficient because StreamingLayerScoreMoments
                # normalizes by the accumulated weight. The moments retain one
                # vector and one scalar per group, not all client gradients.
                current_score_moments.update(
                    layer_grads, weight=float(client_sizes[cid])
                )

            del local_model
            if device.type == "cuda":
                torch.cuda.empty_cache()

        # Every client uploads one complete local model update/state. These are
        # logical communication bytes in this single-process simulator.
        round_upload_bytes = sum(_state_dict_nbytes(state) for state in client_states)
        cumulative_download_bytes += round_download_bytes
        cumulative_upload_bytes += round_upload_bytes

        # All four algorithms optimize the same empirical objective and therefore
        # use p_k=n_k/sum_j n_j. For the archived equal-size IID split this is
        # exactly identical to uniform averaging.
        avg_state = aggregate_state_dicts(client_states, client_sizes)
        global_model.load_state_dict(avg_state, strict=True)

        train_loss = weighted_mean(client_losses, client_sizes)
        test_loss, test_acc = evaluate(global_model, test_loader, device)

        round_flops = estimate_round_flops(
            algo=algo,
            cfg=cfg,
            profile=flop_profile,
            client_sizes=client_sizes,
            test_examples=len(test_loader.dataset),
            round_idx0=round_idx0,
            selected_layers=selected_layers,
            newly_selected_layers=newly_selected_layers,
            collect_scoring_grads=collect_scoring_grads,
        )
        cumulative_flops.add_(round_flops)
        full_fedsophia_round_flops = estimate_round_flops(
            algo="fedsophia",
            cfg=cfg,
            profile=flop_profile,
            client_sizes=client_sizes,
            test_examples=len(test_loader.dataset),
            round_idx0=round_idx0,
            selected_layers=[],
            newly_selected_layers=[],
            collect_scoring_grads=False,
        )
        cumulative_full_fedsophia_flops.add_(full_fedsophia_round_flops)
        estimated_flops_ratio_vs_full_fedsophia = cumulative_flops.total / max(
            cumulative_full_fedsophia_flops.total, 1
        )
        estimated_flops_saving_vs_full_fedsophia = (
            1.0 - estimated_flops_ratio_vs_full_fedsophia
        )

        # Synchronize before wall-clock and peak-memory readings so all CUDA work
        # belonging to this round has completed.
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            round_peak_gpu_allocated_mb = torch.cuda.max_memory_allocated(device) / (1024.0 ** 2)
            round_peak_gpu_reserved_mb = torch.cuda.max_memory_reserved(device) / (1024.0 ** 2)
        else:
            round_peak_gpu_allocated_mb = 0.0
            round_peak_gpu_reserved_mb = 0.0

        round_time_sec = time.perf_counter() - round_start_time
        total_round_time_sec += round_time_sec
        total_hessian_time_sec += round_hessian_time_sec
        total_hessian_param_elements += round_hessian_param_elements
        total_peak_gpu_allocated_mb_rounds += round_peak_gpu_allocated_mb
        total_peak_gpu_reserved_mb_rounds += round_peak_gpu_reserved_mb
        if round_hessian_param_elements > 0:
            hessian_active_round_count += 1
            hessian_round_peak_gpu_allocated_mb_sum += round_peak_gpu_allocated_mb
            hessian_round_peak_gpu_reserved_mb_sum += round_peak_gpu_reserved_mb

        completed_regular_hessian_rounds = 1 + round_idx0 // cfg.hessian_interval
        full_hessian_budget_elements = (
            completed_regular_hessian_rounds
            * cfg.num_clients
            * cfg.local_steps
            * all_hessian_params_per_estimate
        )
        hessian_param_budget_ratio = total_hessian_param_elements / max(
            full_hessian_budget_elements, 1
        )
        hessian_param_saving_ratio = 1.0 - hessian_param_budget_ratio
        round_full_hessian_elements = cfg.num_clients * cfg.local_steps * all_hessian_params_per_estimate
        hessian_param_coverage_ratio_round = round_hessian_param_elements / max(
            round_full_hessian_elements, 1
        )
        # This is exactly (1/R) * sum_r rho_H^r when a full-client,
        # full-local-step, full-parameter GNB pass is defined as rho_H^r = 1.
        avg_hessian_param_coverage_ratio_all_rounds = total_hessian_param_elements / max(
            one_based_round * round_full_hessian_elements, 1
        )
        experiment_peak_gpu_allocated_mb = max(
            experiment_peak_gpu_allocated_mb, round_peak_gpu_allocated_mb
        )
        experiment_peak_gpu_reserved_mb = max(
            experiment_peak_gpu_reserved_mb, round_peak_gpu_reserved_mb
        )
        avg_round_time_sec = total_round_time_sec / one_based_round
        avg_peak_gpu_allocated_mb_round = total_peak_gpu_allocated_mb_rounds / one_based_round
        avg_peak_gpu_reserved_mb_round = total_peak_gpu_reserved_mb_rounds / one_based_round
        avg_hessian_round_peak_gpu_allocated_mb = (
            hessian_round_peak_gpu_allocated_mb_sum / max(hessian_active_round_count, 1)
        )
        avg_hessian_round_peak_gpu_reserved_mb = (
            hessian_round_peak_gpu_reserved_mb_sum / max(hessian_active_round_count, 1)
        )
        communication_bytes_round = round_upload_bytes + round_download_bytes
        communication_bytes_total = cumulative_upload_bytes + cumulative_download_bytes

        if target_reached_round == 0 and test_acc >= cfg.target_accuracy:
            target_reached_round = one_based_round
            target_reached_time_sec = total_round_time_sec
            target_reached_communication_bytes = communication_bytes_total

        if algo in ("fedsophia", "dynahybrid"):
            optimizer_state_bytes_total_clients = _optimizer_state_nbytes(
                client_sophia_states
            )
        elif algo == "dynahybrid_adahessian":
            optimizer_state_bytes_total_clients = _adahessian_state_nbytes(
                client_adahessian_states
            )
        elif algo == "dynahybrid_lbfgs":
            optimizer_state_bytes_total_clients = _lbfgs_estimator_state_nbytes(
                client_lbfgs_states
            )
        else:
            optimizer_state_bytes_total_clients = 0

        if algo in dynamic_algorithms:
            row_selected_layers = ";".join(selected_layers)
            row_selected_k = selected_k
        elif algo == "fedsophia":
            row_selected_layers = "ALL"
            row_selected_k = len(get_layers(cfg.model_name, cfg.grouping_mode))
        else:
            row_selected_layers = "NONE"
            row_selected_k = 0

        row = {
            "dataset": cfg.dataset,
            "model": cfg.model_name,
            "algo": algo,
            "partition": cfg.partition,
            "dirichlet_alpha": cfg.dirichlet_alpha,
            "dirichlet_min_size": cfg.dirichlet_min_size,
            "partition_seed": cfg.seed,
            "lr_schedule": cfg.lr_schedule,
            "lr_multiplier": lr_multiplier,
            "sophia_lr_schedule": cfg.sophia_lr_schedule,
            "sophia_lr_multiplier": sophia_lr_multiplier,
            "fedavg_lr_round": round_cfg.fedavg_lr,
            "sophia_lr_round": round_cfg.sophia_lr,
            "round": one_based_round,
            "train_loss": train_loss,
            "test_loss": test_loss,
            "test_acc": test_acc,
            "selected_k": row_selected_k,
            "selected_layers": row_selected_layers,
            "newly_selected_layers": ";".join(newly_selected_layers) if algo in dynamic_algorithms else "",
            "coverage_threshold": cfg.score_coverage_threshold if algo in dynamic_algorithms else "",
            "selection_strategy": cfg.selection_strategy if algo in dynamic_algorithms else "",
            "k_strategy": cfg.k_strategy if algo in dynamic_algorithms else "",
            "fixed_k": cfg.fixed_second_order_layers if algo in dynamic_algorithms and cfg.k_strategy == "fixed" else "",
            "selected_coverage": last_selected_coverage if algo in dynamic_algorithms and selected_k > 0 else "",
            "scores": repr(last_scores) if algo in dynamic_algorithms else "",
            "normalized_scores": repr(last_normalized_scores) if algo in dynamic_algorithms else "",
            "ranked_layers": ";".join(last_ranked_layers) if algo in dynamic_algorithms else "",
            "scoring_grads_collected": collect_scoring_grads,
            "scoring_target_round": next_round if collect_scoring_grads else "",
            "round_time_sec": round_time_sec,
            "elapsed_train_time_sec": total_round_time_sec,
            "avg_round_time_sec": avg_round_time_sec,
            "hessian_time_sec_round": round_hessian_time_sec,
            "hessian_time_sec_total": total_hessian_time_sec,
            "hessian_time_ratio": total_hessian_time_sec / max(total_round_time_sec, 1e-12),
            "hessian_param_elements_round": round_hessian_param_elements,
            "hessian_param_elements_total": total_hessian_param_elements,
            "all_hessian_param_elements_per_estimate": all_hessian_params_per_estimate,
            "hessian_param_coverage_ratio_round": hessian_param_coverage_ratio_round,
            "avg_hessian_param_coverage_ratio_all_rounds": avg_hessian_param_coverage_ratio_all_rounds,
            "full_hessian_budget_elements": full_hessian_budget_elements,
            "hessian_param_budget_ratio": hessian_param_budget_ratio,
            "hessian_param_saving_ratio": hessian_param_saving_ratio,
            "estimated_forward_backward_flops_round": round_flops.forward_backward,
            "estimated_forward_backward_flops_total": cumulative_flops.forward_backward,
            "estimated_hessian_gnb_flops_round": round_flops.hessian_gnb,
            "estimated_hessian_gnb_flops_total": cumulative_flops.hessian_gnb,
            "estimated_hessian_adahessian_flops_round": round_flops.hessian_adahessian,
            "estimated_hessian_adahessian_flops_total": cumulative_flops.hessian_adahessian,
            "estimated_hessian_lbfgs_flops_round": round_flops.hessian_lbfgs,
            "estimated_hessian_lbfgs_flops_total": cumulative_flops.hessian_lbfgs,
            "estimated_optimizer_flops_round": round_flops.optimizer,
            "estimated_optimizer_flops_total": cumulative_flops.optimizer,
            "estimated_scoring_flops_round": round_flops.scoring,
            "estimated_scoring_flops_total": cumulative_flops.scoring,
            "estimated_aggregation_flops_round": round_flops.aggregation,
            "estimated_aggregation_flops_total": cumulative_flops.aggregation,
            "estimated_evaluation_flops_round": round_flops.evaluation,
            "estimated_evaluation_flops_total": cumulative_flops.evaluation,
            "estimated_total_flops_round": round_flops.total,
            "estimated_total_flops_total": cumulative_flops.total,
            "estimated_total_tflops_total": cumulative_flops.total / 1e12,
            "estimated_full_fedsophia_flops_budget_total": cumulative_full_fedsophia_flops.total,
            "estimated_flops_ratio_vs_full_fedsophia": estimated_flops_ratio_vs_full_fedsophia,
            "estimated_flops_saving_vs_full_fedsophia": estimated_flops_saving_vs_full_fedsophia,
            "flops_forward_per_sample": flop_profile.forward_per_sample,
            "flops_counting_convention": "2_per_mac_backward_2x_forward_selected_gnb_suffix",
            "streaming_score_clients": (
                current_score_moments.client_count if current_score_moments is not None else 0
            ),
            "streaming_score_retained_bytes": (
                current_score_moments.stored_tensor_bytes()
                if current_score_moments is not None else 0
            ),
            "peak_gpu_allocated_mb_round": round_peak_gpu_allocated_mb,
            "peak_gpu_reserved_mb_round": round_peak_gpu_reserved_mb,
            "peak_gpu_allocated_mb_experiment": experiment_peak_gpu_allocated_mb,
            "peak_gpu_reserved_mb_experiment": experiment_peak_gpu_reserved_mb,
            "avg_peak_gpu_allocated_mb_round": avg_peak_gpu_allocated_mb_round,
            "avg_peak_gpu_reserved_mb_round": avg_peak_gpu_reserved_mb_round,
            "avg_hessian_round_peak_gpu_allocated_mb": avg_hessian_round_peak_gpu_allocated_mb,
            "avg_hessian_round_peak_gpu_reserved_mb": avg_hessian_round_peak_gpu_reserved_mb,
            "peak_process_rss_mb": _peak_process_rss_mb(),
            "optimizer_state_bytes_total_clients": optimizer_state_bytes_total_clients,
            "lbfgs_history_pairs_total": (
                sum(len(client_state.history.s) for client_state in client_lbfgs_states)
                if algo == "dynahybrid_lbfgs" else 0
            ),
            "lbfgs_history_elements_total": (
                sum(client_state.stored_pair_elements() for client_state in client_lbfgs_states)
                if algo == "dynahybrid_lbfgs" else 0
            ),
            "lbfgs_damped_pairs_total": (
                sum(client_state.history.damped_pairs for client_state in client_lbfgs_states)
                if algo == "dynahybrid_lbfgs" else 0
            ),
            "lbfgs_rejected_pairs_total": (
                sum(client_state.history.rejected_pairs for client_state in client_lbfgs_states)
                if algo == "dynahybrid_lbfgs" else 0
            ),
            "model_payload_bytes": model_payload_bytes,
            "upload_bytes_round": round_upload_bytes,
            "download_bytes_round": round_download_bytes,
            "communication_bytes_round": communication_bytes_round,
            "upload_bytes_total": cumulative_upload_bytes,
            "download_bytes_total": cumulative_download_bytes,
            "communication_bytes_total": communication_bytes_total,
            "target_accuracy": cfg.target_accuracy,
            "target_reached": target_reached_round > 0,
            "rounds_to_target": target_reached_round,
            "time_to_target_sec": target_reached_time_sec,
            "communication_to_target_bytes": target_reached_communication_bytes,
        }
        rows.append(row)


        # 每轮结束立即写入CSV
        if cfg.save_csv:

            out_csv = os.path.join(
                cfg.output_dir,
                f"{algo}_metrics.csv"
            )

            append_metric_csv(
                out_csv,
                row
            )

            # Rewrite a one-row summary after every completed round. If a job is
            # interrupted, this file still describes all completed work.
            summary_row = {
                "dataset": cfg.dataset,
                "model": cfg.model_name,
                "algo": algo,
                "partition": cfg.partition,
                "dirichlet_alpha": cfg.dirichlet_alpha,
                "dirichlet_min_size": cfg.dirichlet_min_size,
                "partition_seed": cfg.seed,
                "completed_rounds": one_based_round,
                "planned_rounds": cfg.num_rounds,
                "is_complete": one_based_round == cfg.num_rounds,
                "total_train_time_sec": total_round_time_sec,
                "avg_round_time_sec": avg_round_time_sec,
                "total_hessian_time_sec": total_hessian_time_sec,
                "hessian_time_ratio": total_hessian_time_sec / max(total_round_time_sec, 1e-12),
                "hessian_param_elements_total": total_hessian_param_elements,
                "all_hessian_param_elements_per_estimate": all_hessian_params_per_estimate,
                "hessian_param_coverage_ratio_last_round": hessian_param_coverage_ratio_round,
                "avg_hessian_param_coverage_ratio_all_rounds": avg_hessian_param_coverage_ratio_all_rounds,
                "full_hessian_budget_elements": full_hessian_budget_elements,
                "hessian_param_budget_ratio": hessian_param_budget_ratio,
                "hessian_param_saving_ratio": hessian_param_saving_ratio,
                "estimated_forward_backward_flops_total": cumulative_flops.forward_backward,
                "estimated_hessian_gnb_flops_total": cumulative_flops.hessian_gnb,
                "estimated_hessian_adahessian_flops_total": cumulative_flops.hessian_adahessian,
                "estimated_hessian_lbfgs_flops_total": cumulative_flops.hessian_lbfgs,
                "estimated_optimizer_flops_total": cumulative_flops.optimizer,
                "estimated_scoring_flops_total": cumulative_flops.scoring,
                "estimated_aggregation_flops_total": cumulative_flops.aggregation,
                "estimated_evaluation_flops_total": cumulative_flops.evaluation,
                "estimated_total_flops_total": cumulative_flops.total,
                "estimated_total_tflops_total": cumulative_flops.total / 1e12,
                "estimated_full_fedsophia_flops_budget_total": cumulative_full_fedsophia_flops.total,
                "estimated_flops_ratio_vs_full_fedsophia": estimated_flops_ratio_vs_full_fedsophia,
                "estimated_flops_saving_vs_full_fedsophia": estimated_flops_saving_vs_full_fedsophia,
                "flops_forward_per_sample": flop_profile.forward_per_sample,
                "flops_counting_convention": "2_per_mac_backward_2x_forward_selected_gnb_suffix",
                "streaming_score_retained_bytes": (
                    current_score_moments.stored_tensor_bytes()
                    if current_score_moments is not None else 0
                ),
                "peak_gpu_allocated_mb": experiment_peak_gpu_allocated_mb,
                "peak_gpu_reserved_mb": experiment_peak_gpu_reserved_mb,
                "avg_peak_gpu_allocated_mb_round": avg_peak_gpu_allocated_mb_round,
                "avg_peak_gpu_reserved_mb_round": avg_peak_gpu_reserved_mb_round,
                "avg_hessian_round_peak_gpu_allocated_mb": avg_hessian_round_peak_gpu_allocated_mb,
                "avg_hessian_round_peak_gpu_reserved_mb": avg_hessian_round_peak_gpu_reserved_mb,
                "peak_process_rss_mb": _peak_process_rss_mb(),
                "optimizer_state_bytes_total_clients": optimizer_state_bytes_total_clients,
                "model_payload_bytes": model_payload_bytes,
                "upload_bytes_total": cumulative_upload_bytes,
                "download_bytes_total": cumulative_download_bytes,
                "communication_bytes_total": communication_bytes_total,
                "target_accuracy": cfg.target_accuracy,
                "target_reached": target_reached_round > 0,
                "rounds_to_target": target_reached_round,
                "time_to_target_sec": target_reached_time_sec,
                "communication_to_target_bytes": target_reached_communication_bytes,
            }
            write_metrics_csv(summary_csv, [summary_row])

        if cfg.print_every_round:
            print(
                f"[{algo}] Round {one_based_round:03d}/{cfg.num_rounds} | "
                f"train_loss={train_loss:.4f} | test_loss={test_loss:.4f} | "
                f"test_acc={test_acc * 100:.2f}% | K={row['selected_k']} | "
                f"layers={row['selected_layers']} | round_time={round_time_sec:.2f}s | "
                f"hessian_time={round_hessian_time_sec:.2f}s | "
                f"peak_gpu={round_peak_gpu_allocated_mb:.0f}MiB"
            )

        if algo in dynamic_algorithms and collect_scoring_grads:
            print(
                f"[DynaHybrid] Round {one_based_round}: collected online-mean "
                f"layer gradients for selection at round {next_round}."
            )

        prev_score_moments = current_score_moments


    return rows
