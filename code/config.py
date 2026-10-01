"""
config.py

Configuration and architecture-aware group utilities for the archived paper.

All experimental hyperparameters are supplied explicitly by ``main.py`` command-line
arguments.  The ``None`` values below are deliberate: this file must not silently
choose learning rates, optimizer coefficients, training lengths, or selection budgets.
Each run saves the fully resolved values to ``<output_dir>/config.json`` before training.

DynaHybrid dynamically determines the number K of second-order layers every fixed number
of rounds from round 2 onward. Raw layer scores are normalized into proportions, sorted in
descending order, and the smallest K whose cumulative contribution reaches gamma is
selected. K is constrained to [min_second_order_layers, max_second_order_layers].
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
import json
import os
import random

import numpy as np
import torch


@dataclass
class ExperimentConfig:
    # ---------- basic ----------
    seed: Optional[int] = None
    data_dir: Optional[str] = None
    output_dir: Optional[str] = None
    device: Optional[str] = None
    num_workers: Optional[int] = None

    # ---------- dataset / FL setting ----------
    dataset: Optional[str] = None
    model_name: Optional[str] = None
    num_classes: Optional[int] = None
    num_clients: Optional[int] = None
    partition: Optional[str] = None
    iid: Optional[bool] = None
    # Dirichlet label-skew partition parameters. They are required only when
    # partition="dirichlet" and are recorded in config.json for reproducibility.
    dirichlet_alpha: Optional[float] = None
    dirichlet_min_size: Optional[int] = None
    train_subset: Optional[int] = None
    test_subset: Optional[int] = None
    dataset_download: bool = False

    # ---------- communication / local training ----------
    num_rounds: Optional[int] = None
    # Every client still performs 40 ordinary optimization steps per round.
    # On a Hessian-active round, only local step 1 estimates Hessian; the other
    # 39 steps reuse the refreshed h.
    local_steps: Optional[int] = None
    batch_size: Optional[int] = None
    test_batch_size: Optional[int] = None
    image_size: Optional[int] = None

    # ---------- FedAvg SGD update ----------
    # With weight decay, the first-order update is:
    #   theta <- theta - fedavg_lr * (g + weight_decay * theta)
    # equivalently, in the implementation below:
    #   theta <- theta - fedavg_lr * weight_decay * theta
    #   theta <- theta - fedavg_lr * g
    fedavg_lr: Optional[float] = None
    fedprox_mu: Optional[float] = None
    # Optional global gradient-norm clipping. None preserves the original
    # update; pass a finite threshold only when explicitly required.
    max_grad_norm: Optional[float] = None
    # Learning-rate schedule.
    # "constant": no decay.
    # "multistep": decay after the specified communication rounds.
    lr_schedule: Optional[str] = None
    lr_milestones: Optional[Tuple[int, ...]] = None
    lr_decay_gamma: Optional[float] = None

    # ---------- Fed-Sophia hyperparameters ----------
    sophia_lr: Optional[float] = None
    # Keep the second-order schedule independent from the first-order schedule.
    # This lets ResNet FedAvg/FedProx (and Hybrid's SGD layers) decay while
    # full FedSophia and Hybrid's selected Sophia layers keep a fixed eta.
    sophia_lr_schedule: Optional[str] = None

    # Shared weight decay coefficient lambda.
    # FedAvg uses theta <- theta - eta * (g + lambda * theta).
    # Fed-Sophia follows Algorithm 1 line 15: theta <- theta - eta * lambda * theta,
    # then Algorithm 1 line 16: theta <- theta - eta * clip(m / max(h, epsilon), rho).
    weight_decay: Optional[float] = None
    beta1: Optional[float] = None
    beta2: Optional[float] = None
    epsilon: Optional[float] = None
    rho: Optional[float] = None
    # Selected from the interval ablation (B setting). With zero-based indexing,
    # tau=10 refreshes at communication rounds 1, 11, 21, ...
    hessian_interval: Optional[int] = None

    # ---------- L-BFGS curvature-estimator hyperparameters ----------
    # These control only the construction of diag(B_k). Selected-layer
    # parameter updates still use sophia_lr, beta1, beta2, epsilon and rho.
    lbfgs_history_size: Optional[int] = None
    lbfgs_curvature_eps: Optional[float] = None
    lbfgs_damping: Optional[bool] = None
    adahessian_hutchinson_samples: Optional[int] = None
    adahessian_hessian_batch_size: Optional[int] = None

    # ---------- DynaHybrid dynamic layer selection ----------
    grouping_mode: Optional[str] = None

    # On scheduled selection rounds, use the previous round's client-mean
    # gradients to compute the scale-normalized cross-client dispersion:
    #   g_bar_l = sum_k p_k g_k_l
    #   score_l = sum_k p_k ||g_k_l - g_bar_l||^2
    #             / (||g_bar_l||^2 + epsilon_score)
    # The same p_k=n_k/sum_j n_j weights are used for scoring and aggregation.
    # Under an equal-size IID partition this reduces exactly to 1/num_clients.
    # The implementation uses streaming sufficient statistics, so it does not
    # retain every client's full gradient vector simultaneously.
    # Then normalized_score_l = score_l / sum_j score_j.
    # Then sort normalized scores in descending order and choose the smallest K such that
    # their cumulative contribution reaches score_coverage_threshold.
    first_score_round: Optional[int] = None
    score_interval: Optional[int] = None
    score_coverage_threshold: Optional[float] = None
    min_second_order_layers: Optional[int] = None
    max_second_order_layers: Optional[int] = None

    # Ablation controls. The paper's default method is dynamic scoring with
    # dynamic K. ``fixed`` freezes the score-based selection made at round 2;
    # ``random`` redraws K layers at each selection round using a local,
    # deterministic RNG. ``fixed`` K keeps the score ranking but always takes
    # exactly fixed_second_order_layers layers.
    selection_strategy: Optional[str] = None
    k_strategy: Optional[str] = None
    fixed_second_order_layers: Optional[int] = None
    manual_second_order_layers: Tuple[str, ...] = ()

    # Number of curvature estimates within each Hessian-refresh event.
    # False: estimate once at local step 1 and reuse h for the remaining steps.
    # True: estimate at every local step (40 estimates when local_steps=40).
    # DynaHybrid-Sophia uses GNB; DynaHybrid-LBFGS uses same-batch probes.
    hybrid_hessian_every_local_step_on_refresh: Optional[bool] = None

    # If True, DynaHybrid keeps the gradient EMA m for all layers even when a layer is currently using SGD.
    # This makes a newly selected layer have a meaningful m state instead of starting from zero.
    hybrid_track_m_for_all_layers: Optional[bool] = None

    # False is the paper setting: scheduled GNB refreshes are restricted to the
    # currently selected groups. A group newly entering the selected set is
    # refreshed immediately before its first curvature-aware update.
    hybrid_refresh_hessian_for_all_layers: Optional[bool] = None

    # False is the paper setting: round 1 uses only SGD and performs no GNB
    # warm-up. Newly selected groups receive an immediate refresh when needed.
    hybrid_warmup_hessian_round1: Optional[bool] = None

    # ---------- logging ----------
    print_every_round: Optional[bool] = None
    save_csv: Optional[bool] = None
    # Efficiency is compared at the same accuracy target. 0.84 means 84%.
    target_accuracy: Optional[float] = None


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_config(cfg: ExperimentConfig, path: str) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, ensure_ascii=False, indent=2)


# -------------------- architecture-aware layer grouping --------------------

MODEL_LAYERS: Dict[str, List[str]] = {
    "alexnet": ["conv1", "conv2", "conv3", "conv4", "conv5", "fc1", "fc2", "fc3"],
    "resnet18": ["stem", "layer1", "layer2", "layer3", "layer4", "classifier"],
    "googlenet": [
        "stem", "inception3", "inception4_early",
        "inception4_late", "inception5", "classifier",
    ],
}


def get_layers(model_name: str, grouping_mode: str) -> List[str]:
    """Return deterministic optimization groups for the selected architecture."""
    model_name = model_name.lower()
    if model_name not in MODEL_LAYERS:
        raise ValueError(
            f"Unknown model_name={model_name!r}; use alexnet, resnet18, or googlenet."
        )
    if model_name == "alexnet" and grouping_mode == "7groups":
        return ["conv1", "conv2", "conv3", "conv4", "conv5", "fc1", "fc2_fc3"]
    return list(MODEL_LAYERS[model_name])


def get_layer_name(
    param_name: str,
    model_name: str,
    grouping_mode: str,
) -> Optional[str]:
    """Map a named parameter to one architecture-level optimization group."""
    model_name = model_name.lower()
    if model_name == "alexnet":
        mapping = {
            "features.0.": "conv1", "features.3.": "conv2",
            "features.6.": "conv3", "features.8.": "conv4",
            "features.10.": "conv5", "classifier.1.": "fc1",
        }
        for prefix, layer in mapping.items():
            if param_name.startswith(prefix):
                return layer
        if param_name.startswith("classifier.4."):
            return "fc2" if grouping_mode == "8groups" else "fc2_fc3"
        if param_name.startswith("classifier.6."):
            return "fc3" if grouping_mode == "8groups" else "fc2_fc3"
    elif model_name == "resnet18":
        if param_name.startswith(("conv1.", "bn1.")):
            return "stem"
        for layer in ("layer1", "layer2", "layer3", "layer4"):
            if param_name.startswith(layer + "."):
                return layer
        if param_name.startswith("fc."):
            return "classifier"
    elif model_name == "googlenet":
        if param_name.startswith(("conv1.", "conv2.", "conv3.", "maxpool")):
            return "stem"
        if param_name.startswith(("inception3a.", "inception3b.")):
            return "inception3"
        if param_name.startswith(("inception4a.", "inception4b.", "inception4c.")):
            return "inception4_early"
        if param_name.startswith(("inception4d.", "inception4e.")):
            return "inception4_late"
        if param_name.startswith(("inception5a.", "inception5b.")):
            return "inception5"
        if param_name.startswith("fc."):
            return "classifier"
    return None
