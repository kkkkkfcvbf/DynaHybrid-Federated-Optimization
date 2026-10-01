"""
dynahybrid_adahessian.py

DynaHybrid local optimizer with an AdaHessian-style Hutchinson diagonal
Hessian estimator. The optimizer update is intentionally identical to the
archived DynaHybrid-Sophia path; only the local h_hat estimator changes.

Fed-Sophia update:
    m_i^k = beta1 * m_i^{k-1} + (1 - beta1) * g_i^k
    if k mod tau == 0:
        h_hat_i^k = Estimator(theta_i^k) using Hutchinson diag(H)
        h_i^k = beta2 * h_i^{k-tau} + (1 - beta2) * h_hat_i^k
    else:
        h_i^k = h_i^{k-1}
    theta_i^k = theta_i^k - eta * lambda * theta_i^k
    theta_i^{k+1} = theta_i^k - eta * clip(m_i^k / max(h_i^k, epsilon), rho)

AdaHessian/Hutchinson diagonal estimator:
    logits = phi(theta, x_b)
    loss = CE(logits, y_b)
    g = grad(loss, theta)
    draw Rademacher vector v
    Hv = grad(<g, v>, theta)
    return |v ⊙ Hv|
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Set, Tuple
import time

import torch
import torch.nn as nn

from config import ExperimentConfig, get_layer_name, get_layers
from algorithms.fedavg import (
    LayerGradDict,
    TensorDict,
    collect_layer_grad_vectors,
    detach_layer_grad_dict_cpu,
    detach_state_dict_cpu,
    infinite_loader,
    update_layer_grad_running_mean,
)


@dataclass
class AdaHessianClientState:
    """Per-client EMA states m and h. Stored on CPU between local trainings."""
    m: Dict[str, torch.Tensor] = field(default_factory=dict)
    h: Dict[str, torch.Tensor] = field(default_factory=dict)

    def ensure_initialized(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            if name not in self.m:
                self.m[name] = torch.zeros_like(param.detach(), device="cpu")
            if name not in self.h:
                self.h[name] = torch.zeros_like(param.detach(), device="cpu")

    def to_device_dicts(self, model: nn.Module, device: torch.device) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        self.ensure_initialized(model)
        m_dev: Dict[str, torch.Tensor] = {}
        h_dev: Dict[str, torch.Tensor] = {}
        for name, _ in model.named_parameters():
            m_dev[name] = self.m[name].to(device=device, non_blocking=True)
            h_dev[name] = self.h[name].to(device=device, non_blocking=True)
        return m_dev, h_dev

    def save_from_device_dicts(self, m_dev: Dict[str, torch.Tensor], h_dev: Dict[str, torch.Tensor]) -> None:
        for name, tensor in m_dev.items():
            self.m[name] = tensor.detach().cpu().clone()
        for name, tensor in h_dev.items():
            self.h[name] = tensor.detach().cpu().clone()


def _layer_allowed(name: str, cfg: ExperimentConfig, layers: Optional[Set[str]]) -> bool:
    if layers is None:
        return True
    layer = get_layer_name(name, cfg.model_name, cfg.grouping_mode)
    return layer in layers


def _rademacher_like(tensor: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(tensor).bernoulli_(0.5).mul_(2.0).sub_(1.0)


def hutchinson_hessian_diag(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    cfg: ExperimentConfig,
    device: torch.device,
    hessian_layers: Optional[Set[str]] = None,
) -> Dict[str, torch.Tensor]:
    """
    Hutchinson diagonal Hessian estimator used by AdaHessian-style optimizers.
    hessian_layers=None means estimate for all trainable parameters.

    The raw Hutchinson estimate is v * (H v). Neural-network Hessians may be
    indefinite, while the Sophia denominator h must be non-negative. Following
    AdaHessian practice, this function returns the diagonal magnitude.
    """
    model.train()
    hessian_batch_size = int(
        getattr(cfg, "adahessian_hessian_batch_size", 0) or 0
    )
    if hessian_batch_size > 0 and hessian_batch_size < int(x.size(0)):
        x = x[:hessian_batch_size]
        y = y[:hessian_batch_size]

    named_params: List[Tuple[str, torch.Tensor]] = []
    for name, param in model.named_parameters():
        if param.requires_grad and _layer_allowed(name, cfg, hessian_layers):
            named_params.append((name, param))

    if len(named_params) == 0:
        return {}

    params = [p for _, p in named_params]

    samples = int(getattr(cfg, "adahessian_hutchinson_samples", 1) or 1)
    samples = max(samples, 1)
    accum = {name: torch.zeros_like(param.detach()) for name, param in named_params}

    for sample_idx in range(samples):
        logits = model(x)
        loss = nn.functional.cross_entropy(logits, y, reduction="mean")
        grads = torch.autograd.grad(
            loss,
            params,
            retain_graph=True,
            create_graph=True,
            allow_unused=True,
        )
        vectors = [
            _rademacher_like(param) if grad is not None else torch.zeros_like(param)
            for (_, param), grad in zip(named_params, grads)
        ]
        grad_dot_vector = sum(
            (grad * vector).sum()
            for grad, vector in zip(grads, vectors)
            if grad is not None
        )
        if not torch.is_tensor(grad_dot_vector):
            continue
        hvps = torch.autograd.grad(
            grad_dot_vector,
            params,
            retain_graph=sample_idx != samples - 1,
            create_graph=False,
            allow_unused=True,
        )
        for (name, param), vector, hvp in zip(named_params, vectors, hvps):
            if hvp is None:
                estimate = torch.zeros_like(param.detach())
            else:
                estimate = (hvp.detach() * vector.detach()).abs()
            accum[name].add_(estimate)

    h_hat: Dict[str, torch.Tensor] = {}
    for name, value in accum.items():
        h_hat[name] = value.div(float(samples))
    return h_hat


def train_one_client_adahessian_or_hybrid(
    model: nn.Module,
    client_loader: Iterable[Tuple[torch.Tensor, torch.Tensor]],
    state: AdaHessianClientState,
    cfg: ExperimentConfig,
    device: torch.device,
    round_idx0: int,
    sophia_layers: Optional[Set[str]],
    sgd_for_non_sophia: bool,
    force_sgd_all_params: bool = False,
    hessian_layers_override: Optional[Set[str]] = None,
    newly_selected_layers: Optional[Set[str]] = None,
    collect_scoring_grads: bool = False,
    hessian_every_local_step_on_refresh: bool = False,
) -> Tuple[TensorDict, float, LayerGradDict, float, int]:
    """
    Local training for DynaHybrid-AdaHessian.

    Args:
        sophia_layers:
            None means all layers use Fed-Sophia.
            Empty set means no layer uses Fed-Sophia.
            Non-empty set means only these layer groups use Fed-Sophia.
        sgd_for_non_sophia:
            True for DynaHybrid, False for full Fed-Sophia.
        force_sgd_all_params:
            Used by DynaHybrid round 1: all parameters are updated by SGD even if h/m warm-up is enabled.
        hessian_layers_override:
            Controls which layers Hutchinson diag(H) is computed for on Hessian-refresh rounds.
            None means same as sophia_layers if sophia_layers is not None, otherwise all layers.
        newly_selected_layers:
            DynaHybrid layers entering the selected set this round. They receive an
            immediate Hessian estimate even outside the regular refresh schedule.
        hessian_every_local_step_on_refresh:
            True evaluates Hutchinson diag(H) at every local step of a
            Hessian-refresh communication round. DynaHybrid leaves this False
            and uses its one-estimate-per-refresh cost-saving design.
    """
    model.train()
    state.ensure_initialized(model)
    m_dev, h_dev = state.to_device_dicts(model, device)

    loss_fn = nn.CrossEntropyLoss()
    batch_iter = infinite_loader(client_loader)
    total_loss = 0.0
    total_seen = 0
    layer_grad_mean: LayerGradDict = {}
    layer_grad_count = 0
    hessian_time_sec = 0.0
    hessian_param_elements = 0
    hessian_cuda_events: List[Tuple[torch.cuda.Event, torch.cuda.Event]] = []
    newly_selected_layers = newly_selected_layers or set()

    hessian_refresh_round = (round_idx0 % cfg.hessian_interval == 0)
    immediate_new_layer_refresh = len(newly_selected_layers) > 0
    should_compute_hessian = hessian_refresh_round or immediate_new_layer_refresh

    for _local_step in range(cfg.local_steps):
        x, y = next(batch_iter)
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        model.zero_grad(set_to_none=True)
        logits = model(x)
        loss = loss_fn(logits, y)
        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"Non-finite local loss before update at round {round_idx0 + 1}, "
                f"local step {_local_step + 1}."
            )
        loss.backward()
        if cfg.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), cfg.max_grad_norm, error_if_nonfinite=True
            )

        # Only DynaHybrid rounds immediately preceding a selection round need
        # scoring gradients. m still uses param.grad on every local step.
        if collect_scoring_grads:
            layer_grad_count += 1
            update_layer_grad_running_mean(
                layer_grad_mean,
                collect_layer_grad_vectors(model, cfg, move_to_cpu=False),
                layer_grad_count,
            )

        # DynaHybrid-AdaHessian evaluates Hutchinson diag(H) only at local step
        # 0 by default and reuses h for the remaining local steps.
        h_hat: Dict[str, torch.Tensor] = {}
        compute_hessian_this_step = should_compute_hessian and (
            hessian_every_local_step_on_refresh or _local_step == 0
        )
        if compute_hessian_this_step:
            if immediate_new_layer_refresh and not hessian_refresh_round:
                # Between regular refreshes, compute only for layers that have just
                # entered the selected set.
                hessian_layers = newly_selected_layers
            elif hessian_layers_override is not None:
                hessian_layers = hessian_layers_override
            elif sophia_layers is None:
                hessian_layers = None  # all layers
            else:
                hessian_layers = sophia_layers

            if hessian_layers is None or len(hessian_layers) > 0:
                # CUDA kernels are asynchronous. CUDA events measure the actual GPU
                # elapsed time of the complete estimator; perf_counter is used on CPU.
                if device.type == "cuda":
                    hessian_start = torch.cuda.Event(enable_timing=True)
                    hessian_end = torch.cuda.Event(enable_timing=True)
                    hessian_start.record()
                    h_hat = hutchinson_hessian_diag(
                        model, x, y, cfg, device, hessian_layers=hessian_layers
                    )
                    hessian_param_elements += sum(tensor.numel() for tensor in h_hat.values())
                    hessian_end.record()
                    hessian_cuda_events.append((hessian_start, hessian_end))
                else:
                    hessian_start_time = time.perf_counter()
                    h_hat = hutchinson_hessian_diag(
                        model, x, y, cfg, device, hessian_layers=hessian_layers
                    )
                    hessian_param_elements += sum(tensor.numel() for tensor in h_hat.values())
                    hessian_time_sec += time.perf_counter() - hessian_start_time

        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.grad is None:
                    continue

                layer = get_layer_name(name, cfg.model_name, cfg.grouping_mode)
                is_sophia_param = (
                    sophia_layers is None or (layer in sophia_layers if layer is not None else False)
                )

                # In DynaHybrid, keep m for all layers if enabled, so newly selected layers do not start from zero.
                should_update_m = is_sophia_param or (
                    sgd_for_non_sophia and cfg.hybrid_track_m_for_all_layers
                )
                if should_update_m:
                    # m_i^k = beta1 * m_i^{k-1} + (1 - beta1) * g_i^k
                    m_dev[name].mul_(cfg.beta1).add_(param.grad.detach(), alpha=(1.0 - cfg.beta1))

                # h_i^k = beta2 * h_i^{k-tau} + (1 - beta2) * h_hat_i^k on k mod tau == 0.
                if compute_hessian_this_step and name in h_hat:
                    if not torch.isfinite(h_hat[name]).all():
                        raise FloatingPointError(
                            f"Non-finite Hessian estimate for {name!r} at round {round_idx0 + 1}."
                        )
                    if layer in newly_selected_layers:
                        # Do not mix a newly selected layer with a stale/zero history.
                        h_dev[name].copy_(h_hat[name])
                    else:
                        h_dev[name].mul_(cfg.beta2).add_(h_hat[name], alpha=(1.0 - cfg.beta2))

                if force_sgd_all_params:
                    # DynaHybrid first round uses SGD parameter update.
                    # Add the same FedAvg weight decay so first-order layers are comparable.
                    if cfg.weight_decay != 0.0:
                        param.add_(param.detach(), alpha=-cfg.fedavg_lr * cfg.weight_decay)
                    param.add_(param.grad.detach(), alpha=-cfg.fedavg_lr)
                    continue

                if is_sophia_param:
                    # theta_i^k = theta_i^k - eta * lambda * theta_i^k
                    if cfg.weight_decay != 0.0:
                        param.add_(param.detach(), alpha=-cfg.sophia_lr * cfg.weight_decay)

                    # theta_i^{k+1} = theta_i^k - eta * clip(m / max(h, eps), rho)
                    denom = torch.maximum(h_dev[name], torch.full_like(h_dev[name], cfg.epsilon))
                    step = torch.clamp(m_dev[name] / denom, min=-cfg.rho, max=cfg.rho)
                    param.add_(step, alpha=-cfg.sophia_lr)
                else:
                    if sgd_for_non_sophia:
                        # Other layers are first-order FedAvg SGD with the same weight decay.
                        if cfg.weight_decay != 0.0:
                            param.add_(param.detach(), alpha=-cfg.fedavg_lr * cfg.weight_decay)
                        param.add_(param.grad.detach(), alpha=-cfg.fedavg_lr)
                    else:
                        raise RuntimeError("Non-Sophia parameter encountered while sgd_for_non_sophia=False")

                if not torch.isfinite(param).all():
                    raise FloatingPointError(
                        f"Non-finite parameter {name!r} after round {round_idx0 + 1}, "
                        f"local step {_local_step + 1}."
                    )

        bs = int(x.size(0))
        total_loss += float(loss.item()) * bs
        total_seen += bs

    if hessian_cuda_events:
        # One synchronization per client avoids perturbing training with a blocking
        # synchronization after every local Hessian estimate.
        torch.cuda.synchronize(device)
        hessian_time_sec += sum(
            start.elapsed_time(end) for start, end in hessian_cuda_events
        ) / 1000.0

    # DynaHybrid scoring keeps the 40-step running mean on the training device.
    # Transfer only the final per-layer mean once per client, instead of moving
    # a full model gradient from GPU to CPU at every local step.
    if collect_scoring_grads:
        layer_grad_mean = detach_layer_grad_dict_cpu(layer_grad_mean)

    state.save_from_device_dicts(m_dev, h_dev)
    return (
        detach_state_dict_cpu(model),
        total_loss / max(total_seen, 1),
        layer_grad_mean,
        hessian_time_sec,
        hessian_param_elements,
    )


def train_one_client_dynahybrid_adahessian(
    model: nn.Module,
    client_loader: Iterable[Tuple[torch.Tensor, torch.Tensor]],
    state: AdaHessianClientState,
    cfg: ExperimentConfig,
    device: torch.device,
    round_idx0: int,
    selected_adahessian_layers: List[str],
    newly_selected_adahessian_layers: List[str],
    collect_scoring_grads: bool = False,
) -> Tuple[TensorDict, float, LayerGradDict, float, int]:
    """Run DynaHybrid with Hutchinson diag(H) replacing GNB h_hat."""
    selected_set: Set[str] = set(selected_adahessian_layers)
    newly_selected_set: Set[str] = set(newly_selected_adahessian_layers)
    all_layers: Set[str] = set(get_layers(cfg.model_name, cfg.grouping_mode))
    force_sgd = round_idx0 == 0

    if cfg.hybrid_refresh_hessian_for_all_layers:
        hessian_override: Optional[Set[str]] = all_layers
        newly_selected_set = set()
    else:
        hessian_override = selected_set
    if force_sgd and not cfg.hybrid_warmup_hessian_round1:
        hessian_override = set()

    return train_one_client_adahessian_or_hybrid(
        model=model,
        client_loader=client_loader,
        state=state,
        cfg=cfg,
        device=device,
        round_idx0=round_idx0,
        sophia_layers=selected_set,
        sgd_for_non_sophia=True,
        force_sgd_all_params=force_sgd,
        hessian_layers_override=hessian_override,
        newly_selected_layers=newly_selected_set,
        collect_scoring_grads=collect_scoring_grads,
        hessian_every_local_step_on_refresh=(
            cfg.hybrid_hessian_every_local_step_on_refresh
        ),
    )
