"""FedProx local update derived directly from this archive's FedAvg implementation."""

from typing import Iterable, Tuple
import torch
import torch.nn as nn
from config import ExperimentConfig
from algorithms.fedavg import (
    LayerGradDict, TensorDict, collect_layer_grad_vectors,
    detach_state_dict_cpu, infinite_loader, update_layer_grad_running_mean,
)

def train_one_client_fedprox(
    model: nn.Module, client_loader: Iterable[Tuple[torch.Tensor, torch.Tensor]],
    cfg: ExperimentConfig, device: torch.device,
    collect_scoring_grads: bool = False,
) -> Tuple[TensorDict, float, LayerGradDict]:
    """FedAvg SGD plus (mu/2)||theta-theta_global||^2 on each client."""
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    batch_iter = infinite_loader(client_loader)
    global_params = {name: p.detach().clone() for name, p in model.named_parameters()}
    total_loss, total_seen = 0.0, 0
    layer_grad_mean: LayerGradDict = {}
    layer_grad_count = 0
    for _ in range(cfg.local_steps):
        x, y = next(batch_iter)
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        model.zero_grad(set_to_none=True)
        loss = loss_fn(model(x), y)
        if not torch.isfinite(loss):
            raise FloatingPointError("FedProx encountered a non-finite local loss.")
        loss.backward()
        if cfg.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), cfg.max_grad_norm, error_if_nonfinite=True
            )
        if collect_scoring_grads:
            layer_grad_count += 1
            update_layer_grad_running_mean(
                layer_grad_mean,
                collect_layer_grad_vectors(model, cfg),
                layer_grad_count,
            )
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.grad is None:
                    continue
                # Evaluate every term at the same pre-update parameter value:
                # theta <- (1-eta*lambda) theta
                #          - eta*g - eta*mu*(theta-theta_global).
                proximal_delta = param.detach().clone() - global_params[name]
                if cfg.weight_decay != 0.0:
                    param.mul_(1.0 - cfg.fedavg_lr * cfg.weight_decay)
                if cfg.fedprox_mu != 0.0:
                    param.add_(proximal_delta, alpha=-cfg.fedavg_lr * cfg.fedprox_mu)
                param.add_(param.grad.detach(), alpha=-cfg.fedavg_lr)
        bs = int(x.size(0))
        total_loss += float(loss.item()) * bs
        total_seen += bs
    return detach_state_dict_cpu(model), total_loss / max(total_seen, 1), layer_grad_mean
