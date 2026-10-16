"""
fedavg.py

Manual FedAvg client update and server aggregation.
The first-order local update supports weight decay:
    theta <- theta - eta * (g + lambda * theta)
which is implemented as two explicit steps:
    theta <- theta - eta * lambda * theta
    theta <- theta - eta * g
The server only performs weighted parameter averaging and does not train.
"""

from __future__ import annotations

from typing import Dict, Iterable, Iterator, List, Tuple

import torch
import torch.nn as nn

from config import ExperimentConfig, get_layer_name, get_layers

TensorDict = Dict[str, torch.Tensor]
LayerGradDict = Dict[str, torch.Tensor]


def infinite_loader(loader: Iterable[Tuple[torch.Tensor, torch.Tensor]]) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
    """Cycle through a PyTorch DataLoader forever."""
    while True:
        for batch in loader:
            yield batch


def detach_state_dict_cpu(model: nn.Module) -> TensorDict:
    """Return a CPU copy of model parameters/buffers."""
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
#保存客户端训练后的模型参数

def aggregate_state_dicts(client_states: List[TensorDict], client_weights: List[float]) -> TensorDict:
    """Weighted FedAvg aggregation on the server."""
    if len(client_states) == 0:
        raise ValueError("client_states is empty")
    if len(client_states) != len(client_weights):
        raise ValueError("client_states and client_weights must have the same length")

    total = float(sum(client_weights))
    normalized = [w / total for w in client_weights]
    keys = client_states[0].keys()

    avg_state: TensorDict = {}
    for key in keys:
        first = client_states[0][key]
        if not torch.is_floating_point(first):
            # For integer buffers, e.g. num_batches_tracked, copy from the first client.
            avg_state[key] = first.clone()
            continue
        acc = torch.zeros_like(first, dtype=first.dtype)
        for state, weight in zip(client_states, normalized):
            acc.add_(state[key].to(dtype=first.dtype), alpha=float(weight))
        avg_state[key] = acc
    return avg_state
#服务器聚合参数

def collect_layer_grad_vectors(
    model: nn.Module,
    cfg: ExperimentConfig,
    *,
    move_to_cpu: bool = True,
) -> LayerGradDict:
    """
    Flatten and concatenate gradients by architecture-defined layer.

    ``move_to_cpu=True`` preserves the original behavior. DynaHybrid uses
    ``move_to_cpu=False`` while accumulating the local-step mean so that the
    40 gradients are averaged on the GPU and transferred to the CPU only once
    after the client's local training has finished.
    """
    pieces: Dict[str, List[torch.Tensor]] = {
        layer: [] for layer in get_layers(cfg.model_name, cfg.grouping_mode)
    }
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        layer = get_layer_name(name, cfg.model_name, cfg.grouping_mode)
        if layer is None:
            continue
        grad = param.grad.detach().flatten()
        if move_to_cpu:
            grad = grad.cpu()
        pieces[layer].append(grad)

    out: LayerGradDict = {}
    for layer, tensors in pieces.items():
        if len(tensors) > 0:
            out[layer] = torch.cat(tensors, dim=0)
    return out


def detach_layer_grad_dict_cpu(layer_grads: LayerGradDict) -> LayerGradDict:
    """Copy one already-averaged layer-gradient dictionary to CPU once."""
    return {
        layer: grad.detach().cpu().clone()
        for layer, grad in layer_grads.items()
    }
#每层所有参数梯度拼成一个向量。

def average_layer_grad_dicts(dicts: List[LayerGradDict]) -> LayerGradDict:
    """Average a list of layer-gradient dictionaries produced during local steps."""
    if len(dicts) == 0:
        return {}
    layers = set().union(*[d.keys() for d in dicts])
    out: LayerGradDict = {}
    for layer in layers:
        vals = [d[layer] for d in dicts if layer in d]
        if len(vals) > 0:
            out[layer] = torch.stack(vals, dim=0).mean(dim=0)
    return out


def update_layer_grad_running_mean(
    running_mean: LayerGradDict,
    current: LayerGradDict,
    count: int,
) -> None:
    """Update one stored mean per layer instead of retaining all local-step copies."""
    if count < 1:
        raise ValueError("count must be at least 1")
    for layer, grad in current.items():
        if layer not in running_mean:
            running_mean[layer] = grad.clone()
        else:
            # In-place linear interpolation implements
            # mean <- mean + (grad - mean) / count without allocating the
            # full-size temporary tensor produced by ``grad - mean``.
            running_mean[layer].lerp_(grad, 1.0 / float(count))
#平均各客户端梯度

def train_one_client_fedavg(
    model: nn.Module,
    client_loader: Iterable[Tuple[torch.Tensor, torch.Tensor]],
    cfg: ExperimentConfig,
    device: torch.device,
    collect_scoring_grads: bool = False,
) -> Tuple[TensorDict, float, LayerGradDict]:
    """
    Local first-order FedAvg update with weight decay:
        g = grad(loss)
        theta <- theta - eta * (g + lambda * theta)
    This matches torch.optim.SGD(..., weight_decay=lambda, momentum=0).
    Returns updated model state, average local loss, and average layer gradients for scoring.
    """
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    batch_iter = infinite_loader(client_loader)

    total_loss = 0.0
    total_seen = 0
    layer_grad_mean: LayerGradDict = {}
    layer_grad_count = 0

    for _ in range(cfg.local_steps):
        x, y = next(batch_iter)
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        model.zero_grad(set_to_none=True)
        logits = model(x)
        loss = loss_fn(logits, y)
        if not torch.isfinite(loss):
            raise FloatingPointError("FedAvg encountered a non-finite local loss.")
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
            for param in model.parameters():
                if param.grad is None:
                    continue

                # FedAvg + weight decay:
                # theta <- theta - eta * lambda * theta
                # theta <- theta - eta * g
                # This is equivalent to theta <- theta - eta * (g + lambda * theta).
                if cfg.weight_decay != 0.0:
                    param.add_(param.detach(), alpha=-cfg.fedavg_lr * cfg.weight_decay)
                param.add_(param.grad.detach(), alpha=-cfg.fedavg_lr)

        bs = int(x.size(0))
        total_loss += float(loss.item()) * bs
        total_seen += bs

    return detach_state_dict_cpu(model), total_loss / max(total_seen, 1), layer_grad_mean
