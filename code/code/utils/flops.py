"""Deterministic analytical FLOP estimates for all supported architectures.

The counters are deliberately named ``estimated_*``.  They use the common
deep-learning convention of two FLOPs per multiply-accumulate, approximate a
parameter-gradient backward pass as twice the corresponding forward work, and
also count the explicit optimizer/scoring/aggregation tensor arithmetic in this
archive.  Data movement, random sampling, comparisons, indexing, Python and
CSV I/O are not FLOPs and are therefore excluded.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence, Set

import torch
import torch.nn as nn

from config import ExperimentConfig, get_layer_name, get_layers


@dataclass(frozen=True)
class ModelFlopProfile:
    forward_per_sample: int
    layer_forward_per_sample: Dict[str, int]
    parameter_elements_by_layer: Dict[str, int]
    total_parameter_elements: int
    num_classes: int
    model_name: str
    grouping_mode: str

    @classmethod
    def from_config_and_model(cls, cfg: ExperimentConfig, model: nn.Module) -> "ModelFlopProfile":
        """Profile Conv2d/Linear FLOPs using actual module output shapes."""
        layers = get_layers(cfg.model_name, cfg.grouping_mode)
        costs: Dict[str, int] = {layer: 0 for layer in layers}
        hooks = []
        for module_name, module in model.named_modules():
            if not isinstance(module, (nn.Conv2d, nn.Linear)):
                continue
            layer = get_layer_name(module_name + ".weight", cfg.model_name, cfg.grouping_mode)
            if layer is None:
                continue

            def hook(_module, _inputs, output, *, layer=layer):
                if isinstance(_module, nn.Conv2d):
                    outputs = int(output.numel())
                    kh, kw = _module.kernel_size
                    per_output = 2 * (_module.in_channels // _module.groups) * kh * kw
                    if _module.bias is not None:
                        per_output += 1
                else:
                    outputs = int(output.numel())
                    per_output = 2 * _module.in_features + (1 if _module.bias is not None else 0)
                costs[layer] += outputs * per_output

            hooks.append(module.register_forward_hook(hook))

        was_training = model.training
        model.eval()
        device = next(model.parameters()).device
        with torch.no_grad():
            model(torch.zeros(1, 3, cfg.image_size, cfg.image_size, device=device))
        if was_training:
            model.train()
        for handle in hooks:
            handle.remove()

        params = {layer: 0 for layer in layers}
        for name, param in model.named_parameters():
            layer = get_layer_name(name, cfg.model_name, cfg.grouping_mode)
            if layer is not None:
                params[layer] += int(param.numel())

        # Parameterized convolution/linear kernels dominate model FLOPs. The
        # excluded ReLU/pooling/dropout comparisons are stated in METRICS.md.
        return cls(
            forward_per_sample=int(sum(costs.values())),
            layer_forward_per_sample=costs,
            parameter_elements_by_layer=params,
            total_parameter_elements=int(sum(params.values())),
            num_classes=cfg.num_classes,
            model_name=cfg.model_name,
            grouping_mode=cfg.grouping_mode,
        )

    @property
    def loss_per_sample(self) -> int:
        # Approximate softmax cross entropy: max/subtract/exp/sum/div/log.
        return 5 * self.num_classes

    def ordinary_train_flops(self, examples: int) -> int:
        # One forward plus an approximately 2x-forward parameter/input backward.
        return int(examples) * (3 * self.forward_per_sample + self.loss_per_sample)

    def evaluation_flops(self, examples: int) -> int:
        return int(examples) * (self.forward_per_sample + self.loss_per_sample)

    def gnb_flops(self, examples: int, layers: Optional[Set[str]]) -> int:
        """Estimate one GNB forward/backward for all or selected layers.

        For selected layers, backward work is counted over the parameterized
        suffix from the earliest selected layer to the output. This reflects
        the dependency path traversed by autograd and is more conservative than
        counting only selected parameter elements.
        """
        if layers is not None and len(layers) == 0:
            return 0
        ordered = [
            layer for layer in get_layers(self.model_name, self.grouping_mode)
            if layer in self.layer_forward_per_sample
        ]
        if layers is None:
            backward_forward_equivalent = self.forward_per_sample
        else:
            indices = [ordered.index(layer) for layer in layers if layer in ordered]
            if not indices:
                return 0
            first = min(indices)
            backward_forward_equivalent = sum(
                self.layer_forward_per_sample[layer] for layer in ordered[first:]
            )
        per_sample = (
            self.forward_per_sample
            + 2 * backward_forward_equivalent
            + self.loss_per_sample
        )
        return int(examples) * int(per_sample)

    def layer_parameter_count(self, layers: Optional[Iterable[str]]) -> int:
        if layers is None:
            return self.total_parameter_elements
        return sum(self.parameter_elements_by_layer.get(layer, 0) for layer in layers)


@dataclass
class RoundFlops:
    forward_backward: int = 0
    hessian_gnb: int = 0
    hessian_adahessian: int = 0
    optimizer: int = 0
    scoring: int = 0
    aggregation: int = 0
    evaluation: int = 0

    @property
    def total(self) -> int:
        return (
            self.forward_backward
            + self.hessian_gnb
            + self.hessian_adahessian
            + self.optimizer
            + self.scoring
            + self.aggregation
            + self.evaluation
        )

    def add_(self, other: "RoundFlops") -> None:
        self.forward_backward += other.forward_backward
        self.hessian_gnb += other.hessian_gnb
        self.hessian_adahessian += other.hessian_adahessian
        self.optimizer += other.optimizer
        self.scoring += other.scoring
        self.aggregation += other.aggregation
        self.evaluation += other.evaluation


def examples_for_local_steps(dataset_size: int, batch_size: int, local_steps: int) -> int:
    """Number of examples consumed when a finite loader is cycled forever."""
    if dataset_size <= 0 or batch_size <= 0 or local_steps <= 0:
        return 0
    full, remainder = divmod(dataset_size, batch_size)
    batch_sizes = [batch_size] * full + ([remainder] if remainder else [])
    return sum(batch_sizes[index % len(batch_sizes)] for index in range(local_steps))


def first_batch_size(dataset_size: int, batch_size: int) -> int:
    return min(dataset_size, batch_size) if dataset_size > 0 else 0


def estimate_round_flops(
    *,
    algo: str,
    cfg: ExperimentConfig,
    profile: ModelFlopProfile,
    client_sizes: Sequence[int],
    test_examples: int,
    round_idx0: int,
    selected_layers: Sequence[str],
    newly_selected_layers: Sequence[str],
    collect_scoring_grads: bool,
) -> RoundFlops:
    """Estimate all arithmetic categories for one completed FL round."""
    out = RoundFlops()
    processed = sum(
        examples_for_local_steps(size, cfg.batch_size, cfg.local_steps)
        for size in client_sizes
    )
    out.forward_backward = profile.ordinary_train_flops(processed)
    out.evaluation = profile.evaluation_flops(test_examples)
    out.aggregation = 2 * profile.total_parameter_elements * len(client_sizes)

    all_n = profile.total_parameter_elements
    selected = set(selected_layers)
    selected_n = profile.layer_parameter_count(selected)
    steps_all_clients = cfg.local_steps * len(client_sizes)

    if algo == "fedavg":
        out.optimizer = steps_all_clients * 4 * all_n
    elif algo == "fedprox":
        out.optimizer = steps_all_clients * 7 * all_n
    elif algo == "fedsophia":
        # m EMA (3) + weight decay (2) + max/div/clip/update (6).
        out.optimizer = steps_all_clients * 11 * all_n
    elif algo in ("dynahybrid", "dynahybrid_adahessian"):
        if round_idx0 == 0:
            # All-layer m tracking plus all-SGD update during round-1 warm-up.
            out.optimizer = steps_all_clients * (3 * all_n + 4 * all_n)
        else:
            out.optimizer = steps_all_clients * (
                3 * all_n + 8 * selected_n + 4 * (all_n - selected_n)
            )

    # GNB schedule mirrors algorithms/fedsophia.py and dynahybrid.py.
    regular_refresh = round_idx0 % cfg.hessian_interval == 0
    if algo == "fedsophia" and regular_refresh:
        for size in client_sizes:
            examples = examples_for_local_steps(size, cfg.batch_size, cfg.local_steps)
            out.hessian_gnb += profile.gnb_flops(examples, None)
            out.hessian_gnb += 5 * profile.total_parameter_elements * cfg.local_steps
    elif algo == "dynahybrid":
        immediate = len(newly_selected_layers) > 0
        if regular_refresh or immediate:
            if immediate and not regular_refresh:
                hessian_layers: Optional[Set[str]] = set(newly_selected_layers)
            elif cfg.hybrid_refresh_hessian_for_all_layers:
                hessian_layers = None
            else:
                hessian_layers = selected

            if hessian_layers is None or len(hessian_layers) > 0:
                h_n = profile.layer_parameter_count(hessian_layers)

                # Number of GNB Hessian estimates actually executed at this refresh.
                num_hessian_estimates = max(
                    int(getattr(cfg, "hybrid_hessian_estimates", 1)),
                    1,
                )

                for size in client_sizes:
                    one_estimate_flops = (
                        profile.gnb_flops(
                            first_batch_size(size, cfg.batch_size),
                            hessian_layers,
                        )
                        + 5 * h_n
                    )

                    out.hessian_gnb += (
                        num_hessian_estimates * one_estimate_flops
                    )

    elif algo == "dynahybrid_adahessian":
        immediate = len(newly_selected_layers) > 0
        if regular_refresh or immediate:
            if immediate and not regular_refresh:
                hessian_layers = set(newly_selected_layers)
            elif cfg.hybrid_refresh_hessian_for_all_layers:
                hessian_layers = None
            else:
                hessian_layers = selected

            if hessian_layers is None or len(hessian_layers) > 0:
                h_n = profile.layer_parameter_count(hessian_layers)
                samples = max(
                    int(getattr(cfg, "adahessian_hutchinson_samples", 1) or 1),
                    1,
                )
                num_hessian_estimates = (
                    cfg.local_steps
                    if cfg.hybrid_hessian_every_local_step_on_refresh else 1
                )
                for size in client_sizes:
                    # One Hutchinson sample uses one selected-layer forward/first
                    # backward plus one selected-layer Hessian-vector backward.
                    one_sample = (
                        2 * profile.gnb_flops(
                            first_batch_size(size, cfg.batch_size),
                            hessian_layers,
                        )
                        + 6 * h_n
                    )
                    out.hessian_adahessian += (
                        num_hessian_estimates * samples * one_sample
                    )

    if algo in ("dynahybrid", "dynahybrid_adahessian") and collect_scoring_grads:
        n_clients = len(client_sizes)
        # Local running mean: lerp costs subtraction, multiply and addition for
        # local steps 2..J.  Server moments cost A += w*g and ||g||^2 per client,
        # followed by ||A||^2 and the final scalar arithmetic.
        out.scoring = (
            n_clients * max(cfg.local_steps - 1, 0) * 3 * all_n
            + n_clients * 4 * all_n
            + 2 * all_n
            + 2 * len(profile.parameter_elements_by_layer)
        )
    return out
