"""DynaHybrid layer scoring, including an exact streaming-moment form."""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import torch
from config import ExperimentConfig, get_layers
from algorithms.fedavg import LayerGradDict


@dataclass
class StreamingLayerScoreMoments:
    """Sufficient statistics for the weighted client-gradient variance.

    For every layer this stores only

        weighted_sum[l] = sum_i w_i g_i,l
        weighted_sq_norm_sum[l] = sum_i w_i ||g_i,l||^2

    instead of retaining every client gradient.  Dividing both quantities by
    the corresponding per-layer weight gives A_l and B_l in the documented
    streaming formula.
    """

    weighted_sum: LayerGradDict = field(default_factory=dict)
    weighted_sq_norm_sum: Dict[str, float] = field(default_factory=dict)
    weight_sum_by_layer: Dict[str, float] = field(default_factory=dict)
    total_weight: float = 0.0
    client_count: int = 0

    def update(self, layer_grads: LayerGradDict, weight: float = 1.0) -> None:
        weight = float(weight)
        if weight <= 0.0:
            raise ValueError("streaming score weight must be positive")
        for layer, grad in layer_grads.items():
            grad_cpu = grad.detach().cpu()
            if layer not in self.weighted_sum:
                self.weighted_sum[layer] = grad_cpu.clone().mul_(weight)
                self.weighted_sq_norm_sum[layer] = weight * float(
                    torch.dot(grad_cpu, grad_cpu).item()
                )
                self.weight_sum_by_layer[layer] = weight
            else:
                self.weighted_sum[layer].add_(grad_cpu, alpha=weight)
                self.weighted_sq_norm_sum[layer] += weight * float(
                    torch.dot(grad_cpu, grad_cpu).item()
                )
                self.weight_sum_by_layer[layer] += weight
        self.total_weight += weight
        self.client_count += 1

    def compute_scores(self, cfg: ExperimentConfig) -> Dict[str, float]:
        """Return the same weighted variance ratio without materializing clients."""
        if self.total_weight <= 0.0:
            return {}
        scores: Dict[str, float] = {}
        eps = 1e-12
        for layer in get_layers(cfg.model_name, cfg.grouping_mode):
            if layer not in self.weighted_sum:
                continue
            layer_weight = self.weight_sum_by_layer[layer]
            # ||sum_i w_i g_i / W||^2 = ||sum_i w_i g_i||^2 / W^2.
            # Computing it this way avoids allocating another full mean vector.
            weighted_sum = self.weighted_sum[layer]
            mean_sq_norm = float(torch.dot(weighted_sum, weighted_sum).item()) / (
                layer_weight * layer_weight
            )
            second_moment = self.weighted_sq_norm_sum[layer] / layer_weight
            # Round-off can make a theoretically non-negative variance slightly
            # negative, so clamp only at zero.
            numerator = max(second_moment - mean_sq_norm, 0.0)
            score = numerator / (mean_sq_norm + eps)
            scores[layer] = score if torch.isfinite(torch.tensor(score)) else 0.0
        return scores

    def stored_tensor_bytes(self) -> int:
        """Bytes retained by the vector sufficient statistic A_l."""
        return sum(t.numel() * t.element_size() for t in self.weighted_sum.values())


def compute_scores_from_prev_round(
    prev_client_layer_grads: List[LayerGradDict],
    cfg: ExperimentConfig
) -> Dict[str, float]:
    """
    Dynamic layer score:

        Score_l =
        E_i(||g_i^l - g_bar^l||^2)
        -------------------------
              ||g_bar^l||^2 + eps

    where:
        g_i^l : client i gradient of layer l
        g_bar^l : mean gradient over clients

    This legacy materialized helper is retained only for compatibility. The
    training path uses StreamingLayerScoreMoments with sample-size weights.
    """

    scores: Dict[str, float] = {}

    eps = 1e-12

    for layer in get_layers(cfg.model_name, cfg.grouping_mode):

        grads = [
            g[layer]
            for g in prev_client_layer_grads
            if layer in g
        ]

        if len(grads) == 0:
            continue


        # [num_clients, num_parameters_in_layer]
        stack = torch.stack(grads, dim=0)


        # mean gradient
        mean_grad = stack.mean(dim=0)


        # numerator:
        # E_i(||g_i - g_bar||^2)
        numerator = torch.mean(
            torch.linalg.vector_norm(
                stack - mean_grad.unsqueeze(0),
                dim=1
            ) ** 2
        )


        # denominator:
        # ||g_bar||^2 + epsilon
        denominator = (
            torch.linalg.vector_norm(mean_grad) ** 2
            + eps
        )


        score_tensor = numerator / denominator


        if not torch.isfinite(score_tensor):
            score = 0.0
        else:
            score = max(float(score_tensor.item()), 0.0)


        scores[layer] = score


    return scores

def normalize_layer_scores(scores: Dict[str, float]) -> Dict[str, float]:
    """
    Normalize raw non-negative layer scores into proportions that sum to 1.

    Formula:
        normalized_score_l = Score_l / sum_j Score_j

    If all scores are zero, use a uniform distribution so dynamic K remains defined.
    """
    if len(scores) == 0:
        return {}

    cleaned: Dict[str, float] = {}
    for layer, value in scores.items():
        value = float(value)
        if not torch.isfinite(torch.tensor(value)) or value < 0.0:
            value = 0.0
        cleaned[layer] = value

    total = sum(cleaned.values())
    if total <= 1e-30:
        uniform = 1.0 / len(cleaned)
        return {layer: uniform for layer in cleaned}

    return {layer: value / total for layer, value in cleaned.items()}


def select_dynamic_layers_by_coverage(
    scores: Dict[str, float],
    coverage_threshold: float,
    min_k: int,
    max_k: int,
) -> Tuple[List[str], int, Dict[str, float], float, List[str]]:
    """
    Dynamically select the number K and the second-order layers.

    Steps:
    1. Normalize raw scores across layers.
    2. Sort normalized scores in descending order.
    3. Find the smallest raw K whose cumulative contribution is at least gamma.
    4. Constrain K to [min_k, max_k] and to the number of available layers.

    Returns:
        selected_layers,
        selected_k,
        normalized_scores,
        selected_cumulative_coverage,
        ranked_layers

    Note: if the unconstrained K is larger than max_k, the returned cumulative coverage
    can be below gamma because max_k is a hard upper bound requested by the experiment.
    """
    if not 0.0 < coverage_threshold <= 1.0:
        raise ValueError("coverage_threshold must be in (0, 1].")
    if min_k < 1:
        raise ValueError("min_k must be at least 1.")
    if max_k < min_k:
        raise ValueError("max_k must be greater than or equal to min_k.")

    normalized = normalize_layer_scores(scores)
    if len(normalized) == 0:
        return [], 0, {}, 0.0, []

    # Secondary key gives deterministic ordering when two proportions are equal.
    ranked_items = sorted(normalized.items(), key=lambda item: (-item[1], item[0]))
    ranked_layers = [layer for layer, _ in ranked_items]

    cumulative = 0.0
    raw_k = len(ranked_items)
    for index, (_, proportion) in enumerate(ranked_items, start=1):
        cumulative += proportion
        if cumulative + 1e-12 >= coverage_threshold:
            raw_k = index
            break

    available = len(ranked_items)
    effective_min = min(min_k, available)
    effective_max = min(max_k, available)
    selected_k = min(max(raw_k, effective_min), effective_max)
    selected_layers = ranked_layers[:selected_k]
    selected_coverage = sum(normalized[layer] for layer in selected_layers)

    return selected_layers, selected_k, normalized, selected_coverage, ranked_layers
