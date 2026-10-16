"""DynaHybrid wrapper separated from the shared Sophia optimizer core."""

from typing import Iterable, List, Optional, Set, Tuple
import torch
import torch.nn as nn
from config import ExperimentConfig, get_layers
from algorithms.fedavg import LayerGradDict, TensorDict
from algorithms.fedsophia import SophiaClientState, train_one_client_sophia_or_hybrid

def train_one_client_dynahybrid(
    model: nn.Module,
    client_loader: Iterable[Tuple[torch.Tensor, torch.Tensor]],
    state: SophiaClientState,
    cfg: ExperimentConfig,
    device: torch.device,
    round_idx0: int,
    selected_sophia_layers: List[str],
    newly_selected_sophia_layers: List[str],
    collect_scoring_grads: bool = False,
) -> Tuple[TensorDict, float, LayerGradDict, float, int]:
    """
    DynaHybrid:
    - selected_sophia_layers use Fed-Sophia updates.
    - all other layers use FedAvg SGD updates.
    - round 1 uses SGD updates for all parameters as requested.
    """
    selected_set: Set[str] = set(selected_sophia_layers)
    newly_selected_set: Set[str] = set(newly_selected_sophia_layers)
    all_layers: Set[str] = set(get_layers(cfg.model_name, cfg.grouping_mode))

    one_based_round = round_idx0 + 1
    force_sgd = (one_based_round == 1)

    hessian_override: Optional[Set[str]] = None
    if cfg.hybrid_refresh_hessian_for_all_layers:
        hessian_override = all_layers
        # All-layer mode already refreshed every layer on the preceding scheduled
        # round, so selection changes do not trigger an extra Hessian pass.
        newly_selected_set = set()
    else:
        hessian_override = selected_set

    # If user disables round-1 warm-up, do not compute GNB on DynaHybrid round 1.
    if force_sgd and not cfg.hybrid_warmup_hessian_round1:
        hessian_override = set()

    return train_one_client_sophia_or_hybrid(
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
