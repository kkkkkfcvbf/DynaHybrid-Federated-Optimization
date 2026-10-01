"""Command-line entry point for the archived experiments.

Paper-affecting hyperparameters are required command-line arguments.  This is
intentional: an archived command completely identifies a run, while the fully
resolved values are also written to ``config.json`` before training starts.
"""

import argparse
import os
from typing import Dict, List
from config import ExperimentConfig, ensure_dir, get_layers, save_config, set_seed
from train import build_dataloaders, run_federated_experiment
from utils.logger import write_metrics_csv

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--rounds", type=int, required=True)
    parser.add_argument("--clients", type=int, required=True)
    parser.add_argument("--local_steps", type=int, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--test_batch_size", type=int, required=True)
    parser.add_argument("--fedavg_lr", type=float, required=True)
    parser.add_argument("--fedprox_mu", type=float, required=True)
    parser.add_argument(
        "--max_grad_norm", type=float, required=True,
        help="Global clipping threshold; pass 0 to disable clipping.",
    )
    parser.add_argument(
        "--lr_schedule",
        choices=["constant", "multistep"],
        required=True,
    )

    parser.add_argument(
        "--lr_milestones",
        type=str,
        required=True,
        help="Comma-separated rounds, for example 50,75.",
    )

    parser.add_argument(
        "--lr_decay_gamma",
        type=float,
        required=True,
    )
    parser.add_argument(
        "--hybrid_hessian_estimates",
        type=int,
        choices=[1, 40],
        required=True,
        help=(
            "Number of curvature estimates per DynaHybrid refresh event. "
            "For GNB this controls GNB passes; for L-BFGS it controls reversible "
            "same-batch probes. 1 estimates once; 40 estimates at every local step."
        ),
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--dataset", choices=["cifar10", "cifar100", "mnist"], required=True)
    parser.add_argument(
        "--partition", choices=["iid", "dirichlet"], required=True,
        help="Client data partition: equal-size IID or Dirichlet label skew.",
    )
    parser.add_argument(
        "--dirichlet_alpha", type=float, default=None,
        help="Dirichlet concentration alpha; required with --partition dirichlet.",
    )
    parser.add_argument(
        "--dirichlet_min_size", type=int, default=None,
        help="Minimum samples per client; required with --partition dirichlet.",
    )
    parser.add_argument(
        "--model", dest="model_name",
        choices=["alexnet", "resnet18", "googlenet"], required=True,
    )
    parser.add_argument("--sophia_lr", type=float, required=True)
    parser.add_argument(
        "--sophia_lr_schedule",
        choices=["constant", "multistep"],
        required=True,
        help="Schedule for Sophia updates, independent of --lr_schedule.",
    )
    parser.add_argument("--weight_decay", type=float, required=True, help="Shared weight decay lambda.")
    parser.add_argument("--image_size", type=int, required=True)
    parser.add_argument("--beta1", type=float, required=True)
    parser.add_argument("--beta2", type=float, required=True)
    parser.add_argument("--epsilon", type=float, required=True)
    parser.add_argument("--rho", type=float, required=True)
    parser.add_argument("--tau", type=int, required=True)
    parser.add_argument(
        "--lbfgs_history_size", type=int, default=3,
        help="Curvature pairs retained for the joint selected-layer block.",
    )
    parser.add_argument(
        "--lbfgs_curvature_eps", type=float, default=0.2,
        help="Powell damping threshold in y^T s >= eps * s^T B s.",
    )
    parser.add_argument(
        "--lbfgs_damping", choices=["true", "false"], default="true",
        help="Damp weak-curvature pairs instead of rejecting them.",
    )
    parser.add_argument(
        "--adahessian_hutchinson_samples", type=int, default=1,
        help="Number of Rademacher Hutchinson samples for DynaHybrid-AdaHessian.",
    )
    parser.add_argument(
        "--adahessian_hessian_batch_size", type=int, default=0,
        help=(
            "Mini-batch size used only for DynaHybrid-AdaHessian Hessian-vector "
            "products. 0 means use the full training mini-batch."
        ),
    )
    parser.add_argument(
        "--coverage_threshold", "--gamma", dest="coverage_threshold", type=float, required=True,
        help="Dynamic-K cumulative normalized-score threshold gamma.",
    )
    parser.add_argument("--first_score_round", type=int, required=True)
    parser.add_argument("--score_interval", type=int, required=True)
    parser.add_argument("--min_k", type=int, required=True)
    parser.add_argument("--max_k", type=int, required=True)
    parser.add_argument(
        "--selection_strategy", choices=["dynamic", "fixed", "random", "manual"], required=True,
        help="DynaHybrid layer policy: dynamic scores, first score-based set frozen, or random redraw.",
    )
    parser.add_argument(
        "--k_strategy", choices=["dynamic", "fixed"], required=True,
        help="Choose K by score coverage or use --fixed_k.",
    )
    parser.add_argument("--fixed_k", type=int, required=True, help="K for fixed-K ablations; recorded even with dynamic K.")
    parser.add_argument(
        "--fixed_layers", type=str, default=None,
        help="Comma-separated layer groups used with --selection_strategy manual.",
    )
    parser.add_argument(
        "--hybrid_hessian_scope",
        choices=["all", "selected"],
        required=True,
        help="DynaHybrid Hessian scope: all layers, or selected layers only.",
    )
    parser.add_argument("--hybrid_track_m_for_all_layers", choices=["true", "false"], required=True)
    parser.add_argument("--hybrid_warmup_hessian_round1", choices=["true", "false"], required=True)
    parser.add_argument("--print_every_round", choices=["true", "false"], required=True)
    parser.add_argument("--save_csv", choices=["true", "false"], required=True)
    parser.add_argument("--train_subset", type=int, required=True, help="Training samples; pass -1 for the complete dataset.")
    parser.add_argument("--test_subset", type=int, required=True, help="Test samples; pass -1 for the complete dataset.")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--device", type=str, required=True)
    parser.add_argument("--grouping_mode", type=str, required=True, choices=["7groups", "8groups"])
    parser.add_argument("--num_workers", type=int, required=True)
    parser.add_argument(
        "--target_accuracy", type=float, required=True,
        help="Accuracy target used for rounds/time/communication-to-target metrics.",
    )
    parser.add_argument("--download", action="store_true", help="Let torchvision download the selected dataset.")
    parser.add_argument(
        "--algos",
        type=str,
        required=True,
        help=(
            "Comma-separated: fedavg,fedprox,fedsophia,dynahybrid,"
            "dynahybrid_lbfgs,dynahybrid_adahessian"
        ),
    )
    return parser.parse_args()


def apply_args_to_config(cfg: ExperimentConfig, args: argparse.Namespace) -> ExperimentConfig:
    if args.rounds is not None:
        cfg.num_rounds = args.rounds
    if args.clients is not None:
        cfg.num_clients = args.clients
    if args.local_steps is not None:
        cfg.local_steps = args.local_steps
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.test_batch_size is not None:
        cfg.test_batch_size = args.test_batch_size
    if args.fedavg_lr is not None:
        cfg.fedavg_lr = args.fedavg_lr
    if args.fedprox_mu is not None:
        cfg.fedprox_mu = args.fedprox_mu
    if args.max_grad_norm is not None:
        cfg.max_grad_norm = None if args.max_grad_norm == 0.0 else args.max_grad_norm
    if args.lr_schedule is not None:
        cfg.lr_schedule = args.lr_schedule

    if args.lr_milestones is not None:
        cfg.lr_milestones = tuple(
            int(value.strip())
            for value in args.lr_milestones.split(",")
            if value.strip()
        )
    if args.hybrid_hessian_estimates is not None:
        cfg.hybrid_hessian_every_local_step_on_refresh = (
            args.hybrid_hessian_estimates == 40
        )

    if args.lr_decay_gamma is not None:
        cfg.lr_decay_gamma = args.lr_decay_gamma
    if args.seed is not None:
        cfg.seed = args.seed
    if args.data_dir is not None:
        cfg.data_dir = args.data_dir
    if args.dataset is not None:
        cfg.dataset = args.dataset.lower()
        cfg.partition = args.partition
        cfg.iid = args.partition == "iid"
        cfg.dirichlet_alpha = args.dirichlet_alpha
        cfg.dirichlet_min_size = args.dirichlet_min_size
        cfg.num_classes = 100 if cfg.dataset == "cifar100" else 10
    if args.model_name is not None:
        cfg.model_name = args.model_name.lower()
    if args.sophia_lr is not None:
        cfg.sophia_lr = args.sophia_lr
    if args.sophia_lr_schedule is not None:
        cfg.sophia_lr_schedule = args.sophia_lr_schedule
    if args.weight_decay is not None:
        cfg.weight_decay = args.weight_decay
    if args.image_size is not None:
        cfg.image_size = args.image_size
    if args.beta1 is not None:
        cfg.beta1 = args.beta1
    if args.beta2 is not None:
        cfg.beta2 = args.beta2
    if args.epsilon is not None:
        cfg.epsilon = args.epsilon
    if args.rho is not None:
        cfg.rho = args.rho
    if args.tau is not None:
        cfg.hessian_interval = args.tau
    cfg.lbfgs_history_size = args.lbfgs_history_size
    cfg.lbfgs_curvature_eps = args.lbfgs_curvature_eps
    cfg.lbfgs_damping = args.lbfgs_damping == "true"
    cfg.adahessian_hutchinson_samples = args.adahessian_hutchinson_samples
    cfg.adahessian_hessian_batch_size = args.adahessian_hessian_batch_size
    if args.coverage_threshold is not None:
        cfg.score_coverage_threshold = args.coverage_threshold
    if args.first_score_round is not None:
        cfg.first_score_round = args.first_score_round
    if args.score_interval is not None:
        cfg.score_interval = args.score_interval
    if args.min_k is not None:
        cfg.min_second_order_layers = args.min_k
    if args.max_k is not None:
        cfg.max_second_order_layers = args.max_k
    if args.selection_strategy is not None:
        cfg.selection_strategy = args.selection_strategy
    if args.k_strategy is not None:
        cfg.k_strategy = args.k_strategy
    if args.fixed_k is not None:
        cfg.fixed_second_order_layers = args.fixed_k
    if args.fixed_layers is not None:
        cfg.manual_second_order_layers = tuple(
            layer.strip() for layer in args.fixed_layers.split(",") if layer.strip()
        )
    if args.hybrid_hessian_scope is not None:
        cfg.hybrid_refresh_hessian_for_all_layers = args.hybrid_hessian_scope == "all"
    cfg.hybrid_track_m_for_all_layers = args.hybrid_track_m_for_all_layers == "true"
    cfg.hybrid_warmup_hessian_round1 = args.hybrid_warmup_hessian_round1 == "true"
    cfg.print_every_round = args.print_every_round == "true"
    cfg.save_csv = args.save_csv == "true"
    if args.train_subset is not None:
        cfg.train_subset = None if args.train_subset == -1 else args.train_subset
    if args.test_subset is not None:
        cfg.test_subset = None if args.test_subset == -1 else args.test_subset
    if args.output_dir is not None:
        cfg.output_dir = args.output_dir
    if args.device is not None:
        cfg.device = args.device
    if args.grouping_mode is not None:
        cfg.grouping_mode = args.grouping_mode
    if args.num_workers is not None:
        cfg.num_workers = args.num_workers
    if args.target_accuracy is not None:
        cfg.target_accuracy = args.target_accuracy
    if getattr(args, "download", False):
        cfg.dataset_download = True

    if cfg.score_interval < 1:
        raise ValueError("score_interval must be at least 1.")
    if cfg.max_grad_norm is not None and cfg.max_grad_norm <= 0.0:
        raise ValueError("max_grad_norm must be positive when provided.")
    if not 0.0 < cfg.score_coverage_threshold <= 1.0:
        raise ValueError("score_coverage_threshold/gamma must be in (0, 1].")
    if cfg.min_second_order_layers < 1:
        raise ValueError("min_second_order_layers/min_k must be at least 1.")
    if cfg.max_second_order_layers < cfg.min_second_order_layers:
        raise ValueError("max_second_order_layers/max_k must be >= min_second_order_layers/min_k.")
    if cfg.fixed_second_order_layers < 1:
        raise ValueError("fixed_second_order_layers/fixed_k must be at least 1.")
    if cfg.selection_strategy == "manual":
        valid_layers = set(get_layers(cfg.model_name, cfg.grouping_mode))
        if not cfg.manual_second_order_layers:
            raise ValueError("--fixed_layers is required with --selection_strategy manual.")
        unknown_layers = set(cfg.manual_second_order_layers) - valid_layers
        if unknown_layers:
            raise ValueError(
                f"Unknown --fixed_layers for {cfg.model_name}: {sorted(unknown_layers)}; "
                f"valid layers are {sorted(valid_layers)}."
            )
    if not 0.0 < cfg.target_accuracy <= 1.0:
        raise ValueError("target_accuracy must be in (0, 1].")
    if cfg.partition == "dirichlet":
        if cfg.dirichlet_alpha is None or cfg.dirichlet_alpha <= 0.0:
            raise ValueError(
                "--dirichlet_alpha must be positive with --partition dirichlet."
            )
        if cfg.dirichlet_min_size is None or cfg.dirichlet_min_size < 1:
            raise ValueError(
                "--dirichlet_min_size must be at least 1 with --partition dirichlet."
            )
    elif cfg.dirichlet_alpha is not None or cfg.dirichlet_min_size is not None:
        raise ValueError(
            "Dirichlet arguments are valid only with --partition dirichlet."
        )
    if cfg.lbfgs_history_size < 1:
        raise ValueError("lbfgs_history_size must be at least 1.")
    if not 0.0 < cfg.lbfgs_curvature_eps < 1.0:
        raise ValueError("lbfgs_curvature_eps must be in (0, 1).")
    if cfg.adahessian_hutchinson_samples < 1:
        raise ValueError("adahessian_hutchinson_samples must be at least 1.")
    if cfg.adahessian_hessian_batch_size < 0:
        raise ValueError("adahessian_hessian_batch_size must be non-negative.")
    allowed_algorithms = {
        "fedavg", "fedprox", "fedsophia", "dynahybrid",
        "dynahybrid_lbfgs", "dynahybrid_adahessian",
    }
    requested_algorithms = {
        value.strip().lower() for value in args.algos.split(",") if value.strip()
    }
    unknown_algorithms = requested_algorithms - allowed_algorithms
    if not requested_algorithms:
        raise ValueError("--algos must contain at least one algorithm.")
    if unknown_algorithms:
        raise ValueError(
            f"Unknown algorithms: {sorted(unknown_algorithms)}; "
            f"valid values are {sorted(allowed_algorithms)}."
        )
    return cfg


def main() -> None:
    args = parse_args()
    cfg = apply_args_to_config(ExperimentConfig(), args)
    ensure_dir(cfg.output_dir)
    save_config(cfg, os.path.join(cfg.output_dir, "config.json"))
    set_seed(cfg.seed)

    client_loaders, client_sizes, test_loader = build_dataloaders(cfg)
    print(f"Client sample sizes: {client_sizes}")

    all_rows: List[Dict[str, object]] = []
    algos = [a.strip().lower() for a in args.algos.split(",") if a.strip()]
    for algo in algos:
        # Reset seed before every algorithm to use the same initialization and partition randomness as much as possible.
        set_seed(cfg.seed)
        rows = run_federated_experiment(algo, cfg, client_loaders, client_sizes, test_loader)
        all_rows.extend(rows)

    if cfg.save_csv and len(all_rows) > 0:
        write_metrics_csv(os.path.join(cfg.output_dir, "all_metrics.csv"), all_rows)
        print(f"Saved all metrics: {os.path.join(cfg.output_dir, 'all_metrics.csv')}")


if __name__ == "__main__":
    main()
