# DynaHybrid federated-learning implementation

This archive contains the implementation used by the paper *Dynamic Layer-wise
Hybrid Optimization for Computation-Efficient Federated Learning*.

## Implemented methods

- `fedavg`: local SGD and sample-size-weighted full-model aggregation.
- `fedprox`: local SGD with the FedProx proximal term.
- `fedsophia`: full-model Fed-Sophia with the GNB diagonal estimator.
- `dynahybrid-sophia`: DynaHybrid with the Fed-Sophia second-order branch.
- `dynahybrid-adahessian`: DynaHybrid with a Hutchinson diagonal-Hessian estimator.

The archive intentionally does not expose methods that are not evaluated in the
paper. Every paper-affecting setting is supplied through the command line and the
resolved configuration is saved as `config.json` in the output directory.

## Paper-aligned DynaHybrid settings

The main DynaHybrid-Sophia experiments use:

- first selection at communication round 2;
- selection interval 10;
- normalized-score coverage threshold 0.6;
- dynamic layer ranking and dynamic layer count;
- AlexNet layer-count bounds 2--4;
- ResNet-18 and GoogLeNet layer-count bounds 1--3;
- Hessian estimation only for selected layers;
- one Hessian estimate at the first local step of a refresh event;
- no round-1 Hessian warm-up;
- first-moment tracking for every layer;
- `beta1=0.9`, `beta2=0.95`, `rho=0.003`, and weight decay `1e-4`.

All experiments use 10 clients, full participation, 100 communication rounds,
40 local steps per round, batch size 128, and seeds 42--44. IID experiments use
an exhaustive equal-size random partition. Non-IID experiments use Dirichlet
label skew with `alpha=0.5` and minimum client size 10. Task-specific learning
rates and the FedProx coefficient must be passed explicitly, as stated in the
paper.

## Environment

Python 3.10 or newer is recommended. Install dependencies with:

```bash
python -m pip install -r requirements.txt
```

Run the implementation from this directory. Complete dataset files are
included under `./datasets`; alternatively, point `--data_dir` to another
torchvision-compatible copy or add `--download` to obtain a fresh copy.

## Command pattern

The following pattern shows the paper settings. Replace the task-specific
learning rates, model, dataset, seed, layer-count bounds, target accuracy, and
output directory with the values for the run being reproduced.

```bash
python main.py \
  --dataset cifar100 --model alexnet --partition iid \
  --algos fedsophia,dynahybrid \
  --rounds 100 --clients 10 --local_steps 40 \
  --batch_size 128 --test_batch_size 256 \
  --train_subset 50000 --test_subset 10000 \
  --seed 42 --data_dir ./datasets \
  --fedavg_lr 0.10 --fedprox_mu 0.001 --max_grad_norm 0 \
  --lr_schedule constant --lr_milestones 50,75 --lr_decay_gamma 0.2 \
  --sophia_lr 0.01 --sophia_lr_schedule constant \
  --weight_decay 0.0001 --image_size 224 \
  --beta1 0.9 --beta2 0.95 --epsilon 1e-8 --rho 0.003 --tau 10 \
  --coverage_threshold 0.6 --first_score_round 2 --score_interval 10 \
  --min_k 2 --max_k 4 --selection_strategy dynamic \
  --k_strategy dynamic --fixed_k 3 \
  --hybrid_hessian_scope selected --hybrid_hessian_estimates 1 \
  --hybrid_track_m_for_all_layers true \
  --hybrid_warmup_hessian_round1 false \
  --print_every_round true --save_csv true \
  --output_dir ./results/cifar100_alexnet_seed42 \
  --device cuda:0 --grouping_mode 8groups --num_workers 2 \
  --target_accuracy 0.50
```

For `dynahybrid_adahessian`, additionally provide
`--adahessian_hutchinson_samples` and `--adahessian_hessian_batch_size`.

## Outputs

Each run writes:

- `config.json`: resolved experiment configuration;
- `partition_summary.json` and `partition_indices.npz`: exact client partition;
- `<algorithm>_metrics.csv`: one durable row per completed round;
- `<algorithm>_compute_summary.csv`: latest cumulative computation summary;
- `all_metrics.csv`: combined rows when multiple algorithms are requested.

FLOP values are analytical estimates, not hardware-counter measurements. The
counting convention is recorded in every metrics file.

## Verification

Run:

```bash
python -m unittest discover -s tests -v
python -m compileall -q .
```

The tests check layer mappings, weighted score computation, dynamic selection,
the round-1 all-SGD/no-Hessian rule, and deterministic data partitions.
