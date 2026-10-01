# Dynamic Layerwise Hybrid Optimization for Computation-Efficient Federated Learning

This archive contains the code, commands, logs, and figure-generation scripts used to reproduce the experimental results in the paper.

The archive is organized for reproducibility review. A reviewer can either regenerate the paper figures directly from the archived logs, or rerun the experiments from source code with the same random seeds and hyperparameters.

## 1. Archive structure

```text
钱开苹ISPA2026/
|
|-- README.md
|-- requirements.txt
|
|-- code/                         # Training and evaluation source code
|   |-- main.py                    # Main entry point for all experiments
|   |-- train.py                   # Federated training loop
|   |-- test.py                    # Evaluation utilities
|   |-- config.py                  # Configuration, layer grouping, seeding
|   |-- algorithms/                # FedAvg, FedProx, FedSophia, DynaHybrid variants
|   |-- models/                    # AlexNet, ResNet-18, GoogLeNet factory
|   |-- utils/                     # Logging, scoring, FLOP utilities
|   `-- datasets/                  # Archived local dataset copies used by the code
|
|-- commands/                     # Exact commands corresponding to paper figures/tables
|   |-- figure2_commands.md        # Figure 2 IID accuracy curves
|   |-- figure3_commands.md        # Figure 3 Hessian-estimator transfer curves
|   |-- table1_commands.md         # Table I final IID accuracy
|   |-- table2_commands.md         # Table II compute/time comparison
|   |-- table3_commands.md         # Table III Non-IID comparison
|   `-- table4_commands.md         # Table IV ablation experiments
|
|-- logs/                         # Archived terminal logs for paper results
|   |-- figure2.logs/              # Logs for Figure 2 and Table I
|   |-- figure3.logs/              # Logs for Figure 3
|   |-- table1.logs/               # Logs for Table I
|   |-- table2.logs/               # Logs for Table II / FLOP summaries
|   |-- table3.logs/               # Logs for Table III Non-IID results
|   `-- table4.logs/               # Logs for Table IV ablation results
|
|-- figures/
|   |-- figure1.pdf                # Framework schematic
|   |-- figure2.pdf                # Main IID accuracy curves
|   |-- figure3.pdf                # Hessian-estimator transfer curves
|   `-- figure_generation_scripts/
|       |-- plot_fig1.py
|       |-- plot_fig2.py            # Regenerates figure2.pdf from logs/figure2.logs
|       `-- plot_fig3.py            # Regenerates figure3.pdf from logs/figure3.logs
|
`-- paper/                         # Submitted manuscript PDF
```

## 2. Original experimental environment

The archived experiments were run on a Linux GPU server with NVIDIA GPUs. The observed GPU environment for these experiments was:

```text
GPU: NVIDIA GeForce RTX 4090, 24 GB memory
CUDA driver/runtime shown by nvidia-smi: CUDA 12.2
Python: 3.10 series on the training server
```

The Python package versions required by this archive are listed in `requirements.txt`:

```text
torch==2.5.1
torchvision==0.20.1
numpy==1.26.4
pandas==2.2.3
matplotlib==3.9.2
```

To record the local reviewer environment before reproducing the results, run:

```bash
python --version
python -c "import torch, torchvision, numpy, pandas, matplotlib; print('torch', torch.__version__); print('torchvision', torchvision.__version__); print('numpy', numpy.__version__); print('pandas', pandas.__version__); print('matplotlib', matplotlib.__version__); print('cuda available', torch.cuda.is_available()); print('cuda version', torch.version.cuda)"
nvidia-smi
```

## 3. Installation

Unzip the archive and enter the project root:

```bash
unzip 钱开苹ISPA2026.zip
cd 钱开苹ISPA2026
```

Create and activate a Python environment. One possible setup is:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

If the reviewer uses a conda environment, the equivalent setup is:

```bash
conda create -n dynahybrid python=3.10 -y
conda activate dynahybrid
pip install -r requirements.txt
```

The datasets used by the experiments are archived under `code/datasets/`. The training commands below use `--data_dir ./datasets` when run from the `code/` directory. If the reviewer prefers a different dataset directory, replace `--data_dir ./datasets` with that path.

## 4. Random seeds and deterministic settings

The reported multi-seed experiments use seeds:

```text
42, 43, 44
```

The code sets Python, NumPy, PyTorch CPU, and PyTorch CUDA seeds in `code/config.py`. It also sets deterministic cuDNN behavior when possible:

```python
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

Minor numerical differences can still occur across GPU models, CUDA/cuDNN versions, and PyTorch builds. The archived logs are therefore included as the exact record used to generate the paper tables and figures.

## 5. Fast reproduction: regenerate figures from archived logs

This is the fastest way to verify that the archived logs reproduce the submitted paper figures.

Run from the archive root:

```bash
python figures/figure_generation_scripts/plot_fig2.py
python figures/figure_generation_scripts/plot_fig3.py
```

The scripts read:

```text
logs/figure2.logs/
logs/figure3.logs/
```

and regenerate:

```text
figures/figure2.pdf
figures/figure3.pdf
```

To also export PNG previews, run:

```bash
python figures/figure_generation_scripts/plot_fig2.py --png figures/fig2.png
python figures/figure_generation_scripts/plot_fig3.py --png figures/fig3.png
```

Figure 1 is the framework schematic and can be regenerated with:

```bash
python figures/figure_generation_scripts/plot_fig1.py
```

## 6. Full experiment reproduction from source code

All training commands are stored in `commands/`. Each command records the dataset, model, partition type, algorithm list, learning rates, FedProx coefficient, random seed, local steps, batch size, selected-layer strategy, and output directory.

Before running training, enter the source-code directory:

```bash
cd code
```

Select the GPU. For example, to use physical GPU 0:

```bash
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

The commands inside `commands/*.md` are written so that `CUDA_VISIBLE_DEVICES` selects the physical GPU and the Python process uses `--device cuda:0`.

### 6.1 Figure 2, Table I, and Table II: IID main experiments

Use:

```text
commands/figure2_commands.md
commands/table1_commands.md
commands/table2_commands.md
```

These files contain the same IID runs. They cover 9 model-dataset pairs:

```text
AlexNet  + MNIST
AlexNet  + CIFAR-10
AlexNet  + CIFAR-100
ResNet18 + MNIST
ResNet18 + CIFAR-10
ResNet18 + CIFAR-100
GoogLeNet + MNIST
GoogLeNet + CIFAR-10
GoogLeNet + CIFAR-100
```

Algorithms:

```text
FedAvg, FedProx, FedSophia, DynaHybrid-Sophia
```

Seeds:

```text
42, 43, 44
```

The archived logs for these results are in:

```text
logs/figure2.logs/
logs/table1.logs/
logs/table2.logs/
```

### 6.2 Figure 3: estimator transferability

Use:

```text
commands/figure3_commands.md
```

This reproduces three representative comparisons:

```text
MNIST + ResNet18
CIFAR-10 + AlexNet
CIFAR-100 + AlexNet
```

Algorithms:

```text
FedSophia, DynaHybrid-Sophia, DynaHybrid-AdaHessian
```

Archived logs:

```text
logs/figure3.logs/
```

Regenerate the plotted PDF after training with:

```bash
python figures/figure_generation_scripts/plot_fig3.py
```

### 6.3 Table III: Non-IID experiments

Use:

```text
commands/table3_commands.md
```

Partition setting:

```text
Dirichlet label skew, alpha = 0.5, minimum client size = 10
```

Representative tasks:

```text
MNIST + ResNet18
MNIST + GoogLeNet
CIFAR-10 + ResNet18
CIFAR-10 + GoogLeNet
CIFAR-100 + AlexNet
CIFAR-100 + ResNet18
```

Archived logs:

```text
logs/table3.logs/
```

### 6.4 Table IV: ablation experiments

Use:

```text
commands/table4_commands.md
```

Ablation task:

```text
CIFAR-10 + AlexNet
```

Ablation factors:

```text
Ranking: fixed selected layers vs. dynamic layer scores
Count: fixed selected-layer count vs. dynamic selected-layer count
Refresh: 1 Hessian estimate vs. 40 Hessian estimates
```

Archived logs:

```text
logs/table4.logs/
```

## 7. Core command format

A typical command has the following structure:

```bash
cd code
python main.py \
  --dataset cifar10 \
  --model alexnet \
  --partition iid \
  --algos fedavg,fedprox,fedsophia,dynahybrid \
  --rounds 100 \
  --clients 10 \
  --local_steps 40 \
  --batch_size 128 \
  --test_batch_size 256 \
  --train_subset 50000 \
  --test_subset 10000 \
  --seed 42 \
  --data_dir ./datasets \
  --fedavg_lr 0.05 \
  --fedprox_mu 0.01 \
  --max_grad_norm 0 \
  --lr_schedule constant \
  --lr_milestones 50,75 \
  --lr_decay_gamma 0.2 \
  --sophia_lr 0.01 \
  --sophia_lr_schedule constant \
  --weight_decay 0.0001 \
  --image_size 224 \
  --beta1 0.9 \
  --beta2 0.95 \
  --epsilon 1e-8 \
  --rho 0.003 \
  --tau 10 \
  --coverage_threshold 0.6 \
  --first_score_round 2 \
  --score_interval 10 \
  --min_k 2 \
  --max_k 4 \
  --selection_strategy dynamic \
  --k_strategy dynamic \
  --fixed_k 3 \
  --hybrid_hessian_scope selected \
  --hybrid_hessian_estimates 1 \
  --hybrid_track_m_for_all_layers true \
  --hybrid_warmup_hessian_round1 false \
  --print_every_round true \
  --save_csv true \
  --output_dir ../results/example_cifar10_alexnet_seed42 \
  --device cuda:0 \
  --grouping_mode 8groups \
  --num_workers 2 \
  --target_accuracy 0.84
```

The full paper commands in `commands/` should be used for exact reproduction.

## 8. Outputs produced by a training run

Each training run writes outputs under the specified `--output_dir`, including:

```text
config.json                    # Fully resolved command-line configuration
all_metrics.csv                # Per-round metrics for all algorithms in the run
<algorithm>_metrics.csv         # Per-round metrics for one algorithm
<algorithm>_compute_summary.csv # Computation and timing summary when applicable
```

When `--print_every_round true` is used, the terminal log records every round, including train loss, test loss, test accuracy, selected layer count, selected layer names, Hessian time, round time, and peak GPU memory. The archived `logs/` directory contains these records.

## 9. Mapping from paper items to files

| Paper item | Commands | Archived logs | Regeneration script / data |
|---|---|---|---|
| Figure 1 | Not training-based | Not applicable | `figures/figure_generation_scripts/plot_fig1.py` |
| Figure 2 | `commands/figure2_commands.md` | `logs/figure2.logs/` | `figures/figure_generation_scripts/plot_fig2.py` |
| Figure 3 | `commands/figure3_commands.md` | `logs/figure3.logs/` | `figures/figure_generation_scripts/plot_fig3.py` |
| Table I | `commands/table1_commands.md` | `logs/table1.logs/` | final accuracies parsed from logs |
| Table II | `commands/table2_commands.md` | `logs/table2.logs/` | FLOP/time summaries parsed from logs |
| Table III | `commands/table3_commands.md` | `logs/table3.logs/` | final accuracies parsed from logs |
| Table IV | `commands/table4_commands.md` | `logs/table4.logs/` | ablation final accuracies parsed from logs |

## 10. Notes for reviewers

1. The fastest independent check is to regenerate `figure2.pdf` and `figure3.pdf` from archived logs using Section 5.
2. Full retraining is computationally expensive. Each 100-round federated run uses 10 clients and 40 local steps per communication round.
3. The archived logs are the exact logs used to produce the submitted figures and tables.
4. If a different GPU, CUDA version, or PyTorch build is used, small numerical differences are expected. Use the same seeds and commands to compare trends and final accuracies.
5. The command-line configuration is intentionally explicit. Paper-affecting hyperparameters are passed as arguments rather than hidden in source code.
