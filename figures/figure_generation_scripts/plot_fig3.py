"""Generate Figure 3 from logs/figure3.logs or logs/fig3logs."""

from __future__ import annotations
import argparse
import re
from collections import defaultdict
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FILE_RE = re.compile(r"^(?P<dataset>cifar10|cifar100|mnist)_(?P<model>alexnet|resnet18|googlenet)_seed(?P<seed>\d+)\.log$")
LINE_RE = re.compile(r"\[(?P<method>[^\]]+)\]\s+Round\s+(?P<round>\d+)\s*/\s*100\s+\|.*?test_acc=(?P<acc>[0-9.]+)%")

def project_root() -> Path:
    here = Path(__file__).resolve()
    if here.parent.name == "figure_generation_scripts":
        return here.parents[2]
    if here.parent.name == "figures":
        return here.parents[1]
    return Path.cwd()

def resolve_data_root(value: Path | None, figure_no: int) -> Path:
    candidates = []
    if value is not None:
        candidates.append(value.expanduser())
    root = project_root()
    candidates += [
        root / "logs" / f"figure{figure_no}.logs",
        root / "logs" / f"fig{figure_no}logs",
        root / "logs" / f"fig{figure_no}.logs",
        Path.cwd() / "logs" / f"figure{figure_no}.logs",
        Path.cwd() / "logs" / f"fig{figure_no}logs",
    ]
    for path in candidates:
        path = path.resolve()
        if path.is_dir() and list(path.glob("*.log")):
            return path
    checked = "\n".join(str(p) for p in candidates)
    raise FileNotFoundError(f"Could not find Figure {figure_no} log directory. Checked:\n{checked}")

def parse_logs(log_dir: Path, method_order):
    data = defaultdict(lambda: defaultdict(dict))
    for path in sorted(log_dir.glob("*.log")):
        fm = FILE_RE.match(path.name.lower())
        if not fm:
            continue
        dataset = fm.group("dataset")
        model = fm.group("model")
        seed = int(fm.group("seed"))
        per_method = defaultdict(dict)
        text = path.read_text(encoding="utf-8", errors="ignore")
        for m in LINE_RE.finditer(text):
            method = m.group("method").strip().lower()
            round_id = int(m.group("round"))
            acc = float(m.group("acc"))
            per_method[method][round_id] = acc
        for method, rounds in per_method.items():
            if method in method_order:
                missing = sorted(set(range(1, 101)) - set(rounds))
                if missing:
                    raise ValueError(f"{path.name} / {method} missing rounds: {missing[:8]}")
                data[(dataset, model)][method][seed] = np.array([rounds[i] for i in range(1, 101)], dtype=float)
    return data

def validate(data, panels, method_order):
    for dataset, model, _ in panels:
        key = (dataset, model)
        if key not in data:
            raise ValueError(f"Missing logs for {dataset}/{model}")
        for method in method_order:
            seeds = data[key].get(method, {})
            if sorted(seeds) != [42, 43, 44]:
                raise ValueError(f"Expected seeds 42,43,44 for {dataset}/{model}/{method}; found {sorted(seeds)}")

def apply_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Liberation Serif", "DejaVu Serif"],
        "font.size": 8.0,
        "axes.titlesize": 9.0,
        "axes.labelsize": 8.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 6.0,
        "axes.linewidth": 0.7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

PANELS = [
    ("mnist", "resnet18", "(a) MNIST-ResNet18"),
    ("cifar10", "alexnet", "(b) CIFAR10-AlexNet"),
    ("cifar100", "alexnet", "(c) CIFAR100-AlexNet"),
]
METHOD_ORDER = ["fedsophia", "dynahybrid-sophia", "dynahybrid-adahessian"]
METHOD_LABEL = {"fedsophia":"FedSophia", "dynahybrid-sophia":"DynaHybrid-Sophia", "dynahybrid-adahessian":"DynaHybrid-AdaHessian"}
METHOD_STYLE = {
    "fedsophia": {"color":"#009E73", "linestyle":":", "linewidth":1.75},
    "dynahybrid-sophia": {"color":"#D55E00", "linestyle":"--", "linewidth":2.25},
    "dynahybrid-adahessian": {"color":"#CC79A7", "linestyle":"-", "linewidth":1.65},
}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--png", type=Path, default=None)
    args = parser.parse_args()
    log_dir = resolve_data_root(args.data_root, 3)
    data = parse_logs(log_dir, METHOD_ORDER)
    validate(data, PANELS, METHOD_ORDER)
    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.45), sharex=False, sharey=False)
    x = np.arange(1, 101, dtype=float)
    for axis, (dataset, model, title) in zip(axes.flat, PANELS):
        key = (dataset, model)
        for method in METHOD_ORDER:
            curves = np.vstack([data[key][method][seed] for seed in sorted(data[key][method])])
            axis.plot(x, curves.mean(axis=0), label=METHOD_LABEL[method], **METHOD_STYLE[method])
        axis.set_xlim(0, 100); axis.set_ylim(0, 100)
        axis.set_xticks([0,20,40,60,80,100]); axis.set_yticks([0,20,40,60,80,100])
        axis.set_xlabel("Communication Round", labelpad=2)
        axis.set_ylabel("Test Accuracy (%)", labelpad=2)
        axis.grid(True, linestyle=":", linewidth=0.55, alpha=0.55)
        axis.tick_params(direction="in", length=2.5, width=0.6)
        axis.legend(loc="lower right", frameon=True, framealpha=0.92, edgecolor="#666666", fancybox=False, fontsize=5.8, handlelength=2.2, borderpad=0.30, labelspacing=0.22)
        axis.text(0.5, -0.32, title, transform=axis.transAxes, ha="center", va="top", fontsize=8.5)
    fig.subplots_adjust(left=0.085, right=0.99, bottom=0.26, top=0.98, hspace=0.0, wspace=0.36)
    out = (args.output or (project_root()/"figures"/"figure3.pdf")).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, format="pdf", bbox_inches="tight", pad_inches=0.02)
    if args.png:
        png = args.png.resolve(); png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(png, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"Saved: {out}")
if __name__ == "__main__":
    main()
