"""Generate Figure 2 from logs/figure2.logs or logs/fig2logs."""
from __future__ import annotations
import argparse
import re
from collections import defaultdict
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

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
    ("cifar10", "alexnet", "(a) CIFAR10-AlexNet"),
    ("cifar100", "alexnet", "(b) CIFAR100-AlexNet"),
    ("mnist", "alexnet", "(c) MNIST-AlexNet"),
    ("cifar10", "resnet18", "(d) CIFAR10-ResNet18"),
    ("cifar100", "resnet18", "(e) CIFAR100-ResNet18"),
    ("mnist", "resnet18", "(f) MNIST-ResNet18"),
    ("cifar10", "googlenet", "(g) CIFAR10-GoogleNet"),
    ("cifar100", "googlenet", "(h) CIFAR100-GoogleNet"),
    ("mnist", "googlenet", "(i) MNIST-GoogleNet"),
]
METHOD_ORDER = ["fedavg", "fedprox", "fedsophia", "dynahybrid-sophia"]
METHOD_LABEL = {"fedavg":"FedAvg", "fedprox":"FedProx", "fedsophia":"FedSophia", "dynahybrid-sophia":"Dynahybrid-sophia"}
METHOD_STYLE = {
    "fedavg": {
        "color": "#0072B2",  # 蓝色
        "linestyle": "-",
        "linewidth": 1.75,
    },
    "fedprox": {
        "color": "#D62728",  # 红色
        "linestyle": "-",
        "linewidth": 1.75,
    },
    "fedsophia": {
        "color": "#8E44AD",  # 紫色
        "linestyle": "-",
        "linewidth": 1.75,
    },
    "dynahybrid-sophia": {
        "color": "#009E73",  # 绿色
        "linestyle": "-",
        "linewidth": 1.75,
    },
}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--png", type=Path, default=None)
    args = parser.parse_args()
    log_dir = resolve_data_root(args.data_root, 2)
    data = parse_logs(log_dir, METHOD_ORDER)
    validate(data, PANELS, METHOD_ORDER)
    apply_style()
    fig, axes = plt.subplots(3, 3, figsize=(7.16, 7.55), sharex=False, sharey=False)
    x = np.arange(1, 101, dtype=float)

    for axis, (dataset, model, title) in zip(axes.flat, PANELS):
        key = (dataset, model)
        means = {}
        for method in METHOD_ORDER:
            curves = np.vstack([data[key][method][seed] for seed in sorted(data[key][method])])
            mean_curve = curves.mean(axis=0)
            means[method] = mean_curve
            axis.plot(x, mean_curve, label=METHOD_LABEL[method], **METHOD_STYLE[method])

        axis.set_xlim(0, 100)
        axis.set_ylim(0, 100)
        axis.set_xticks([0, 20, 40, 60, 80, 100])
        axis.set_yticks([0, 20, 40, 60, 80, 100])
        axis.set_xlabel("Communication Round", labelpad=2)
        axis.set_ylabel("Test Accuracy (%)", labelpad=2)
        axis.grid(True, linestyle=":", linewidth=0.55, alpha=0.55)
        axis.tick_params(direction="in", length=2.5, width=0.6)
        axis.legend(loc="lower right", frameon=True, framealpha=0.92, edgecolor="#666666",
                    fancybox=False, fontsize=5.8, handlelength=2.2, borderpad=0.30, labelspacing=0.22)
        axis.text(0.5, -0.26, title, transform=axis.transAxes, ha="center", va="top", fontsize=8.5)

        # ========== 只对 MNIST 三个子图加局部放大 ==========
        if dataset == "mnist":
            # 放大区域：横坐标 0-20，纵坐标 80-100（更高放大比例）
            x1, x2 = 0, 20
            y1, y2 = 80, 100

            # 1. 主图上画浅灰矩形框
            rect = Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=0.8, edgecolor="#555555", linestyle="--",
                facecolor="#e8e8e8", alpha=0.55, zorder=0
            )
            axis.add_patch(rect)

            # 2. 创建 inset 放大图（更大、居中，无刻度数字）
            axins = inset_axes(axis, width="52%", height="42%", loc="center", borderpad=0.3)
            axins.set_facecolor("#f7f7f7")

            for method in METHOD_ORDER:
                axins.plot(x, means[method], **METHOD_STYLE[method])

            axins.set_xlim(x1, x2)
            axins.set_ylim(y1, y2)

            # 去掉所有刻度数字和刻度线
            axins.set_xticks([])
            axins.set_yticks([])
            axins.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

            axins.grid(True, linestyle=":", linewidth=0.4, alpha=0.5)
            for spine in axins.spines.values():
                spine.set_linewidth(0.7)
                spine.set_edgecolor("#444444")

            # 3. 画虚线连接主图区域与 inset
            mark_inset(axis, axins, loc1=2, loc2=4, fc="none", ec="#555555",
                       linestyle="--", linewidth=0.7)

    fig.subplots_adjust(left=0.085, right=0.99, bottom=0.075, top=0.985, hspace=0.58, wspace=0.36)
    out = (args.output or (project_root() / "figures" / "figure2.pdf")).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, format="pdf", bbox_inches="tight", pad_inches=0.02)
    if args.png:
        png = args.png.resolve()
        png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(png, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"Saved: {out}")

if __name__ == "__main__":
    main()