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


# ============================================================
# File / log parsing
# ============================================================

FILE_RE = re.compile(
    r"^(?P<dataset>cifar10|cifar100|mnist)_"
    r"(?P<model>alexnet|resnet18|googlenet)_"
    r"seed(?P<seed>\d+)\.log$"
)

LINE_RE = re.compile(
    r"\[(?P<method>[^\]]+)\]\s+"
    r"Round\s+(?P<round>\d+)\s*/\s*100\s+\|"
    r".*?test_acc=(?P<acc>[0-9.]+)%"
)


# ============================================================
# Project root
# ============================================================

def project_root() -> Path:
    here = Path(__file__).resolve()

    if here.parent.name == "figure_generation_scripts":
        return here.parents[2]

    if here.parent.name == "figures":
        return here.parents[1]

    return Path.cwd()


# ============================================================
# Locate Figure 2 log directory
# ============================================================

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
    raise FileNotFoundError(
        f"Could not find Figure {figure_no} log directory. Checked:\n{checked}"
    )


# ============================================================
# Parse logs
# ============================================================

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
            if method not in method_order:
                continue

            missing = sorted(set(range(1, 101)) - set(rounds))
            if missing:
                raise ValueError(
                    f"{path.name} / {method} missing rounds: {missing[:8]}"
                )

            data[(dataset, model)][method][seed] = np.array(
                [rounds[i] for i in range(1, 101)],
                dtype=float,
            )

    return data


# ============================================================
# Validate data
# ============================================================

def validate(data, panels, method_order):
    for dataset, model, _ in panels:
        key = (dataset, model)

        if key not in data:
            raise ValueError(f"Missing logs for {dataset}/{model}")

        for method in method_order:
            seeds = data[key].get(method, {})
            if sorted(seeds) != [42, 43, 44]:
                raise ValueError(
                    f"Expected seeds 42,43,44 for {dataset}/{model}/{method}; "
                    f"found {sorted(seeds)}"
                )


# ============================================================
# Matplotlib style
# ============================================================

def apply_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": [
            "Times New Roman",
            "Times",
            "Liberation Serif",
            "DejaVu Serif",
        ],
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


# ============================================================
# Figure panels
# ============================================================

PANELS = [
    ("cifar10",  "alexnet",   "(a) CIFAR10-AlexNet"),
    ("cifar100", "alexnet",   "(b) CIFAR100-AlexNet"),
    ("mnist",    "alexnet",   "(c) MNIST-AlexNet"),
    ("cifar10",  "resnet18",  "(d) CIFAR10-ResNet18"),
    ("cifar100", "resnet18",  "(e) CIFAR100-ResNet18"),
    ("mnist",    "resnet18",  "(f) MNIST-ResNet18"),
    ("cifar10",  "googlenet", "(g) CIFAR10-GoogleNet"),
    ("cifar100", "googlenet", "(h) CIFAR100-GoogleNet"),
    ("mnist",    "googlenet", "(i) MNIST-GoogleNet"),
]


# ============================================================
# Methods
# ============================================================

METHOD_ORDER = [
    "fedavg",
    "fedprox",
    "fedsophia",
    "dynahybrid-sophia",
]

METHOD_LABEL = {
    "fedavg": "FedAvg",
    "fedprox": "FedProx",
    "fedsophia": "FedSophia",
    "dynahybrid-sophia": "Dynahybrid-sophia",
}


# ============================================================
# Line styles
# ============================================================

METHOD_STYLE = {
    "fedavg": {
        "color": "#0072B2",
        "linestyle": "-",
        "linewidth": 1.2,
    },
    "fedprox": {
        "color": "#D62728",
        "linestyle": "-",
        "linewidth": 1.2,
    },
    "fedsophia": {
        "color": "#8E44AD",
        "linestyle": "-",
        "linewidth": 1.2,
    },
    "dynahybrid-sophia": {
        "color": "#009E73",
        "linestyle": "-",
        "linewidth": 1.2,
    },
}


# ============================================================
# Marker styles
# ============================================================

METHOD_MARKER = {
    "fedavg": "^",              # triangle
    "fedprox": "o",             # circle
    "fedsophia": "s",           # square
    "dynahybrid-sophia": "*",   # star
}

METHOD_MARKER_SIZE = {
    "fedavg": 5.2,
    "fedprox": 5.0,
    "fedsophia": 5.0,
    "dynahybrid-sophia": 6.0,
}

# Markers shown at these fixed rounds for all panels
MARKER_ROUNDS = [20, 40, 60, 80, 100]


# ============================================================
# Elbow detection for MNIST curves
# ============================================================

def find_elbow_index(curve: np.ndarray, search_end: int = 30) -> int | None:
    """
    Detect an elbow point in the early training stage:
    find the point where the curve changes from rapid growth
    to a flatter trend.

    Returns a 0-based index, or None if no valid elbow is found.
    """
    y = np.asarray(curve, dtype=float)
    n = len(y)

    if n < 5:
        return None

    k = min(search_end, n)
    y_early = y[:k]

    # simple smoothing
    kernel = np.array([1.0, 1.0, 1.0], dtype=float) / 3.0
    y_pad = np.pad(y_early, (1, 1), mode="edge")
    y_smooth = np.convolve(y_pad, kernel, mode="valid")

    dy = np.diff(y_smooth)
    if len(dy) == 0:
        return None

    peak_idx = int(np.argmax(dy))
    peak_val = float(dy[peak_idx])

    if peak_val <= 0:
        return None

    # threshold for "flattening"
    threshold = peak_val * 0.18

    for j in range(peak_idx + 1, len(dy)):
        if dy[j] <= threshold:
            elbow_idx = j + 1
            return min(max(elbow_idx, 0), n - 1)

    # fallback
    elbow_idx = peak_idx + 1
    return min(max(elbow_idx, 0), n - 1)


# ============================================================
# Plot curve + markers
# ============================================================

def plot_curve_with_markers(
    axis,
    x,
    mean_curve,
    method,
    label=None,
    extra_marker_indices=None,
):
    marker_indices = [r - 1 for r in MARKER_ROUNDS]

    if extra_marker_indices is not None:
        marker_indices.extend(extra_marker_indices)

    marker_indices = sorted(set(
        idx for idx in marker_indices
        if 0 <= idx < len(mean_curve)
    ))

    axis.plot(
        x,
        mean_curve,
        label=label,
        marker=METHOD_MARKER[method],
        markevery=marker_indices,
        markersize=METHOD_MARKER_SIZE[method],
        markeredgewidth=0.9,
        **METHOD_STYLE[method],
    )


# ============================================================
# Main
# ============================================================

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

    fig, axes = plt.subplots(
        3, 3,
        figsize=(7.16, 7.55),
        sharex=False,
        sharey=False,
    )

    # data rounds: 1~100
    x = np.arange(1, 101, dtype=float)

    for axis, (dataset, model, title) in zip(axes.flat, PANELS):
        key = (dataset, model)
        means = {}
        elbow_indices = {}

        # ----------------------------------------------------
        # Compute mean curves and elbow points first
        # ----------------------------------------------------
        for method in METHOD_ORDER:
            curves = np.vstack([
                data[key][method][seed]
                for seed in sorted(data[key][method])
            ])
            mean_curve = curves.mean(axis=0)
            means[method] = mean_curve

            if dataset == "mnist":
                elbow_indices[method] = find_elbow_index(mean_curve)
            else:
                elbow_indices[method] = None

        # ----------------------------------------------------
        # Plot four methods
        # ----------------------------------------------------
        for method in METHOD_ORDER:
            extra_indices = []
            if dataset == "mnist" and elbow_indices[method] is not None:
                extra_indices.append(elbow_indices[method])

            plot_curve_with_markers(
                axis=axis,
                x=x,
                mean_curve=means[method],
                method=method,
                label=METHOD_LABEL[method],
                extra_marker_indices=extra_indices,
            )

        # ----------------------------------------------------
        # Axis settings
        # ----------------------------------------------------
        axis.set_xlim(0, 100)
        axis.set_ylim(0, 100)
        axis.set_xticks([0, 20, 40, 60, 80, 100])
        axis.set_yticks([0, 20, 40, 60, 80, 100])

        axis.set_xlabel("Communication Round", labelpad=2)
        axis.set_ylabel("Test Accuracy (%)", labelpad=2)

        axis.grid(True, linestyle=":", linewidth=0.55, alpha=0.55)
        axis.tick_params(direction="in", length=2.5, width=0.6)

        axis.legend(
            loc="lower right",
            frameon=True,
            framealpha=0.92,
            edgecolor="#666666",
            fancybox=False,
            fontsize=5.8,
            handlelength=2.2,
            borderpad=0.30,
            labelspacing=0.22,
        )

        axis.text(
            0.5, -0.26, title,
            transform=axis.transAxes,
            ha="center", va="top",
            fontsize=8.5,
        )

        # ----------------------------------------------------
        # Add local zoom only for MNIST panels
        # ----------------------------------------------------
        if dataset == "mnist":
            x1, x2 = 0, 20
            y1, y2 = 80, 100

            rect = Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=0.8,
                edgecolor="#555555",
                linestyle="--",
                facecolor="#e8e8e8",
                alpha=0.55,
                zorder=0,
            )
            axis.add_patch(rect)

            axins = inset_axes(
                axis,
                width="52%",
                height="42%",
                loc="center",
                borderpad=0.3,
            )
            axins.set_facecolor("#f7f7f7")

            for method in METHOD_ORDER:
                extra_indices = []
                if elbow_indices[method] is not None:
                    extra_indices.append(elbow_indices[method])

                plot_curve_with_markers(
                    axis=axins,
                    x=x,
                    mean_curve=means[method],
                    method=method,
                    extra_marker_indices=extra_indices,
                )

            axins.set_xlim(x1, x2)
            axins.set_ylim(y1, y2)

            axins.set_xticks([])
            axins.set_yticks([])
            axins.tick_params(
                left=False,
                bottom=False,
                labelleft=False,
                labelbottom=False,
            )

            axins.grid(True, linestyle=":", linewidth=0.4, alpha=0.5)

            for spine in axins.spines.values():
                spine.set_linewidth(0.7)
                spine.set_edgecolor("#444444")

            mark_inset(
                axis, axins,
                loc1=2, loc2=4,
                fc="none", ec="#555555",
                linestyle="--", linewidth=0.7,
            )

    fig.subplots_adjust(
        left=0.085,
        right=0.99,
        bottom=0.075,
        top=0.985,
        hspace=0.58,
        wspace=0.36,
    )

    out = (args.output or (project_root() / "figures" / "figure2.pdf")).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(
        out,
        format="pdf",
        bbox_inches="tight",
        pad_inches=0.02,
    )

    if args.png:
        png = args.png.resolve()
        png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            png,
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.02,
        )

    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()