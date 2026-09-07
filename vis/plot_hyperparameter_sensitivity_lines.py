#!/usr/bin/env python3
"""Draw VMH-P hyperparameter sensitivity curves.

The solid curve is ACC. The shaded band is a visual trend band around each
curve, and the starred marker denotes the main-table default setting for each
hyperparameter.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


OUT_DIR = Path(__file__).resolve().parent
ABS_TREND_BAND = 0.30
DELTA_TREND_BAND = 0.16
BAND_ALPHA = 0.22
LINE_WIDTH = 2.35
MARKER_SIZE = 5.8

HOST_STYLE = {
    "CLMLF": {"color": "#4C78A8", "marker": "o"},
    "D2R": {"color": "#C44E52", "marker": "s"},
    "SPP-SCL": {"color": "#55A868", "marker": "^"},
}

DATA = {
    "K": {
        "x": [2, 3, 4, 5],
        "default": 3,
        "xlabel": r"$K$",
        "values": {
            "CLMLF": [(72.84, 72.79), (73.35, 73.32), (73.44, 73.44), (73.35, 73.32)],
            "D2R": [(77.34, 77.38), (77.39, 77.41), (77.49, 77.51), (77.48, 77.50)],
            "SPP-SCL": [(73.58, 73.64), (73.59, 73.67), (73.75, 73.82), (73.00, 73.10)],
        },
    },
    "c": {
        "x": [0.10, 0.25, 0.50, 1.00],
        "default": 0.25,
        "xlabel": r"$c$",
        "values": {
            "CLMLF": [(73.51, 73.49), (73.35, 73.32), (73.52, 73.46), (73.67, 73.58)],
            "D2R": [(77.46, 77.49), (77.39, 77.41), (77.34, 77.36), (77.40, 77.42)],
            "SPP-SCL": [(74.10, 74.20), (73.59, 73.67), (73.50, 73.56), (73.22, 73.28)],
        },
    },
    r"$\eta$": {
        "x": [0.05, 0.10, 0.20, 0.40, 0.50],
        "default": 0.50,
        "xlabel": r"$\eta$",
        "values": {
            "CLMLF": [(72.37, 72.12), (72.89, 72.84), (72.99, 72.81), (73.51, 73.49), (73.35, 73.32)],
            "D2R": [(77.25, 77.28), (77.30, 77.33), (77.07, 77.09), (77.46, 77.47), (77.39, 77.41)],
            "SPP-SCL": [(73.65, 73.72), (73.40, 73.47), (73.57, 73.66), (73.52, 73.58), (73.59, 73.67)],
        },
    },
    r"$\lambda_{\mathrm{str}}$": {
        "x": [0.10, 0.30, 0.50, 0.70],
        "default": 0.10,
        "xlabel": r"$\lambda_{\mathrm{str}}$",
        "values": {
            "CLMLF": [(73.35, 73.32), (73.74, 73.72), (73.61, 73.56), (73.51, 73.52)],
            "D2R": [(77.39, 77.41), (76.98, 77.02), (77.32, 77.34), (77.48, 77.51)],
            "SPP-SCL": [(73.59, 73.67), (73.63, 73.71), (73.73, 73.82), (73.89, 73.99)],
        },
    },
}


def metric_arrays(pairs: list[tuple[float, float]]) -> tuple[list[float], list[float]]:
    acc = [a for a, _ in pairs]
    f1 = [f for _, f in pairs]
    return acc, f1


def tick_labels(xs: list[float | int]) -> list[str]:
    labels = []
    for x in xs:
        if isinstance(x, int) or float(x).is_integer():
            labels.append(str(int(x)))
        else:
            labels.append(f"{x:.2f}".rstrip("0").rstrip("."))
    return labels


def plot_positions(xs: list[float | int]) -> list[float]:
    return [float(i) for i in range(len(xs))]


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "dejavuserif",
            "axes.linewidth": 1.0,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def draw_absolute() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 5.8), constrained_layout=True)
    axes = axes.ravel()

    for panel_idx, (ax, (name, spec)) in enumerate(zip(axes, DATA.items())):
        xs = spec["x"]
        xpos = plot_positions(xs)
        default = spec["default"]
        default_pos = xpos[xs.index(default)]
        all_scores = []

        ax.axvline(default_pos, color="#8A8A8A", linestyle=":", linewidth=1.15, zorder=0)
        for host, pairs in spec["values"].items():
            style = HOST_STYLE[host]
            acc, _ = metric_arrays(pairs)
            point_band = [ABS_TREND_BAND for _ in acc]
            low = [a - b for a, b in zip(acc, point_band)]
            high = [a + b for a, b in zip(acc, point_band)]
            all_scores.extend(low + high)

            ax.fill_between(xpos, low, high, color=style["color"], alpha=BAND_ALPHA, linewidth=0)
            ax.plot(
                xpos,
                acc,
                color=style["color"],
                marker=style["marker"],
                markersize=MARKER_SIZE,
                linewidth=LINE_WIDTH,
                markeredgecolor="white",
                markeredgewidth=0.65,
                label=host,
            )

            default_idx = xs.index(default)
            ax.scatter(
                [default_pos],
                [acc[default_idx]],
                color=style["color"],
                marker="*",
                s=105,
                edgecolor="white",
                linewidth=0.8,
                zorder=5,
            )

        margin = max(0.18, (max(all_scores) - min(all_scores)) * 0.10)
        ax.set_ylim(min(all_scores) - margin, max(all_scores) + margin)
        ax.set_xlabel(spec["xlabel"])
        ax.set_ylabel("Performance (%)" if panel_idx % 2 == 0 else "")
        ax.set_xlim(-0.35, len(xs) - 0.65)
        ax.set_xticks(xpos)
        ax.set_xticklabels(tick_labels(xs))
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.75, alpha=0.82)
        ax.tick_params(axis="both", width=0.95, length=4.0, color="#222222")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#222222")
        ax.spines["bottom"].set_color("#222222")

    handles = [
        Line2D(
            [0],
            [0],
            color=style["color"],
            marker=style["marker"],
            linewidth=LINE_WIDTH,
            markersize=MARKER_SIZE,
            markeredgecolor="white",
            markeredgewidth=0.65,
            label=host,
        )
        for host, style in HOST_STYLE.items()
    ]
    handles.append(
        Line2D([0], [0], color="#8A8A8A", marker="*", linestyle=":", linewidth=1.0, markersize=8, label="Main setting")
    )
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=True, bbox_to_anchor=(0.5, 1.045))

    for ext in ("pdf", "png", "svg"):
        fig.savefig(OUT_DIR / f"hyper_sensitivity_lines_abs.{ext}", dpi=400, bbox_inches="tight")
    plt.close(fig)


def draw_delta() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 5.8), constrained_layout=True)
    axes = axes.ravel()

    for panel_idx, (ax, (name, spec)) in enumerate(zip(axes, DATA.items())):
        xs = spec["x"]
        xpos = plot_positions(xs)
        default = spec["default"]
        default_pos = xpos[xs.index(default)]
        all_delta = []

        ax.axhline(0.0, color="#5A5A5A", linestyle=":", linewidth=1.0, zorder=0)
        ax.axvline(default_pos, color="#8A8A8A", linestyle=":", linewidth=1.15, zorder=0)

        for host, pairs in spec["values"].items():
            style = HOST_STYLE[host]
            acc, _ = metric_arrays(pairs)
            default_idx = xs.index(default)
            acc_base = acc[default_idx]
            acc_delta = [v - acc_base for v in acc]
            point_band = [DELTA_TREND_BAND for _ in acc_delta]
            low = [a - b for a, b in zip(acc_delta, point_band)]
            high = [a + b for a, b in zip(acc_delta, point_band)]
            all_delta.extend(low + high)

            ax.fill_between(xpos, low, high, color=style["color"], alpha=BAND_ALPHA, linewidth=0)
            ax.plot(
                xpos,
                acc_delta,
                color=style["color"],
                marker=style["marker"],
                markersize=MARKER_SIZE,
                linewidth=LINE_WIDTH,
                markeredgecolor="white",
                markeredgewidth=0.65,
                label=host,
            )
            ax.scatter(
                [default_pos],
                [0.0],
                color=style["color"],
                marker="*",
                s=105,
                edgecolor="white",
                linewidth=0.8,
                zorder=5,
            )

        margin = max(0.10, (max(all_delta) - min(all_delta)) * 0.15)
        ax.set_ylim(min(all_delta) - margin, max(all_delta) + margin)
        ax.set_xlabel(spec["xlabel"])
        ax.set_ylabel(r"$\Delta$ Performance" if panel_idx % 2 == 0 else "")
        ax.set_xlim(-0.35, len(xs) - 0.65)
        ax.set_xticks(xpos)
        ax.set_xticklabels(tick_labels(xs))
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.75, alpha=0.82)
        ax.tick_params(axis="both", width=0.95, length=4.0, color="#222222")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#222222")
        ax.spines["bottom"].set_color("#222222")

    handles = [
        Line2D(
            [0],
            [0],
            color=style["color"],
            marker=style["marker"],
            linewidth=LINE_WIDTH,
            markersize=MARKER_SIZE,
            markeredgecolor="white",
            markeredgewidth=0.65,
            label=host,
        )
        for host, style in HOST_STYLE.items()
    ]
    handles.append(
        Line2D([0], [0], color="#8A8A8A", marker="*", linestyle=":", linewidth=1.0, markersize=8, label="Main setting")
    )
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=True, bbox_to_anchor=(0.5, 1.045))

    for ext in ("pdf", "png", "svg"):
        fig.savefig(OUT_DIR / f"hyper_sensitivity_lines_delta.{ext}", dpi=400, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    configure_matplotlib()
    draw_absolute()
    draw_delta()


if __name__ == "__main__":
    main()
