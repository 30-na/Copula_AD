"""Comparison plots for the three-method evaluation (F-theory / block
bootstrap / Bayesian posterior odds), for either the mice or UCR
comparison table. Two panels: (a) false-alarm rate per method (box/strip
across all combos), (b) detection rate per method, same layout.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLOR_INK = "#0b0b0b"
COLOR_MUTED = "#6b6a65"
METHOD_COLORS = {"f": "#b0462f", "bb": "#2f6ab0", "bayes": "#2f8a4e"}
METHOD_LABELS = {"f": "F-distribution\n(theory)", "bb": "Block\nbootstrap", "bayes": "Bayesian\nposterior odds"}

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
        "font.size": 10.5,
        "axes.labelsize": 10.5,
        "axes.titlesize": 11,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "axes.linewidth": 0.8,
    }
)


def _style_axis(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(COLOR_MUTED)
    ax.spines["bottom"].set_color(COLOR_MUTED)
    ax.tick_params(colors=COLOR_MUTED)


def plot_comparison(table: pd.DataFrame, title: str, output_path: str, alpha_target: float = 0.05) -> None:
    methods = ["f", "bb", "bayes"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8))

    for col, (metric, metric_label) in enumerate([("false_alarm_rate", "False-alarm rate"), ("detection_rate", "Detection rate")]):
        ax = axes[col]
        _style_axis(ax)
        data = [table[f"{m}_{metric}"].dropna().to_numpy() for m in methods]
        positions = np.arange(1, len(methods) + 1)
        bp = ax.boxplot(data, positions=positions, widths=0.5, patch_artist=True, showmeans=True,
                         medianprops={"color": COLOR_INK}, meanprops={"marker": "D", "markerfacecolor": "white", "markeredgecolor": COLOR_INK, "markersize": 5})
        for patch, m in zip(bp["boxes"], methods):
            patch.set_facecolor(METHOD_COLORS[m])
            patch.set_alpha(0.35)
            patch.set_edgecolor(METHOD_COLORS[m])
        for i, m in enumerate(methods):
            y = table[f"{m}_{metric}"].dropna().to_numpy()
            x = np.random.default_rng(0).normal(positions[i], 0.06, size=len(y))
            ax.scatter(x, y, s=14, color=METHOD_COLORS[m], alpha=0.6, zorder=3)
        if metric == "false_alarm_rate":
            ax.axhline(alpha_target, color=COLOR_MUTED, linewidth=1.0, linestyle="--", label=f"target (alpha={alpha_target})")
            ax.legend(loc="upper right", fontsize=8.5, frameon=False)
        ax.set_xticks(positions)
        ax.set_xticklabels([METHOD_LABELS[m] for m in methods])
        ax.set_ylabel(metric_label)
        ax.set_ylim(-0.02, 1.02)
        ax.set_title(f"({'a' if col == 0 else 'b'}) {metric_label} by method", loc="left")

    fig.suptitle(title, fontsize=12.5, color=COLOR_INK, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"Saved {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    table = pd.read_csv(args.table)
    plot_comparison(table, args.title, args.output)


if __name__ == "__main__":
    main()
