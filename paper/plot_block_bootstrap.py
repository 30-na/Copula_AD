"""Visualize the block-bootstrap calibration result:
  (a) empirical null distribution (histogram) vs. the theoretical
      F-distribution density, with both sets of control limits marked
  (b) the real test data's variance-ratio trace, panel-b style, with
      BOTH sets of control limits drawn so you can see directly which
      points each one would flag
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from algorithm import fit_reference_model
from test_block_bootstrap import CONFIG, N_TRAIN, WINDOW_SIZE, SEIZURE_NUMBER, ALPHA, f_critical_values

COLOR_INK = "#0b0b0b"
COLOR_MUTED = "#6b6a65"
COLOR_LINE = "#1a1a1a"
COLOR_F = "#b0462f"
COLOR_EMP = "#2f6ab0"
COLOR_ANOMALY_SPAN = "#c9c7c0"

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 10.5,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.linewidth": 0.8,
    }
)


def _style_axis(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(COLOR_MUTED)
    ax.spines["bottom"].set_color(COLOR_MUTED)
    ax.tick_params(colors=COLOR_MUTED)


def main() -> None:
    values = pd.read_csv(f"data/mice/{CONFIG}/timeseries_seizure{SEIZURE_NUMBER}.csv")["value"].to_numpy(float)
    labels = pd.read_csv(f"data/mice/{CONFIG}/labels_seizure{SEIZURE_NUMBER}.csv")["is_anomaly"].to_numpy(int)
    window_results = pd.read_csv(f"results/mice/{CONFIG}/window_results_seizure{SEIZURE_NUMBER}.csv")
    t_star = np.load("results/mice/figs/block_bootstrap_t_star.npy")

    train_values = values[:N_TRAIN]
    model = fit_reference_model(train_values, 3, 3)

    empirical_lower, empirical_upper = np.percentile(t_star, [100 * ALPHA / 2, 100 * (1 - ALPHA / 2)])
    f_lower, f_upper = f_critical_values(WINDOW_SIZE, model["reference_df"], ALPHA)

    fig, axes = plt.subplots(2, 1, figsize=(9, 7.5), height_ratios=(1, 1.1))

    # Panel (a): empirical null histogram vs theoretical F density
    ax = axes[0]
    _style_axis(ax)
    ax.hist(t_star, bins=80, density=True, color="#cfd8dc", edgecolor="none", label="Empirical null (block bootstrap)")
    x = np.linspace(0.01, max(t_star.max(), f_upper) * 1.05, 2000)
    f_pdf = stats.f.pdf(x, WINDOW_SIZE, model["reference_df"])
    ax.plot(x, f_pdf, color=COLOR_F, linewidth=1.6, label=f"Theoretical F({WINDOW_SIZE}, {model['reference_df']}) density")
    ax.axvline(f_lower, color=COLOR_F, linewidth=1.1, linestyle="--")
    ax.axvline(f_upper, color=COLOR_F, linewidth=1.1, linestyle="--", label="F-distribution limits")
    ax.axvline(empirical_lower, color=COLOR_EMP, linewidth=1.1, linestyle=":")
    ax.axvline(empirical_upper, color=COLOR_EMP, linewidth=1.1, linestyle=":", label="Empirical limits")
    ax.set_xlim(0, max(t_star.max(), f_upper) * 1.05)
    ax.set_xlabel("Variance ratio T")
    ax.set_ylabel("Density")
    ax.set_title("(a) Empirical null distribution vs. theoretical F distribution", loc="left")
    ax.legend(loc="upper right", fontsize=8, frameon=False)

    # Panel (b): real test data, both control limit sets overlaid
    ax = axes[1]
    _style_axis(ax)
    anomaly_positions = np.where(labels[N_TRAIN:] == 1)[0]
    anomaly_span = (int(anomaly_positions.min()), int(anomaly_positions.max()))
    ax.axvspan(*anomaly_span, color=COLOR_ANOMALY_SPAN, alpha=0.6, linewidth=0, label="Seizure")
    ax.axhline(1.0, color=COLOR_MUTED, linewidth=0.8, linestyle=":")
    ax.axhline(f_lower, color=COLOR_F, linewidth=1.1, linestyle="--")
    ax.axhline(f_upper, color=COLOR_F, linewidth=1.1, linestyle="--", label="F-distribution limits (current pipeline)")
    ax.axhline(empirical_lower, color=COLOR_EMP, linewidth=1.1, linestyle=":")
    ax.axhline(empirical_upper, color=COLOR_EMP, linewidth=1.1, linestyle=":", label="Empirical block-bootstrap limits")

    x = window_results["window_start"] + WINDOW_SIZE // 2
    ratio = window_results["variance_ratio"]
    ax.plot(x, ratio, color=COLOR_LINE, linewidth=0.8, zorder=3)
    ax.scatter(x, ratio, s=10, color=COLOR_LINE, zorder=4)
    ax.set_ylim(0, min(ratio.max() * 1.1, 40))
    ax.set_xlabel("Sample (test region)")
    ax.set_ylabel("Variance ratio")
    ax.set_title("(b) Real test data: which points each calibration flags", loc="left")
    ax.legend(loc="upper right", fontsize=8, frameon=False)

    fig.suptitle(f"Block-bootstrap calibration — Seizure {SEIZURE_NUMBER}, channel 1, window={WINDOW_SIZE}", fontsize=12, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    output_path = "results/mice/figs/block_bootstrap_seizure1.png"
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
