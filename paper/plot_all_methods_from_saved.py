"""Build the (a)-(g) combined comparison figure for every noise-category
dataset directly from the already-computed CSVs in
results/ucr/comparison/all_methods_timeseries/ and the thresholds table
-- no recomputation, just plotting.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

TS_DIR = Path("results/ucr/comparison/all_methods_timeseries")
THRESH_PATH = Path("results/ucr/comparison/all_methods_thresholds.csv")
OUT_DIR = Path("results/ucr/comparison/figures/all_methods")

COLOR_INK = "#0b0b0b"
COLOR_MUTED = "#6b6a65"
COLOR_LINE = "#1a1a1a"
COLOR_ANOMALY_SPAN = "#c9c7c0"
COLOR_LIMIT = "#8a8a8a"
COLOR_DISCORD = "#c0392b"

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
        "font.size": 10.0,
        "axes.labelsize": 10.0,
        "axes.titlesize": 10.0,
        "xtick.labelsize": 9.0,
        "ytick.labelsize": 9.0,
        "legend.fontsize": 8.0,
        "axes.linewidth": 0.8,
    }
)


def _style_axis(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(COLOR_MUTED)
    ax.spines["bottom"].set_color(COLOR_MUTED)
    ax.tick_params(colors=COLOR_MUTED)
    ax.set_xticks([])


def _panel(ax, x, y, threshold, anomaly_span, title, ylabel, label):
    _style_axis(ax)
    ax.axvspan(*anomaly_span, color=COLOR_ANOMALY_SPAN, alpha=0.6, linewidth=0)
    ax.axhline(threshold, color=COLOR_LIMIT, linewidth=1.1, linestyle="--", label=f"threshold (alpha=0.05)")
    mask = ~np.isnan(y)
    ax.plot(x[mask], y[mask], color=COLOR_LINE, linewidth=0.9, zorder=3, label=label)
    sig = mask & (y > threshold)
    ax.fill_between(x, 0, np.nan_to_num(y), where=sig, color=COLOR_DISCORD, alpha=0.3, linewidth=0, zorder=2)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontsize=10.0, color=COLOR_INK)
    ax.legend(loc="upper right", fontsize=7.5, frameon=False)


def plot_one(dataset_name: str, thresholds: dict) -> Path:
    df = pd.read_csv(TS_DIR / f"{dataset_name}.csv")
    x = df["time"].to_numpy()
    anomaly_positions = np.where(df["is_anomaly"].to_numpy() == 1)[0]
    anomaly_span = (int(anomaly_positions.min()), int(anomaly_positions.max()))

    fig, axes = plt.subplots(7, 1, figsize=(8, 11.5), height_ratios=[1] + [1.05] * 6)

    ax = axes[0]
    _style_axis(ax)
    ax.plot(x, df["value"].to_numpy(), color=COLOR_INK, linewidth=0.6)
    ax.axvspan(*anomaly_span, color=COLOR_ANOMALY_SPAN, alpha=0.6, linewidth=0, label="Anomaly")
    y_min, y_max = ax.get_ylim()
    ax.set_ylim(y_min, y_max + 0.18 * (y_max - y_min))
    ax.set_ylabel("Value")
    ax.set_title(f"(a) {dataset_name}  [test region]", loc="left", fontsize=10.0, color=COLOR_INK)
    ax.legend(loc="upper right", fontsize=8, frameon=True, facecolor="white", edgecolor="none")

    specs = [
        ("b", "merlin_nn_distance", "merlin", "MERLIN Discord Profile", "NN Distance", "NN distance"),
        ("c", "ours_variance_ratio", "ours", "Our Method: Bootstrap-Calibrated", "Variance Ratio", "Variance ratio"),
        ("d", "naive_rolling_std", "naive_rolling_std", "Naive Baseline: Rolling Std", "Rolling Std", "movstd"),
        ("e", "cusum", "cusum", "CUSUM on Variance Ratios", "CUSUM", "Cumulative sum"),
        ("f", "autoencoder_error", "autoencoder", "Autoencoder Reconstruction Error", "Recon. MSE", "AE error"),
        ("g", "lstm_error", "lstm", "LSTM Forecasting Error", "Forecast MSE", "LSTM error"),
    ]
    for ax, (letter, col, method_key, name, ylabel, label) in zip(axes[1:], specs):
        y = df[col].to_numpy()
        threshold = thresholds[(dataset_name, method_key)]
        _panel(ax, x, y, threshold, anomaly_span, f"({letter}) {name} (w=100)", ylabel, label)
    axes[-1].set_xlabel("Sample")

    fig.tight_layout()
    fig.subplots_adjust(right=1.0 - fig.subplotpars.left)
    output_path = OUT_DIR / f"{dataset_name}_win100.png"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for attempt in range(3):
        try:
            fig.savefig(output_path, dpi=200)
            break
        except OSError:
            if attempt == 2:
                raise
            import time
            time.sleep(0.5)
    plt.close(fig)
    return output_path


def main() -> None:
    thresh_df = pd.read_csv(THRESH_PATH)
    thresholds = {(row["dataset"], row["method"]): row["threshold"] for _, row in thresh_df.iterrows()}
    dataset_names = sorted(p.stem for p in TS_DIR.glob("*.csv"))
    for i, name in enumerate(dataset_names, 1):
        out_path = plot_one(name, thresholds)
        print(f"[{i}/{len(dataset_names)}] {name} -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
