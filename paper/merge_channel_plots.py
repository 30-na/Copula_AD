"""Merge the 4 channels' already-computed mice results into one 4-row
comparison figure per (seizure, window size) -- no re-fitting, no
re-scoring. Just reads the CSVs run_mice.py already saved per channel
and re-plots them stacked, sharing one time axis, for a fair side-by-side
look at how the innovation-variance-ratio signal differs by channel.

Assumes the folder-naming convention from run_mice.py: channel 1 has no
"_chN" suffix, channels 2-4 do (e.g. 2000hz_pre60s_win50_ntrain1000_ch3).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLOR_INK = "#0b0b0b"
COLOR_MUTED = "#6b6a65"
COLOR_LINE = "#1a1a1a"
COLOR_LIMIT = "#8a8a8a"
COLOR_ANOMALY_SPAN = "#c9c7c0"

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
        "font.size": 10.5,
        "axes.labelsize": 10.5,
        "axes.titlesize": 10.5,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "legend.fontsize": 9.5,
        "axes.linewidth": 0.8,
    }
)


def _style_axis(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(COLOR_MUTED)
    ax.spines["bottom"].set_color(COLOR_MUTED)
    ax.tick_params(colors=COLOR_MUTED)


def f_critical_values(window_size: int, reference_df: int, alpha: float) -> tuple[float, float]:
    return (
        float(stats.f.ppf(alpha / 2, window_size, reference_df)),
        float(stats.f.ppf(1 - alpha / 2, window_size, reference_df)),
    )


def merge_channels(
    base_config: str, seizure_number: int, n_train: int, window_size: int, alpha: float, output_path: Path,
) -> None:
    data_dir = Path("data/mice") / base_config
    results_dir = Path("results/mice") / base_config
    fig, axes = plt.subplots(4, 1, figsize=(8, 8.5), sharex=True)

    for row, channel in enumerate([1, 2, 3, 4]):
        tag = f"seizure{seizure_number}" + (f"_ch{channel}" if channel != 1 else "")

        labels = pd.read_csv(data_dir / f"labels_{tag}.csv")["is_anomaly"].to_numpy(int)
        results = pd.read_csv(results_dir / f"window_results_{tag}.csv")

        anomaly_positions = np.where(labels == 1)[0]
        anomaly_span = (int(anomaly_positions.min()), int(anomaly_positions.max())) if len(anomaly_positions) else None

        # reconstruct reference_df the same way fit_reference_model does:
        # it isn't saved directly, so back it out from the F-critical values
        # being consistent -- simplest is to just recompute it from n_train
        # and the model order, but the order isn't saved either. Instead we
        # rebuild control limits straight from the printed relationship:
        # reference_df = n_train - n_fixed_params. n_fixed_params isn't in
        # the CSV, so approximate control limits aren't redrawn here --
        # only the ratio trace and the significance markers (which the
        # original run already decided via p_value) are shown.
        ax = axes[row]
        _style_axis(ax)
        if anomaly_span is not None:
            ax.axvspan(*anomaly_span, color=COLOR_ANOMALY_SPAN, alpha=0.6, linewidth=0)
        ax.axhline(1.0, color=COLOR_MUTED, linewidth=1.0, linestyle=":")
        x = n_train + results["window_start"] + window_size // 2
        significant = results["p_value"] < alpha
        ratio = results["variance_ratio"]
        ax.plot(x, ratio, color=COLOR_LINE, linewidth=1.0, zorder=3)
        ax.scatter(x[significant], ratio[significant], s=14, color=COLOR_LINE, zorder=4)
        ax.scatter(x[~significant], ratio[~significant], s=14, facecolors="white", edgecolors=COLOR_LINE, linewidths=0.8, zorder=4)
        ax.set_xlim(0, len(labels))
        ax.set_ylabel("Variance Ratio")
        ax.set_title(f"Channel {channel}", loc="left", fontsize=10.5, color=COLOR_INK)

    axes[-1].set_xlabel("Sample")
    fig.suptitle(f"Seizure {seizure_number} — window={window_size}", fontsize=11, color=COLOR_INK, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seizure-index", type=int, default=0, help="0-indexed.")
    parser.add_argument("--n-train", type=int, default=1000)
    parser.add_argument("--window-size", type=int, default=100)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    seizure_number = args.seizure_index + 1
    base_config = "2000hz_pre60s_ntrain1000"
    if args.window_size != 100:
        base_config = f"2000hz_pre60s_win{args.window_size}_ntrain1000"
    output_path = Path(args.output) if args.output else Path(f"results/mice/figs/channels_seizure{seizure_number}_win{args.window_size}.png")
    merge_channels(base_config, seizure_number, args.n_train, args.window_size, args.alpha, output_path)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
