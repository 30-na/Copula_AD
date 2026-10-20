"""MERLIN-style baseline figure: discord (nearest-neighbor distance)
profile computed over the full test region at a FIXED subsequence
length (matching our method's fixed window_size), analogous to the
3-panel theory-vs-bootstrap figures but for the MERLIN baseline.

Uses stumpy's STOMP implementation to compute the EXACT nearest-neighbor
distance for every window position (what MERLIN's pruning search
approximates efficiently) so we can plot the full profile, not just
the single reported discord.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import stumpy
from scipy import stats

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from algorithm import fit_reference_model, make_windows, score_window, variance_ratio_pvalue
from compare_methods import build_block_bootstrap_null, valid_normal_starts
from run_ucr import find_train_end

WINDOW_SIZE = 100
ALPHA = 0.05
N_BOOTSTRAP = 2000

COLOR_INK = "#0b0b0b"
COLOR_MUTED = "#6b6a65"
COLOR_LINE = "#1a1a1a"
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
    ax.set_xticks([])


def main() -> None:
    import sys
    dataset_name = sys.argv[1] if len(sys.argv) > 1 else "ucr_003_DISTORTED3sddb40"
    window_size = int(sys.argv[2]) if len(sys.argv) > 2 else WINDOW_SIZE

    dataset_dir = Path("data/ucr_prepared/noise") / dataset_name
    raw_dir = Path("data/ucr_raw")
    train_end = find_train_end(raw_dir, dataset_name)

    ts = pd.read_csv(dataset_dir / "timeseries.csv")
    labels = pd.read_csv(dataset_dir / "labels.csv")
    values = ts["value"].to_numpy(dtype=float)
    is_anomaly = labels["is_anomaly"].to_numpy(dtype=int)

    train_values = values[:train_end]
    test_values = values[train_end:]
    test_is_anomaly = is_anomaly[train_end:]

    stumpy.config.STUMPY_EXCL_ZONE_DENOM = 1  # exclusion zone = window_size, matching MERLIN's non-self-match rule |p-q| >= L
    mp = stumpy.stump(test_values, m=window_size)
    profile = mp[:, 0].astype(float)  # nearest-neighbor distance per window start position

    discord_pos = int(np.argmax(profile))
    discord_dist = profile[discord_pos]

    anomaly_positions = np.where(test_is_anomaly == 1)[0]
    anomaly_start_pos = int(anomaly_positions.min())
    anomaly_end_pos = int(anomaly_positions.max())

    x_series = np.arange(len(test_values))
    x_profile = np.arange(len(profile)) + window_size // 2  # center each window's x position

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = fit_reference_model(train_values, 3, 3)
        windows = make_windows(test_values, window_size)
        rows = []
        for i, w in enumerate(windows):
            sigma2_w, _ = score_window(w, model)
            ratio, _ = variance_ratio_pvalue(sigma2_w, window_size, model)
            start, end = i * window_size, (i + 1) * window_size
            rows.append({"window_start": start, "variance_ratio": ratio})
        results = pd.DataFrame(rows)

        starts = valid_normal_starts(test_values, test_is_anomaly, window_size)
        rng = np.random.default_rng(0)
        t_star = build_block_bootstrap_null(test_values, starts, model, window_size, N_BOOTSTRAP, rng)
    bb_upper = float(np.percentile(t_star, 100 * (1 - ALPHA)))
    ratio = results["variance_ratio"].to_numpy()
    x_ratio = results["window_start"].to_numpy() + window_size // 2
    sig_bb = ratio > bb_upper

    # Naive baseline inspired by Wu & Keogh's "one-liner" (movstd(TS,k) > b), but
    # unlike their brute-force threshold fit to each dataset's true labels, this
    # threshold comes only from the training region, so it's a fair comparison
    # (same no-label-peeking rule as our own bootstrap threshold).
    train_rolling_std = pd.Series(train_values).rolling(window=window_size).std().dropna().to_numpy()
    naive_threshold = float(np.percentile(train_rolling_std, 100 * (1 - ALPHA)))
    test_rolling_std = pd.Series(test_values).rolling(window=window_size, center=True).std().to_numpy()
    x_naive = np.arange(len(test_rolling_std))
    sig_naive = test_rolling_std > naive_threshold

    fig, axes = plt.subplots(4, 1, figsize=(8, 6.6), height_ratios=(1, 1.1, 1.1, 1.1))

    ax = axes[0]
    _style_axis(ax)
    ax.plot(x_series, test_values, color=COLOR_INK, linewidth=0.7)
    ax.axvspan(anomaly_start_pos, anomaly_end_pos, color=COLOR_ANOMALY_SPAN, alpha=0.6, linewidth=0, label="Anomaly")
    y_min, y_max = ax.get_ylim()
    ax.set_ylim(y_min, y_max + 0.18 * (y_max - y_min))
    ax.set_ylabel("Value")
    ax.set_title(f"(a) {dataset_name}  [test region]", loc="left", fontsize=10.5, color=COLOR_INK)
    ax.legend(loc="upper right", fontsize=9, frameon=True, facecolor="white", edgecolor="none")

    ax = axes[1]
    _style_axis(ax)
    ax.axvspan(anomaly_start_pos, anomaly_end_pos, color=COLOR_ANOMALY_SPAN, alpha=0.6, linewidth=0)
    ax.plot(x_profile, profile, color=COLOR_LINE, linewidth=1.0, zorder=3, label="Nearest-neighbor distance (discord profile)")
    ax.scatter([x_profile[discord_pos]], [discord_dist], s=40, color="#c0392b", zorder=4, label=f"MERLIN discord (dist={discord_dist:.2f})")
    ax.set_ylabel("NN Distance")
    ax.set_xlabel("Sample")
    ax.set_title(f"(b) MERLIN Discord Profile (w={window_size}, fixed)", loc="left", fontsize=10.5, color=COLOR_INK)
    ax.legend(loc="upper right", fontsize=8.5, frameon=False)

    ax = axes[2]
    _style_axis(ax)
    ax.axvspan(anomaly_start_pos, anomaly_end_pos, color=COLOR_ANOMALY_SPAN, alpha=0.6, linewidth=0)
    ax.axhline(1.0, color=COLOR_MUTED, linewidth=1.0, linestyle=":")
    ax.axhline(bb_upper, color="#8a8a8a", linewidth=1.1, linestyle="--", label=f"Bootstrap control limit (alpha={ALPHA})")
    ax.plot(x_ratio, ratio, color=COLOR_LINE, linewidth=1.2, zorder=3, label="Innovation variance ratio")
    ax.scatter(x_ratio[sig_bb], ratio[sig_bb], s=16, color=COLOR_LINE, zorder=4)
    ax.scatter(x_ratio[~sig_bb], ratio[~sig_bb], s=16, facecolors="white", edgecolors=COLOR_LINE, linewidths=0.9, zorder=4)
    ax.set_ylabel("Variance Ratio")
    ax.set_xlabel("Sample")
    ax.set_title(f"(c) Our Method: Bootstrap-Calibrated Limits (w={window_size}, one-sided)", loc="left", fontsize=10.5, color=COLOR_INK)
    ax.legend(loc="upper right", fontsize=8.5, frameon=False)

    ax = axes[3]
    _style_axis(ax)
    ax.axvspan(anomaly_start_pos, anomaly_end_pos, color=COLOR_ANOMALY_SPAN, alpha=0.6, linewidth=0)
    ax.axhline(naive_threshold, color="#8a8a8a", linewidth=1.1, linestyle="--", label=f"Training-set threshold (alpha={ALPHA})")
    ax.plot(x_naive, test_rolling_std, color=COLOR_LINE, linewidth=0.9, zorder=3, label="Raw rolling std (movstd)")
    ax.fill_between(x_naive, 0, test_rolling_std, where=sig_naive, color="#c0392b", alpha=0.35, linewidth=0, zorder=2)
    ax.set_ylabel("Rolling Std")
    ax.set_xlabel("Sample")
    ax.set_title(f"(d) Naive Baseline: Rolling Std, No Pre-Whitening (w={window_size})", loc="left", fontsize=10.5, color=COLOR_INK)
    ax.legend(loc="upper right", fontsize=8.5, frameon=False)

    fig.tight_layout()
    fig.subplots_adjust(right=1.0 - fig.subplotpars.left)
    output_path = Path(f"results/ucr/comparison/figures/merlin_{dataset_name}_win{window_size}.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)

    hit = anomaly_start_pos - 100 <= x_profile[discord_pos] <= anomaly_end_pos + 100
    print(f"Saved {output_path}")
    print(f"discord at x={x_profile[discord_pos]}, true anomaly=[{anomaly_start_pos},{anomaly_end_pos}], hit={hit}")


if __name__ == "__main__":
    main()
