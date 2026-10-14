"""Same theory-vs-bootstrap illustrative figure as
run_real_data_bootstrap.py, but for one UCR benchmark dataset instead of
mice EEG. Two-sided limits (matching the main comparison table).
"""

from __future__ import annotations

import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from algorithm import fit_reference_model, make_windows, score_window, variance_ratio_pvalue
from compare_methods import build_block_bootstrap_null, valid_normal_starts
from run_ucr import find_train_end

ALPHA = 0.05
N_BOOTSTRAP = 2000

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
    ax.set_xticks([])


def f_critical_values(window_size: int, reference_df: int, alpha: float) -> tuple[float, float]:
    return (
        float(stats.f.ppf(alpha / 2, window_size, reference_df)),
        float(stats.f.ppf(1 - alpha / 2, window_size, reference_df)),
    )


def plot_theory_vs_bootstrap(
    values: np.ndarray, train_end: int, is_anomaly: np.ndarray, results: pd.DataFrame,
    model: dict, window_size: int, alpha: float, bb_lower: float, bb_upper: float, dataset_name: str, output_path: Path,
    one_sided: bool = False, ylim_max: float | None = None, zoom_pad: int | None = None,
) -> None:
    x_series = np.arange(len(values))
    anomaly_positions = np.where(is_anomaly == 1)[0]
    anomaly_start_pos = int(anomaly_positions.min())
    anomaly_end_pos = int(anomaly_positions.max())
    x = train_end + results["window_start"] + window_size // 2

    x_lo, x_hi = (None, None)
    if zoom_pad is not None:
        x_lo = max(0, anomaly_start_pos - zoom_pad)
        x_hi = min(len(values) - 1, anomaly_end_pos + zoom_pad)

    fig, axes = plt.subplots(3, 1, figsize=(8, 4.7), height_ratios=(1, 1.1, 1.1))

    ax = axes[0]
    _style_axis(ax)
    ax.plot(x_series, values, color=COLOR_INK, linewidth=0.7)
    ax.axvspan(anomaly_start_pos, anomaly_end_pos, color=COLOR_ANOMALY_SPAN, alpha=0.6, linewidth=0, label="Anomaly")
    ax.axvline(train_end, color=COLOR_MUTED, linewidth=0.9, linestyle=":")
    if zoom_pad is not None:
        ax.set_xlim(x_lo, x_hi)
        mask = (x_series >= x_lo) & (x_series <= x_hi)
        y_min, y_max = float(values[mask].min()), float(values[mask].max())
    else:
        y_min, y_max = ax.get_ylim()
    ax.set_ylim(y_min, y_max + 0.18 * (y_max - y_min))
    ax.set_ylabel("Value")
    ax.set_title(f"(a) {dataset_name}", loc="left", fontsize=10.5, color=COLOR_INK)
    ax.legend(loc="upper right", fontsize=9, frameon=True, facecolor="white", edgecolor="none")

    ratio = results["variance_ratio"].to_numpy()
    if one_sided:
        f_lower = -np.inf
        f_upper = float(stats.f.ppf(1 - alpha, window_size, model["reference_df"]))
    else:
        f_lower, f_upper = f_critical_values(window_size, model["reference_df"], alpha)

    ax = axes[1]
    _style_axis(ax)
    ax.axvspan(anomaly_start_pos, anomaly_end_pos, color=COLOR_ANOMALY_SPAN, alpha=0.6, linewidth=0)
    ax.axhline(1.0, color=COLOR_MUTED, linewidth=1.0, linestyle=":")
    if not one_sided:
        ax.axhline(f_lower, color=COLOR_LIMIT, linewidth=1.1, linestyle="--", label=f"control limits (alpha={alpha})")
    ax.axhline(f_upper, color=COLOR_LIMIT, linewidth=1.1, linestyle="--", label=f"control limit (alpha={alpha})" if one_sided else None)
    sig_f = (ratio > f_upper) if one_sided else (results["p_value"].to_numpy() < alpha)
    ax.plot(x, ratio, color=COLOR_LINE, linewidth=1.2, zorder=3, label="Innovation variance ratio")
    ax.scatter(x[sig_f], ratio[sig_f], s=16, color=COLOR_LINE, zorder=4)
    ax.scatter(x[~sig_f], ratio[~sig_f], s=16, facecolors="white", edgecolors=COLOR_LINE, linewidths=0.9, zorder=4)
    ax.set_ylabel("Variance Ratio")
    ax.set_title("(b) Theoretical F-Distribution Limits" + (" (one-sided)" if one_sided else ""), loc="left", fontsize=10.5, color=COLOR_INK)
    ax.legend(loc="upper right", fontsize=8.5, frameon=False)
    if zoom_pad is not None:
        ax.set_xlim(x_lo, x_hi)

    ax = axes[2]
    _style_axis(ax)
    ax.axvspan(anomaly_start_pos, anomaly_end_pos, color=COLOR_ANOMALY_SPAN, alpha=0.6, linewidth=0)
    ax.axhline(1.0, color=COLOR_MUTED, linewidth=1.0, linestyle=":")
    if not one_sided:
        ax.axhline(bb_lower, color=COLOR_LIMIT, linewidth=1.1, linestyle="--", label=f"control limits (alpha={alpha})")
    ax.axhline(bb_upper, color=COLOR_LIMIT, linewidth=1.1, linestyle="--", label=f"control limit (alpha={alpha})" if one_sided else None)
    sig_bb = (ratio > bb_upper) if one_sided else ((ratio < bb_lower) | (ratio > bb_upper))
    ax.plot(x, ratio, color=COLOR_LINE, linewidth=1.2, zorder=3, label="Innovation variance ratio")
    ax.scatter(x[sig_bb], ratio[sig_bb], s=16, color=COLOR_LINE, zorder=4)
    ax.scatter(x[~sig_bb], ratio[~sig_bb], s=16, facecolors="white", edgecolors=COLOR_LINE, linewidths=0.9, zorder=4)
    ax.set_ylabel("Variance Ratio")
    ax.set_xlabel("Sample")
    ax.set_title("(c) Bootstrap-Calibrated Limits" + (" (one-sided)" if one_sided else ""), loc="left", fontsize=10.5, color=COLOR_INK)
    ax.legend(loc="upper right", fontsize=8.5, frameon=False)
    if zoom_pad is not None:
        ax.set_xlim(x_lo, x_hi)

    if zoom_pad is not None:
        in_range = (x >= x_lo) & (x <= x_hi)
        ratio_range = ratio[in_range]
        shared_min = min(ratio_range.min(), 0 if one_sided else f_lower, 0 if one_sided else bb_lower)
        shared_max = max(ratio_range.max(), f_upper, bb_upper)
    else:
        shared_min = min(ratio.min(), 0 if one_sided else f_lower, 0 if one_sided else bb_lower)
        shared_max = ylim_max if ylim_max is not None else max(ratio.max(), f_upper, bb_upper)
    pad = 0.08 * (shared_max - shared_min)
    axes[1].set_ylim(shared_min - pad, shared_max + pad)
    axes[2].set_ylim(shared_min - pad, shared_max + pad)

    if zoom_pad is None:
        clipped = np.where(ratio > shared_max)[0]
        if len(clipped) > 0:
            x_offset = 0.02 * (x.max() - x.min())
            values_str = ", ".join(f"{ratio[i]:.1f}" for i in clipped)
            label = f"off-scale: {values_str} ↑"
            for ax in (axes[1], axes[2]):
                ax.text(
                    x[clipped].max() + x_offset, shared_max - 0.04 * (shared_max - shared_min),
                    label, ha="left", va="top", fontsize=7.5, color=COLOR_INK,
                )

    fig.tight_layout()
    fig.subplots_adjust(right=1.0 - fig.subplotpars.left)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(3):
        try:
            fig.savefig(output_path, dpi=220)
            break
        except OSError:
            if attempt == 2:
                raise
            time.sleep(0.5)
    plt.close(fig)


def main() -> None:
    import sys
    dataset_name = sys.argv[1] if len(sys.argv) > 1 else "ucr_008_DISTORTEDCIMIS44AirTemperature4"
    window_size = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    one_sided = sys.argv[3] == "1s" if len(sys.argv) > 3 else False
    arg4 = sys.argv[4] if len(sys.argv) > 4 else None
    ylim_max = None
    zoom_pad = None
    if arg4 is not None:
        if arg4.startswith("zoomw"):
            n_windows = int(arg4.split(":")[1]) if ":" in arg4 else 15
            zoom_pad = n_windows * window_size
        elif arg4.startswith("zoom"):
            zoom_pad = int(arg4.split(":")[1]) if ":" in arg4 else 300
        else:
            ylim_max = float(arg4)
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

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = fit_reference_model(train_values, 3, 3)
        windows = make_windows(test_values, window_size)
        rows = []
        for i, w in enumerate(windows):
            sigma2_w, _ = score_window(w, model)
            ratio, p_value = variance_ratio_pvalue(sigma2_w, window_size, model)
            start, end = i * window_size, (i + 1) * window_size
            rows.append({
                "window_start": start, "variance_ratio": ratio, "p_value": p_value,
                "actual_anomaly": int(test_is_anomaly[start:end].any()),
            })
        results = pd.DataFrame(rows)

        starts = valid_normal_starts(test_values, test_is_anomaly, window_size)
        rng = np.random.default_rng(0)
        t_star = build_block_bootstrap_null(test_values, starts, model, window_size, N_BOOTSTRAP, rng)
    if one_sided:
        bb_lower, bb_upper = -np.inf, np.percentile(t_star, 100 * (1 - ALPHA))
    else:
        bb_lower, bb_upper = np.percentile(t_star, [100 * ALPHA / 2, 100 * (1 - ALPHA / 2)])

    suffix = "_1sided" if one_sided else ""
    suffix += "_zoom" if (ylim_max is not None or zoom_pad is not None) else ""
    output_path = Path(f"results/ucr/comparison/figures/ucr_theory_vs_bootstrap_{dataset_name}_win{window_size}{suffix}.png")
    plot_theory_vs_bootstrap(
        values, train_end, is_anomaly, results, model, window_size, ALPHA, bb_lower, bb_upper, dataset_name, output_path,
        one_sided=one_sided, ylim_max=ylim_max, zoom_pad=zoom_pad,
    )
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
