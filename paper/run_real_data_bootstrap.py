"""Paper Results section: real data (mice EEG) showing theoretical
F-distribution control limits vs. bootstrap-calibrated control limits.

Produces, for seizure 1 / channel 1 (the illustrative example):
  - figs/real_data_bootstrap.png: 3-panel figure in the SAME style as
    run_simulation.py's plot_results_figure_split -- (a) the real EEG
    excerpt with the seizure shaded, (b) the innovation variance ratio
    with THEORETICAL F-distribution limits, (c) the SAME ratio series
    with BOOTSTRAP-CALIBRATED limits. Same R_w values in (b) and (c);
    only the control limits (and therefore which points are flagged)
    differ.

And, across all 7 seizures x 4 channels (28 combos):
  - real_data_summary_table.csv: false-alarm rate and detection rate,
    F-distribution theory vs. block bootstrap, per combo and averaged.

No Bayesian posterior odds here -- scoped to match the paper's stated
contribution (bootstrap only).
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from algorithm import fit_reference_model, make_windows, score_window, variance_ratio_pvalue
from compare_methods import build_block_bootstrap_null, valid_normal_starts

CONFIG = "2000hz_pre60s_ntrain1000"
N_TRAIN = 1000
WINDOW_SIZE = 100
ALPHA = 0.05
N_BOOTSTRAP = 2000

COLOR_INK = "#0b0b0b"
COLOR_MUTED = "#6b6a65"
COLOR_LINE = "#1a1a1a"   # near-black, matching run_simulation.py's grayscale convention
COLOR_LIMIT = "#8a8a8a"  # mid-gray, matching run_simulation.py's grayscale convention
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
    full_values: np.ndarray, n_train: int, test_is_anomaly: np.ndarray, results: pd.DataFrame,
    model: dict, window_size: int, alpha: float, bb_lower: float, bb_upper: float, output_path: Path,
) -> None:
    """3-panel figure, same layout/style as run_simulation.py's
    plot_results_figure_split: (a) series, (b) ratio + F-theory limits,
    (c) SAME ratio + bootstrap limits.
    """
    x_series = np.arange(len(full_values))
    anomaly_positions = np.where(test_is_anomaly == 1)[0]
    anomaly_start_pos = n_train + int(anomaly_positions.min())
    anomaly_end_pos = n_train + int(anomaly_positions.max())
    x = n_train + results["window_start"] + window_size // 2

    fig, axes = plt.subplots(3, 1, figsize=(8, 4.7), height_ratios=(1, 1.1, 1.1))

    ax = axes[0]
    _style_axis(ax)
    ax.plot(x_series, full_values, color=COLOR_INK, linewidth=0.7)
    ax.axvspan(anomaly_start_pos, anomaly_end_pos, color=COLOR_ANOMALY_SPAN, alpha=0.6, linewidth=0, label="Seizure")
    ax.axvline(n_train, color=COLOR_MUTED, linewidth=0.9, linestyle=":")
    y_min, y_max = ax.get_ylim()
    ax.set_ylim(y_min, y_max + 0.18 * (y_max - y_min))
    ax.set_ylabel("Value")
    ax.set_title("(a) Mouse EEG Time Series", loc="left", fontsize=10.5, color=COLOR_INK)
    ax.legend(loc="upper right", fontsize=9, frameon=True, facecolor="white", edgecolor="none")

    ratio = results["variance_ratio"].to_numpy()
    f_lower, f_upper = f_critical_values(window_size, model["reference_df"], alpha)

    # (b) F-distribution theory limits
    ax = axes[1]
    _style_axis(ax)
    ax.axvspan(anomaly_start_pos, anomaly_end_pos, color=COLOR_ANOMALY_SPAN, alpha=0.6, linewidth=0)
    ax.axhline(1.0, color=COLOR_MUTED, linewidth=1.0, linestyle=":")
    ax.axhline(f_lower, color=COLOR_LIMIT, linewidth=1.1, linestyle="--", label=f"control limits (alpha={alpha})")
    ax.axhline(f_upper, color=COLOR_LIMIT, linewidth=1.1, linestyle="--")
    sig_f = results["p_value"].to_numpy() < alpha
    ax.plot(x, ratio, color=COLOR_LINE, linewidth=1.2, zorder=3, label="Innovation variance ratio")
    ax.scatter(x[sig_f], ratio[sig_f], s=16, color=COLOR_LINE, zorder=4)
    ax.scatter(x[~sig_f], ratio[~sig_f], s=16, facecolors="white", edgecolors=COLOR_LINE, linewidths=0.9, zorder=4)
    ax.set_ylabel("Variance Ratio")
    ax.set_title("(b) Theoretical F-Distribution Limits", loc="left", fontsize=10.5, color=COLOR_INK)
    ax.legend(loc="upper right", fontsize=8.5, frameon=False)

    # (c) bootstrap-calibrated limits, SAME ratio values
    ax = axes[2]
    _style_axis(ax)
    ax.axvspan(anomaly_start_pos, anomaly_end_pos, color=COLOR_ANOMALY_SPAN, alpha=0.6, linewidth=0)
    ax.axhline(1.0, color=COLOR_MUTED, linewidth=1.0, linestyle=":")
    ax.axhline(bb_lower, color=COLOR_LIMIT, linewidth=1.1, linestyle="--", label=f"control limits (alpha={alpha})")
    ax.axhline(bb_upper, color=COLOR_LIMIT, linewidth=1.1, linestyle="--")
    sig_bb = (ratio < bb_lower) | (ratio > bb_upper)
    ax.plot(x, ratio, color=COLOR_LINE, linewidth=1.2, zorder=3, label="Innovation variance ratio")
    ax.scatter(x[sig_bb], ratio[sig_bb], s=16, color=COLOR_LINE, zorder=4)
    ax.scatter(x[~sig_bb], ratio[~sig_bb], s=16, facecolors="white", edgecolors=COLOR_LINE, linewidths=0.9, zorder=4)
    ax.set_ylabel("Variance Ratio")
    ax.set_xlabel("Sample")
    ax.set_title("(c) Bootstrap-Calibrated Limits", loc="left", fontsize=10.5, color=COLOR_INK)
    ax.legend(loc="upper right", fontsize=8.5, frameon=False)

    # shared y-axis across (b)/(c), same convention as plot_results_figure_split
    shared_min = min(ratio.min(), f_lower, bb_lower)
    shared_max = max(ratio.max(), f_upper, bb_upper)
    pad = 0.08 * (shared_max - shared_min)
    axes[1].set_ylim(shared_min - pad, shared_max + pad)
    axes[2].set_ylim(shared_min - pad, shared_max + pad)

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


def run_one_combo(seizure_number: int, channel: int) -> dict | None:
    tag = f"seizure{seizure_number}" + (f"_ch{channel}" if channel != 1 else "")
    values = pd.read_csv(f"data/mice/{CONFIG}/timeseries_{tag}.csv")["value"].to_numpy(float)
    labels = pd.read_csv(f"data/mice/{CONFIG}/labels_{tag}.csv")["is_anomaly"].to_numpy(int)

    train_values = values[:N_TRAIN]
    test_values = values[N_TRAIN:]
    test_is_anomaly = labels[N_TRAIN:]

    model = fit_reference_model(train_values, 3, 3)
    windows = make_windows(test_values, WINDOW_SIZE)
    rows = []
    for i, w in enumerate(windows):
        sigma2_w, _ = score_window(w, model)
        ratio, p_value = variance_ratio_pvalue(sigma2_w, WINDOW_SIZE, model)
        start, end = i * WINDOW_SIZE, (i + 1) * WINDOW_SIZE
        rows.append({
            "window_index": i, "window_start": start, "window_end": end,
            "sigma2_w": sigma2_w, "variance_ratio": ratio, "p_value": p_value,
            "actual_anomaly": int(test_is_anomaly[start:end].any()),
        })
    results = pd.DataFrame(rows)

    starts = valid_normal_starts(test_values, test_is_anomaly, WINDOW_SIZE)
    if len(starts) < 20:
        return None
    rng = np.random.default_rng(0)
    t_star = build_block_bootstrap_null(test_values, starts, model, WINDOW_SIZE, N_BOOTSTRAP, rng)
    bb_lower, bb_upper = np.percentile(t_star, [100 * ALPHA / 2, 100 * (1 - ALPHA / 2)])

    normal_mask = results["actual_anomaly"].to_numpy() == 0
    anomaly_mask = results["actual_anomaly"].to_numpy() == 1
    n_normal, n_anomaly = int(normal_mask.sum()), int(anomaly_mask.sum())

    flagged_f = results["p_value"].to_numpy() < ALPHA
    fa_f, det_f = int((flagged_f & normal_mask).sum()), int((flagged_f & anomaly_mask).sum())

    ratio = results["variance_ratio"].to_numpy()
    flagged_bb = (ratio < bb_lower) | (ratio > bb_upper)
    fa_bb, det_bb = int((flagged_bb & normal_mask).sum()), int((flagged_bb & anomaly_mask).sum())

    return {
        "seizure": seizure_number, "channel": channel,
        "f_false_alarm_rate": fa_f / n_normal, "f_detection_rate": det_f / n_anomaly,
        "bb_false_alarm_rate": fa_bb / n_normal, "bb_detection_rate": det_bb / n_anomaly,
        "values": values, "test_is_anomaly": test_is_anomaly, "results": results,
        "model": model, "bb_lower": bb_lower, "bb_upper": bb_upper,
    }


def main() -> None:
    # illustrative figure: seizure 1, channel 1
    example = run_one_combo(1, 1)
    plot_theory_vs_bootstrap(
        example["values"], N_TRAIN, example["test_is_anomaly"], example["results"], example["model"],
        WINDOW_SIZE, ALPHA, example["bb_lower"], example["bb_upper"],
        Path("results/mice/comparison/figures/real_data_bootstrap.png"),
    )
    print("Saved results/mice/comparison/figures/real_data_bootstrap.png")

    # summary table: all 28 combos
    rows = [example]
    for seizure_number in range(1, 8):
        for channel in range(1, 5):
            if seizure_number == 1 and channel == 1:
                continue  # already computed above
            r = run_one_combo(seizure_number, channel)
            if r is not None:
                rows.append(r)
            print(f"seizure{seizure_number} ch{channel} done", flush=True)

    table = pd.DataFrame([
        {k: v for k, v in r.items() if k in ("seizure", "channel", "f_false_alarm_rate", "f_detection_rate", "bb_false_alarm_rate", "bb_detection_rate")}
        for r in rows
    ]).sort_values(["seizure", "channel"])
    table.to_csv("results/mice/comparison/real_data_summary_table.csv", index=False)
    print(f"\nSaved results/mice/comparison/real_data_summary_table.csv ({len(table)} rows)")

    print("\n=== Mean across all 28 combos ===")
    print(f"F-distribution theory: false alarms {table['f_false_alarm_rate'].mean():.2%}, detected {table['f_detection_rate'].mean():.2%}")
    print(f"Block bootstrap:       false alarms {table['bb_false_alarm_rate'].mean():.2%}, detected {table['bb_detection_rate'].mean():.2%}")


if __name__ == "__main__":
    main()
