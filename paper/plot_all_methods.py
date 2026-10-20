"""Combined (a)-(g) comparison figure for one UCR noise-category dataset:
  (a) raw test-region series
  (b) MERLIN discord profile (nearest-neighbor distance, exact via stumpy)
  (c) Our method: bootstrap-calibrated innovation variance ratio (one-sided)
  (d) Naive baseline: raw rolling std, no pre-whitening
  (e) CUSUM on our own window-level variance ratios
  (f) Autoencoder reconstruction error
  (g) LSTM forecasting error

All methods use the SAME fixed window_size grid and the SAME fairness
rule for thresholds (set from the training segment only, target alpha).
"""

from __future__ import annotations

# stumpy (numba) must be imported before numpy/pandas/torch: importing
# numpy+pandas first causes a native DLL conflict with torch on Windows.
import stumpy

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from algorithm import fit_reference_model, make_windows, score_window, variance_ratio_pvalue
from baselines import cusum_on_ratios, naive_rolling_std
from compare_methods import build_block_bootstrap_null, valid_normal_starts
from dl_baselines import autoencoder_scores, lstm_scores
from run_ucr import find_train_end

ALPHA = 0.05
N_BOOTSTRAP = 2000

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


def _flagged_line_panel(ax, x, y, threshold, anomaly_span, title, ylabel, label, one_sided_lower=None):
    _style_axis(ax)
    ax.axvspan(*anomaly_span, color=COLOR_ANOMALY_SPAN, alpha=0.6, linewidth=0)
    ax.axhline(threshold, color=COLOR_LIMIT, linewidth=1.1, linestyle="--", label=f"threshold (alpha={ALPHA})")
    sig = y > threshold
    ax.plot(x, y, color=COLOR_LINE, linewidth=1.0, zorder=3, label=label)
    ax.scatter(x[sig], y[sig], s=14, color=COLOR_LINE, zorder=4)
    ax.scatter(x[~sig], y[~sig], s=14, facecolors="white", edgecolors=COLOR_LINE, linewidths=0.8, zorder=4)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontsize=10.0, color=COLOR_INK)
    ax.legend(loc="upper right", fontsize=7.5, frameon=False)
    return sig


def run_dataset(dataset_name: str, window_size: int) -> dict:
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

    anomaly_positions = np.where(test_is_anomaly == 1)[0]
    anomaly_start_pos = int(anomaly_positions.min())
    anomaly_end_pos = int(anomaly_positions.max())

    # (b) MERLIN discord profile
    stumpy.config.STUMPY_EXCL_ZONE_DENOM = 1
    mp = stumpy.stump(test_values, m=window_size)
    merlin_profile = mp[:, 0].astype(float)
    merlin_discord_pos = int(np.argmax(merlin_profile))
    x_merlin = np.arange(len(merlin_profile)) + window_size // 2

    # (c) our method
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = fit_reference_model(train_values, 3, 3)
        windows = make_windows(test_values, window_size)
        rows = []
        for i, w in enumerate(windows):
            sigma2_w, _ = score_window(w, model)
            ratio, _ = variance_ratio_pvalue(sigma2_w, window_size, model)
            rows.append({"window_start": i * window_size, "variance_ratio": ratio,
                         "actual_anomaly": int(test_is_anomaly[i * window_size:(i + 1) * window_size].any())})
        results = pd.DataFrame(rows)
        starts = valid_normal_starts(test_values, test_is_anomaly, window_size)
        rng = np.random.default_rng(0)
        t_star = build_block_bootstrap_null(test_values, starts, model, window_size, N_BOOTSTRAP, rng)
    bb_upper = float(np.percentile(t_star, 100 * (1 - ALPHA)))
    ratio = results["variance_ratio"].to_numpy()
    x_ratio = results["window_start"].to_numpy() + window_size // 2
    normal_mask = results["actual_anomaly"].to_numpy() == 0
    anom_mask = results["actual_anomaly"].to_numpy() == 1

    # (d) naive rolling std
    naive_curve, naive_threshold = naive_rolling_std(train_values, test_values, window_size)
    x_naive = np.arange(len(naive_curve))

    # (e) CUSUM on our own window ratios
    cusum_curve, cusum_h = cusum_on_ratios(train_values, ratio, model, window_size)

    # (f) autoencoder, (g) lstm
    ae_err, ae_threshold = autoencoder_scores(train_values, test_values, window_size)
    lstm_err, lstm_threshold = lstm_scores(train_values, test_values, window_size)

    def rate(sig_window_level: np.ndarray) -> tuple[float, float]:
        n_normal, n_anom = int(normal_mask.sum()), int(anom_mask.sum())
        fa = int((sig_window_level & normal_mask).sum()) / n_normal if n_normal else float("nan")
        det = int((sig_window_level & anom_mask).sum()) / n_anom if n_anom else float("nan")
        return fa, det

    metrics = {
        "dataset": dataset_name,
        "merlin_hit": int(anomaly_start_pos - 100 <= x_merlin[merlin_discord_pos] <= anomaly_end_pos + 100),
        "ours_fa": rate(ratio > bb_upper)[0], "ours_det": rate(ratio > bb_upper)[1],
        "naive_fa": None, "naive_det": None,  # window-unaligned grid, reported separately if needed
        "cusum_fa": rate(cusum_curve > cusum_h)[0], "cusum_det": rate(cusum_curve > cusum_h)[1],
        "ae_fa": rate(ae_err > ae_threshold)[0], "ae_det": rate(ae_err > ae_threshold)[1],
        "lstm_fa": rate(lstm_err > lstm_threshold)[0], "lstm_det": rate(lstm_err > lstm_threshold)[1],
    }

    # ---- figure ----
    fig, axes = plt.subplots(7, 1, figsize=(8, 11.5), height_ratios=[1] + [1.05] * 6)
    anomaly_span = (anomaly_start_pos, anomaly_end_pos)

    ax = axes[0]
    _style_axis(ax)
    ax.plot(np.arange(len(test_values)), test_values, color=COLOR_INK, linewidth=0.6)
    ax.axvspan(*anomaly_span, color=COLOR_ANOMALY_SPAN, alpha=0.6, linewidth=0, label="Anomaly")
    y_min, y_max = ax.get_ylim()
    ax.set_ylim(y_min, y_max + 0.18 * (y_max - y_min))
    ax.set_ylabel("Value")
    ax.set_title(f"(a) {dataset_name}  [test region]", loc="left", fontsize=10.0, color=COLOR_INK)
    ax.legend(loc="upper right", fontsize=8, frameon=True, facecolor="white", edgecolor="none")

    ax = axes[1]
    _style_axis(ax)
    ax.axvspan(*anomaly_span, color=COLOR_ANOMALY_SPAN, alpha=0.6, linewidth=0)
    ax.plot(x_merlin, merlin_profile, color=COLOR_LINE, linewidth=0.8, label="NN distance")
    ax.scatter([x_merlin[merlin_discord_pos]], [merlin_profile[merlin_discord_pos]], s=35, color=COLOR_DISCORD, zorder=4, label="MERLIN discord")
    ax.set_ylabel("NN Distance")
    ax.set_title(f"(b) MERLIN Discord Profile (w={window_size})", loc="left", fontsize=10.0, color=COLOR_INK)
    ax.legend(loc="upper right", fontsize=7.5, frameon=False)

    _flagged_line_panel(axes[2], x_ratio, ratio, bb_upper, anomaly_span,
                        f"(c) Our Method: Bootstrap-Calibrated (w={window_size})", "Variance Ratio", "Variance ratio")

    _flagged_line_panel(axes[3], x_naive, naive_curve, naive_threshold, anomaly_span,
                        f"(d) Naive Baseline: Rolling Std (w={window_size})", "Rolling Std", "movstd")

    _flagged_line_panel(axes[4], x_ratio, cusum_curve, cusum_h, anomaly_span,
                        f"(e) CUSUM on Variance Ratios (w={window_size})", "CUSUM", "Cumulative sum")

    _flagged_line_panel(axes[5], x_ratio, ae_err, ae_threshold, anomaly_span,
                        f"(f) Autoencoder Reconstruction Error (w={window_size})", "Recon. MSE", "AE error")

    ax = axes[6]
    _flagged_line_panel(ax, x_ratio, lstm_err, lstm_threshold, anomaly_span,
                        f"(g) LSTM Forecasting Error (w={window_size})", "Forecast MSE", "LSTM error")
    ax.set_xlabel("Sample")

    fig.tight_layout()
    fig.subplots_adjust(right=1.0 - fig.subplotpars.left)
    output_path = Path(f"results/ucr/comparison/figures/all_methods/{dataset_name}_win{window_size}.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
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

    metrics["figure_path"] = str(output_path)
    return metrics


def main() -> None:
    import sys
    dataset_name = sys.argv[1] if len(sys.argv) > 1 else "ucr_003_DISTORTED3sddb40"
    window_size = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    metrics = run_dataset(dataset_name, window_size)
    print(metrics)


if __name__ == "__main__":
    main()
