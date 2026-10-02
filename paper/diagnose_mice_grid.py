"""One combined figure: 4 columns (channels 1-4), 5 rows --
raw timeseries, the method's own F-ratio output, then the three
assumption-diagnostic plots below, each titled with its test statistic.
Reuses diagnose_mice.py's test logic and the already-cached
data/results CSVs; only new work is recomputing residuals per channel
(cheap -- reuses algorithm.score_window).

Rows:
  1. Raw timeseries (the full extracted excerpt for that channel)
  2. Variance ratio (the method's own output -- same as run_mice.py's panel b)
  3. ACF of pooled normal-test-window residuals -- model adequacy (whiteness)
  4. QQ-plot of the same residuals against Normal -- Gaussian innovations
  5. sigma2_w per window over time, normal windows only -- stationarity
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import acf

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from algorithm import fit_reference_model, make_windows, score_window
from diagnose_mice import residual_diagnostics, stationarity_check

COLOR_INK = "#0b0b0b"
COLOR_MUTED = "#6b6a65"
COLOR_LINE = "#1a1a1a"
COLOR_LIMIT = "#8a8a8a"
COLOR_ANOMALY_SPAN = "#c9c7c0"

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
        "font.size": 9.5,
        "axes.labelsize": 9.5,
        "axes.titlesize": 9.0,
        "xtick.labelsize": 8.0,
        "ytick.labelsize": 8.0,
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


def diagnose_channel(config: str, seizure_number: int, channel: int, n_train: int, window_size: int, max_p: int, max_q: int, n_sample_windows: int, seed: int):
    tag = f"seizure{seizure_number}" + (f"_ch{channel}" if channel != 1 else "")
    values = pd.read_csv(f"data/mice/{config}/timeseries_{tag}.csv")["value"].to_numpy(float)
    labels = pd.read_csv(f"data/mice/{config}/labels_{tag}.csv")["is_anomaly"].to_numpy(int)
    window_results = pd.read_csv(f"results/mice/{config}/window_results_{tag}.csv")

    train_values = values[:n_train]
    test_values = values[n_train:]

    model = fit_reference_model(train_values, max_p, max_q)
    windows = make_windows(test_values, window_size)
    normal_flags = window_results["actual_anomaly"].to_numpy()[: len(windows)] == 0
    normal_indices = np.where(normal_flags)[0]
    rng = np.random.default_rng(seed)
    sample_indices = rng.choice(normal_indices, size=min(n_sample_windows, len(normal_indices)), replace=False)
    pooled_residuals = np.concatenate([score_window(windows[i], model)[1] for i in sample_indices])

    resid_stats = residual_diagnostics(pooled_residuals, "test")
    stat = stationarity_check(window_results)

    return {
        "values": values,
        "labels": labels,
        "residuals": pooled_residuals,
        "window_results": window_results,
        "resid_stats": resid_stats,
        "stat": stat,
        "model": model,
    }


def plot_grid(config: str, seizure_number: int, n_train: int, window_size: int, max_p: int, max_q: int, n_sample_windows: int, alpha: float, output_path: str) -> None:
    fig, axes = plt.subplots(5, 4, figsize=(15, 11))

    for col, channel in enumerate([1, 2, 3, 4]):
        d = diagnose_channel(config, seizure_number, channel, n_train, window_size, max_p, max_q, n_sample_windows, seed=0)
        values = d["values"]
        labels = d["labels"]
        residuals = d["residuals"]
        rs = d["resid_stats"]
        stat = d["stat"]
        wr_all = d["window_results"]

        anomaly_positions = np.where(labels == 1)[0]
        anomaly_span = (int(anomaly_positions.min()), int(anomaly_positions.max())) if len(anomaly_positions) else None

        # Row 1: raw timeseries
        ax = axes[0, col]
        _style_axis(ax)
        if anomaly_span is not None:
            ax.axvspan(*anomaly_span, color=COLOR_ANOMALY_SPAN, alpha=0.6, linewidth=0)
        ax.plot(np.arange(len(values)), values, color=COLOR_INK, linewidth=0.5)
        ax.axvline(n_train, color=COLOR_MUTED, linewidth=0.8, linestyle=":")
        ax.set_title(f"Channel {channel}\nRaw excerpt", fontsize=9.5, color=COLOR_INK)
        if col == 0:
            ax.set_ylabel("Value")

        # Row 2: variance ratio -- the method's own output (same as run_mice.py panel b)
        ax = axes[1, col]
        _style_axis(ax)
        if anomaly_span is not None:
            ax.axvspan(*anomaly_span, color=COLOR_ANOMALY_SPAN, alpha=0.6, linewidth=0)
        ax.axhline(1.0, color=COLOR_MUTED, linewidth=0.8, linestyle=":")
        lower, upper = f_critical_values(window_size, d["model"]["reference_df"], alpha)
        ax.axhline(lower, color=COLOR_LIMIT, linewidth=1.0, linestyle="--")
        ax.axhline(upper, color=COLOR_LIMIT, linewidth=1.0, linestyle="--")
        x = n_train + wr_all["window_start"] + window_size // 2
        significant = wr_all["p_value"] < alpha
        ratio = wr_all["variance_ratio"]
        ax.plot(x, ratio, color=COLOR_LINE, linewidth=0.8, zorder=3)
        ax.scatter(x[significant], ratio[significant], s=8, color=COLOR_LINE, zorder=4)
        ax.scatter(x[~significant], ratio[~significant], s=8, facecolors="white", edgecolors=COLOR_LINE, linewidths=0.6, zorder=4)
        ax.set_xlim(0, len(values))
        n_flagged = int(significant.sum())
        ax.set_title(f"Variance ratio (method output)\n{n_flagged}/{len(wr_all)} windows flagged", fontsize=9.0)
        if col == 0:
            ax.set_ylabel("Variance ratio")

        # Row 3: ACF of residuals (whiteness / model adequacy)
        ax = axes[2, col]
        _style_axis(ax)
        n = len(residuals)
        nlags = 20
        acf_vals = acf(residuals, nlags=nlags, fft=True)
        conf = 1.96 / np.sqrt(n)
        ax.bar(np.arange(nlags + 1), acf_vals, color=COLOR_INK, width=0.6)
        ax.axhline(conf, color=COLOR_LIMIT, linewidth=1.0, linestyle="--")
        ax.axhline(-conf, color=COLOR_LIMIT, linewidth=1.0, linestyle="--")
        ax.axhline(0, color=COLOR_MUTED, linewidth=0.6)
        verdict = "white noise OK" if rs["white_noise_ok"] else "AUTOCORRELATED"
        ax.set_title(f"ACF of residuals\nLjung-Box p={rs['ljung_box_p']:.4f} ({verdict})", fontsize=9.0)
        if col == 0:
            ax.set_ylabel("ACF")

        # Row 4: QQ-plot (Gaussianity)
        ax = axes[3, col]
        _style_axis(ax)
        (osm, osr), (slope, intercept, r) = stats.probplot(residuals, dist="norm")
        ax.scatter(osm, osr, s=6, color=COLOR_INK, alpha=0.5)
        ax.plot(osm, slope * osm + intercept, color=COLOR_LIMIT, linewidth=1.2, linestyle="--")
        verdict = "Gaussian OK" if rs["gaussian_ok"] else "NOT GAUSSIAN"
        ax.set_title(f"QQ-plot vs Normal\nJarque-Bera p={rs['jarque_bera_p']:.4f} ({verdict})", fontsize=9.0)
        if col == 0:
            ax.set_ylabel("Ordered residuals")

        # Row 5: sigma2_w over time (stationarity)
        ax = axes[4, col]
        _style_axis(ax)
        wr = d["window_results"]
        normal = wr[wr.actual_anomaly == 0]
        ax.plot(normal["window_index"], normal["sigma2_w"], color=COLOR_INK, linewidth=0.6)
        ax.scatter(normal["window_index"], normal["sigma2_w"], s=4, color=COLOR_INK, alpha=0.5)
        verdict = "HETEROGENEOUS" if stat["block_variance_heterogeneous"] else "homogeneous"
        ax.set_title(f"sigma2_w over time (normal windows)\nLevene p={stat['levene_p']:.4f} ({verdict})", fontsize=9.0)
        ax.set_xlabel("Window index")
        if col == 0:
            ax.set_ylabel("sigma2_w")

    fig.suptitle(f"Assumption diagnostics — Seizure {seizure_number}, window={window_size}, n_train={n_train}", fontsize=12, color=COLOR_INK, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"Saved {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="2000hz_pre60s_ntrain1000")
    parser.add_argument("--seizure-index", type=int, default=0)
    parser.add_argument("--n-train", type=int, default=1000)
    parser.add_argument("--window-size", type=int, default=100)
    parser.add_argument("--max-p", type=int, default=3)
    parser.add_argument("--max-q", type=int, default=3)
    parser.add_argument("--n-sample-windows", type=int, default=40)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    seizure_number = args.seizure_index + 1
    output_path = args.output or f"results/mice/figs/diagnostics_seizure{seizure_number}_win{args.window_size}.png"
    plot_grid(args.config, seizure_number, args.n_train, args.window_size, args.max_p, args.max_q, args.n_sample_windows, args.alpha, output_path)


if __name__ == "__main__":
    main()
