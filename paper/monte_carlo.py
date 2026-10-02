"""Robustness Evaluation, baseline case (paper Sec. 8.6): does Algorithm 1
actually hold its nominal Type I error and deliver real power, under IDEAL
conditions (Gaussian innovations, correctly selected order)? Script 09/
run_simulation.py runs the algorithm ONCE -- one run is one noisy draw and
cannot answer this. This script runs it many times (--n-reps) on fresh
random data with the SAME settings each time, and pools every window's
p-value across all replications.

This is the baseline/sanity-check step only: Gaussian innovations, no
distribution or window-size sweep yet (that comes later).

Outputs, under --output-dir (default results/monte_carlo):
  - summary.csv        one row: empirical Type I error and power, with
                        exact (Clopper-Pearson) 95% confidence intervals
  - figures/summary.png two panels: (a) pooled p-value calibration check
                        for normal windows (should be ~Uniform(0,1) if the
                        test is well calibrated), (b) the Type I error /
                        power numbers as text
"""

from __future__ import annotations

import argparse
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from algorithm import run_algorithm
from simulate import add_variance_burst, parse_coeffs, simulate_arma

COLOR_INK = "#0b0b0b"
COLOR_MUTED = "#6b6a65"
COLOR_BW_LINE = "#1a1a1a"   # near-black: the data line, matching run_simulation.py's grayscale figures
COLOR_BW_LIMIT = "#8a8a8a"  # mid-gray: the reference/limit line, matching run_simulation.py's grayscale figures

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


def run_one_replication(
    n_total: int, n0: int, phi: list[float], theta: list[float], sigma: float,
    anomaly_start: int, anomaly_duration: int, anomaly_strength: float,
    window_size: int, alpha: float, max_p: int, max_q: int,
    data_seed: int, anomaly_seed: int,
) -> pd.DataFrame | None:
    """One Monte Carlo replication: fresh data, fresh fit, fresh scoring.

    Returns a DataFrame with columns p_value, variance_ratio, actual_anomaly
    (one row per test window), or None if the ARMA fit failed for this draw
    (rare; the caller counts and skips these).
    """
    clean_values = simulate_arma(n_total, phi, theta, sigma, data_seed)
    values, is_anomaly = add_variance_burst(clean_values, anomaly_start, anomaly_duration, anomaly_strength, anomaly_seed)

    train_values = values[:n0]
    test_values = values[n0:]
    test_is_anomaly = is_anomaly[n0:]

    try:
        results, _, _ = run_algorithm(train_values, test_values, window_size, alpha, max_p, max_q)
    except Exception:
        return None

    actual_anomaly = []
    for _, row in results.iterrows():
        start, end = int(row["window_start"]), int(row["window_end"])
        actual_anomaly.append(int(test_is_anomaly[start:end].any()))
    results = results.copy()
    results["actual_anomaly"] = actual_anomaly
    return results[["p_value", "variance_ratio", "actual_anomaly"]]


def exact_ci(successes: int, n: int) -> tuple[float, float]:
    """Clopper-Pearson exact 95% confidence interval for a proportion."""
    if n == 0:
        return (0.0, 1.0)
    low = stats.beta.ppf(0.025, successes, n - successes + 1) if successes > 0 else 0.0
    high = stats.beta.ppf(0.975, successes + 1, n - successes) if successes < n else 1.0
    return float(low), float(high)


def plot_summary(pooled: pd.DataFrame, alpha: float, n_reps_ok: int, output_path: Path) -> None:
    normal = pooled[pooled["actual_anomaly"] == 0]
    anomalous = pooled[pooled["actual_anomaly"] == 1]

    n_normal = len(normal)
    n_false_positive = int((normal["p_value"] < alpha).sum())
    type1_rate = n_false_positive / n_normal if n_normal else float("nan")
    type1_low, type1_high = exact_ci(n_false_positive, n_normal)

    n_anomalous = len(anomalous)
    n_detected = int((anomalous["p_value"] < alpha).sum())
    power = n_detected / n_anomalous if n_anomalous else float("nan")
    power_low, power_high = exact_ci(n_detected, n_anomalous)

    # P value plot (Davidson & MacKinnon 1998, "Graphical Methods for
    # Investigating the Size and Power of Hypothesis Tests"): the empirical
    # CDF of the normal-window p-values, plotted against the nominal level
    # itself. Under a correctly calibrated test the p-values are Uniform(0,1),
    # so this traces the 45-degree line; its height at x=alpha is exactly
    # the Type I error at that alpha. One panel only -- the numeric
    # Type I error / power summary lives in summary.csv, not in the figure,
    # so it can be pasted as a table separately (e.g. into Overleaf).
    # Canvas width chosen to match results_split_bw.png's font size on the
    # page: that figure is figsize width 8in at \includegraphics[width=1
    # \linewidth]. Extra width here (vs. a bare 4in square) reserves room
    # for the legend beside the axes instead of below it, so this one is
    # meant to be placed at width=0.7\linewidth: 8in * 0.7 = 5.6in keeps
    # the same canvas-width-to-LaTeX-width ratio (same effective font size
    # once both are typeset). No bbox_inches="tight" on save -- that
    # auto-crop silently changes the saved canvas size away from the
    # figsize we set, which is exactly what broke this ratio last time.
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    _style_axis(ax)
    sorted_p = np.sort(normal["p_value"].to_numpy())
    ecdf = np.arange(1, len(sorted_p) + 1) / len(sorted_p)
    ax.plot([0, 1], [0, 1], color=COLOR_BW_LIMIT, linewidth=1.2, linestyle="--", label="Uniform(0,1) (perfectly calibrated)")
    ax.plot(sorted_p, ecdf, color=COLOR_BW_LINE, linewidth=1.6, label="Empirical CDF of p-values")
    ax.axvline(alpha, color=COLOR_MUTED, linewidth=1.0, linestyle=":")
    ax.scatter([alpha], [type1_rate], color=COLOR_BW_LINE, s=40, zorder=5, label=f"Type I error at alpha={alpha}")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xlabel("alpha (nominal significance level)")
    ax.set_ylabel("Fraction of normal windows flagged")
    ax.set_title("P Value Plot", loc="left", fontsize=10.5, color=COLOR_INK)
    # Legend placed to the RIGHT of the axes, not inside it -- at this
    # canvas size an inside legend (fixed font/marker size, same as
    # everywhere else) would overlap the data curve rather than shrinking
    # to fit.
    ax.legend(loc="center left", bbox_to_anchor=(1.06, 0.5), fontsize=8.5, frameon=False)

    # Manual margins instead of tight_layout(): reserves a fixed right
    # strip for the legend so the canvas stays EXACTLY figsize (5.6, 4.0)
    # -- no auto-crop, no surprise resizing.
    fig.subplots_adjust(left=0.13, right=0.58, top=0.90, bottom=0.14)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(3):
        try:
            # No bbox_inches="tight" here -- that auto-crops the canvas to
            # the content bounding box, which silently changes the saved
            # image's width away from the figsize we set (this broke the
            # font-size-matching ratio with results_split_bw.png before).
            # subplots_adjust above already reserves room for the legend.
            fig.savefig(output_path, dpi=220)
            break
        except OSError:
            if attempt == 2:
                raise
            time.sleep(0.5)
    plt.close(fig)

    return {
        "type1_rate": type1_rate, "type1_ci_low": type1_low, "type1_ci_high": type1_high,
        "power": power, "power_ci_low": power_low, "power_ci_high": power_high,
        "n_normal_windows": n_normal, "n_anomalous_windows": n_anomalous,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-total", type=int, default=1600)
    parser.add_argument("--n0", type=int, default=600, help="Anomaly-free training length (paper's n0).")
    parser.add_argument("--phi", type=str, default="0.6")
    parser.add_argument("--theta", type=str, default="0.3")
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--anomaly-duration", type=int, default=140)
    parser.add_argument("--anomaly-strength", type=float, default=0.6)
    parser.add_argument("--anomaly-start", type=int, default=None, help="Defaults to centering in [n0, n_total).")
    parser.add_argument("--window-size", type=int, default=20)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--max-p", type=int, default=3)
    parser.add_argument("--max-q", type=int, default=3)
    parser.add_argument("--n-reps", type=int, default=200, help='"Simulation times" -- number of Monte Carlo replications.')
    parser.add_argument("--seed", type=int, default=123, help="Base seed; replication i uses seed + i.")
    parser.add_argument("--anomaly-seed", type=int, default=456, help="Base anomaly seed; replication i uses anomaly_seed + i.")
    parser.add_argument("--output-dir", default="results/monte_carlo")
    args = parser.parse_args()

    if not 0 < args.n0 < args.n_total:
        raise ValueError("n0 must be strictly between 0 and n_total.")
    post_n0_length = args.n_total - args.n0
    if args.anomaly_duration > post_n0_length:
        raise ValueError(
            f"anomaly_duration ({args.anomaly_duration}) is longer than the test segment "
            f"(n_total - n0 = {post_n0_length}); lower --anomaly-duration or raise --n-total."
        )
    anomaly_start = args.anomaly_start
    if anomaly_start is None:
        anomaly_start = args.n0 + (post_n0_length - args.anomaly_duration) // 2
    if anomaly_start < args.n0 or anomaly_start + args.anomaly_duration > args.n_total:
        raise ValueError("anomaly_start/anomaly_duration place the anomaly outside [n0, n_total).")

    phi = parse_coeffs(args.phi)
    theta = parse_coeffs(args.theta)

    print(f"Running {args.n_reps} replications (phi={phi}, theta={theta}, sigma={args.sigma}, n0={args.n0}, window_size={args.window_size})...")
    all_results = []
    n_failed = 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i in range(args.n_reps):
            rep = run_one_replication(
                args.n_total, args.n0, phi, theta, args.sigma,
                anomaly_start, args.anomaly_duration, args.anomaly_strength,
                args.window_size, args.alpha, args.max_p, args.max_q,
                data_seed=args.seed + i, anomaly_seed=args.anomaly_seed + i,
            )
            if rep is None:
                n_failed += 1
                continue
            all_results.append(rep)
            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{args.n_reps} done")

    if n_failed:
        print(f"Warning: {n_failed}/{args.n_reps} replications failed to fit and were skipped.")

    pooled = pd.concat(all_results, ignore_index=True)
    n_reps_ok = args.n_reps - n_failed

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stats_summary = plot_summary(pooled, args.alpha, n_reps_ok, output_dir / "figures" / "summary.png")

    summary_row = {
        "n_reps": n_reps_ok, "n_total": args.n_total, "n0": args.n0,
        "phi": args.phi, "theta": args.theta, "sigma": args.sigma,
        "window_size": args.window_size, "alpha": args.alpha,
        "anomaly_duration": args.anomaly_duration, "anomaly_strength": args.anomaly_strength,
        **stats_summary,
    }
    # Same Windows file-lock retry as the PNG save (a viewer/editor can
    # briefly hold the CSV open).
    for attempt in range(3):
        try:
            pd.DataFrame([summary_row]).to_csv(output_dir / "summary.csv", index=False)
            break
        except PermissionError:
            if attempt == 2:
                raise
            time.sleep(0.5)

    print(f"Type I error: {stats_summary['type1_rate']:.2%} "
          f"[{stats_summary['type1_ci_low']:.2%}, {stats_summary['type1_ci_high']:.2%}] "
          f"(nominal alpha = {args.alpha:.0%}), pooled over {stats_summary['n_normal_windows']} normal windows")
    print(f"Power:        {stats_summary['power']:.2%} "
          f"[{stats_summary['power_ci_low']:.2%}, {stats_summary['power_ci_high']:.2%}], "
          f"pooled over {stats_summary['n_anomalous_windows']} anomalous windows")
    print(f"Results saved under {output_dir}")


if __name__ == "__main__":
    main()
