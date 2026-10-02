"""Run the paper's Algorithm 1 on simulated data and produce the paper
figure. Everything is tunable from one command: simulation parameters
(phi, theta, sigma, n0, anomaly settings) and algorithm parameters (window
size, alpha, AIC search bounds) are all CLI flags here -- no need to run
simulate.py separately first (though the generated data is still saved to
--data-dir, in case you want to reuse or inspect it).

Outputs:
  --data-dir   (default data/simulation): the generated series + labels
  --output-dir (default results/simulation):
    - window_results.csv   one row per test window
    - model.json            the fitted reference model
    - figures/summary.png   the paper figure (8 panels, see plot_summary_figure)
"""

from __future__ import annotations

import argparse
import json
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

# ---------------------------------------------------------------------------
# Raw-variance baseline: the plain sample variance of each window, scored
# against the plain sample variance of the training segment, with the same
# two-sided F-test. No ARMA modeling. This is the SPC-literature comparison
# point (Alwan & Roberts 1988): raw variance mixes real noise-variance
# changes with the process's own autocorrelation, so it's a noisier
# estimator than the innovation-variance estimator for the same window
# length. algorithm.py doesn't know about this -- it's a comparison, not
# part of Algorithm 1 -- so it lives here instead.
# ---------------------------------------------------------------------------


def raw_variance_reference(train_values: np.ndarray) -> tuple[float, int]:
    """Args: train_values, 1D array. Returns: (variance, degrees of freedom)."""
    return float(np.var(train_values, ddof=1)), len(train_values) - 1


def raw_variance_window(window_values: np.ndarray, ref_variance: float, ref_df: int) -> tuple[float, float, float]:
    """Args: one window's values, the training reference variance and its df.
    Returns: (window_variance, ratio, two-sided p-value).
    """
    variance = float(np.var(window_values, ddof=1))
    window_df = len(window_values) - 1
    ratio = variance / ref_variance
    lower_tail = stats.f.cdf(ratio, window_df, ref_df)
    upper_tail = stats.f.sf(ratio, window_df, ref_df)
    p_value = min(1.0, 2.0 * min(lower_tail, upper_tail))
    return variance, ratio, p_value


def add_evaluation_columns(
    results: pd.DataFrame,
    test_values: np.ndarray,
    test_is_anomaly: np.ndarray,
    train_values: np.ndarray,
) -> pd.DataFrame:
    """Add ground-truth and raw-variance-baseline columns to algorithm.py's
    output. algorithm.py knows nothing about labels or baselines -- that's
    evaluation, kept separate from detection.

    Args:
        results: algorithm.run_algorithm's results table.
        test_values: test segment, 1D array (same one passed to run_algorithm).
        test_is_anomaly: 0/1 ground-truth flag per point, aligned to test_values.
        train_values: training segment, 1D array (for the raw-variance reference).

    Returns:
        results, with four columns added: raw_variance_w, raw_variance_ratio,
        raw_variance_p_value, actual_anomaly.
    """
    ref_variance, ref_df = raw_variance_reference(train_values)
    raw_variance, raw_ratio, raw_p_value, actual_anomaly = [], [], [], []
    for _, row in results.iterrows():
        start, end = int(row["window_start"]), int(row["window_end"])
        variance, ratio, p_value = raw_variance_window(test_values[start:end], ref_variance, ref_df)
        raw_variance.append(variance)
        raw_ratio.append(ratio)
        raw_p_value.append(p_value)
        actual_anomaly.append(int(test_is_anomaly[start:end].any()))

    results = results.copy()
    results["raw_variance_w"] = raw_variance
    results["raw_variance_ratio"] = raw_ratio
    results["raw_variance_p_value"] = raw_p_value
    results["actual_anomaly"] = actual_anomaly
    return results


# ---------------------------------------------------------------------------
# Figure: left column (a)-(d) the series/variance/ratio/p-value; right
# column (e)-(h) the distributional checks Algorithm 1's derivation rests
# on. Print-friendly styling: serif type, no gridlines, thin axes.
# ---------------------------------------------------------------------------

COLOR_PROPOSED = "#2a78d6"
COLOR_RAW_VARIANCE = "#eb6834"
COLOR_INK = "#0b0b0b"
COLOR_MUTED = "#6b6a65"
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


def _style_hist_axis(ax) -> None:
    """Print-friendly spine style, keeping normal numeric x-ticks (for histograms)."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(COLOR_MUTED)
    ax.spines["bottom"].set_color(COLOR_MUTED)
    ax.tick_params(colors=COLOR_MUTED)


def _style_axis(ax) -> None:
    """Same as _style_hist_axis, but hides x-ticks (time is a direction, not values to read)."""
    _style_hist_axis(ax)
    ax.set_xticks([])


def format_model_summary(model: dict) -> str:
    """Args: the fitted reference model (fit_reference_model's output).
    Returns: one-line summary for the figure title -- order, every fixed
    parameter, and sigma0^2.
    """
    p, d, q = model["order"]
    param_parts = [f"{name}={value:.3f}" for name, value in zip(model["param_names"], model["fixed_params"])]
    params_text = ", ".join(param_parts) if param_parts else "no AR/MA terms"
    return f"Fitted reference model: ARMA(p,d,q)=({p},{d},{q})  |  {params_text}  |  sigma0^2={model['reference_sigma2']:.3f}"


def f_critical_values(window_size: int, reference_df: int, alpha: float) -> tuple[float, float]:
    """Two-sided decision boundary on R_w for panel (g): the test flags a
    window when R_w falls below F_{alpha/2} or above F_{1-alpha/2}.
    """
    return (
        float(stats.f.ppf(alpha / 2, window_size, reference_df)),
        float(stats.f.ppf(1 - alpha / 2, window_size, reference_df)),
    )


def n_bins(sample) -> int:
    """Rice rule (2*N^(1/3)), clamped to [20, 90] so the histogram shape
    stays readable whether there are 30 windows or 300."""
    return int(np.clip(round(2.0 * len(sample) ** (1 / 3)), 20, 90))


def plot_assumption_checks(
    fig, gs, results: pd.DataFrame, residuals_by_window: list[np.ndarray], model: dict, window_size: int, alpha: float
) -> None:
    """Right-hand column: empirical vs. theoretical checks of the three
    distributional claims Algorithm 1 depends on, built ONLY from windows
    that are truly normal (actual_anomaly == 0):
      (e) v_t / sqrt(sigma0^2 F_t^*)  ~  N(0,1)           [assumption A2]
      (f) n * R_w = n * sigma_w_hat^2/sigma0^2  ~ chi^2_n
      (g) R_w  ~  F(n, n0)
      (h) empirical false-positive rate vs. nominal alpha
    """
    normal_mask = (results["actual_anomaly"] == 0).to_numpy()
    normal_positions = np.where(normal_mask)[0]

    # Re-standardize each window's residuals by the REFERENCE sigma0^2
    # instead of that window's own sigma_w^2 (which would trivially sum to
    # n by construction): v_t/sqrt(sigma0^2 F_t^*) = std_t * sqrt(R_w).
    ratios = results["variance_ratio"].to_numpy()
    pooled_residuals = (
        np.concatenate([residuals_by_window[i] * np.sqrt(ratios[i]) for i in normal_positions])
        if len(normal_positions)
        else np.array([])
    )

    chi2_stat = window_size * results.loc[normal_mask, "variance_ratio"].to_numpy()
    f_stat = results.loc[normal_mask, "variance_ratio"].to_numpy()

    hist_kwargs = dict(density=True, color=COLOR_PROPOSED, alpha=0.55, edgecolor="white", linewidth=0.4)

    def draw_two_sided_cutoffs(ax, lower: float, upper: float, label: str) -> None:
        for i, cut in enumerate((lower, upper)):
            ax.axvline(cut, color=COLOR_RAW_VARIANCE, linewidth=1.1, linestyle="--", label=label if i == 0 else None)

    ax = fig.add_subplot(gs[0, 1])
    _style_hist_axis(ax)
    if len(pooled_residuals):
        ax.hist(pooled_residuals, bins=n_bins(pooled_residuals), **hist_kwargs)
        grid = np.linspace(*ax.get_xlim(), 400)
        ax.plot(grid, stats.norm.pdf(grid), color=COLOR_INK, linewidth=1.2, label="N(0,1)")
    ax.set_title("(e) Standardized Residuals", loc="left", fontsize=10.5, color=COLOR_INK)
    ax.set_xlabel(r"$v_t / \sqrt{\hat\sigma_0^2 F_t^*}$", fontsize=9)
    ax.legend(loc="upper right", fontsize=8, frameon=False)

    ax = fig.add_subplot(gs[1, 1])
    _style_hist_axis(ax)
    if len(chi2_stat):
        ax.hist(chi2_stat, bins=n_bins(chi2_stat), **hist_kwargs)
        grid = np.linspace(max(0.0, ax.get_xlim()[0]), ax.get_xlim()[1], 400)
        ax.plot(grid, stats.chi2.pdf(grid, df=window_size), color=COLOR_INK, linewidth=1.2, label=f"chi2({window_size})")
    ax.set_title("(f) Window Chi-Square Statistic", loc="left", fontsize=10.5, color=COLOR_INK)
    ax.set_xlabel(r"$n\,\hat\sigma_w^2/\hat\sigma_0^2$", fontsize=9)
    ax.legend(loc="upper right", fontsize=8, frameon=False)

    ax = fig.add_subplot(gs[2, 1])
    _style_hist_axis(ax)
    if len(f_stat):
        ax.hist(f_stat, bins=n_bins(f_stat), **hist_kwargs)
        grid = np.linspace(max(0.0, ax.get_xlim()[0]), ax.get_xlim()[1], 400)
        ax.plot(
            grid, stats.f.pdf(grid, window_size, model["reference_df"]),
            color=COLOR_INK, linewidth=1.2, label=f"F({window_size},{model['reference_df']})",
        )
        draw_two_sided_cutoffs(ax, *f_critical_values(window_size, model["reference_df"], alpha), rf"two-sided $\alpha$={alpha}")
    ax.set_title("(g) Variance Ratio (Normal Windows)", loc="left", fontsize=10.5, color=COLOR_INK)
    ax.set_xlabel(r"$R_w = \hat\sigma_w^2/\hat\sigma_0^2$", fontsize=9)
    ax.legend(loc="upper right", fontsize=8, frameon=False)

    ax = fig.add_subplot(gs[3, 1])
    ax.axis("off")
    n_normal = int(normal_mask.sum())
    n_false_positive = int((results.loc[normal_mask, "p_value"] < alpha).sum())
    if n_normal > 0:
        observed_rate = n_false_positive / n_normal
        ci_low = stats.beta.ppf(0.025, n_false_positive, n_normal - n_false_positive + 1) if n_false_positive > 0 else 0.0
        ci_high = stats.beta.ppf(0.975, n_false_positive + 1, n_normal - n_false_positive) if n_false_positive < n_normal else 1.0
        summary_text = (
            "(h) Empirical False-Positive Rate\n\n"
            f"{n_false_positive} / {n_normal} normal windows\n"
            f"= {observed_rate:.1%}\n\n"
            f"nominal alpha = {alpha:.0%}\n"
            f"95% exact CI = [{ci_low:.1%}, {ci_high:.1%}]"
        )
    else:
        summary_text = "(h) Empirical False-Positive Rate\n\nNo normal test windows."
    ax.text(0.0, 0.85, summary_text, fontsize=9.5, color=COLOR_INK, va="top", linespacing=1.8)


def plot_summary_figure(
    test_values: np.ndarray,
    test_is_anomaly: np.ndarray,
    results: pd.DataFrame,
    residuals_by_window: list[np.ndarray],
    model: dict,
    raw_ref: tuple[float, int],
    window_size: int,
    alpha: float,
    output_path: Path,
) -> None:
    """Build and save the 8-panel paper figure. Args: the test segment and
    its ground-truth flags, the scored results table, each window's
    standardized residuals, the fitted model, the raw-variance reference
    (variance, df), the window size and alpha, and where to save the PNG.
    """
    x_series = np.arange(len(test_values))
    anomaly_positions = np.where(test_is_anomaly == 1)[0]
    anomaly_start_pos = int(anomaly_positions.min()) if len(anomaly_positions) else None
    anomaly_end_pos = int(anomaly_positions.max()) if len(anomaly_positions) else None

    fig = plt.figure(figsize=(13, 10.5))
    gs = fig.add_gridspec(4, 2, width_ratios=(1.6, 1), hspace=0.55, wspace=0.32)

    plot_assumption_checks(fig, gs, results, residuals_by_window, model, window_size, alpha)

    ax = fig.add_subplot(gs[0, 0])
    _style_axis(ax)
    ax.plot(x_series, test_values, color=COLOR_INK, linewidth=0.9)
    if anomaly_start_pos is not None:
        ax.axvspan(anomaly_start_pos, anomaly_end_pos, color=COLOR_ANOMALY_SPAN, alpha=0.6, linewidth=0, label="Anomaly period")
    # Headroom above the tallest spike, so the "upper right" legend never
    # overlaps a data point that happens to reach the top of the axes.
    y_min, y_max = ax.get_ylim()
    ax.set_ylim(y_min, y_max + 0.18 * (y_max - y_min))
    ax.set_ylabel("Value")
    ax.set_title("(a) Simulated Time Series", loc="left", fontsize=10.5, color=COLOR_INK)
    ax.legend(loc="upper right", fontsize=9, frameon=True, facecolor="white", edgecolor="none")

    x = results["window_start"] + window_size // 2

    ax = fig.add_subplot(gs[1, 0])
    _style_axis(ax)
    if anomaly_start_pos is not None:
        ax.axvspan(anomaly_start_pos, anomaly_end_pos, color=COLOR_ANOMALY_SPAN, alpha=0.6, linewidth=0)
    ax.axhline(model["reference_sigma2"], color=COLOR_PROPOSED, linewidth=1.0, linestyle=":")
    ax.axhline(raw_ref[0], color=COLOR_RAW_VARIANCE, linewidth=1.0, linestyle=":")
    for column, color, label in (("sigma2_w", COLOR_PROPOSED, "Innovation variance"), ("raw_variance_w", COLOR_RAW_VARIANCE, "Raw variance")):
        ax.plot(x, results[column], color=color, marker="o", ms=4.5, linewidth=1.3, label=label)
    ax.set_ylabel("Variance")
    ax.set_title("(b) Estimated Innovation Variance", loc="left", fontsize=10.5, color=COLOR_INK)
    ax.legend(loc="upper right", fontsize=9, frameon=False)

    ax = fig.add_subplot(gs[2, 0])
    _style_axis(ax)
    if anomaly_start_pos is not None:
        ax.axvspan(anomaly_start_pos, anomaly_end_pos, color=COLOR_ANOMALY_SPAN, alpha=0.6, linewidth=0)
    ax.axhline(1.0, color=COLOR_MUTED, linewidth=1.0, linestyle=":")
    for column, color, label in (("variance_ratio", COLOR_PROPOSED, "Innovation variance ratio"), ("raw_variance_ratio", COLOR_RAW_VARIANCE, "Raw variance ratio")):
        ax.plot(x, results[column], color=color, marker="o", ms=4.5, linewidth=1.3, label=label)
    ax.set_ylabel("Variance Ratio")
    ax.set_title("(c) Variance Ratio", loc="left", fontsize=10.5, color=COLOR_INK)
    ax.legend(loc="upper right", fontsize=9, frameon=False)

    ax = fig.add_subplot(gs[3, 0])
    _style_axis(ax)
    if anomaly_start_pos is not None:
        ax.axvspan(anomaly_start_pos, anomaly_end_pos, color=COLOR_ANOMALY_SPAN, alpha=0.6, linewidth=0)
    ax.axhline(alpha, color=COLOR_MUTED, linewidth=1.0, linestyle="--", label=f"alpha = {alpha}")
    p = results["p_value"]
    significant = p < alpha
    ax.plot(x, p, color=COLOR_PROPOSED, linewidth=1.3, zorder=2)
    ax.scatter(x[significant], p[significant], s=30, color=COLOR_PROPOSED, zorder=3)
    ax.scatter(x[~significant], p[~significant], s=30, facecolors="white", edgecolors=COLOR_PROPOSED, linewidths=1.2, zorder=3)
    ax.set_ylim(-0.02, 1.02)
    ax.set_ylabel("Two-Sided p-Value")
    ax.set_xlabel("Time")
    ax.set_title("(d) Statistical Significance", loc="left", fontsize=10.5, color=COLOR_INK)
    ax.legend(loc="upper right", fontsize=9, frameon=False)

    fig.suptitle(format_model_summary(model), fontsize=9, color=COLOR_MUTED, y=0.995)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # panel (h) is text-only, harmless for tight_layout
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))

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


COLOR_BW_LINE = "#1a1a1a"   # near-black: the data line + markers, both panels, in the grayscale version
COLOR_BW_LIMIT = "#8a8a8a"  # mid-gray: the dashed control-limit lines, both panels, in the grayscale version


def plot_results_figure_split(
    test_values: np.ndarray,
    test_is_anomaly: np.ndarray,
    results: pd.DataFrame,
    model: dict,
    raw_ref: tuple[float, int],
    window_size: int,
    alpha: float,
    output_path: Path,
    grayscale: bool = False,
) -> None:
    """Three-panel alternative to plot_results_figure: instead of overlaying
    both ratios (and both control-limit bands) on one axis -- where the two
    bands can sit almost on top of each other and become hard to tell
    apart -- each method gets its OWN panel with its OWN control limits.
    (a) series, (b) innovation-variance ratio + its limits, (c) raw-variance
    ratio + its limits. Same x-axis across all three, so timing is still
    directly comparable.

    grayscale=True swaps the blue/orange palette for near-black (data line)
    / mid-gray (control limits) in BOTH panels -- panels are already told
    apart by title, so color no longer needs to distinguish them; instead
    color now distinguishes the data line from its control limits.
    """
    b_line_color = COLOR_BW_LINE if grayscale else COLOR_PROPOSED
    b_limit_color = COLOR_BW_LIMIT if grayscale else COLOR_PROPOSED
    c_line_color = COLOR_BW_LINE if grayscale else COLOR_RAW_VARIANCE
    c_limit_color = COLOR_BW_LIMIT if grayscale else COLOR_RAW_VARIANCE
    x_series = np.arange(len(test_values))
    anomaly_positions = np.where(test_is_anomaly == 1)[0]
    anomaly_start_pos = int(anomaly_positions.min()) if len(anomaly_positions) else None
    anomaly_end_pos = int(anomaly_positions.max()) if len(anomaly_positions) else None
    x = results["window_start"] + window_size // 2

    fig, axes = plt.subplots(3, 1, figsize=(8, 4.7), height_ratios=(1, 1.1, 1.1))

    ax = axes[0]
    _style_axis(ax)
    ax.plot(x_series, test_values, color=COLOR_INK, linewidth=0.9)
    if anomaly_start_pos is not None:
        ax.axvspan(anomaly_start_pos, anomaly_end_pos, color=COLOR_ANOMALY_SPAN, alpha=0.6, linewidth=0, label="Anomaly period")
    y_min, y_max = ax.get_ylim()
    ax.set_ylim(y_min, y_max + 0.18 * (y_max - y_min))
    ax.set_ylabel("Value")
    ax.set_title("(a) Simulated Time Series", loc="left", fontsize=10.5, color=COLOR_INK)
    ax.legend(loc="upper right", fontsize=9, frameon=True, facecolor="white", edgecolor="none")

    # (b) innovation-variance ratio, its own control limits only.
    ax = axes[1]
    _style_axis(ax)
    if anomaly_start_pos is not None:
        ax.axvspan(anomaly_start_pos, anomaly_end_pos, color=COLOR_ANOMALY_SPAN, alpha=0.6, linewidth=0)
    ax.axhline(1.0, color=COLOR_MUTED, linewidth=1.0, linestyle=":")
    lower, upper = f_critical_values(window_size, model["reference_df"], alpha)
    ax.axhline(lower, color=b_limit_color, linewidth=1.1, linestyle="--", label=f"control limits (alpha={alpha})")
    ax.axhline(upper, color=b_limit_color, linewidth=1.1, linestyle="--")
    significant = results["p_value"] < alpha
    ratio = results["variance_ratio"]
    ax.plot(x, ratio, color=b_line_color, linewidth=1.4, zorder=3, label="Innovation variance ratio")
    ax.scatter(x[significant], ratio[significant], s=18, color=b_line_color, zorder=4)
    ax.scatter(x[~significant], ratio[~significant], s=18, facecolors="white", edgecolors=b_line_color, linewidths=1.0, zorder=4)
    ax.set_ylabel("Variance Ratio")
    ax.set_title("(b) Innovation Variance Ratio", loc="left", fontsize=10.5, color=COLOR_INK)
    ax.legend(loc="upper right", fontsize=8.5, frameon=False)

    # (c) raw-variance ratio, its own control limits only.
    ax = axes[2]
    _style_axis(ax)
    if anomaly_start_pos is not None:
        ax.axvspan(anomaly_start_pos, anomaly_end_pos, color=COLOR_ANOMALY_SPAN, alpha=0.6, linewidth=0)
    ax.axhline(1.0, color=COLOR_MUTED, linewidth=1.0, linestyle=":")
    raw_lower, raw_upper = f_critical_values(window_size - 1, raw_ref[1], alpha)
    ax.axhline(raw_lower, color=c_limit_color, linewidth=1.1, linestyle="--", label=f"control limits (alpha={alpha})")
    ax.axhline(raw_upper, color=c_limit_color, linewidth=1.1, linestyle="--")
    raw_significant = results["raw_variance_p_value"] < alpha
    raw_ratio = results["raw_variance_ratio"]
    ax.plot(x, raw_ratio, color=c_line_color, linewidth=1.1, zorder=3, label="Raw variance ratio")
    ax.scatter(x[raw_significant], raw_ratio[raw_significant], s=14, color=c_line_color, zorder=4)
    ax.scatter(x[~raw_significant], raw_ratio[~raw_significant], s=14, facecolors="white", edgecolors=c_line_color, linewidths=0.9, zorder=4)
    ax.set_ylabel("Variance Ratio")
    ax.set_xlabel("Time")
    ax.set_title("(c) Raw Variance Ratio", loc="left", fontsize=10.5, color=COLOR_INK)
    ax.legend(loc="upper right", fontsize=8.5, frameon=False)

    # Shared y-axis across (b) and (c) so the two ratios are directly
    # comparable at a glance, not just relative to their own panel.
    shared_min = min(ratio.min(), raw_ratio.min(), lower, raw_lower)
    shared_max = max(ratio.max(), raw_ratio.max(), upper, raw_upper)
    pad = 0.08 * (shared_max - shared_min)
    axes[1].set_ylim(shared_min - pad, shared_max + pad)
    axes[2].set_ylim(shared_min - pad, shared_max + pad)

    fig.tight_layout()
    # tight_layout gives the y-label a wider left margin than the empty
    # right side, so the plot area sits left-of-center. Mirror the left
    # margin onto the right so the axes block itself is centered.
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


def plot_results_figure(
    test_values: np.ndarray,
    test_is_anomaly: np.ndarray,
    results: pd.DataFrame,
    model: dict,
    raw_ref: tuple[float, int],
    window_size: int,
    alpha: float,
    output_path: Path,
) -> None:
    """Two-panel, control-chart style figure for the paper's Results
    section (separate from plot_summary_figure's 8-panel diagnostic
    figure, which is unchanged). Panel (a): the raw series with the true
    anomaly shaded. Panel (b): the variance ratio R_w per window, proposed
    method vs. the raw-variance baseline, EACH with its own two-sided
    control limits -- they are not the same band, because the two
    estimators have different degrees of freedom (the raw sample variance
    loses one d.f. estimating the window mean; the innovation-variance
    estimator doesn't, since c/phi/theta are already fixed from training).
    A point outside its own band is exactly what its own p_value < alpha
    flags.

    Args: the test segment and its ground-truth flags, the scored results
    table, the fitted model, the raw-variance reference (variance, df),
    the window size and alpha, and where to save the PNG. Returns:
    nothing (saves the figure to output_path).
    """
    x_series = np.arange(len(test_values))
    anomaly_positions = np.where(test_is_anomaly == 1)[0]
    anomaly_start_pos = int(anomaly_positions.min()) if len(anomaly_positions) else None
    anomaly_end_pos = int(anomaly_positions.max()) if len(anomaly_positions) else None

    fig, axes = plt.subplots(2, 1, figsize=(8, 6), height_ratios=(1, 1.3))

    ax = axes[0]
    _style_axis(ax)
    ax.plot(x_series, test_values, color=COLOR_INK, linewidth=0.9)
    if anomaly_start_pos is not None:
        ax.axvspan(anomaly_start_pos, anomaly_end_pos, color=COLOR_ANOMALY_SPAN, alpha=0.6, linewidth=0, label="Anomaly period")
    # Headroom above the tallest spike, so the "upper right" legend never
    # overlaps a data point that happens to reach the top of the axes.
    y_min, y_max = ax.get_ylim()
    ax.set_ylim(y_min, y_max + 0.18 * (y_max - y_min))
    ax.set_ylabel("Value")
    ax.set_title("(a) Simulated Time Series", loc="left", fontsize=10.5, color=COLOR_INK)
    ax.legend(loc="upper right", fontsize=9, frameon=True, facecolor="white", edgecolor="none")

    ax = axes[1]
    _style_axis(ax)
    if anomaly_start_pos is not None:
        ax.axvspan(anomaly_start_pos, anomaly_end_pos, color=COLOR_ANOMALY_SPAN, alpha=0.6, linewidth=0)

    x = results["window_start"] + window_size // 2
    ax.axhline(1.0, color=COLOR_MUTED, linewidth=1.0, linestyle=":")

    # Innovation-variance ratio: its own control limits, F(window_size, n0).
    lower, upper = f_critical_values(window_size, model["reference_df"], alpha)
    ax.axhline(lower, color=COLOR_PROPOSED, linewidth=1.1, linestyle="--", label=f"control limits, innovation (alpha={alpha})")
    ax.axhline(upper, color=COLOR_PROPOSED, linewidth=1.1, linestyle="--")

    # Raw-variance ratio: its OWN control limits, F(window_size - 1, n0 - 1)
    # -- one fewer degree of freedom each side, since the raw sample
    # variance also has to estimate the window/training mean.
    raw_lower, raw_upper = f_critical_values(window_size - 1, raw_ref[1], alpha)
    ax.axhline(raw_lower, color=COLOR_RAW_VARIANCE, linewidth=1.1, linestyle="--", label=f"control limits, raw (alpha={alpha})")
    ax.axhline(raw_upper, color=COLOR_RAW_VARIANCE, linewidth=1.1, linestyle="--")

    raw_significant = results["raw_variance_p_value"] < alpha
    raw_ratio = results["raw_variance_ratio"]
    ax.plot(x, raw_ratio, color=COLOR_RAW_VARIANCE, linewidth=1.1, zorder=2, label="Raw variance ratio")
    ax.scatter(x[raw_significant], raw_ratio[raw_significant], s=26, color=COLOR_RAW_VARIANCE, zorder=3)
    ax.scatter(x[~raw_significant], raw_ratio[~raw_significant], s=26, facecolors="white", edgecolors=COLOR_RAW_VARIANCE, linewidths=1.1, zorder=3)

    significant = results["p_value"] < alpha
    ratio = results["variance_ratio"]
    ax.plot(x, ratio, color=COLOR_PROPOSED, linewidth=1.4, zorder=4, label="Innovation variance ratio")
    ax.scatter(x[significant], ratio[significant], s=32, color=COLOR_PROPOSED, zorder=5)
    ax.scatter(x[~significant], ratio[~significant], s=32, facecolors="white", edgecolors=COLOR_PROPOSED, linewidths=1.2, zorder=5)

    ax.set_ylabel("Variance Ratio")
    ax.set_xlabel("Time")
    ax.set_title("(b) Variance Ratio with Control Limits", loc="left", fontsize=10.5, color=COLOR_INK)
    ax.legend(loc="upper right", fontsize=8, frameon=False)

    fig.suptitle(format_model_summary(model), fontsize=9, color=COLOR_MUTED, y=0.98)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))

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
    """Read CLI flags, simulate data, run Algorithm 1, save results + figure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-total", type=int, default=1600)
    parser.add_argument("--n0", type=int, default=600, help="Anomaly-free training length (paper's n0).")
    parser.add_argument("--phi", type=str, default="0.6")
    parser.add_argument("--theta", type=str, default="0.3")
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--anomaly-duration", type=int, default=140)
    parser.add_argument("--anomaly-strength", type=float, default=.6)
    parser.add_argument("--anomaly-start", type=int, default=None, help="Defaults to centering in [n0, n_total).")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--anomaly-seed", type=int, default=6545)
    parser.add_argument("--window-size", type=int, default=20)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--max-p", type=int, default=3)
    parser.add_argument("--max-q", type=int, default=3)
    parser.add_argument("--data-dir", default="data/simulation")
    parser.add_argument("--output-dir", default="results/simulation")
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
    if anomaly_start < args.n0:
        raise ValueError(f"anomaly_start ({anomaly_start}) must be >= n0 ({args.n0}).")
    if anomaly_start + args.anomaly_duration > args.n_total:
        raise ValueError("anomaly_start + anomaly_duration exceeds n_total.")

    phi = parse_coeffs(args.phi)
    theta = parse_coeffs(args.theta)
    clean_values = simulate_arma(args.n_total, phi, theta, args.sigma, args.seed)
    values, is_anomaly = add_variance_burst(clean_values, anomaly_start, args.anomaly_duration, args.anomaly_strength, args.anomaly_seed)

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    index = pd.RangeIndex(args.n_total, name="t")
    pd.DataFrame({"value": clean_values}, index=index).to_csv(data_dir / "simulated_clean.csv")
    pd.DataFrame({"value": values}, index=index).to_csv(data_dir / "simulated_with_anomaly.csv")
    pd.DataFrame({"is_anomaly": is_anomaly}, index=index).to_csv(data_dir / "simulated_anomaly_labels.csv")

    train_values = values[: args.n0]
    test_values = values[args.n0 :]
    test_is_anomaly = is_anomaly[args.n0 :]
    print(f"Training segment: {len(train_values)} points, test segment: {len(test_values)} points")

    results, model, residuals_by_window = run_algorithm(
        train_values, test_values, args.window_size, args.alpha, args.max_p, args.max_q
    )
    print(f"Selected order (p, d, q) = {model['order']}, AIC = {model['aic']:.2f}")
    print(f"Reference sigma0^2 = {model['reference_sigma2']:.4f}")

    results = add_evaluation_columns(results, test_values, test_is_anomaly, train_values)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_dir / "window_results.csv", index=False)

    model_record = {**model, "fixed_params": model["fixed_params"].tolist()}
    (output_dir / "model.json").write_text(json.dumps(model_record, indent=2), encoding="utf-8")

    n_true = int(results["actual_anomaly"].sum())
    n_normal = int((results["actual_anomaly"] == 0).sum())
    n_detected = int(((results["actual_anomaly"] == 1) & (results["p_value"] < args.alpha)).sum())
    n_false_alarms = int(((results["actual_anomaly"] == 0) & (results["p_value"] < args.alpha)).sum())
    n_raw_detected = int(((results["actual_anomaly"] == 1) & (results["raw_variance_p_value"] < args.alpha)).sum())
    n_raw_false_alarms = int(((results["actual_anomaly"] == 0) & (results["raw_variance_p_value"] < args.alpha)).sum())

    plot_summary_figure(
        test_values, test_is_anomaly, results, residuals_by_window, model,
        raw_variance_reference(train_values), args.window_size, args.alpha,
        output_dir / "figures" / "summary.png",
    )
    plot_results_figure(
        test_values, test_is_anomaly, results, model,
        raw_variance_reference(train_values), args.window_size, args.alpha,
        output_dir / "figures" / "results.png",
    )
    plot_results_figure_split(
        test_values, test_is_anomaly, results, model,
        raw_variance_reference(train_values), args.window_size, args.alpha,
        output_dir / "figures" / "results_split.png",
    )
    plot_results_figure_split(
        test_values, test_is_anomaly, results, model,
        raw_variance_reference(train_values), args.window_size, args.alpha,
        output_dir / "figures" / "results_split_bw.png",
        grayscale=True,
    )

    print(f"Test windows: {len(results)} total, {n_true} overlap the true anomaly")
    print(f"Proposed method: detected {n_detected}/{n_true}, false alarms {n_false_alarms}/{n_normal}")
    print(f"Raw variance:    detected {n_raw_detected}/{n_true}, false alarms {n_raw_false_alarms}/{n_normal}")
    print(f"Results saved under {output_dir}")


if __name__ == "__main__":
    main()
