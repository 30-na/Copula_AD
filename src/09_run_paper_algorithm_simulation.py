"""Run the paper's Algorithm 1 (fixed-dynamics innovation-variance test) on
simulated data, step by step, matching the paper's Methodology section as
closely as possible. No train/train-A/train-B split trickery, no seasonal
terms, no empirical calibration -- just the plain algorithm from the paper:

    1. Split the series into a training segment (assumed anomaly-free) and
       a test segment.
    2. On the training segment, pick ARMA(p, q) orders with AIC, fit the
       model, and record the reference innovation variance sigma0^2.
    3. Cut the test segment into non-overlapping windows of length n.
    4. For each window, KEEP phi/theta FIXED at their training-segment
       values, run the Kalman filter, and estimate the window's own
       innovation variance sigma_w^2 (this is the "concentrated scale"
       trick: only sigma^2 is re-estimated per window, nothing else).
    5. Form the ratio R_w = sigma_w^2 / sigma0^2 and compute a TWO-SIDED
       p-value from the F(n, n0) distribution (paper Eq. 19): both a big
       increase and a big decrease in variance count as anomalous.
    6. Flag window w as anomalous if p_w < alpha.

This script also computes a comparison baseline: the plain sample variance
of each raw window (no ARIMA, no accounting for autocorrelation), scored
against the plain sample variance of the raw training segment with the
same nominal F-test. This is what monitoring raw variance directly, instead
of modeling the process first, gets you. The classic argument for why this
is not a fair comparison for autocorrelated data is well established in the
statistical process control (SPC) literature: Alwan & Roberts (1988,
"Time-Series Modeling for Statistical Process Control", Journal of Business
& Economic Statistics) showed that applying standard control-chart limits
to raw autocorrelated data is "seriously misleading", and the standard fix
is to fit a time-series model and monitor its residuals instead, because
the residuals are (approximately) independent while the raw data are not.
Because the raw window variance mixes genuine noise-variance changes with
the process's own autocorrelation, it is a noisier (higher-variance)
estimator than the innovation-variance estimator for the same window
length -- that is the effect this script's comparison plot is meant to
show, not a claim that raw variance can never detect anything.

This script is intentionally simple/flat (no classes, no multiprocessing)
so it is easy to read side by side with the paper.

Outputs (under --output-dir):
  - window_results.csv       one row per test window: both methods' variance
                              ratios and p-values, for the raw data used
                              in `figures/summary.png`
  - fixed_model.json         the fitted reference model (phi, theta, sigma0^2)
  - figures/summary.png      8-panel figure for the paper. Left column
                              (a)-(d): the test series, per-window variance,
                              variance ratio and p-value, our method against
                              the raw-variance baseline. Right column
                              (e)-(h): the distributional checks Algorithm 1
                              rests on -- standardized residuals vs. N(0,1),
                              n*R_w vs. chi^2_n, R_w vs. F(n, n0), and the
                              empirical false-positive rate vs. alpha.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.arima.model import ARIMA

# Differencing order. The paper's assumption (A1) is that the normal process
# is stationary and invertible, so the model is a pure ARMA(p, q): d = 0.
D_ORDER = 0

os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".matplotlib"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Step 1: load the simulated series and its anomaly labels.
# ---------------------------------------------------------------------------

def load_simulation_data(input_path: Path, labels_path: Path) -> tuple[pd.Series, pd.DataFrame]:
    series_df = pd.read_csv(input_path, index_col=0, parse_dates=True)
    labels_df = pd.read_csv(labels_path, index_col=0, parse_dates=True)
    series = series_df.iloc[:, 0].astype(float)
    return series, labels_df


# ---------------------------------------------------------------------------
# Step 2: pick the training/test split. The split point is a fraction of the
# whole series (train_fraction), NOT "wherever the anomaly happens to
# start". This matters for two reasons:
#   - the training segment should be reasonably large, since it has to
#     support both order selection (AIC) and stable parameter estimates;
#   - in a real deployment you would not know exactly when an anomaly
#     starts, so the split should be a modeling decision, not something
#     read off the labels. Here the anomaly is deliberately placed so it
#     sits in the MIDDLE of the resulting test segment (normal data both
#     before and after it), which gives a much more realistic test: the
#     algorithm has to stay quiet on normal test windows on both sides of
#     the anomaly, not just "everything after the anomaly starts".
# We still sanity-check that no anomalous point leaked into the training
# segment, since that would violate the "training segment is anomaly-free"
# assumption stated in the paper's Section 4.2.
# ---------------------------------------------------------------------------

def split_train_test(
    series: pd.Series, labels: pd.DataFrame, train_fraction: float
) -> tuple[pd.Series, pd.Series]:
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be strictly between 0 and 1.")

    train_end = int(round(len(series) * train_fraction))
    train_segment = series.iloc[:train_end]
    test_segment = series.iloc[train_end:]

    anomaly_flags = labels["is_anomaly"].to_numpy(dtype=int)
    if anomaly_flags[:train_end].sum() > 0:
        raise ValueError(
            "The training segment contains anomalous points; lower --train-fraction "
            "or move the anomaly later so the training segment stays anomaly-free."
        )
    return train_segment, test_segment


# ---------------------------------------------------------------------------
# Step 3: select ARMA(p, q) orders on the training segment using AIC (paper
# Section 4.3: "orders p and q are selected using AIC"). The differencing
# order is fixed at d = 0 -- the paper's assumption (A1) is that the normal
# process is stationary and invertible, so no differencing is applied.
# ---------------------------------------------------------------------------

def select_and_fit_reference_model(
    train_segment: pd.Series,
    max_p: int,
    max_q: int,
) -> dict:
    best_aic = np.inf
    best_fit = None
    best_order = None

    train_values = train_segment.to_numpy(dtype=float)

    for p in range(max_p + 1):
        for q in range(max_q + 1):
            if p == 0 and q == 0:
                continue  # skip the trivial "no dynamics" model
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", ConvergenceWarning)
                    warnings.simplefilter("ignore", UserWarning)
                    fit = ARIMA(train_values, order=(p, D_ORDER, q)).fit()
            except Exception:
                continue  # this (p, q) combination failed to fit; skip it
            if np.isfinite(fit.aic) and fit.aic < best_aic:
                best_aic = fit.aic
                best_fit = fit
                best_order = (p, D_ORDER, q)

    if best_fit is None:
        raise RuntimeError("No ARMA(p, q) model could be fit on the training segment.")

    # best_fit.param_names looks like [const, ar.L1, ..., ma.L1, ..., sigma2]
    # (the constant is present because d = 0 makes statsmodels default to
    # trend='c'). We need sigma0^2 (the reference innovation variance) and
    # the REST of the parameters -- c_hat, the phi's and the theta's --
    # which are held fixed per window, as in Algorithm 1 line 5.
    param_names = list(best_fit.param_names)
    params = np.asarray(best_fit.params, dtype=float)
    sigma_index = param_names.index("sigma2")

    reference_sigma2 = float(params[sigma_index])
    fixed_param_names = [name for name in param_names if name != "sigma2"]
    fixed_params = np.delete(params, sigma_index)

    # Reference degrees of freedom for the F-test denominator: n0, the
    # training segment length, matching the paper's Eq. 15 exactly.
    #
    # CAVEAT: sigma0^2 is computed on the same data that produced phi_hat
    # and theta_hat, so chi^2_{n0} is optimistic -- the honest df is closer
    # to n0 - p - q - 1. The effect is small when n0 is large, but it
    # inflates the observed false-positive rate slightly above alpha.
    reference_df = len(train_values)

    return {
        "order": best_order,
        "param_names": fixed_param_names,
        "fixed_params": fixed_params,
        "reference_sigma2": reference_sigma2,
        "reference_df": reference_df,
        "aic": float(best_aic),
    }


# ---------------------------------------------------------------------------
# Step 4: cut the test segment into non-overlapping windows of length
# window_size, matching Algorithm 1 line 3 ("Divide the test segment into
# nonoverlapping windows of length n").
# ---------------------------------------------------------------------------

def make_nonoverlapping_windows(series: pd.Series, window_size: int) -> list[pd.Series]:
    # A trailing partial window is dropped: every window must hold exactly n
    # points for F(n, n0) in Eq. 18 to be the right reference distribution.
    windows = []
    n_windows = len(series) // window_size
    for i in range(n_windows):
        start = i * window_size
        end = start + window_size
        windows.append(series.iloc[start:end])
    return windows


# ---------------------------------------------------------------------------
# Step 5: for one window, hold phi/theta FIXED at the reference values and
# re-estimate only sigma_w^2 via the Kalman filter (paper Eq. 12-13). This
# is what statsmodels calls "concentrate_scale": sigma^2 is analytically
# concentrated out of the likelihood, so .filter() does not re-optimize
# anything -- it just runs the Kalman filter once with the fixed phi/theta
# and reads off the resulting scale estimate.
# ---------------------------------------------------------------------------

def fixed_dynamics_window_variance(window: pd.Series, model_info: dict) -> tuple[float, np.ndarray]:
    model = ARIMA(
        window.to_numpy(dtype=float),
        order=model_info["order"],
        concentrate_scale=True,
    )
    if list(model.param_names) != model_info["param_names"]:
        raise RuntimeError("Parameter mismatch between reference model and window model.")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = model.filter(model_info["fixed_params"])

    # statsmodels' `scale` with concentrate_scale=True is exactly Eq. 13,
    # (1/n) * sum_{t in w} v_t^2 / F_t^*, summed over ALL n points of the
    # window. The process is stationary (assumption A1) and d = 0, so the
    # Kalman filter starts from the stationary prior a_1 = 0, P_1 = Q_0
    # (paper Sec. 3.3) and there is no diffuse warm-up to discard: n is
    # simply the window length, exactly as in the paper.
    sigma2_w = float(result.scale)

    # statsmodels' standardized_forecasts_error is v_t / sqrt(sigma2_w * F_t^*),
    # i.e. it is standardized using THIS window's own re-estimated sigma2_w
    # (by construction, the sum of its squares is always exactly n -- a
    # tautology, not something you can check). To get the version the
    # paper's assumption (A2) actually makes a falsifiable claim about --
    # v_t standardized by the REFERENCE/null sigma0^2 -- multiply by
    # sqrt(sigma2_w / sigma0^2) = sqrt(R_w):
    #   v_t/sqrt(sigma0^2 F_t^*) = [v_t/sqrt(sigma2_w F_t^*)] * sqrt(sigma2_w/sigma0^2)
    # The caller does that multiplication (it needs R_w, computed next).
    standardized_residuals = np.asarray(
        result.filter_results.standardized_forecasts_error[0], dtype=float
    )

    return sigma2_w, standardized_residuals


# ---------------------------------------------------------------------------
# Step 5 (continued): variance ratio R_w and its two-sided p-value.
# Paper Eq. 17 and 19:
#   R_w = sigma_w^2 / sigma0^2  ~  F(n, n0)   under H0
#   p_w = 2 * min( F_cdf(R_w), 1 - F_cdf(R_w) )
# ---------------------------------------------------------------------------

def variance_ratio_two_sided_pvalue(sigma2_w: float, window_size: int, model_info: dict) -> tuple[float, float]:
    ratio = sigma2_w / model_info["reference_sigma2"]
    lower_tail = stats.f.cdf(ratio, window_size, model_info["reference_df"])
    upper_tail = stats.f.sf(ratio, window_size, model_info["reference_df"])
    p_value = min(1.0, 2.0 * min(lower_tail, upper_tail))
    return ratio, p_value


# ---------------------------------------------------------------------------
# Raw-variance baseline for comparison: skip the ARIMA model entirely and
# just take the plain sample variance of the raw window values, then run
# the same kind of two-sided F-test against the plain sample variance of
# the raw training segment. This is what "just look at the variance"
# means in practice, and it is the standard comparison point in the SPC
# literature (see the module docstring): it silently assumes the raw
# values are independent, which they are NOT here because of the AR(1)
# autocorrelation. We use n - 1 degrees of freedom (not n) because, unlike
# our innovation-variance estimator, the raw sample variance also has to
# estimate the window mean.
# ---------------------------------------------------------------------------

def compute_raw_variance_reference(train_segment: pd.Series) -> tuple[float, int]:
    values = train_segment.to_numpy(dtype=float)
    return float(np.var(values, ddof=1)), len(values) - 1


def raw_variance_window_pvalue(
    window: pd.Series, reference_variance: float, reference_df: int
) -> tuple[float, float, float]:
    values = window.to_numpy(dtype=float)
    window_variance = float(np.var(values, ddof=1))
    window_df = len(values) - 1
    ratio = window_variance / reference_variance
    lower_tail = stats.f.cdf(ratio, window_df, reference_df)
    upper_tail = stats.f.sf(ratio, window_df, reference_df)
    p_value = min(1.0, 2.0 * min(lower_tail, upper_tail))
    return window_variance, ratio, p_value


# ---------------------------------------------------------------------------
# Put steps 4-6 together: score every test window.
# ---------------------------------------------------------------------------

def score_test_windows(
    test_segment: pd.Series,
    labels: pd.DataFrame,
    model_info: dict,
    raw_variance_reference: tuple[float, int],
    window_size: int,
    alpha: float,
) -> tuple[pd.DataFrame, list[np.ndarray]]:
    windows = make_nonoverlapping_windows(test_segment, window_size)
    anomaly_times = labels.index[labels["is_anomaly"] == 1]
    raw_ref_variance, raw_ref_df = raw_variance_reference

    rows = []
    h0_standardized_residuals_by_window = []
    for i, window in enumerate(windows):
        sigma2_w, standardized_residuals = fixed_dynamics_window_variance(window, model_info)
        ratio, p_value = variance_ratio_two_sided_pvalue(sigma2_w, window_size, model_info)
        # Re-standardize by the REFERENCE sigma0^2 instead of this window's
        # own sigma2_w (see the comment in fixed_dynamics_window_variance) --
        # this is the quantity assumption (A2) actually predicts is N(0,1).
        h0_standardized_residuals_by_window.append(standardized_residuals * np.sqrt(ratio))
        raw_variance_w, raw_ratio, raw_p_value = raw_variance_window_pvalue(
            window, raw_ref_variance, raw_ref_df
        )

        window_start = window.index[0]
        window_end = window.index[-1]
        window_midpoint = window.index[len(window) // 2]
        # Plain integer position of the window within the test segment (0-based).
        # Used on the x-axis instead of the (synthetic, meaningless) timestamps.
        window_midpoint_index = i * window_size + len(window) // 2
        has_true_anomaly = bool(
            ((anomaly_times >= window_start) & (anomaly_times <= window_end)).any()
        )

        rows.append(
            {
                "window_start": window_start,
                "window_end": window_end,
                "window_midpoint": window_midpoint,
                "window_midpoint_index": window_midpoint_index,
                "n": len(window),
                "sigma2_w": sigma2_w,
                "variance_ratio": ratio,
                "p_value": p_value,
                "raw_variance_w": raw_variance_w,
                "raw_variance_ratio": raw_ratio,
                "raw_variance_p_value": raw_p_value,
                "predicted_anomaly": int(p_value < alpha),
                "actual_anomaly": int(has_true_anomaly),
            }
        )

    return pd.DataFrame(rows), h0_standardized_residuals_by_window


# ---------------------------------------------------------------------------
# Figure for the paper: 2 stacked panels, sized to be roughly square
# overall. Only the TEST segment is shown (the training segment belongs in
# the caption/text, not drawn). Panel labels are a bare "(a)"/"(b)" tag,
# nothing more -- descriptive text belongs in the figure caption, not
# printed on the plot. The x-axis is a plain sample count, not the
# simulation's synthetic timestamps (those numbers carry no real meaning),
# and its numeric ticks are hidden -- only the axis title "Time" remains,
# since only the relative flow of time matters here, not specific values.
#
# Styling choices, aimed at a plain "print figure in a journal" look
# rather than a dashboard: serif type (matches a LaTeX-typeset paper), no
# gridlines, thin axis lines, and a plain, print-safe categorical pair
# (blue / orange) with a neutral gray anomaly span instead of red/pink.
# ---------------------------------------------------------------------------

COLOR_PROPOSED = "#2a78d6"      # blue
COLOR_RAW_VARIANCE = "#eb6834"  # orange
COLOR_INK = "#0b0b0b"           # primary ink (series line, axis text)
COLOR_MUTED = "#6b6a65"         # reference line / spines / ticks
COLOR_ANOMALY_SPAN = "#c9c7c0"  # neutral gray highlight, not red/pink

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
    """Same print-friendly spine style as _style_axis, but keeps normal
    numeric x-ticks (histograms need them; the time-indexed panels don't).
    """
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(COLOR_MUTED)
    ax.spines["bottom"].set_color(COLOR_MUTED)
    ax.tick_params(colors=COLOR_MUTED)


def _style_axis(ax) -> None:
    """Shared, print-friendly axis style: no top/right box, no gridlines."""
    _style_hist_axis(ax)
    ax.set_xticks([])  # "Time" is a direction here, not a set of values to read off


def format_model_summary(model_info: dict) -> str:
    """One-line summary of every estimated reference-model parameter, for
    the figure title: ARMA order, each fixed parameter (const/ar/ma), and
    the reference innovation variance sigma0^2.
    """
    p, d, q = model_info["order"]
    param_parts = [
        f"{name}={value:.3f}"
        for name, value in zip(model_info["param_names"], model_info["fixed_params"])
    ]
    params_text = ", ".join(param_parts) if param_parts else "no AR/MA terms"
    return (
        f"Fitted reference model: ARMA(p,d,q)=({p},{d},{q})  |  "
        f"{params_text}  |  sigma0^2={model_info['reference_sigma2']:.3f}"
    )


def f_critical_values(window_size: int, reference_df: int, alpha: float) -> tuple[float, float]:
    """The two-sided decision boundary on R_w, drawn on panel (g): the test
    flags window w when R_w falls below F_{alpha/2}(n, n0) or above
    F_{1-alpha/2}(n, n0) -- exactly the p_w < alpha rule of Eq. 19, read on
    the ratio scale.
    """
    return (
        float(stats.f.ppf(alpha / 2, window_size, reference_df)),
        float(stats.f.ppf(1 - alpha / 2, window_size, reference_df)),
    )


def plot_assumption_checks(
    fig,
    gs,
    window_results: pd.DataFrame,
    h0_standardized_residuals_by_window: list[np.ndarray],
    model_info: dict,
    window_size: int,
    alpha: float,
) -> None:
    """Right-hand column: empirical vs. theoretical distribution checks for
    the three distributional claims Algorithm 1 depends on (paper Sec.
    4.4-4.5) -- built ONLY from windows that are truly normal (actual_anomaly
    == 0), since the anomalous windows are exactly where H0 is expected to
    fail and would bias the check:
      (e) v_t / sqrt(sigma0^2 F_t^*)  ~  N(0,1)          [assumption A2]
      (f) n * R_w = n * sigma_w_hat^2/sigma0^2  ~ chi^2_n [Eq. 15]
      (g) R_w  ~  F(n, n0)                                [Eq. 18]
    """
    normal_mask = (window_results["actual_anomaly"] == 0).to_numpy()
    normal_positions = np.where(normal_mask)[0]

    pooled_residuals = np.concatenate(
        [h0_standardized_residuals_by_window[i] for i in normal_positions]
    ) if len(normal_positions) else np.array([])

    chi2_stat = (window_results.loc[normal_mask, "n"] * window_results.loc[normal_mask, "variance_ratio"]).to_numpy()
    f_stat = window_results.loc[normal_mask, "variance_ratio"].to_numpy()

    hist_kwargs = dict(density=True, color=COLOR_PROPOSED, alpha=0.55, edgecolor="white", linewidth=0.4)

    def n_bins(sample) -> int:
        # Bin count scales with sample size (Rice rule, 2*N^(1/3)), clamped so
        # the shape stays readable whether there are 90 windows or 900.
        return int(np.clip(round(2.0 * len(sample) ** (1 / 3)), 20, 90))

    def draw_two_sided_cutoffs(ax, lower: float, upper: float, label: str) -> None:
        """Vertical lines at the two-sided alpha/2 critical values: anything
        outside this band is what the test flags as anomalous."""
        for i, cut in enumerate((lower, upper)):
            ax.axvline(
                cut, color=COLOR_RAW_VARIANCE, linewidth=1.1, linestyle="--",
                label=label if i == 0 else None,
            )

    # (e) standardized residuals vs. N(0,1)
    ax = fig.add_subplot(gs[0, 1])
    _style_hist_axis(ax)
    if len(pooled_residuals):
        ax.hist(pooled_residuals, bins=n_bins(pooled_residuals), **hist_kwargs)
        grid = np.linspace(*ax.get_xlim(), 400)
        ax.plot(grid, stats.norm.pdf(grid), color=COLOR_INK, linewidth=1.2, label="N(0,1)")
    ax.set_title("(e) Standardized Residuals", loc="left", fontsize=10.5, color=COLOR_INK)
    ax.set_xlabel(r"$v_t / \sqrt{\hat\sigma_0^2 F_t^*}$", fontsize=9)
    ax.legend(loc="upper right", fontsize=8, frameon=False)

    # (f) n * R_w vs. chi^2_n
    ax = fig.add_subplot(gs[1, 1])
    _style_hist_axis(ax)
    if len(chi2_stat):
        ax.hist(chi2_stat, bins=n_bins(chi2_stat), **hist_kwargs)
        grid = np.linspace(max(0.0, ax.get_xlim()[0]), ax.get_xlim()[1], 400)
        ax.plot(grid, stats.chi2.pdf(grid, df=window_size), color=COLOR_INK, linewidth=1.2, label=f"chi2({window_size})")
    ax.set_title("(f) Window Chi-Square Statistic", loc="left", fontsize=10.5, color=COLOR_INK)
    ax.set_xlabel(r"$n\,\hat\sigma_w^2/\hat\sigma_0^2$", fontsize=9)
    ax.legend(loc="upper right", fontsize=8, frameon=False)

    # (g) R_w vs. F(n, n0)
    ax = fig.add_subplot(gs[2, 1])
    _style_hist_axis(ax)
    if len(f_stat):
        ax.hist(f_stat, bins=n_bins(f_stat), **hist_kwargs)
        grid = np.linspace(max(0.0, ax.get_xlim()[0]), ax.get_xlim()[1], 400)
        ax.plot(
            grid, stats.f.pdf(grid, window_size, model_info["reference_df"]),
            color=COLOR_INK, linewidth=1.2, label=f"F({window_size},{model_info['reference_df']})",
        )
        draw_two_sided_cutoffs(
            ax, *f_critical_values(window_size, model_info["reference_df"], alpha),
            rf"two-sided $\alpha$={alpha}",
        )
    ax.set_title("(g) Variance Ratio (Normal Windows)", loc="left", fontsize=10.5, color=COLOR_INK)
    ax.set_xlabel(r"$R_w = \hat\sigma_w^2/\hat\sigma_0^2$", fontsize=9)
    ax.legend(loc="upper right", fontsize=8, frameon=False)

    # (h) does the empirical false-positive rate on normal windows match
    # the nominal alpha? -- a Clopper-Pearson exact CI puts the small
    # sample size (relatively few normal windows) in context: a rate that
    # looks high can still be pure binomial noise around alpha.
    ax = fig.add_subplot(gs[3, 1])
    ax.axis("off")
    n_normal = int(normal_mask.sum())
    n_false_positive = int((window_results.loc[normal_mask, "p_value"] < alpha).sum())
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
    test_segment: pd.Series,
    labels: pd.DataFrame,
    window_results: pd.DataFrame,
    h0_standardized_residuals_by_window: list[np.ndarray],
    model_info: dict,
    raw_variance_reference: tuple[float, int],
    window_size: int,
    alpha: float,
    output_path: Path,
) -> None:
    # Work in plain sample position (0, 1, 2, ...) instead of the
    # synthetic simulation timestamps.
    x_series = np.arange(len(test_segment))
    anomaly_flags = labels.loc[test_segment.index, "is_anomaly"].to_numpy()
    anomaly_positions = np.where(anomaly_flags == 1)[0]
    anomaly_start_pos = int(anomaly_positions.min()) if len(anomaly_positions) else None
    anomaly_end_pos = int(anomaly_positions.max()) if len(anomaly_positions) else None

    fig = plt.figure(figsize=(13, 10.5))
    gs = fig.add_gridspec(4, 2, width_ratios=(1.6, 1), hspace=0.55, wspace=0.32)

    plot_assumption_checks(
        fig, gs, window_results, h0_standardized_residuals_by_window, model_info, window_size, alpha
    )

    # (a) test-segment series only, no training data drawn. The series
    # itself needs no legend entry (the axis labels already say what it
    # is); only the shaded span needs one, since nothing else identifies it.
    ax = fig.add_subplot(gs[0, 0])
    _style_axis(ax)
    ax.plot(x_series, test_segment.to_numpy(), color=COLOR_INK, linewidth=0.9)
    if anomaly_start_pos is not None:
        ax.axvspan(
            anomaly_start_pos, anomaly_end_pos, color=COLOR_ANOMALY_SPAN, alpha=0.6,
            linewidth=0, label="Anomaly period",
        )
    ax.set_ylabel("Value")
    ax.set_title("(a) Simulated Time Series", loc="left", fontsize=10.5, color=COLOR_INK)
    ax.legend(loc="upper right", fontsize=9, frameon=True, facecolor="white", edgecolor="none")

    x = window_results["window_midpoint_index"]

    # (b) estimated innovation variance per window: proposed method's
    # sigma_w^2 vs. the raw-variance baseline's window sample variance,
    # each against its own training-segment reference level (unlabeled
    # dotted lines, same idea as the ratio=1 reference line in panel (c)).
    ax = fig.add_subplot(gs[1, 0])
    _style_axis(ax)
    if anomaly_start_pos is not None:
        ax.axvspan(anomaly_start_pos, anomaly_end_pos, color=COLOR_ANOMALY_SPAN, alpha=0.6, linewidth=0)
    ax.axhline(model_info["reference_sigma2"], color=COLOR_PROPOSED, linewidth=1.0, linestyle=":")
    ax.axhline(raw_variance_reference[0], color=COLOR_RAW_VARIANCE, linewidth=1.0, linestyle=":")

    for column, color, label in (
        ("sigma2_w", COLOR_PROPOSED, "Innovation variance"),
        ("raw_variance_w", COLOR_RAW_VARIANCE, "Raw variance"),
    ):
        ax.plot(x, window_results[column], color=color, marker="o", ms=4.5, linewidth=1.3, label=label)

    ax.set_ylabel("Variance")
    ax.set_title("(b) Estimated Innovation Variance", loc="left", fontsize=10.5, color=COLOR_INK)
    ax.legend(loc="upper right", fontsize=9, frameon=False)

    # (c) variance ratio: proposed method vs. the raw-variance baseline.
    ax = fig.add_subplot(gs[2, 0])
    _style_axis(ax)
    if anomaly_start_pos is not None:
        ax.axvspan(anomaly_start_pos, anomaly_end_pos, color=COLOR_ANOMALY_SPAN, alpha=0.6, linewidth=0)
    ax.axhline(1.0, color=COLOR_MUTED, linewidth=1.0, linestyle=":")

    for column, color, label in (
        ("variance_ratio", COLOR_PROPOSED, "Innovation variance ratio"),
        ("raw_variance_ratio", COLOR_RAW_VARIANCE, "Raw variance ratio"),
    ):
        ax.plot(x, window_results[column], color=color, marker="o", ms=4.5, linewidth=1.3, label=label)

    ax.set_ylabel("Variance Ratio")
    ax.set_title("(c) Variance Ratio", loc="left", fontsize=10.5, color=COLOR_INK)
    ax.legend(loc="upper right", fontsize=9, frameon=False)

    # (d) two-sided p-value for the PROPOSED method only. The raw-variance
    # ratio has no p-value here: its F-reference distribution assumes
    # i.i.d. data, which is false for this autocorrelated series, so that
    # number is not a valid statistic (see the earlier discussion) and is
    # left out rather than plotted as if it were one.
    ax = fig.add_subplot(gs[3, 0])
    _style_axis(ax)
    if anomaly_start_pos is not None:
        ax.axvspan(anomaly_start_pos, anomaly_end_pos, color=COLOR_ANOMALY_SPAN, alpha=0.6, linewidth=0)
    ax.axhline(alpha, color=COLOR_MUTED, linewidth=1.0, linestyle="--", label=f"alpha = {alpha}")

    p = window_results["p_value"]
    significant = p < alpha
    ax.plot(x, p, color=COLOR_PROPOSED, linewidth=1.3, zorder=2)
    # Significant (p < alpha): solid marker. Not significant: hollow marker.
    ax.scatter(x[significant], p[significant], s=30, color=COLOR_PROPOSED, zorder=3)
    ax.scatter(
        x[~significant], p[~significant], s=30, facecolors="white",
        edgecolors=COLOR_PROPOSED, linewidths=1.2, zorder=3,
    )

    ax.set_ylim(-0.02, 1.02)
    ax.set_ylabel("Two-Sided p-Value")
    ax.set_xlabel("Time")
    ax.set_title("(d) Statistical Significance", loc="left", fontsize=10.5, color=COLOR_INK)
    ax.legend(loc="upper right", fontsize=9, frameon=False)

    fig.suptitle(format_model_summary(model_info), fontsize=9, color=COLOR_MUTED, y=0.995)
    with warnings.catch_warnings():
        # panel (h) is a bare text axis (axis("off")), which tight_layout
        # cannot size itself against -- harmless, the other 7 axes still
        # drive the layout correctly.
        warnings.simplefilter("ignore", UserWarning)
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # On Windows, something (an editor preview, antivirus scan) sometimes
    # holds the PNG open for an instant right when we try to overwrite it.
    # A couple of short retries clears this without failing the whole run.
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/simulation/simulated_with_anomaly.csv")
    parser.add_argument("--labels", default="data/simulation/simulated_anomaly_labels.csv")
    parser.add_argument("--output-dir", default="results/paper_algorithm_simulation")
    parser.add_argument("--max-p", type=int, default=3)
    parser.add_argument("--max-q", type=int, default=3)
    parser.add_argument("--window-size", type=int, default=100)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument(
        "--train-fraction", type=float, default=0.4,
        help="Fraction of the series used for training; the rest is the test segment.",
    )
    args = parser.parse_args()

    series, labels = load_simulation_data(Path(args.input), Path(args.labels))
    train_segment, test_segment = split_train_test(series, labels, args.train_fraction)
    print(f"Training segment: {len(train_segment)} points, test segment: {len(test_segment)} points")

    model_info = select_and_fit_reference_model(train_segment, args.max_p, args.max_q)
    print(f"Selected order (p, d, q) = {model_info['order']}, AIC = {model_info['aic']:.2f}")
    print(f"Reference sigma0^2 = {model_info['reference_sigma2']:.4f}")

    raw_variance_reference = compute_raw_variance_reference(train_segment)
    print(f"Raw-variance reference = {raw_variance_reference[0]:.4f}")

    window_results, h0_standardized_residuals_by_window = score_test_windows(
        test_segment, labels, model_info, raw_variance_reference, args.window_size, args.alpha
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    window_results.to_csv(output_dir / "window_results.csv", index=False)

    model_record = {
        "order": model_info["order"],
        "param_names": model_info["param_names"],
        "fixed_params": model_info["fixed_params"].tolist(),
        "reference_sigma2": model_info["reference_sigma2"],
        "reference_df": model_info["reference_df"],
        "aic": model_info["aic"],
        "window_size": args.window_size,
        "alpha": args.alpha,
    }
    (output_dir / "fixed_model.json").write_text(json.dumps(model_record, indent=2), encoding="utf-8")

    n_true_anomaly_windows = int(window_results["actual_anomaly"].sum())
    n_normal_windows = int((window_results["actual_anomaly"] == 0).sum())
    n_detected_true = int(
        ((window_results["actual_anomaly"] == 1) & (window_results["p_value"] < args.alpha)).sum()
    )
    n_false_alarms = int(
        ((window_results["actual_anomaly"] == 0) & (window_results["p_value"] < args.alpha)).sum()
    )
    n_raw_detected_true = int(
        ((window_results["actual_anomaly"] == 1) & (window_results["raw_variance_p_value"] < args.alpha)).sum()
    )
    n_raw_false_alarms = int(
        ((window_results["actual_anomaly"] == 0) & (window_results["raw_variance_p_value"] < args.alpha)).sum()
    )

    # Detection counts are printed here (for you to put in the figure
    # caption) rather than drawn onto the figure itself.
    plot_summary_figure(
        test_segment, labels, window_results, h0_standardized_residuals_by_window,
        model_info, raw_variance_reference, args.window_size, args.alpha,
        output_dir / "figures" / "summary.png",
    )

    print(f"Model: ARMA{model_info['order']}, alpha = {args.alpha}")
    print(f"Test windows: {len(window_results)} total, {n_true_anomaly_windows} overlap the true anomaly")
    print(f"Proposed method: detected {n_detected_true}/{n_true_anomaly_windows}, false alarms {n_false_alarms}/{n_normal_windows}")
    print(f"Raw variance:    detected {n_raw_detected_true}/{n_true_anomaly_windows}, false alarms {n_raw_false_alarms}/{n_normal_windows}")
    print(f"Results saved under {output_dir}")


if __name__ == "__main__":
    main()
