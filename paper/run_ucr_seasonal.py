"""Experiment: does adding a seasonal component (and/or letting AIC choose
the differencing order d, instead of algorithm.py's fixed d=0) fix the high
false-alarm rate the plain algorithm shows on periodic UCR series (ECG,
temperature, respiration)? For each dataset, runs BOTH:
  - "plain": ARIMA(p,d,q), d now searched by AIC too (0..max_d)
  - "seasonal": SARIMA(p,d,q)x(1,0,1,s), same d search, where the seasonal
    period s is estimated automatically from the training segment's
    dominant FFT frequency, and the seasonal AR/MA order is fixed at
    (1,0,1) rather than grid-searched (keeps runtime manageable -- this is
    an experiment, not the paper's main method).

This is intentionally a separate, self-contained script -- NOT a change to
algorithm.py, which is the paper's already-described core method. Nothing
here is meant to replace run_ucr.py; it's a side-by-side comparison to see
whether the seasonal extension is worth pursuing further.

Outputs, under --output-dir (default results/ucr_seasonal/<category>):
  - comparison_table.csv  one row per dataset, plain vs. seasonal side by
                          side: order, detection rate, false-alarm rate
"""

from __future__ import annotations

import argparse
import re
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.arima.model import ARIMA

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from algorithm import variance_ratio_pvalue

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


def _style_axis(ax, hide_xticks: bool = False) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(COLOR_MUTED)
    ax.spines["bottom"].set_color(COLOR_MUTED)
    ax.tick_params(colors=COLOR_MUTED)
    if hide_xticks:
        ax.set_xticks([])


def f_critical_values(window_size: int, reference_df: int, alpha: float) -> tuple[float, float]:
    return (
        float(stats.f.ppf(alpha / 2, window_size, reference_df)),
        float(stats.f.ppf(1 - alpha / 2, window_size, reference_df)),
    )


def find_train_end(raw_dir: Path, dataset_folder_name: str) -> int:
    """Recover train_end from the original raw UCR filename (see run_ucr.py)."""
    m = re.match(r"ucr_(\d+)_", dataset_folder_name)
    if not m:
        raise ValueError(f"Could not parse an index from {dataset_folder_name!r}.")
    idx = m.group(1)
    matches = list(raw_dir.glob(f"{idx}_UCR_Anomaly_*.txt"))
    if not matches:
        raise FileNotFoundError(f"No raw file found for index {idx} in {raw_dir}.")
    m2 = re.search(r"_(\d+)_(\d+)_(\d+)\.txt$", matches[0].name)
    if not m2:
        raise ValueError(f"Could not parse train_end from {matches[0].name!r}.")
    return int(m2.group(1))


def estimate_period(train_values: np.ndarray, min_period: int = 4, max_period: int | None = None) -> int | None:
    """Dominant period in the training segment via FFT peak power.

    Args:
        train_values: anomaly-free training segment.
        min_period: shortest period considered (avoids spurious high-frequency noise).
        max_period: longest period considered (defaults to n/4, so at least
            4 full cycles fit in the training segment).

    Returns:
        Estimated period in samples, or None if no meaningful peak is found.
    """
    n = len(train_values)
    if max_period is None:
        max_period = n // 4
    detrended = train_values - train_values.mean()
    freqs = np.fft.rfftfreq(n)
    power = np.abs(np.fft.rfft(detrended)) ** 2

    with np.errstate(divide="ignore"):
        periods = np.where(freqs > 0, 1.0 / np.where(freqs > 0, freqs, 1), np.inf)
    valid = (periods >= min_period) & (periods <= max_period)
    if not valid.any():
        return None
    best = np.argmax(power[valid])
    return int(round(periods[valid][best]))


def fit_reference_model_flexible(
    train_values: np.ndarray, max_p: int, max_d: int, max_q: int, seasonal_period: int | None = None
) -> dict:
    """Same idea as algorithm.fit_reference_model, but the AIC search now
    also covers d (0..max_d) -- algorithm.py fixes d=0 per the paper's
    stationarity assumption (A1); this is the experiment checking whether
    letting AIC choose d changes anything on real (not guaranteed-
    stationary) UCR data. Optionally also attaches a fixed seasonal_order=
    (1, 0, 1, seasonal_period) term (seasonal_period=None -> no seasonal
    component, i.e. the "plain, but with d searched" variant).
    """
    seasonal_order = (1, 0, 1, seasonal_period) if seasonal_period else None
    best = None
    for p in range(max_p + 1):
        for d in range(max_d + 1):
            for q in range(max_q + 1):
                if p == 0 and q == 0:
                    continue  # skip the trivial "no dynamics" model (any d)
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", ConvergenceWarning)
                        warnings.simplefilter("ignore", UserWarning)
                        kwargs = {"order": (p, d, q)}
                        if seasonal_order is not None:
                            kwargs["seasonal_order"] = seasonal_order
                        fit = ARIMA(train_values, **kwargs).fit()
                except Exception:
                    continue
                if not np.isfinite(fit.aic):
                    continue
                if best is None or fit.aic < best["aic"]:
                    param_names = list(fit.param_names)
                    params = np.asarray(fit.params, dtype=float)
                    sigma_index = param_names.index("sigma2")
                    best = {
                        "order": (p, d, q),
                        "seasonal_order": seasonal_order,
                        "param_names": [name for name in param_names if name != "sigma2"],
                        "fixed_params": np.delete(params, sigma_index),
                        "reference_sigma2": float(params[sigma_index]),
                        "reference_df": len(train_values) - (len(param_names) - 1),
                        "aic": float(fit.aic),
                    }
    if best is None:
        raise RuntimeError("No ARIMA model could be fit on the training segment.")
    return best


def score_window_flexible(window_values: np.ndarray, model: dict) -> tuple[float, int]:
    """Same fixed-dynamics concentrated-scale trick as algorithm.score_window.
    Unlike algorithm.py (which can assume d=0, so burn-in is always 0), d>0
    is possible here, so burn-in from differencing is tracked explicitly and
    n_used (window_size - burn_in) is returned alongside sigma2_w.
    """
    kwargs = {"order": model["order"], "concentrate_scale": True}
    if model.get("seasonal_order") is not None:
        kwargs["seasonal_order"] = model["seasonal_order"]
    window_model = ARIMA(window_values, **kwargs)
    if list(window_model.param_names) != model["param_names"]:
        raise RuntimeError("Parameter mismatch between reference model and window model.")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = window_model.filter(model["fixed_params"])
    sigma2_w = float(result.scale)
    burn_in = int(result.filter_results.loglikelihood_burn)
    n_used = max(1, len(window_values) - burn_in)
    return sigma2_w, n_used


def run_flexible_algorithm(
    train_values: np.ndarray, test_values: np.ndarray, window_size: int, alpha: float,
    max_p: int, max_d: int, max_q: int, seasonal_period: int | None = None,
) -> tuple[pd.DataFrame, dict]:
    model = fit_reference_model_flexible(train_values, max_p, max_d, max_q, seasonal_period)
    n_windows = len(test_values) // window_size
    rows = []
    for i in range(n_windows):
        window = test_values[i * window_size : (i + 1) * window_size]
        sigma2_w, n_used = score_window_flexible(window, model)
        ratio, p_value = variance_ratio_pvalue(sigma2_w, n_used, model)
        rows.append(
            {
                "window_index": i, "window_start": i * window_size, "window_end": i * window_size + window_size,
                "sigma2_w": sigma2_w, "n_used": n_used, "variance_ratio": ratio, "p_value": p_value,
                "is_anomalous": int(p_value < alpha),
            }
        )
    return pd.DataFrame(rows), model


def evaluate(results: pd.DataFrame, test_is_anomaly: np.ndarray, alpha: float) -> dict:
    actual_anomaly = []
    for _, row in results.iterrows():
        start, end = int(row["window_start"]), int(row["window_end"])
        actual_anomaly.append(int(test_is_anomaly[start:end].any()))
    results = results.copy()
    results["actual_anomaly"] = actual_anomaly
    n_true = int((results["actual_anomaly"] == 1).sum())
    n_normal = int((results["actual_anomaly"] == 0).sum())
    n_detected = int(((results["actual_anomaly"] == 1) & (results["p_value"] < alpha)).sum())
    n_false_alarms = int(((results["actual_anomaly"] == 0) & (results["p_value"] < alpha)).sum())
    return {
        "n_true_anomaly_windows": n_true, "n_normal_windows": n_normal,
        "n_detected": n_detected, "n_false_alarms": n_false_alarms,
        "detection_rate": n_detected / n_true if n_true else float("nan"),
        "false_alarm_rate": n_false_alarms / n_normal if n_normal else float("nan"),
    }


def _plot_ratio_panel(ax, results: pd.DataFrame, model: dict, window_size: int, alpha: float, title: str, anomaly_span: tuple[int, int] | None) -> None:
    _style_axis(ax, hide_xticks=True)
    if anomaly_span is not None:
        ax.axvspan(anomaly_span[0], anomaly_span[1], color=COLOR_ANOMALY_SPAN, alpha=0.6, linewidth=0)
    ax.axhline(1.0, color=COLOR_MUTED, linewidth=1.0, linestyle=":")
    # Use n_used (window_size minus any differencing burn-in), not the raw
    # window_size -- with d>0 now possible, those can differ.
    n_used = int(results["n_used"].iloc[0]) if "n_used" in results else window_size
    lower, upper = f_critical_values(n_used, model["reference_df"], alpha)
    ax.axhline(lower, color=COLOR_LIMIT, linewidth=1.1, linestyle="--")
    ax.axhline(upper, color=COLOR_LIMIT, linewidth=1.1, linestyle="--")
    x = results["window_start"] + window_size // 2
    significant = results["p_value"] < alpha
    ratio = results["variance_ratio"]
    ax.plot(x, ratio, color=COLOR_LINE, linewidth=1.3, zorder=3)
    ax.scatter(x[significant], ratio[significant], s=18, color=COLOR_LINE, zorder=4)
    ax.scatter(x[~significant], ratio[~significant], s=18, facecolors="white", edgecolors=COLOR_LINE, linewidths=1.0, zorder=4)
    ax.set_ylabel("Variance Ratio")
    ax.set_title(title, loc="left", fontsize=10.5, color=COLOR_INK)


def plot_comparison_figure(
    test_values: np.ndarray, test_is_anomaly: np.ndarray, plain_results: pd.DataFrame, plain_model: dict,
    seasonal_results: pd.DataFrame, seasonal_model: dict, window_size: int, alpha: float, period: int,
    dataset_name: str, output_path: Path,
) -> None:
    """Three panels: (a) raw series with the true anomaly shaded, (b) plain
    algorithm's innovation variance ratio, (c) seasonal variant's ratio --
    same data, same control-limit convention, side by side for comparison.
    """
    anomaly_positions = np.where(test_is_anomaly == 1)[0]
    anomaly_span = (int(anomaly_positions.min()), int(anomaly_positions.max())) if len(anomaly_positions) else None

    fig, axes = plt.subplots(3, 1, figsize=(8, 6.5), height_ratios=(1, 1, 1))

    ax = axes[0]
    _style_axis(ax, hide_xticks=True)
    ax.plot(np.arange(len(test_values)), test_values, color=COLOR_INK, linewidth=0.8)
    if anomaly_span is not None:
        ax.axvspan(*anomaly_span, color=COLOR_ANOMALY_SPAN, alpha=0.6, linewidth=0, label="Anomaly period")
    y_min, y_max = ax.get_ylim()
    ax.set_ylim(y_min, y_max + 0.18 * (y_max - y_min))
    ax.set_ylabel("Value")
    ax.set_title(f"(a) {dataset_name}", loc="left", fontsize=10.5, color=COLOR_INK)
    ax.legend(loc="upper right", fontsize=9, frameon=True, facecolor="white", edgecolor="none")

    _plot_ratio_panel(axes[1], plain_results, plain_model, window_size, alpha, f"(b) Plain: order={plain_model['order']}", anomaly_span)
    _plot_ratio_panel(
        axes[2], seasonal_results, seasonal_model, window_size, alpha,
        f"(c) Seasonal (period={period}): order={seasonal_model['order']}x{seasonal_model['seasonal_order']}",
        anomaly_span,
    )
    axes[2].set_xlabel("Time")

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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--category", default="noise")
    parser.add_argument("--data-dir", default="data/ucr_prepared")
    parser.add_argument("--raw-dir", default="data/ucr_raw")
    parser.add_argument("--window-size", type=int, default=100)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--max-p", type=int, default=2, help="Kept small (vs. 3 in run_ucr.py) since seasonal fits are much slower.")
    parser.add_argument("--max-q", type=int, default=2)
    parser.add_argument("--max-d", type=int, default=1, help="AIC now also searches d = 0..max_d (algorithm.py fixes d=0).")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dataset", default=None, help="Run only this one dataset folder name (skips --limit).")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    data_dir = Path(args.data_dir) / args.category
    raw_dir = Path(args.raw_dir)
    output_dir = Path(args.output_dir) if args.output_dir else Path("results/ucr_seasonal") / args.category
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.dataset:
        dataset_dirs = [data_dir / args.dataset]
    else:
        dataset_dirs = sorted(p for p in data_dir.iterdir() if p.is_dir())
        if args.limit is not None:
            dataset_dirs = dataset_dirs[: args.limit]

    rows = []
    for dataset_dir in dataset_dirs:
        name = dataset_dir.name
        print(f"--- {name} ---")
        row = {"dataset": name}
        try:
            train_end = find_train_end(raw_dir, name)
            ts = pd.read_csv(dataset_dir / "timeseries.csv")
            labels = pd.read_csv(dataset_dir / "labels.csv")
            values = ts["value"].to_numpy(dtype=float)
            is_anomaly = labels["is_anomaly"].to_numpy(dtype=int)

            if is_anomaly[:train_end].sum() > 0:
                row["status"] = "skipped_dirty_training"
                rows.append(row)
                continue

            train_values = values[:train_end]
            test_values = values[train_end:]
            test_is_anomaly = is_anomaly[train_end:]

            # --- plain, but with d now searched by AIC too (0..max_d) ---
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                plain_results, plain_model = run_flexible_algorithm(
                    train_values, test_values, args.window_size, args.alpha,
                    max_p=3, max_d=args.max_d, max_q=3, seasonal_period=None,
                )
            plain_eval = evaluate(plain_results, test_is_anomaly, args.alpha)
            row["order_plain"] = str(plain_model["order"])
            row["false_alarm_rate_plain"] = plain_eval["false_alarm_rate"]
            row["detection_rate_plain"] = plain_eval["detection_rate"]

            # --- seasonal variant, d also searched ---
            period = estimate_period(train_values, max_period=min(len(train_values) // 4, 2000))
            row["estimated_period"] = period
            if period is None or period < 2 or period > len(train_values) // 3:
                row["status"] = "no_usable_period_found"
                rows.append(row)
                print(f"  plain (d={plain_model['order'][1]}): false_alarm_rate={plain_eval['false_alarm_rate']:.3f}; seasonal: skipped (no usable period)")
                continue

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                seasonal_results, seasonal_model = run_flexible_algorithm(
                    train_values, test_values, args.window_size, args.alpha,
                    max_p=args.max_p, max_d=args.max_d, max_q=args.max_q, seasonal_period=period,
                )
            seasonal_eval = evaluate(seasonal_results, test_is_anomaly, args.alpha)
            row["order_seasonal"] = str(seasonal_model["order"])
            row["seasonal_order"] = str(seasonal_model["seasonal_order"])
            row["false_alarm_rate_seasonal"] = seasonal_eval["false_alarm_rate"]
            row["detection_rate_seasonal"] = seasonal_eval["detection_rate"]
            row["status"] = "ok"
            print(
                f"  plain (d={plain_model['order'][1]}): false_alarm_rate={plain_eval['false_alarm_rate']:.3f}  |  "
                f"seasonal (period={period}, d={seasonal_model['order'][1]}): false_alarm_rate={seasonal_eval['false_alarm_rate']:.3f}"
            )
            plot_comparison_figure(
                test_values, test_is_anomaly, plain_results, plain_model,
                seasonal_results, seasonal_model, args.window_size, args.alpha, period,
                name, output_dir / "figures" / f"{name}.png",
            )
        except Exception as exc:
            row["status"] = f"failed: {exc}"
            print(f"  failed: {exc}")
        rows.append(row)

    table = pd.DataFrame(rows)
    for attempt in range(3):
        try:
            table.to_csv(output_dir / "comparison_table.csv", index=False)
            break
        except PermissionError:
            if attempt == 2:
                raise
            time.sleep(0.5)
    print(f"Results saved under {output_dir}")


if __name__ == "__main__":
    main()
