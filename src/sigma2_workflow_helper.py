"""Simple shared helpers for sigma2 detection, evaluation, and plotting."""

from __future__ import annotations

import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from tqdm import tqdm

os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".matplotlib"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.axes import Axes


def read_timeseries(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index.name = "time"
    return df.apply(pd.to_numeric, errors="coerce")


def read_labels(path: Path | None) -> pd.DataFrame | None:
    if path is None or not path.exists():
        return None
    labels = pd.read_csv(path, index_col=0, parse_dates=True)
    labels.index.name = "time"
    return labels


def parse_order(values: list[int]) -> tuple[int, int, int]:
    if len(values) != 3:
        raise ValueError("ARIMA order must be three integers: p d q.")
    return tuple(values)


def estimate_period_autocorr(
    series: pd.Series,
    min_period: int = 2,
    max_period: int | None = None,
    min_autocorr: float = 0.3,
) -> int | None:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if len(clean) < max(8, min_period * 4):
        return None

    values = clean.to_numpy(dtype=float)
    values = values - np.mean(values)
    if not np.isfinite(values).all() or np.std(values) == 0:
        return None

    if max_period is None:
        max_period = min(len(values) // 4, 1000)
    max_period = max(min_period, int(max_period))
    if max_period <= min_period:
        return None

    autocorr = sm.tsa.stattools.acf(values, nlags=max_period, fft=True)
    candidate_lags = [
        lag
        for lag in range(min_period + 1, max_period)
        if autocorr[lag] > autocorr[lag - 1] and autocorr[lag] >= autocorr[lag + 1]
    ]
    if not candidate_lags:
        candidate_lags = list(range(min_period, max_period + 1))

    best_lag = max(candidate_lags, key=lambda lag: autocorr[lag])
    if not np.isfinite(autocorr[best_lag]) or autocorr[best_lag] < min_autocorr:
        return None
    return int(best_lag)


def estimate_period_fft(
    series: pd.Series,
    min_period: int = 2,
    max_period: int | None = None,
    min_power_ratio: float = 3.0,
) -> int | None:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if len(clean) < max(8, min_period * 4):
        return None

    values = clean.to_numpy(dtype=float)
    values = values - np.mean(values)
    if not np.isfinite(values).all() or np.std(values) == 0:
        return None

    if max_period is None:
        max_period = min(len(values) // 4, 1000)
    max_period = max(min_period, int(max_period))
    if max_period <= min_period:
        return None

    frequencies = np.fft.rfftfreq(len(values), d=1.0)
    powers = np.abs(np.fft.rfft(values)) ** 2

    positive = frequencies > 0
    if not positive.any():
        return None

    candidate_frequencies = frequencies[positive]
    candidate_powers = powers[positive]
    candidate_periods = 1.0 / candidate_frequencies

    valid = (candidate_periods >= min_period) & (candidate_periods <= max_period)
    if not valid.any():
        return None

    candidate_periods = candidate_periods[valid]
    candidate_powers = candidate_powers[valid]
    best_idx = int(np.argmax(candidate_powers))
    best_power = float(candidate_powers[best_idx])
    baseline_power = float(np.median(candidate_powers))
    if best_power <= 0:
        return None
    if baseline_power > 0 and best_power / baseline_power < min_power_ratio:
        return None

    best_period = int(round(candidate_periods[best_idx]))
    if best_period < min_period or best_period > max_period:
        return None
    return best_period


def estimate_period(
    series: pd.Series,
    method: str,
) -> int | None:
    if method == "autocorr":
        return estimate_period_autocorr(series)
    if method == "fft":
        return estimate_period_fft(series)
    raise ValueError(f"Unknown period method: {method}")


MIN_WINDOW_SIZE = 50


def preprocess_stationary_series(
    df: pd.DataFrame,
    period_method: str,
    fallback_window_size: int,
    fallback_stride: int,
    preprocessing_mode: str = "seasonal_diff",
    min_window_size: int = MIN_WINDOW_SIZE,
) -> tuple[pd.DataFrame, dict[str, str | int | bool | None]]:
    if len(df.columns) != 1:
        raise ValueError("Input data must contain exactly one value column.")

    value_name = df.columns[0]
    series = df.iloc[:, 0].astype(float)
    adjusted_series = series.copy()
    estimated_period = None
    seasonal_adjustment = False
    effective_period_method = "none"
    window_size = int(fallback_window_size)
    stride = int(fallback_stride)

    if preprocessing_mode == "raw":
        pass
    elif preprocessing_mode == "seasonal_diff":
        estimated_period = estimate_period(series, period_method)
        effective_period_method = period_method
        if estimated_period is not None:
            adjusted_series = series - series.shift(estimated_period)
            seasonal_adjustment = True
            window_size = max(int(min_window_size), 2 * int(estimated_period))
            stride = max(1, int(round(0.25 * window_size)))
    else:
        raise ValueError(f"Unknown preprocessing mode: {preprocessing_mode}")

    processed_df = pd.DataFrame({value_name: adjusted_series}, index=df.index)
    settings: dict[str, str | int | bool | None] = {
        "preprocessing_mode": preprocessing_mode,
        "period_method": effective_period_method,
        "estimated_period": estimated_period,
        "seasonal_adjustment": seasonal_adjustment,
        "window_size": window_size,
        "stride": stride,
    }
    return processed_df, settings


def create_trailing_windows(
    series: pd.Series,
    window_size: int,
    stride: int,
) -> list[pd.Series]:
    if stride <= 0:
        raise ValueError("stride must be positive.")

    clean = series.dropna()
    if len(clean) < window_size:
        raise ValueError("window_size is larger than the time series.")

    windows = []
    end_positions = list(range(window_size - 1, len(clean), stride))
    final_end = len(clean) - 1
    if end_positions[-1] != final_end:
        end_positions.append(final_end)

    for end in end_positions:
        start = end - window_size + 1
        windows.append(clean.iloc[start : end + 1])
    return windows


def fit_arima_window(
    window_values: np.ndarray,
    order: tuple[int, int, int],
) -> tuple[float, np.ndarray]:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            warnings.simplefilter("ignore", UserWarning)
            model = sm.tsa.ARIMA(window_values, order=order)
            result = model.fit()

        params = result.params
        if hasattr(params, "index") and "sigma2" in params.index:
            sigma2 = float(params.loc["sigma2"])
        else:
            sigma2 = float(params[-1])

        predicted_values = np.asarray(result.fittedvalues, dtype=float)
        if len(predicted_values) != len(window_values):
            padded = np.full(len(window_values), np.nan, dtype=float)
            padded[-len(predicted_values) :] = predicted_values
            predicted_values = padded

        residuals = np.asarray(window_values, dtype=float) - predicted_values
        return sigma2, residuals
    except Exception:
        return np.nan, np.full(len(window_values), np.nan, dtype=float)


def compute_sigma2(
    df: pd.DataFrame,
    window_size: int,
    stride: int,
    order: tuple[int, int, int],
) -> tuple[pd.DataFrame, pd.Series]:
    if len(df.columns) != 1:
        raise ValueError("Input data must contain exactly one value column.")

    series = df.iloc[:, 0]
    windows = create_trailing_windows(series, window_size, stride)

    sigma2_values = []
    window_starts = []
    window_ends = []
    residual_sum = np.zeros(len(series), dtype=float)
    residual_count = np.zeros(len(series), dtype=float)

    for window in tqdm(windows, desc="Fitting ARIMA windows"):
        sigma2, residuals = fit_arima_window(window.to_numpy(), order)
        sigma2_values.append(sigma2)
        window_starts.append(window.index[0])
        window_ends.append(window.index[-1])

        positions = series.index.get_indexer(window.index)
        valid = np.isfinite(residuals)
        residual_sum[positions[valid]] += residuals[valid]
        residual_count[positions[valid]] += 1

    with np.errstate(invalid="ignore", divide="ignore"):
        residual_values = residual_sum / residual_count

    residual_values[residual_count == 0] = np.nan
    residual_series = pd.Series(residual_values, index=df.index, name="residual")

    sigma2_df = pd.DataFrame(
        {
            "window_start": window_starts,
            "window_end": window_ends,
            "sigma2": sigma2_values,
        },
        index=pd.Index(window_ends, name="time"),
    )
    return sigma2_df, residual_series


def labels_by_window_overlap(sigma2_df: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    anomaly_times = labels.index[labels["is_anomaly"] == 1]
    if len(anomaly_times) == 0:
        aligned = pd.DataFrame({"is_anomaly": 0}, index=sigma2_df.index)
    else:
        overlaps = []
        for window_start, window_end in zip(sigma2_df["window_start"], sigma2_df["window_end"]):
            has_anomaly = ((anomaly_times >= window_start) & (anomaly_times <= window_end)).any()
            overlaps.append(int(has_anomaly))
        aligned = pd.DataFrame({"is_anomaly": overlaps}, index=sigma2_df.index)

    if "is_train" in labels.columns:
        train_times = labels.index[labels["is_train"] == 1]
        if len(train_times) > 0:
            train_end = train_times.max()
            aligned["is_train"] = (sigma2_df["window_end"] <= train_end).astype(int)

    return aligned


def build_window_results(
    sigma2_df: pd.DataFrame,
    labels: pd.DataFrame | None,
) -> pd.DataFrame:
    results = pd.DataFrame(index=sigma2_df.index)
    results["window_start"] = sigma2_df["window_start"]
    results["window_end"] = sigma2_df["window_end"]
    results["sigma2"] = sigma2_df["sigma2"]
    results["actual_anomaly"] = 0
    results["predicted_anomaly"] = 0

    if labels is not None and "is_anomaly" in labels.columns:
        aligned_labels = labels_by_window_overlap(sigma2_df, labels)
        results["actual_anomaly"] = aligned_labels["is_anomaly"].astype(int)

    valid_scores = results["sigma2"].dropna()
    if not valid_scores.empty:
        results.loc[valid_scores.idxmax(), "predicted_anomaly"] = 1

    return results


def window_midpoint_time(window_start: pd.Timestamp, window_end: pd.Timestamp) -> pd.Timestamp:
    midpoint_seconds = int((window_end - window_start).total_seconds() // 2)
    return window_start + pd.to_timedelta(midpoint_seconds, unit="s")


def compute_ucr_metric(
    window_results: pd.DataFrame,
    anomaly_start: int,
    anomaly_end: int,
    reference_start_time: pd.Timestamp | None = None,
) -> dict[str, float | int]:
    scores = window_results["sigma2"].dropna()
    if scores.empty:
        return {
            "ucr_prediction_index": np.nan,
            "ucr_prediction_score": np.nan,
            "ucr_correct_start": np.nan,
            "ucr_correct_end": np.nan,
            "ucr_correct": 0,
        }

    best_label = scores.idxmax()
    best_row = window_results.loc[best_label]
    best_score = float(best_row["sigma2"])

    window_start = pd.Timestamp(best_row["window_start"])
    window_end = pd.Timestamp(best_row["window_end"])
    prediction_time = window_midpoint_time(window_start, window_end)
    if reference_start_time is None:
        reference_start = pd.Timestamp(window_results["window_start"].iloc[0])
    else:
        reference_start = pd.Timestamp(reference_start_time)
    prediction_index = int((prediction_time - reference_start).total_seconds())

    correct_start = anomaly_start - 100
    correct_end = anomaly_end + 100
    is_correct = int(correct_start <= prediction_index <= correct_end)

    return {
        "ucr_prediction_index": prediction_index,
        "ucr_prediction_score": best_score,
        "ucr_correct_start": correct_start,
        "ucr_correct_end": correct_end,
        "ucr_correct": is_correct,
    }


def anomaly_interval(
    labels: pd.DataFrame,
) -> tuple[pd.Timestamp, pd.Timestamp] | tuple[None, None]:
    anomaly_times = labels.index[labels["is_anomaly"] == 1]
    if len(anomaly_times) == 0:
        return None, None
    return anomaly_times.min(), anomaly_times.max()


def zoom_interval(
    index: pd.Index,
    start_time: pd.Timestamp | None,
    end_time: pd.Timestamp | None,
) -> tuple[pd.Timestamp, pd.Timestamp] | tuple[None, None]:
    if start_time is None or end_time is None or len(index) == 0:
        return None, None

    padding_points = 500
    start_pos = int(index.searchsorted(start_time, side="left"))
    end_pos = int(index.searchsorted(end_time, side="right")) - 1

    zoom_start_pos = max(0, start_pos - padding_points)
    zoom_end_pos = min(len(index) - 1, end_pos + padding_points)

    return pd.Timestamp(index[zoom_start_pos]), pd.Timestamp(index[zoom_end_pos])


def add_anomaly_span(
    ax: Axes,
    start_time: pd.Timestamp | None,
    end_time: pd.Timestamp | None,
) -> None:
    if start_time is not None and end_time is not None:
        ax.axvspan(start_time, end_time, color="red", alpha=0.14, label="anomaly")


def draw_row_legend(legend_ax: Axes, source_axes: Axes | list[Axes]) -> None:
    legend_ax.set_xticks([])
    legend_ax.set_yticks([])
    for spine in legend_ax.spines.values():
        spine.set_visible(False)

    axes = source_axes if isinstance(source_axes, list) else [source_axes]
    handles = []
    labels = []
    seen = set()
    for source_ax in axes:
        axis_handles, axis_labels = source_ax.get_legend_handles_labels()
        for handle, label in zip(axis_handles, axis_labels):
            if label in seen:
                continue
            seen.add(label)
            handles.append(handle)
            labels.append(label)

    if handles:
        legend_ax.legend(
            handles,
            labels,
            loc="center left",
            frameon=True,
            fontsize=9,
        )


def format_time_axis(ax: Axes) -> None:
    locator = mdates.AutoDateLocator(minticks=4, maxticks=7)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))


def style_group_axis(ax: Axes, text: str, facecolor: str = "#f2f4f7") -> None:
    ax.set_facecolor(facecolor)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.text(
        0.5,
        0.5,
        text,
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
        color="#1f2933",
        wrap=True,
    )


def plot_sigma2_scores(ax: Axes, predictions: pd.DataFrame) -> None:
    sigma2_scores = predictions["sigma2"].dropna()
    if sigma2_scores.empty:
        return

    ax.plot(
        sigma2_scores.index,
        sigma2_scores,
        marker="o",
        markersize=3,
        linewidth=0.8,
        color="black",
        label="sigma2",
    )


def plot_sigma2(
    sigma2_df: pd.DataFrame,
    labels: pd.DataFrame | None,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(
        sigma2_df.index,
        sigma2_df["sigma2"],
        marker="o",
        markersize=3,
        linewidth=0.8,
        color="black",
        label="sigma2",
    )

    if labels is not None and "is_anomaly" in labels.columns:
        anomaly_times = labels.index[labels["is_anomaly"] == 1]
        if len(anomaly_times) > 0:
            ax.axvspan(anomaly_times.min(), anomaly_times.max(), color="red", alpha=0.15, label="anomaly")

    ax.set_title("Sigma2 Values From ARIMA Windows")
    ax.set_xlabel("Time")
    ax.set_ylabel("sigma2")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="upper right")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_combined(
    raw_df: pd.DataFrame,
    residual_series: pd.Series,
    predictions: pd.DataFrame,
    labels: pd.DataFrame,
    title: str,
    output_path: Path,
    transformed_df: pd.DataFrame | None = None,
    raw_residual_series: pd.Series | None = None,
    raw_predictions: pd.DataFrame | None = None,
) -> None:
    start_time, end_time = anomaly_interval(labels)
    zoom_start, zoom_end = zoom_interval(raw_df.index, start_time, end_time)

    fig = plt.figure(figsize=(31, 12.5))
    grid = gridspec.GridSpec(
        nrows=4,
        ncols=6,
        figure=fig,
        width_ratios=[0.16, 1.0, 0.92, 1.0, 0.92, 0.24],
        height_ratios=[0.15, 1.0, 1.0, 1.0],
        wspace=0.22,
        hspace=0.36,
    )

    style_group_axis(fig.add_subplot(grid[0, 0]), "", facecolor="#ffffff")
    style_group_axis(fig.add_subplot(grid[0, 1]), "Raw Full Series", facecolor="#e8f1fb")
    style_group_axis(fig.add_subplot(grid[0, 2]), "Raw Zoom Around Anomaly", facecolor="#fdeaea")
    style_group_axis(fig.add_subplot(grid[0, 3]), "Transformed Full Series", facecolor="#e8f5ef")
    style_group_axis(fig.add_subplot(grid[0, 4]), "Transformed Zoom Around Anomaly", facecolor="#eef8ea")
    style_group_axis(fig.add_subplot(grid[0, 5]), "", facecolor="#ffffff")

    row_label_axes = [
        fig.add_subplot(grid[1, 0]),
        fig.add_subplot(grid[2, 0]),
        fig.add_subplot(grid[3, 0]),
    ]
    style_group_axis(row_label_axes[0], "Series")
    style_group_axis(row_label_axes[1], "Residual")
    style_group_axis(row_label_axes[2], "Estimated\nInnovation\nVariance\n(sigma2)")

    legend_axes = [
        fig.add_subplot(grid[1, 5]),
        fig.add_subplot(grid[2, 5]),
        fig.add_subplot(grid[3, 5]),
    ]

    raw_full_ax = fig.add_subplot(grid[1, 1])
    raw_zoom_ax = fig.add_subplot(grid[1, 2])
    transformed_full_ax = fig.add_subplot(grid[1, 3])
    transformed_zoom_ax = fig.add_subplot(grid[1, 4])
    residual_full_ax = fig.add_subplot(grid[2, 1])
    residual_zoom_ax = fig.add_subplot(grid[2, 2])
    transformed_residual_full_ax = fig.add_subplot(grid[2, 3])
    transformed_residual_zoom_ax = fig.add_subplot(grid[2, 4])
    sigma_full_ax = fig.add_subplot(grid[3, 1])
    sigma_zoom_ax = fig.add_subplot(grid[3, 2])
    transformed_sigma_full_ax = fig.add_subplot(grid[3, 3])
    transformed_sigma_zoom_ax = fig.add_subplot(grid[3, 4])

    raw_series = raw_df.iloc[:, 0]
    transformed_series = raw_series if transformed_df is None else transformed_df.iloc[:, 0]
    raw_residual = residual_series if raw_residual_series is None else raw_residual_series
    raw_prediction_scores = predictions if raw_predictions is None else raw_predictions
    transformed_valid_scores = predictions["sigma2"].dropna()
    transformed_max_time = None
    transformed_max_score = None
    if not transformed_valid_scores.empty:
        transformed_max_time = transformed_valid_scores.idxmax()
        transformed_max_score = transformed_valid_scores.loc[transformed_max_time]

    raw_valid_scores = raw_prediction_scores["sigma2"].dropna()
    raw_max_time = None
    raw_max_score = None
    if not raw_valid_scores.empty:
        raw_max_time = raw_valid_scores.idxmax()
        raw_max_score = raw_valid_scores.loc[raw_max_time]

    raw_full_ax.plot(raw_series.index, raw_series, linewidth=0.8, label="raw series")
    add_anomaly_span(raw_full_ax, start_time, end_time)
    raw_full_ax.set_ylabel("value")
    raw_full_ax.grid(True, linestyle="--", alpha=0.3)
    format_time_axis(raw_full_ax)

    transformed_full_ax.plot(
        transformed_series.index,
        transformed_series,
        linewidth=0.8,
        label="transformed series",
    )
    add_anomaly_span(transformed_full_ax, start_time, end_time)
    transformed_full_ax.set_ylabel("value")
    transformed_full_ax.grid(True, linestyle="--", alpha=0.3)
    format_time_axis(transformed_full_ax)

    residual_full_ax.plot(
        raw_residual.index,
        raw_residual,
        linewidth=0.7,
        color="black",
        label="residual",
    )
    add_anomaly_span(residual_full_ax, start_time, end_time)
    residual_full_ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    residual_full_ax.set_ylabel("residual")
    residual_full_ax.grid(True, linestyle="--", alpha=0.3)
    format_time_axis(residual_full_ax)

    transformed_residual_full_ax.plot(
        residual_series.index,
        residual_series,
        linewidth=0.7,
        color="black",
        label="residual",
    )
    add_anomaly_span(transformed_residual_full_ax, start_time, end_time)
    transformed_residual_full_ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    transformed_residual_full_ax.set_ylabel("residual")
    transformed_residual_full_ax.grid(True, linestyle="--", alpha=0.3)
    format_time_axis(transformed_residual_full_ax)

    plot_sigma2_scores(sigma_full_ax, raw_prediction_scores)
    if raw_max_time is not None:
        sigma_full_ax.scatter(
            [raw_max_time],
            [raw_max_score],
            color="red",
            marker="x",
            s=70,
            linewidths=2,
            label="max sigma2",
        )
    add_anomaly_span(sigma_full_ax, start_time, end_time)
    sigma_full_ax.set_ylabel("sigma2")
    sigma_full_ax.grid(True, linestyle="--", alpha=0.3)
    format_time_axis(sigma_full_ax)

    plot_sigma2_scores(transformed_sigma_full_ax, predictions)
    if transformed_max_time is not None:
        transformed_sigma_full_ax.scatter(
            [transformed_max_time],
            [transformed_max_score],
            color="red",
            marker="x",
            s=70,
            linewidths=2,
            label="max sigma2",
        )
    add_anomaly_span(transformed_sigma_full_ax, start_time, end_time)
    transformed_sigma_full_ax.set_ylabel("sigma2")
    transformed_sigma_full_ax.grid(True, linestyle="--", alpha=0.3)
    format_time_axis(transformed_sigma_full_ax)

    if zoom_start is not None and zoom_end is not None:
        raw_zoom = raw_series.loc[zoom_start:zoom_end]
        raw_zoom_ax.plot(raw_zoom.index, raw_zoom, linewidth=0.9, label="raw series")
        add_anomaly_span(raw_zoom_ax, start_time, end_time)
        raw_zoom_ax.set_xlim(zoom_start, zoom_end)
        format_time_axis(raw_zoom_ax)

        transformed_zoom = transformed_series.loc[zoom_start:zoom_end]
        transformed_label = "raw series" if transformed_df is None else "transformed series"
        transformed_zoom_ax.plot(
            transformed_zoom.index,
            transformed_zoom,
            linewidth=0.9,
            label=transformed_label,
        )
        add_anomaly_span(transformed_zoom_ax, start_time, end_time)
        transformed_zoom_ax.set_xlim(zoom_start, zoom_end)
        format_time_axis(transformed_zoom_ax)

        residual_zoom = raw_residual.loc[zoom_start:zoom_end]
        residual_zoom_ax.plot(
            residual_zoom.index,
            residual_zoom,
            linewidth=0.8,
            color="black",
            label="residual",
        )
        add_anomaly_span(residual_zoom_ax, start_time, end_time)
        residual_zoom_ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
        residual_zoom_ax.set_xlim(zoom_start, zoom_end)
        format_time_axis(residual_zoom_ax)

        transformed_residual_zoom = residual_series.loc[zoom_start:zoom_end]
        transformed_residual_zoom_ax.plot(
            transformed_residual_zoom.index,
            transformed_residual_zoom,
            linewidth=0.8,
            color="black",
            label="residual",
        )
        add_anomaly_span(transformed_residual_zoom_ax, start_time, end_time)
        transformed_residual_zoom_ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
        transformed_residual_zoom_ax.set_xlim(zoom_start, zoom_end)
        format_time_axis(transformed_residual_zoom_ax)

        raw_sigma_zoom = raw_prediction_scores.loc[zoom_start:zoom_end]
        plot_sigma2_scores(sigma_zoom_ax, raw_sigma_zoom)
        if raw_max_time is not None and zoom_start <= raw_max_time <= zoom_end:
            sigma_zoom_ax.scatter(
                [raw_max_time],
                [raw_max_score],
                color="red",
                marker="x",
                s=70,
                linewidths=2,
                label="max sigma2",
            )
        add_anomaly_span(sigma_zoom_ax, start_time, end_time)
        sigma_zoom_ax.set_xlim(zoom_start, zoom_end)
        format_time_axis(sigma_zoom_ax)

        transformed_sigma_zoom = predictions.loc[zoom_start:zoom_end]
        plot_sigma2_scores(transformed_sigma_zoom_ax, transformed_sigma_zoom)
        if transformed_max_time is not None and zoom_start <= transformed_max_time <= zoom_end:
            transformed_sigma_zoom_ax.scatter(
                [transformed_max_time],
                [transformed_max_score],
                color="red",
                marker="x",
                s=70,
                linewidths=2,
                label="max sigma2",
            )
        add_anomaly_span(transformed_sigma_zoom_ax, start_time, end_time)
        transformed_sigma_zoom_ax.set_xlim(zoom_start, zoom_end)
        format_time_axis(transformed_sigma_zoom_ax)
    else:
        raw_zoom_ax.text(0.5, 0.5, "No anomaly interval available", ha="center", va="center")
        transformed_zoom_ax.text(0.5, 0.5, "No anomaly interval available", ha="center", va="center")
        residual_zoom_ax.text(0.5, 0.5, "No anomaly interval available", ha="center", va="center")
        transformed_residual_zoom_ax.text(0.5, 0.5, "No anomaly interval available", ha="center", va="center")
        sigma_zoom_ax.text(0.5, 0.5, "No anomaly interval available", ha="center", va="center")
        transformed_sigma_zoom_ax.text(0.5, 0.5, "No anomaly interval available", ha="center", va="center")

    raw_zoom_ax.set_ylabel("value")
    raw_zoom_ax.grid(True, linestyle="--", alpha=0.3)

    transformed_zoom_ax.set_ylabel("value")
    transformed_zoom_ax.grid(True, linestyle="--", alpha=0.3)

    residual_zoom_ax.set_xlabel("time")
    residual_zoom_ax.set_ylabel("residual")
    residual_zoom_ax.grid(True, linestyle="--", alpha=0.3)

    transformed_residual_zoom_ax.set_xlabel("time")
    transformed_residual_zoom_ax.set_ylabel("residual")
    transformed_residual_zoom_ax.grid(True, linestyle="--", alpha=0.3)

    sigma_zoom_ax.set_xlabel("time")
    sigma_zoom_ax.set_ylabel("sigma2")
    sigma_zoom_ax.grid(True, linestyle="--", alpha=0.3)

    transformed_sigma_zoom_ax.set_xlabel("time")
    transformed_sigma_zoom_ax.set_ylabel("sigma2")
    transformed_sigma_zoom_ax.grid(True, linestyle="--", alpha=0.3)

    plot_axes = [
        raw_full_ax,
        raw_zoom_ax,
        transformed_full_ax,
        transformed_zoom_ax,
        residual_full_ax,
        residual_zoom_ax,
        transformed_residual_full_ax,
        transformed_residual_zoom_ax,
        sigma_full_ax,
        sigma_zoom_ax,
        transformed_sigma_full_ax,
        transformed_sigma_zoom_ax,
    ]
    fig.align_ylabels(plot_axes)
    fig.subplots_adjust(left=0.035, right=0.965, top=0.90, bottom=0.075)
    fig.suptitle(title, fontsize=13, fontweight="bold")
    draw_row_legend(legend_axes[0], [raw_full_ax, transformed_full_ax])
    draw_row_legend(legend_axes[1], [residual_full_ax, transformed_residual_full_ax])
    draw_row_legend(legend_axes[2], [sigma_full_ax, transformed_sigma_full_ax])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
