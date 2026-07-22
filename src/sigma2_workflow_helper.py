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


def compute_ucr_metric(
    window_results: pd.DataFrame,
    anomaly_start: int,
    anomaly_end: int,
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
    prediction_time = window_start + (window_end - window_start) / 2
    reference_start = pd.Timestamp(window_results["window_start"].iloc[0])
    prediction_index = int(round((prediction_time - reference_start).total_seconds()))

    anomaly_length = anomaly_end - anomaly_start + 1
    correct_start = min(anomaly_start - anomaly_length, anomaly_start - 100)
    correct_end = max(anomaly_end + anomaly_length, anomaly_end + 100)
    is_correct = int(correct_start < prediction_index < correct_end)

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

    total_duration = index[-1] - index[0]
    anomaly_length = (end_time - start_time) + pd.Timedelta(seconds=1)
    desired_padding = 3 * anomaly_length
    max_zoom_duration = total_duration / 2
    max_padding = max(
        pd.Timedelta(seconds=1),
        (max_zoom_duration - anomaly_length) / 2,
    )
    padding = min(desired_padding, max_padding)

    start = max(index[0], start_time - padding)
    end = min(index[-1], end_time + padding)
    return start, end


def add_anomaly_span(
    ax: Axes,
    start_time: pd.Timestamp | None,
    end_time: pd.Timestamp | None,
) -> None:
    if start_time is not None and end_time is not None:
        ax.axvspan(start_time, end_time, color="red", alpha=0.14, label="anomaly")


def draw_row_legend(legend_ax: Axes, source_ax: Axes) -> None:
    legend_ax.set_xticks([])
    legend_ax.set_yticks([])
    for spine in legend_ax.spines.values():
        spine.set_visible(False)

    handles, labels = source_ax.get_legend_handles_labels()
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
) -> None:
    start_time, end_time = anomaly_interval(labels)
    zoom_start, zoom_end = zoom_interval(raw_df.index, start_time, end_time)

    fig = plt.figure(figsize=(20, 12.5))
    grid = gridspec.GridSpec(
        nrows=4,
        ncols=4,
        figure=fig,
        width_ratios=[0.16, 1.14, 1.0, 0.22],
        height_ratios=[0.15, 1.0, 1.0, 1.0],
        wspace=0.22,
        hspace=0.36,
    )

    style_group_axis(fig.add_subplot(grid[0, 0]), "", facecolor="#ffffff")
    style_group_axis(fig.add_subplot(grid[0, 1]), "Full Series", facecolor="#e8f1fb")
    style_group_axis(fig.add_subplot(grid[0, 2]), "Zoom Around Anomaly Area", facecolor="#fdeaea")
    style_group_axis(fig.add_subplot(grid[0, 3]), "", facecolor="#ffffff")

    row_label_axes = [
        fig.add_subplot(grid[1, 0]),
        fig.add_subplot(grid[2, 0]),
        fig.add_subplot(grid[3, 0]),
    ]
    style_group_axis(row_label_axes[0], "Raw\nData")
    style_group_axis(row_label_axes[1], "Residual")
    style_group_axis(row_label_axes[2], "Estimated\nInnovation\nVariance\n(sigma2)")

    legend_axes = [
        fig.add_subplot(grid[1, 3]),
        fig.add_subplot(grid[2, 3]),
        fig.add_subplot(grid[3, 3]),
    ]

    raw_ax = fig.add_subplot(grid[1, 1])
    raw_zoom_ax = fig.add_subplot(grid[1, 2])
    residual_ax = fig.add_subplot(grid[2, 1])
    residual_zoom_ax = fig.add_subplot(grid[2, 2])
    sigma_ax = fig.add_subplot(grid[3, 1])
    sigma_zoom_ax = fig.add_subplot(grid[3, 2])

    raw_series = raw_df.iloc[:, 0]

    raw_ax.plot(raw_series.index, raw_series, linewidth=0.8, label="raw series")
    add_anomaly_span(raw_ax, start_time, end_time)
    raw_ax.set_ylabel("value")
    raw_ax.grid(True, linestyle="--", alpha=0.3)
    format_time_axis(raw_ax)

    residual_ax.plot(
        residual_series.index,
        residual_series,
        linewidth=0.7,
        color="black",
        label="residual",
    )
    add_anomaly_span(residual_ax, start_time, end_time)
    residual_ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    residual_ax.set_ylabel("residual")
    residual_ax.grid(True, linestyle="--", alpha=0.3)
    format_time_axis(residual_ax)

    plot_sigma2_scores(sigma_ax, predictions)

    max_time = None
    max_score = None
    valid_scores = predictions["sigma2"].dropna()
    if not valid_scores.empty:
        max_time = valid_scores.idxmax()
        max_score = valid_scores.loc[max_time]
        sigma_ax.scatter(
            [max_time],
            [max_score],
            color="red",
            marker="x",
            s=70,
            linewidths=2,
            label="max sigma2",
        )

    add_anomaly_span(sigma_ax, start_time, end_time)
    sigma_ax.set_ylabel("sigma2")
    sigma_ax.grid(True, linestyle="--", alpha=0.3)
    format_time_axis(sigma_ax)

    if zoom_start is not None and zoom_end is not None:
        raw_zoom = raw_series.loc[zoom_start:zoom_end]
        raw_zoom_ax.plot(raw_zoom.index, raw_zoom, linewidth=0.9, label="raw series")
        add_anomaly_span(raw_zoom_ax, start_time, end_time)
        raw_zoom_ax.set_xlim(zoom_start, zoom_end)
        format_time_axis(raw_zoom_ax)

        residual_zoom = residual_series.loc[zoom_start:zoom_end]
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

        sigma_zoom = predictions.loc[zoom_start:zoom_end]
        plot_sigma2_scores(sigma_zoom_ax, sigma_zoom)
        if max_time is not None and zoom_start <= max_time <= zoom_end:
            sigma_zoom_ax.scatter(
                [max_time],
                [max_score],
                color="red",
                marker="x",
                s=70,
                linewidths=2,
                label="max sigma2",
            )
        add_anomaly_span(sigma_zoom_ax, start_time, end_time)
        sigma_zoom_ax.set_xlim(zoom_start, zoom_end)
        format_time_axis(sigma_zoom_ax)
    else:
        raw_zoom_ax.text(0.5, 0.5, "No anomaly interval available", ha="center", va="center")
        residual_zoom_ax.text(0.5, 0.5, "No anomaly interval available", ha="center", va="center")
        sigma_zoom_ax.text(0.5, 0.5, "No anomaly interval available", ha="center", va="center")

    raw_zoom_ax.set_ylabel("value")
    raw_zoom_ax.grid(True, linestyle="--", alpha=0.3)

    residual_zoom_ax.set_xlabel("time")
    residual_zoom_ax.set_ylabel("residual")
    residual_zoom_ax.grid(True, linestyle="--", alpha=0.3)

    sigma_zoom_ax.set_xlabel("time")
    sigma_zoom_ax.set_ylabel("sigma2")
    sigma_zoom_ax.grid(True, linestyle="--", alpha=0.3)

    left_axes = [raw_ax, residual_ax, sigma_ax]
    zoom_axes = [raw_zoom_ax, residual_zoom_ax, sigma_zoom_ax]
    fig.align_ylabels(left_axes)
    fig.align_ylabels(zoom_axes)
    fig.subplots_adjust(left=0.035, right=0.965, top=0.90, bottom=0.075)
    fig.suptitle(title, fontsize=13, fontweight="bold")
    draw_row_legend(legend_axes[0], raw_ax)
    draw_row_legend(legend_axes[1], residual_ax)
    draw_row_legend(legend_axes[2], sigma_ax)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
