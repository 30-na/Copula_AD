"""Create one combined figure for a dataset and its sigma2 result.

The figure contains:
  1. raw time-series data with the true anomaly region,
  2. residual values with the true anomaly region,
  3. sigma2 score with the maximum-score point,
  4. raw data zoomed around the true anomaly,
  5. residual values zoomed around the true anomaly,
  6. sigma2 score zoomed around the true anomaly.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".matplotlib"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import gridspec
from matplotlib.axes import Axes


def read_timeseries(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index.name = "time"
    return df.apply(pd.to_numeric, errors="coerce")


def read_predictions(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index.name = "time"
    if "is_point_score" in df.columns:
        df["is_point_score"] = df["is_point_score"].astype(bool)
    return df


def read_labels(path: Path) -> pd.DataFrame:
    labels = pd.read_csv(path, index_col=0, parse_dates=True)
    labels.index.name = "time"
    return labels


def anomaly_interval(labels: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp] | tuple[None, None]:
    if "is_anomaly" not in labels.columns:
        return None, None

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

    anomaly_length = (end_time - start_time) + pd.Timedelta(seconds=1)
    padding = 3 * anomaly_length

    start = max(index[0], start_time - padding)
    end = min(index[-1], end_time + padding)
    return start, end


def add_anomaly_span(ax, start_time: pd.Timestamp | None, end_time: pd.Timestamp | None) -> None:
    if start_time is not None and end_time is not None:
        ax.axvspan(start_time, end_time, color="red", alpha=0.14, label="true anomaly")


def legend_outside(ax: Axes) -> None:
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(
            handles,
            labels,
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),
            borderaxespad=0.0,
            frameon=True,
            fontsize=9,
        )


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


def residual_for_plot(raw_df: pd.DataFrame, predictions: pd.DataFrame) -> pd.Series | None:
    """Return residuals as observed minus fitted values at prediction timestamps."""
    predicted_columns = [column for column in predictions.columns if column.endswith("_predicted")]
    if predicted_columns:
        residual_by_column = []
        for predicted_column in predicted_columns:
            raw_column = predicted_column[: -len("_predicted")]
            if raw_column not in raw_df.columns:
                continue

            observed = raw_df[raw_column].reindex(predictions.index)
            predicted = predictions[predicted_column]
            residual_by_column.append(observed - predicted)

        if residual_by_column:
            return pd.concat(residual_by_column, axis=1).mean(axis=1)

    if "residual_mean" in predictions.columns:
        return predictions["residual_mean"]
    if "innovation_mean" in predictions.columns:
        return predictions["innovation_mean"]

    return None


def marker_style(index: pd.Index, dense_threshold: int = 1000) -> dict[str, float | str | None]:
    if len(index) > dense_threshold:
        return {"marker": None, "markersize": 0}
    return {"marker": "o", "markersize": 2}


def plot_sigma2_scores(
    ax: Axes,
    predictions: pd.DataFrame,
    label: str = "sigma2",
) -> None:
    sigma2_scores = predictions["score"].dropna()
    if sigma2_scores.empty:
        return

    ax.plot(
        sigma2_scores.index,
        sigma2_scores,
        marker="o",
        markersize=3,
        linewidth=0.8,
        color="black",
        label=label,
    )


def plot_combined(
    raw_df: pd.DataFrame,
    predictions: pd.DataFrame,
    labels: pd.DataFrame,
    title: str,
    output_path: Path,
) -> None:
    start_time, end_time = anomaly_interval(labels)
    zoom_start, zoom_end = zoom_interval(raw_df.index, start_time, end_time)
    raw_columns = list(raw_df.columns)
    residual = residual_for_plot(raw_df, predictions)

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

    for column in raw_columns:
        label = "raw series" if len(raw_columns) == 1 else column
        raw_ax.plot(raw_df.index, raw_df[column], linewidth=0.8, label=label)

    add_anomaly_span(raw_ax, start_time, end_time)

    raw_ax.set_ylabel("value")
    raw_ax.grid(True, linestyle="--", alpha=0.3)
    format_time_axis(raw_ax)

    if residual is not None:
        residual_style = marker_style(residual.index)
        residual_ax.plot(
            residual.index,
            residual,
            marker=residual_style["marker"],
            markersize=residual_style["markersize"],
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
    if "score" in predictions.columns:
        valid_scores = predictions["score"].dropna()
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
        raw_zoom = raw_df.loc[zoom_start:zoom_end]
        for column in raw_columns:
            label = "raw series" if len(raw_columns) == 1 else column
            raw_zoom_ax.plot(raw_zoom.index, raw_zoom[column], linewidth=0.9, label=label)
        add_anomaly_span(raw_zoom_ax, start_time, end_time)
        raw_zoom_ax.set_xlim(zoom_start, zoom_end)
        format_time_axis(raw_zoom_ax)

        if residual is not None:
            residual_zoom = residual.loc[zoom_start:zoom_end]
            residual_zoom_style = marker_style(residual_zoom.index)
            residual_zoom_ax.plot(
                residual_zoom.index,
                residual_zoom,
                marker=residual_zoom_style["marker"],
                markersize=residual_zoom_style["markersize"],
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot raw data and sigma2 result in one figure.")
    parser.add_argument("--raw", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    raw_df = read_timeseries(Path(args.raw))
    predictions = read_predictions(Path(args.predictions))
    labels = read_labels(Path(args.labels))

    plot_combined(
        raw_df=raw_df,
        predictions=predictions,
        labels=labels,
        title=args.title,
        output_path=Path(args.output),
    )
    print(f"Saved combined figure to {args.output}")


if __name__ == "__main__":
    main()
