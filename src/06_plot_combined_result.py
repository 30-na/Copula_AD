"""Create one combined figure for a dataset and its sigma2 result.

The figure contains:
  1. raw time-series data with the true anomaly region,
  2. sigma2 score with threshold and the true anomaly region.
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


def read_timeseries(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index.name = "time"
    return df.apply(pd.to_numeric, errors="coerce")


def read_predictions(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index.name = "time"
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


def plot_combined(
    raw_df: pd.DataFrame,
    predictions: pd.DataFrame,
    labels: pd.DataFrame,
    title: str,
    output_path: Path,
) -> None:
    start_time, end_time = anomaly_interval(labels)
    raw_columns = list(raw_df.columns)

    fig, axes = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(13, 7),
        sharex=False,
        gridspec_kw={"height_ratios": [1.2, 1]},
    )

    raw_ax, sigma_ax = axes

    for column in raw_columns:
        label = "time series" if len(raw_columns) == 1 else column
        raw_ax.plot(raw_df.index, raw_df[column], linewidth=0.8, label=label)

    if start_time is not None and end_time is not None:
        raw_ax.axvspan(start_time, end_time, color="red", alpha=0.14, label="true anomaly")

    raw_ax.set_title("Raw Data")
    raw_ax.set_ylabel("value")
    raw_ax.grid(True, linestyle="--", alpha=0.3)
    raw_ax.legend(loc="upper right")

    sigma_ax.plot(
        predictions.index,
        predictions["score"],
        marker="o",
        markersize=3,
        linewidth=0.8,
        color="black",
        label="sigma2 score",
    )

    if "threshold" in predictions.columns:
        sigma_ax.axhline(
            predictions["threshold"].iloc[0],
            color="red",
            linestyle="--",
            linewidth=1,
            label="threshold",
        )

    if "is_predicted" in predictions.columns:
        flagged = predictions[predictions["is_predicted"] == 1]
        sigma_ax.scatter(
            flagged.index,
            flagged["score"],
            color="red",
            marker="x",
            s=45,
            label="predicted anomaly",
        )

    if start_time is not None and end_time is not None:
        sigma_ax.axvspan(start_time, end_time, color="red", alpha=0.14, label="true anomaly")

    sigma_ax.set_title("Estimate of the Innovation Variance")
    sigma_ax.set_xlabel("time")
    sigma_ax.set_ylabel("sigma2")
    sigma_ax.grid(True, linestyle="--", alpha=0.3)
    sigma_ax.legend(loc="upper right")

    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
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
