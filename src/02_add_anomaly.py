"""Add controlled anomalies to simulated time-series data and plot the result.

The default anomaly is a variance burst, which is a good first test for a
sigma2-based algorithm because sigma2 should increase when residual variance
increases.

Outputs:
  - data/simulated_with_anomaly.csv
  - data/simulated_anomaly_labels.csv
  - results/figures/simulated_with_anomaly.png
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".matplotlib"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_timeseries(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index.name = "time"
    return df


def add_variance_burst(
    df: pd.DataFrame,
    start_index: int,
    duration: int,
    strength: float,
    channels: list[str],
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Increase noise variance in selected channels over a known interval."""
    if start_index < 0:
        raise ValueError("start_index must be nonnegative.")
    if duration <= 0:
        raise ValueError("duration must be positive.")
    if start_index + duration > len(df):
        raise ValueError("start_index + duration exceeds the length of the input data.")

    rng = np.random.default_rng(seed)
    output = df.copy()
    anomaly_mask = np.zeros(len(df), dtype=int)
    anomaly_mask[start_index : start_index + duration] = 1

    if not channels:
        channels = list(df.columns)

    missing = [channel for channel in channels if channel not in df.columns]
    if missing:
        raise ValueError(f"Input data is missing requested channels: {missing}")

    baseline_std = df[channels].std()
    burst_noise = rng.normal(
        loc=0.0,
        scale=strength * baseline_std.to_numpy(),
        size=(duration, len(channels)),
    )
    output.loc[output.index[start_index : start_index + duration], channels] += burst_noise

    labels = pd.DataFrame(
        {
            "is_anomaly": anomaly_mask,
            "anomaly_type": np.where(anomaly_mask == 1, "variance_burst", "normal"),
        },
        index=df.index,
    )
    labels.index.name = "time"
    return output, labels


def plot_anomaly(
    clean_df: pd.DataFrame,
    anomaly_df: pd.DataFrame,
    labels: pd.DataFrame,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(
        nrows=len(anomaly_df.columns),
        ncols=1,
        figsize=(12, 2.4 * len(anomaly_df.columns)),
        sharex=True,
    )

    if len(anomaly_df.columns) == 1:
        axes = [axes]

    anomaly_times = labels.index[labels["is_anomaly"] == 1]
    start_time = anomaly_times.min()
    end_time = anomaly_times.max()

    for ax, column in zip(axes, anomaly_df.columns):
        ax.plot(clean_df.index, clean_df[column], color="gray", linewidth=0.7, alpha=0.5, label="clean")
        ax.plot(anomaly_df.index, anomaly_df[column], color="black", linewidth=0.8, label="with anomaly")
        ax.axvspan(start_time, end_time, color="red", alpha=0.15, label="anomaly window")
        ax.set_ylabel(column)
        ax.grid(True, linestyle="--", alpha=0.3)

    axes[0].set_title("Simulated Time Series With Controlled Variance-Burst Anomaly")
    axes[0].legend(loc="upper right")
    axes[-1].set_xlabel("Time")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Add controlled anomaly to simulated data.")
    parser.add_argument("--input", default="data/simulated_clean.csv")
    parser.add_argument("--output", default="data/simulated_with_anomaly.csv")
    parser.add_argument("--labels-output", default="data/simulated_anomaly_labels.csv")
    parser.add_argument("--plot", default="results/figures/simulated_with_anomaly.png")
    parser.add_argument("--start-index", type=int, default=2500)
    parser.add_argument("--duration", type=int, default=900)
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("--channels", nargs="*", default=None)
    parser.add_argument("--seed", type=int, default=456)
    args = parser.parse_args()

    clean_df = read_timeseries(Path(args.input))
    anomaly_df, labels = add_variance_burst(
        clean_df,
        start_index=args.start_index,
        duration=args.duration,
        strength=args.strength,
        channels=args.channels,
        seed=args.seed,
    )

    output_path = Path(args.output)
    labels_path = Path(args.labels_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    anomaly_df.to_csv(output_path)
    labels.to_csv(labels_path)

    plot_anomaly(clean_df, anomaly_df, labels, Path(args.plot))
    print(f"Saved anomalous data to {output_path}")
    print(f"Saved labels to {labels_path}")
    print(f"Saved plot to {args.plot}")


if __name__ == "__main__":
    main()
