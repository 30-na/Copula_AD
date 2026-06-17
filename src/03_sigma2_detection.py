"""Run a simple sigma2 anomaly-detection workflow on simulated data.

This script:
  1. reads simulated time-series data,
  2. fits ARIMA models on overlapping windows,
  3. extracts sigma2 for each window and channel,
  4. plots sigma2 values with the known anomaly interval.

Outputs:
  - results/sigma2_values.csv
  - results/figures/sigma2_values.png
"""

from __future__ import annotations

import argparse
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
import matplotlib.pyplot as plt


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


def create_windows(
    series: pd.Series,
    window_size: int,
    overlap_size: int,
) -> list[tuple[pd.Timestamp, pd.Timestamp, np.ndarray]]:
    step = window_size - overlap_size
    if step <= 0:
        raise ValueError("overlap_size must be smaller than window_size.")
    if len(series) < window_size:
        raise ValueError("window_size is larger than the time series.")

    clean = series.dropna()
    windows = []
    for start in range(0, len(clean) - window_size + 1, step):
        end = start + window_size
        window = clean.iloc[start:end]
        windows.append((window.index[0], window.index[-1], window.to_numpy()))
    return windows


def fit_sigma2(window: np.ndarray, order: tuple[int, int, int]) -> float:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            warnings.simplefilter("ignore", UserWarning)
            model = sm.tsa.ARIMA(window, order=order)
            result = model.fit()

        params = result.params
        if hasattr(params, "index") and "sigma2" in params.index:
            return float(params.loc["sigma2"])
        return float(params[-1])
    except Exception:
        return np.nan


def compute_sigma2(
    df: pd.DataFrame,
    window_size: int,
    overlap_size: int,
    order: tuple[int, int, int],
) -> pd.DataFrame:
    sigma2_by_channel = {}
    window_start_times = None
    window_end_times = None

    for column in df.columns:
        windows = create_windows(df[column], window_size, overlap_size)
        if window_start_times is None:
            window_start_times = [start_time for start_time, _, _ in windows]
        if window_end_times is None:
            window_end_times = [end_time for _, end_time, _ in windows]

        sigma2_values = [
            fit_sigma2(values, order)
            for _, _, values in tqdm(windows, desc=f"Fitting ARIMA for {column}")
        ]
        sigma2_by_channel[column] = sigma2_values

    sigma2_df = pd.DataFrame(sigma2_by_channel, index=pd.Index(window_end_times, name="time"))
    sigma2_df.insert(0, "window_start", window_start_times)
    sigma2_df.insert(1, "window_end", window_end_times)
    numeric_columns = [column for column in sigma2_by_channel]
    sigma2_df["sigma2_mean"] = sigma2_df[numeric_columns].mean(axis=1)
    return sigma2_df


def plot_sigma2(
    sigma2_df: pd.DataFrame,
    labels: pd.DataFrame | None,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))

    channel_columns = [
        column
        for column in sigma2_df.columns
        if column not in {"window_start", "window_end", "sigma2_mean"}
    ]
    for column in channel_columns:
        ax.plot(
            sigma2_df.index,
            sigma2_df[column],
            marker="o",
            markersize=3,
            linewidth=0.8,
            alpha=0.6,
            label=column,
        )

    ax.plot(
        sigma2_df.index,
        sigma2_df["sigma2_mean"],
        marker="o",
        markersize=4,
        color="black",
        linewidth=1.5,
        label="mean sigma2",
    )

    if labels is not None and "is_anomaly" in labels.columns:
        anomaly_times = labels.index[labels["is_anomaly"] == 1]
        if len(anomaly_times) > 0:
            ax.axvspan(
                anomaly_times.min(),
                anomaly_times.max(),
                color="red",
                alpha=0.15,
                label="true anomaly",
            )

    ax.set_title("Sigma2 Values From Overlapping ARIMA Windows")
    ax.set_xlabel("Time")
    ax.set_ylabel("sigma2")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="upper right")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def parse_order(values: list[int]) -> tuple[int, int, int]:
    if len(values) != 3:
        raise ValueError("ARIMA order must be three integers: p d q.")
    return tuple(values)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute and plot sigma2 values.")
    parser.add_argument("--input", default="data/simulated_with_anomaly.csv")
    parser.add_argument("--labels", default="data/simulated_anomaly_labels.csv")
    parser.add_argument("--output", default="results/sigma2_values.csv")
    parser.add_argument("--plot", default="results/figures/sigma2_values.png")
    parser.add_argument("--window-size", type=int, default=200)
    parser.add_argument("--overlap-size", type=int, default=100)
    parser.add_argument("--order", nargs=3, type=int, default=[1, 0, 1])
    args = parser.parse_args()

    order = parse_order(args.order)
    df = read_timeseries(Path(args.input))
    labels = read_labels(Path(args.labels))

    sigma2_df = compute_sigma2(
        df,
        window_size=args.window_size,
        overlap_size=args.overlap_size,
        order=order,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sigma2_df.to_csv(output_path)
    plot_sigma2(sigma2_df, labels, Path(args.plot))

    print(f"Saved sigma2 values to {output_path}")
    print(f"Saved sigma2 plot to {args.plot}")


if __name__ == "__main__":
    main()
