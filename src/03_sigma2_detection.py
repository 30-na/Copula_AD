"""Run the sigma2 anomaly-detection workflow on simulated data.

This script:
  1. reads simulated time-series data,
  2. fits ARIMA models on overlapping windows,
  3. extracts sigma2, fitted values, and residuals,
  4. saves sigma2 values, predictions, metrics, and a combined plot.

Default outputs:
  - results/simulation/<run_name>/sigma2_values.csv
  - results/simulation/<run_name>/simulation_predictions.csv
  - results/simulation/<run_name>/simulation_metrics.csv
  - results/simulation/<run_name>/figures/combined.png
"""

from __future__ import annotations

import argparse
import importlib.util
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


SRC_DIR = Path(__file__).resolve().parent


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def fit_arima_window_full_stats(
    window: np.ndarray,
    order: tuple[int, int, int],
) -> tuple[float, np.ndarray, np.ndarray]:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            warnings.simplefilter("ignore", UserWarning)
            model = sm.tsa.ARIMA(window, order=order)
            result = model.fit()

        params = result.params
        if hasattr(params, "index") and "sigma2" in params.index:
            sigma2 = float(params.loc["sigma2"])
        else:
            sigma2 = float(params[-1])

        predicted_values = np.asarray(result.fittedvalues, dtype=float)
        if len(predicted_values) != len(window):
            padded = np.full(len(window), np.nan, dtype=float)
            padded[-len(predicted_values) :] = predicted_values
            predicted_values = padded
        residuals = np.asarray(window, dtype=float) - predicted_values
        return sigma2, residuals, predicted_values
    except Exception:
        nan_values = np.full(len(window), np.nan, dtype=float)
        return np.nan, nan_values, nan_values.copy()


def fit_arima_window_stats(window: np.ndarray, order: tuple[int, int, int]) -> tuple[float, float, float]:
    sigma2, residuals, predicted_values = fit_arima_window_full_stats(window, order)
    residual = float(residuals[-1]) if len(residuals) else np.nan
    predicted_value = float(predicted_values[-1]) if len(predicted_values) else np.nan
    return sigma2, residual, predicted_value


def fit_sigma2(window: np.ndarray, order: tuple[int, int, int]) -> float:
    sigma2, _, _ = fit_arima_window_stats(window, order)
    return sigma2


def compute_sigma2(
    df: pd.DataFrame,
    window_size: int,
    overlap_size: int,
    order: tuple[int, int, int],
) -> pd.DataFrame:
    sigma2_by_channel = {}
    residual_by_channel = {}
    predicted_by_channel = {}
    window_start_times = None
    window_end_times = None

    for column in df.columns:
        windows = create_windows(df[column], window_size, overlap_size)
        if window_start_times is None:
            window_start_times = [start_time for start_time, _, _ in windows]
        if window_end_times is None:
            window_end_times = [end_time for _, end_time, _ in windows]

        fitted_stats = [
            fit_arima_window_stats(values, order)
            for _, _, values in tqdm(windows, desc=f"Fitting ARIMA for {column}")
        ]
        sigma2_values = [sigma2 for sigma2, _, _ in fitted_stats]
        residual_values = [residual for _, residual, _ in fitted_stats]
        predicted_values = [predicted for _, _, predicted in fitted_stats]
        sigma2_by_channel[column] = sigma2_values
        residual_by_channel[f"{column}_residual"] = residual_values
        predicted_by_channel[f"{column}_predicted"] = predicted_values

    sigma2_df = pd.DataFrame(sigma2_by_channel, index=pd.Index(window_end_times, name="time"))
    sigma2_df.insert(0, "window_start", window_start_times)
    sigma2_df.insert(1, "window_end", window_end_times)
    for column, values in residual_by_channel.items():
        sigma2_df[column] = values
    for column, values in predicted_by_channel.items():
        sigma2_df[column] = values
    numeric_columns = [column for column in sigma2_by_channel]
    sigma2_df["sigma2_mean"] = sigma2_df[numeric_columns].mean(axis=1)
    residual_columns = [column for column in residual_by_channel]
    predicted_columns = [column for column in predicted_by_channel]
    sigma2_df["residual_mean"] = sigma2_df[residual_columns].mean(axis=1)
    sigma2_df["predicted_mean"] = sigma2_df[predicted_columns].mean(axis=1)
    return sigma2_df


def create_trailing_windows(
    series: pd.Series,
    window_size: int,
    stride: int,
) -> list[tuple[int, int, pd.Timestamp, pd.Timestamp, np.ndarray]]:
    if stride <= 0:
        raise ValueError("stride must be positive.")
    if len(series) < window_size:
        raise ValueError("window_size is larger than the time series.")

    clean = series.dropna()
    windows = []
    end_positions = list(range(window_size - 1, len(clean), stride))
    final_end = len(clean) - 1
    if end_positions[-1] != final_end:
        end_positions.append(final_end)

    for end in end_positions:
        start = end - window_size + 1
        window = clean.iloc[start : end + 1]
        windows.append((start, end, window.index[0], window.index[-1], window.to_numpy()))
    return windows


def compute_sigma2_pointwise(
    df: pd.DataFrame,
    window_size: int,
    stride: int,
    order: tuple[int, int, int],
) -> pd.DataFrame:
    """Compute pointwise residuals and endpoint sigma2 scores.

    Each ARIMA model is fitted on a trailing window. Its sigma2 estimate is
    assigned only to the endpoint of that window. Its fitted values give one
    residual per point inside the window. When windows overlap, residuals and
    fitted values are averaged per point.
    """
    sigma2_by_channel = {}
    residual_by_channel = {}
    predicted_by_channel = {}
    contribution_counts = {}

    for column in df.columns:
        windows = create_trailing_windows(df[column], window_size, stride)
        sigma2_values = np.full(len(df), np.nan, dtype=float)
        residual_sum = np.zeros(len(df), dtype=float)
        predicted_sum = np.zeros(len(df), dtype=float)
        residual_counts = np.zeros(len(df), dtype=float)
        predicted_counts = np.zeros(len(df), dtype=float)
        window_counts = np.zeros(len(df), dtype=float)

        for start, end, _, _, values in tqdm(windows, desc=f"Fitting trailing ARIMA for {column}"):
            sigma2, residuals, predicted_values = fit_arima_window_full_stats(values, order)
            point_slice = slice(start, end + 1)

            if np.isfinite(sigma2):
                sigma2_values[end] = sigma2

            valid_residuals = np.isfinite(residuals)
            residual_sum[start : end + 1][valid_residuals] += residuals[valid_residuals]
            residual_counts[start : end + 1][valid_residuals] += 1

            valid_predictions = np.isfinite(predicted_values)
            predicted_sum[start : end + 1][valid_predictions] += predicted_values[valid_predictions]
            predicted_counts[start : end + 1][valid_predictions] += 1

            window_counts[point_slice] += 1

        with np.errstate(invalid="ignore", divide="ignore"):
            residual_values = residual_sum / residual_counts
            predicted_values = predicted_sum / predicted_counts

        residual_values[residual_counts == 0] = np.nan
        predicted_values[predicted_counts == 0] = np.nan

        sigma2_by_channel[column] = sigma2_values
        residual_by_channel[f"{column}_residual"] = residual_values
        predicted_by_channel[f"{column}_predicted"] = predicted_values
        contribution_counts[f"{column}_window_count"] = window_counts

    sigma2_df = pd.DataFrame(index=df.index.copy())
    sigma2_df.index.name = "time"
    for column, values in residual_by_channel.items():
        sigma2_df[column] = values
    for column, values in predicted_by_channel.items():
        sigma2_df[column] = values
    for column, values in sigma2_by_channel.items():
        sigma2_df[column] = values
    for column, values in contribution_counts.items():
        sigma2_df[column] = values
    numeric_columns = [column for column in sigma2_by_channel]
    sigma2_df["sigma2_mean"] = sigma2_df[numeric_columns].mean(axis=1)
    residual_columns = [column for column in residual_by_channel]
    predicted_columns = [column for column in predicted_by_channel]
    sigma2_df["residual_mean"] = sigma2_df[residual_columns].mean(axis=1)
    sigma2_df["predicted_mean"] = sigma2_df[predicted_columns].mean(axis=1)
    sigma2_df["is_point_score"] = True

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
        if column
        not in {
            "window_start",
            "window_end",
            "sigma2_mean",
            "residual_mean",
            "innovation_mean",
            "predicted_mean",
            "is_point_score",
        }
        and not column.endswith("_residual")
        and not column.endswith("_innovation")
        and not column.endswith("_predicted")
        and not column.endswith("_window_count")
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


def make_run_name(args: argparse.Namespace, order: tuple[int, int, int]) -> str:
    arima_name = "".join(str(value) for value in order)
    if args.score_mode == "pointwise":
        spacing = f"s{args.stride}"
    else:
        spacing = f"overlap{args.overlap_size}"
    return f"simulation_{args.score_mode}_w{args.window_size}_{spacing}_arima{arima_name}"


def configure_output_paths(args: argparse.Namespace, order: tuple[int, int, int]) -> None:
    run_name = args.run_name or make_run_name(args, order)
    run_output_dir = Path(args.run_output_dir) if args.run_output_dir else Path("results") / "simulation" / run_name

    args.run_name = run_name
    args.run_output_dir = str(run_output_dir)
    if args.output is None:
        args.output = str(run_output_dir / "sigma2_values.csv")
    if args.predictions_output is None:
        args.predictions_output = str(run_output_dir / "simulation_predictions.csv")
    if args.metrics_output is None:
        args.metrics_output = str(run_output_dir / "simulation_metrics.csv")
    if args.plot is None:
        args.plot = str(run_output_dir / "figures" / "combined.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute and plot sigma2 values.")
    parser.add_argument("--input", default="data/simulated_with_anomaly.csv")
    parser.add_argument("--labels", default="data/simulated_anomaly_labels.csv")
    parser.add_argument("--run-name", help="Optional simulation run folder name for parameter comparisons.")
    parser.add_argument("--run-output-dir", help="Optional simulation output folder for this run.")
    parser.add_argument("--output", help="Optional sigma2 output CSV path.")
    parser.add_argument("--predictions-output", help="Optional predictions output CSV path.")
    parser.add_argument("--metrics-output", help="Optional metrics output CSV path.")
    parser.add_argument("--plot", help="Optional combined plot output path.")
    parser.add_argument("--window-size", type=int, default=200)
    parser.add_argument("--overlap-size", type=int, default=100)
    parser.add_argument("--score-mode", choices=["window", "pointwise"], default="window")
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--order", nargs=3, type=int, default=[1, 0, 1])
    parser.add_argument("--score-column", default="sigma2_mean")
    parser.add_argument("--k", type=float, default=3.0)
    parser.add_argument("--skip-plot", action="store_true")
    args = parser.parse_args()

    order = parse_order(args.order)
    configure_output_paths(args, order)
    df = read_timeseries(Path(args.input))
    labels = read_labels(Path(args.labels))

    if args.score_mode == "pointwise":
        sigma2_df = compute_sigma2_pointwise(
            df,
            window_size=args.window_size,
            stride=args.stride,
            order=order,
        )
    else:
        sigma2_df = compute_sigma2(
            df,
            window_size=args.window_size,
            overlap_size=args.overlap_size,
            order=order,
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sigma2_df.to_csv(output_path)

    print(f"Run name: {args.run_name}")
    print(f"Simulation output: {args.run_output_dir}")
    print(f"Saved sigma2 values to {output_path}")

    if labels is not None and "is_anomaly" in labels.columns:
        eval_mod = load_module("evaluate_sigma2", SRC_DIR / "05_evaluate_sigma2.py")
        predictions, threshold = eval_mod.compute_predictions(
            sigma2_df,
            labels,
            score_column=args.score_column,
            k=args.k,
        )
        metrics = eval_mod.compute_metrics(predictions)
        metrics.update(
            {
                "threshold": threshold,
                "k": args.k,
                "score_column": args.score_column,
                "window_size": args.window_size,
                "overlap_size": args.overlap_size,
                "score_mode": args.score_mode,
                "stride": args.stride,
                "order": str(order),
            }
        )

        predictions_path = Path(args.predictions_output)
        metrics_path = Path(args.metrics_output)
        predictions_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        predictions.to_csv(predictions_path)
        pd.DataFrame([metrics]).to_csv(metrics_path, index=False)

        print(f"Saved predictions to {predictions_path}")
        print(f"Saved metrics to {metrics_path}")

        if not args.skip_plot:
            plot_mod = load_module("plot_combined_result", SRC_DIR / "06_plot_combined_result.py")
            plot_mod.plot_combined(
                raw_df=df,
                predictions=predictions,
                labels=labels,
                title=f"Simulation {args.run_name}",
                output_path=Path(args.plot),
            )
            print(f"Saved combined plot to {args.plot}")
    elif not args.skip_plot:
        plot_sigma2(sigma2_df, labels, Path(args.plot))
        print(f"Saved sigma2 plot to {args.plot}")


if __name__ == "__main__":
    main()
