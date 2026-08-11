"""Run the shared sigma2 workflow on simulated data.

This script:
  1. reads simulated time-series data,
  2. runs the shared sigma2 detector,
  3. evaluates the result using the UCR-style metric,
  4. saves one window-level results file and a combined plot.

Default outputs:
  - results/simulation/<run_name>/sigma2_values.csv
  - results/simulation/<run_name>/figures/combined.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

from sigma2_workflow_helper import (
    build_window_results,
    compute_sigma2,
    compute_ucr_metric,
    parse_order,
    plot_combined,
    plot_sigma2,
    preprocess_stationary_series,
    read_labels,
    read_timeseries,
)


def make_run_name(args: argparse.Namespace, order: tuple[int, int, int]) -> str:
    arima_name = "".join(str(value) for value in order)
    if args.preprocessing_mode == "raw":
        preproc_name = f"raw-w{args.window_size}-s{args.stride}"
    else:
        preproc_name = args.period_method
    return f"simulation_period-{preproc_name}_arima{arima_name}"


def configure_output_paths(args: argparse.Namespace, order: tuple[int, int, int]) -> None:
    run_name = args.run_name or make_run_name(args, order)
    run_output_dir = Path(args.run_output_dir) if args.run_output_dir else Path("results") / "simulation" / run_name

    args.run_name = run_name
    args.run_output_dir = str(run_output_dir)
    if args.output is None:
        args.output = str(run_output_dir / "sigma2_values.csv")
    if args.plot is None:
        args.plot = str(run_output_dir / "figures" / "combined.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute and plot sigma2 values.")
    parser.add_argument("--input", default="data/simulation/simulated_with_anomaly.csv")
    parser.add_argument("--labels", default="data/simulation/simulated_anomaly_labels.csv")
    parser.add_argument("--run-name", help="Optional simulation run folder name for parameter comparisons.")
    parser.add_argument("--run-output-dir", help="Optional simulation output folder for this run.")
    parser.add_argument("--output", help="Optional sigma2 output CSV path.")
    parser.add_argument("--plot", help="Optional combined plot output path.")
    parser.add_argument(
        "--window-size",
        type=int,
        default=200,
        help="Window size. Used directly in raw mode, otherwise a fallback if no cycle is found.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=50,
        help="Stride. Used directly in raw mode, otherwise a fallback if no cycle is found.",
    )
    parser.add_argument(
        "--preprocessing-mode",
        choices=["raw", "seasonal_diff"],
        default="seasonal_diff",
        help="raw keeps --window-size/--stride and skips seasonal differencing.",
    )
    parser.add_argument("--period-method", choices=["autocorr", "fft"], default="autocorr")
    parser.add_argument("--order", nargs=3, type=int, default=[1, 0, 1])
    parser.add_argument("--skip-plot", action="store_true")
    args = parser.parse_args()

    order = parse_order(args.order)
    configure_output_paths(args, order)
    raw_df = read_timeseries(Path(args.input))
    labels = read_labels(Path(args.labels))
    processed_df, preprocessing = preprocess_stationary_series(
        raw_df,
        period_method=args.period_method,
        fallback_window_size=args.window_size,
        fallback_stride=args.stride,
        preprocessing_mode=args.preprocessing_mode,
    )
    window_size = int(preprocessing["window_size"])
    stride = int(preprocessing["stride"])
    raw_sigma2_df, raw_residual_series = compute_sigma2(
        raw_df,
        window_size=window_size,
        stride=stride,
        order=order,
    )
    sigma2_df = raw_sigma2_df
    residual_series = raw_residual_series
    if bool(preprocessing["seasonal_adjustment"]):
        sigma2_df, residual_series = compute_sigma2(
            processed_df,
            window_size=window_size,
            stride=stride,
            order=order,
        )
    window_results = build_window_results(sigma2_df, labels)
    raw_window_results = window_results
    if bool(preprocessing["seasonal_adjustment"]):
        raw_window_results = build_window_results(raw_sigma2_df, labels)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    window_results.to_csv(output_path)

    print(f"Run name: {args.run_name}")
    print(f"Simulation output: {args.run_output_dir}")
    print(
        "Preprocessing:",
        f"method={preprocessing['period_method']},",
        f"estimated_period={preprocessing['estimated_period']},",
        f"seasonal_adjustment={preprocessing['seasonal_adjustment']},",
        f"window_size={preprocessing['window_size']},",
        f"stride={preprocessing['stride']}",
    )
    print(f"Saved window results to {output_path}")

    if labels is not None and "is_anomaly" in labels.columns:
        anomaly_positions = labels.index[labels["is_anomaly"].astype(int) == 1]
        if len(anomaly_positions) == 0:
            raise ValueError("Label file does not contain any anomaly points.")

        evaluation = compute_ucr_metric(
            window_results,
            anomaly_start=int(labels["is_anomaly"].to_numpy(dtype=int).argmax()),
            anomaly_end=int(len(labels) - 1 - labels["is_anomaly"].to_numpy(dtype=int)[::-1].argmax()),
            reference_start_time=raw_df.index[0],
        )
        print(
            "UCR evaluation:",
            f"predicted_index={evaluation['ucr_prediction_index']},",
            f"correct_interval=({evaluation['ucr_correct_start']}, {evaluation['ucr_correct_end']}),",
            f"detected={evaluation['ucr_correct']}",
        )

        if not args.skip_plot:
            plot_combined(
                raw_df=raw_df,
                residual_series=residual_series,
                predictions=window_results,
                labels=labels,
                raw_residual_series=raw_residual_series,
                raw_predictions=raw_window_results,
                title=f"Simulation {args.run_name}",
                output_path=Path(args.plot),
            )
            print(f"Saved combined plot to {args.plot}")
    elif not args.skip_plot:
        plot_sigma2(window_results, labels, Path(args.plot))
        print(f"Saved sigma2 plot to {args.plot}")


if __name__ == "__main__":
    main()
