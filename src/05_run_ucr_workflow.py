"""Run the shared sigma2 workflow on all prepared UCR datasets.

This script:
  1. reads prepared UCR time-series data and labels,
  2. estimates a period using the selected preprocessing method,
  3. applies seasonal differencing when a period is found,
  4. uses window = 2P and stride = 25% of window,
  5. falls back to the default window and stride when no period is found,
  6. saves one combined 3x3 plot and one sigma2 CSV per dataset,
  7. saves evaluation summaries by category and overall.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from sigma2_workflow_helper import (
    build_window_results,
    compute_sigma2,
    compute_ucr_metric,
    parse_order,
    plot_combined,
    preprocess_stationary_series,
    read_labels,
    read_timeseries,
)


def make_run_name(
    args: argparse.Namespace,
    order: tuple[int, int, int],
) -> str:
    arima_name = "".join(str(value) for value in order)
    base_name = args.run_name or "ucr"
    if args.preprocessing_mode == "raw":
        preproc_name = f"raw-w{args.window_size}-s{args.stride}"
    else:
        preproc_name = args.period_method
    return f"{base_name}_preproc-{preproc_name}_arima{arima_name}"


def build_run_config(
    args: argparse.Namespace,
    order: tuple[int, int, int],
) -> dict[str, str | Path]:
    run_name = make_run_name(args, order)

    if args.run_output_dir:
        run_output_dir = Path(args.run_output_dir) / run_name
    else:
        run_output_dir = Path("results") / run_name

    if args.overall_summary_output is None:
        overall_summary_output = run_output_dir / "overall_evaluation.csv"
    else:
        requested = Path(args.overall_summary_output)
        if requested.suffix.lower() == ".csv":
            overall_summary_output = requested
        else:
            overall_summary_output = requested / run_name / "overall_evaluation.csv"

    return {
        "preprocessing_mode": args.preprocessing_mode,
        "run_name": run_name,
        "run_output_dir": run_output_dir,
        "overall_summary_output": overall_summary_output,
    }


def find_prepared_datasets(prepared_dir: Path) -> list[dict[str, Path | str]]:
    datasets = []
    for timeseries_path in sorted(prepared_dir.glob("by_category/*/*/timeseries.csv")):
        dataset_dir = timeseries_path.parent
        labels_path = dataset_dir / "labels.csv"
        if not labels_path.exists():
            continue

        category = dataset_dir.parent.name
        dataset_name = dataset_dir.name
        datasets.append(
            {
                "category": category,
                "dataset_name": dataset_name,
                "dataset_dir": dataset_dir,
                "timeseries_path": timeseries_path,
                "labels_path": labels_path,
            }
        )

    if not datasets:
        raise FileNotFoundError(f"No prepared UCR datasets found under {prepared_dir}")
    return datasets


def category_output_dir(run_output_dir: Path, category: str) -> Path:
    return run_output_dir / category


def process_one(
    dataset_info: dict[str, Path | str],
    args: argparse.Namespace,
    order: tuple[int, int, int],
    run_config: dict[str, str | Path],
) -> dict[str, str | int | float]:
    category = str(dataset_info["category"])
    dataset_name = str(dataset_info["dataset_name"])
    timeseries_path = Path(dataset_info["timeseries_path"])
    labels_path = Path(dataset_info["labels_path"])
    preprocessing_mode = str(run_config["preprocessing_mode"])
    run_output_dir = Path(run_config["run_output_dir"])

    raw_df = read_timeseries(timeseries_path)
    labels = read_labels(labels_path)
    if labels is None or "is_anomaly" not in labels.columns:
        raise ValueError(f"Labels file is missing 'is_anomaly': {labels_path}")

    processed_df, preprocessing = preprocess_stationary_series(
        raw_df,
        period_method=args.period_method,
        fallback_window_size=args.window_size,
        fallback_stride=args.stride,
        preprocessing_mode=preprocessing_mode,
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

    output_dir = category_output_dir(run_output_dir, category)
    sigma2_path = output_dir / f"{dataset_name}_sigma2_values.csv"
    plot_path = output_dir / f"{dataset_name}_combined.png"
    output_dir.mkdir(parents=True, exist_ok=True)
    window_results.to_csv(sigma2_path)

    anomaly_positions = labels["is_anomaly"].to_numpy(dtype=int)
    if anomaly_positions.sum() == 0:
        raise ValueError(f"Labels file does not contain anomaly points: {labels_path}")

    anomaly_start = int(anomaly_positions.argmax())
    anomaly_end = int(len(anomaly_positions) - 1 - anomaly_positions[::-1].argmax())
    evaluation = compute_ucr_metric(
        window_results,
        anomaly_start=anomaly_start,
        anomaly_end=anomaly_end,
        reference_start_time=raw_df.index[0],
    )

    if not args.skip_plot:
        plot_combined(
            raw_df=raw_df,
            residual_series=residual_series,
            predictions=window_results,
            labels=labels,
            transformed_df=processed_df,
            raw_residual_series=raw_residual_series,
            raw_predictions=raw_window_results,
            title=f"UCR {category} {dataset_name}",
            output_path=plot_path,
        )

    return {
        "preprocessing_mode": preprocessing_mode,
        "category": category,
        "dataset_name": dataset_name,
        "timeseries_path": str(timeseries_path),
        "labels_path": str(labels_path),
        "sigma2_path": str(sigma2_path),
        "figure_path": str(plot_path),
        "period_method": str(preprocessing["period_method"]),
        "estimated_period": preprocessing["estimated_period"],
        "seasonal_adjustment": bool(preprocessing["seasonal_adjustment"]),
        "window_size": int(preprocessing["window_size"]),
        "stride": int(preprocessing["stride"]),
        "order": str(order),
        "n_points": int(len(raw_df)),
        "n_windows": int(len(window_results)),
        "anomaly_start": anomaly_start,
        "anomaly_end": anomaly_end,
        "ucr_prediction_index": evaluation["ucr_prediction_index"],
        "ucr_prediction_score": evaluation["ucr_prediction_score"],
        "ucr_correct_start": evaluation["ucr_correct_start"],
        "ucr_correct_end": evaluation["ucr_correct_end"],
        "ucr_correct": evaluation["ucr_correct"],
    }


def summarize_category(metrics_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for category, group in metrics_df.groupby("category", dropna=False):
        rows.append(
            {
                "category": category,
                "n_datasets": int(len(group)),
                "n_correct": int(group["ucr_correct"].sum()),
                "ucr_accuracy": float(group["ucr_correct"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("category")


def summarize_overall(metrics_df: pd.DataFrame) -> pd.DataFrame:
    category_rows = summarize_category(metrics_df)
    overall_row = pd.DataFrame(
        [
            {
                "category": "overall",
                "n_datasets": int(len(metrics_df)),
                "n_correct": int(metrics_df["ucr_correct"].sum()),
                "ucr_accuracy": float(metrics_df["ucr_correct"].mean()),
            }
        ]
    )
    return pd.concat([category_rows, overall_row], ignore_index=True)


def write_summaries(
    metrics: list[dict[str, str | int | float]],
    run_output_dir: Path,
    overall_summary_output: Path,
) -> None:
    metrics_df = pd.DataFrame(metrics).sort_values(["category", "dataset_name"])

    for category, group in metrics_df.groupby("category", dropna=False):
        category_dir = category_output_dir(run_output_dir, str(category))
        category_dir.mkdir(parents=True, exist_ok=True)
        category_eval_path = category_dir / f"{category}_evaluation.csv"
        group.to_csv(category_eval_path, index=False)
    overall_summary_output.parent.mkdir(parents=True, exist_ok=True)
    summarize_overall(metrics_df).to_csv(overall_summary_output, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run sigma2 workflow on all prepared UCR datasets.")
    parser.add_argument("--prepared-dir", default="data/ucr/prepared")
    parser.add_argument("--run-name", help="Optional base name for parameter comparisons.")
    parser.add_argument("--run-output-dir", help="Optional parent folder for all run folders.")
    parser.add_argument("--overall-summary-output", help="Optional CSV path or parent folder for overall summaries.")
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
    parser.add_argument("--order", nargs=3, type=int, default=[1, 1, 1])
    parser.add_argument("--limit", type=int, help="Optional maximum number of datasets to process.")
    parser.add_argument("--skip-plot", action="store_true")
    args = parser.parse_args()

    order = parse_order(args.order)
    datasets = find_prepared_datasets(Path(args.prepared_dir))
    if args.limit is not None:
        datasets = datasets[: args.limit]
    run_config = build_run_config(args, order)

    print(f"Found {len(datasets)} prepared UCR datasets.")

    run_name = str(run_config["run_name"])
    run_output_dir = Path(run_config["run_output_dir"])
    overall_summary_output = Path(run_config["overall_summary_output"])

    print(f"Run name: {run_name}")
    if args.preprocessing_mode == "raw":
        print(f"Preprocessing: raw, window_size={args.window_size}, stride={args.stride}")
    else:
        print(f"Preprocessing: seasonal_diff, period_method={args.period_method}")
    print(f"Results folder: {run_output_dir}")

    all_metrics = []
    progress_label = "raw" if args.preprocessing_mode == "raw" else args.period_method
    for dataset_info in tqdm(datasets, desc=f"UCR {progress_label}", unit="dataset"):
        metrics = process_one(dataset_info, args, order, run_config)
        all_metrics.append(metrics)

    write_summaries(
        all_metrics,
        run_output_dir,
        overall_summary_output,
    )
    overall_summary = summarize_overall(pd.DataFrame(all_metrics))
    overall_row = overall_summary[overall_summary["category"] == "overall"].iloc[0]
    category_summary = summarize_category(pd.DataFrame(all_metrics))

    print(f"Saved overall summary to {overall_summary_output}")
    for _, row in category_summary.iterrows():
        category_name = str(row["category"])
        print(
            f"Saved category evaluation to "
            f"{category_output_dir(run_output_dir, category_name) / (category_name + '_evaluation.csv')}"
        )
    print(
        "Overall UCR:",
        f"n_datasets={int(overall_row['n_datasets'])},",
        f"n_correct={int(overall_row['n_correct'])},",
        f"ucr_accuracy={float(overall_row['ucr_accuracy']):.4f}",
    )


if __name__ == "__main__":
    main()
