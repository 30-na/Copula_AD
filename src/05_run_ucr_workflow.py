"""Run the shared sigma2 workflow on all prepared UCR datasets.

This script:
  1. reads prepared UCR time-series data and labels,
  2. runs the shared sigma2 detector on every dataset,
  3. saves one sigma2 results CSV and one combined plot per dataset,
  4. saves UCR evaluation summaries by dataset, by category, and overall.

Default outputs:
  - results/<run_name>/<category>/<dataset>_sigma2_values.csv
  - results/<run_name>/<category>/<dataset>_combined.png
  - results/<run_name>/<category>/<category>_evaluation.csv
  - results/<run_name>/overall_evaluation.csv
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
    read_labels,
    read_timeseries,
)


def make_run_name(args: argparse.Namespace, order: tuple[int, int, int]) -> str:
    arima_name = "".join(str(value) for value in order)
    return f"ucr_w{args.window_size}_s{args.stride}_arima{arima_name}"


def configure_output_paths(args: argparse.Namespace, order: tuple[int, int, int]) -> None:
    run_name = args.run_name or make_run_name(args, order)
    run_output_dir = Path(args.run_output_dir) if args.run_output_dir else Path("results") / run_name

    args.run_name = run_name
    args.run_output_dir = str(run_output_dir)
    if args.overall_summary_output is None:
        args.overall_summary_output = str(run_output_dir / "overall_evaluation.csv")


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


def process_one(dataset_info: dict[str, Path | str], args: argparse.Namespace, order: tuple[int, int, int]) -> dict[str, str | int | float]:
    category = str(dataset_info["category"])
    dataset_name = str(dataset_info["dataset_name"])
    timeseries_path = Path(dataset_info["timeseries_path"])
    labels_path = Path(dataset_info["labels_path"])

    df = read_timeseries(timeseries_path)
    labels = read_labels(labels_path)
    if labels is None or "is_anomaly" not in labels.columns:
        raise ValueError(f"Labels file is missing 'is_anomaly': {labels_path}")

    sigma2_df, residual_series = compute_sigma2(
        df,
        window_size=args.window_size,
        stride=args.stride,
        order=order,
    )
    window_results = build_window_results(sigma2_df, labels)

    output_dir = category_output_dir(Path(args.run_output_dir), category)
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
    )

    if not args.skip_plot:
        plot_combined(
            raw_df=df,
            residual_series=residual_series,
            predictions=window_results,
            labels=labels,
            title=f"UCR {category} {dataset_name}",
            output_path=plot_path,
        )

    return {
        "category": category,
        "dataset_name": dataset_name,
        "timeseries_path": str(timeseries_path),
        "labels_path": str(labels_path),
        "sigma2_path": str(sigma2_path),
        "figure_path": str(plot_path),
        "window_size": args.window_size,
        "stride": args.stride,
        "order": str(order),
        "n_points": int(len(df)),
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


def write_summaries(metrics: list[dict[str, str | int | float]], args: argparse.Namespace) -> None:
    metrics_df = pd.DataFrame(metrics).sort_values(["category", "dataset_name"])
    overall_summary_path = Path(args.overall_summary_output)
    overall_summary_path.parent.mkdir(parents=True, exist_ok=True)

    for category, group in metrics_df.groupby("category", dropna=False):
        category_dir = category_output_dir(Path(args.run_output_dir), str(category))
        category_dir.mkdir(parents=True, exist_ok=True)
        category_eval_path = category_dir / f"{category}_evaluation.csv"
        group.to_csv(category_eval_path, index=False)

    summarize_overall(metrics_df).to_csv(overall_summary_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run sigma2 workflow on all prepared UCR datasets.")
    parser.add_argument("--prepared-dir", default="data/ucr/prepared")
    parser.add_argument("--run-name", help="Optional run folder name for parameter comparisons.")
    parser.add_argument("--run-output-dir", help="Optional output folder for this run.")
    parser.add_argument("--overall-summary-output", help="Optional overall summary CSV path.")
    parser.add_argument("--window-size", type=int, default=100)
    parser.add_argument("--stride", type=int, default=50)
    parser.add_argument("--order", nargs=3, type=int, default=[1, 1, 1])
    parser.add_argument("--limit", type=int, help="Optional maximum number of datasets to process.")
    parser.add_argument("--skip-plot", action="store_true")
    args = parser.parse_args()

    order = parse_order(args.order)
    configure_output_paths(args, order)
    datasets = find_prepared_datasets(Path(args.prepared_dir))
    if args.limit is not None:
        datasets = datasets[: args.limit]

    print(f"Found {len(datasets)} prepared UCR datasets.")
    print(f"Run name: {args.run_name}")
    print(f"Results folder: {args.run_output_dir}")

    all_metrics = []
    for dataset_info in tqdm(datasets, desc="UCR datasets", unit="dataset"):
        metrics = process_one(dataset_info, args, order)
        all_metrics.append(metrics)

    write_summaries(all_metrics, args)
    overall_summary = summarize_overall(pd.DataFrame(all_metrics))
    overall_row = overall_summary[overall_summary["category"] == "overall"].iloc[0]
    category_summary = summarize_category(pd.DataFrame(all_metrics))

    print(f"Saved overall summary to {args.overall_summary_output}")
    for _, row in category_summary.iterrows():
        print(
            f"Saved category evaluation to "
            f"{category_output_dir(Path(args.run_output_dir), str(row['category'])) / (str(row['category']) + '_evaluation.csv')}"
        )
    print(
        "Overall UCR:",
        f"n_datasets={int(overall_row['n_datasets'])},",
        f"n_correct={int(overall_row['n_correct'])},",
        f"ucr_accuracy={float(overall_row['ucr_accuracy']):.4f}",
    )


if __name__ == "__main__":
    main()
