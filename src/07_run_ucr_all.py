"""Run the sigma2 workflow for all unique UCR anomaly datasets.

For each dataset this script saves:
  - one combined figure with raw data, residuals, and estimated innovation variance
  - per-dataset sigma2 output from the fitted ARIMA windows
  - benchmark performance summaries

Default outputs:
  - results/figures/<run_name>/
  - results/benchmark/<run_name>/sigma2_values/
  - results/benchmark/<run_name>/ucr_all_summary.csv
  - results/benchmark/<run_name>/ucr_all_overall_summary.csv
  - results/benchmark/<run_name>/ucr_all_by_category_summary.csv
"""

from __future__ import annotations

import argparse
import importlib.util
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


sigma2_mod = load_module("sigma2_detection", ROOT / "src" / "03_sigma2_detection.py")
eval_mod = load_module("evaluate_sigma2", ROOT / "src" / "05_evaluate_sigma2.py")
plot_mod = load_module("plot_combined_result", ROOT / "src" / "06_plot_combined_result.py")


DEFAULT_CATEGORY_METADATA = (
    ROOT
    / "data"
    / "raw"
    / "ucr"
    / "is-it-worth-it-benchmark-main"
    / "is-it-worth-it-benchmark-main"
    / "data"
    / "anomaly_types.csv"
)


def parse_ucr_metadata(path: Path) -> dict[str, int | str]:
    match = re.match(r"^(\d+)_UCR_Anomaly_(.+)_(\d+)_(\d+)_(\d+)\.txt$", path.name)
    if not match:
        raise ValueError(f"Could not parse UCR metadata from {path.name}")
    dataset_id, dataset_name, train_end, anomaly_start, anomaly_end = match.groups()
    return {
        "dataset_id": int(dataset_id),
        "dataset_name": dataset_name,
        "train_end": int(train_end),
        "anomaly_start": int(anomaly_start),
        "anomaly_end": int(anomaly_end),
    }


def find_unique_ucr_files(extract_dir: Path) -> list[Path]:
    files = sorted(extract_dir.rglob("*UCR_Anomaly*.txt"), key=lambda path: path.name)
    by_id: dict[int, Path] = {}

    for path in files:
        metadata = parse_ucr_metadata(path)
        dataset_id = int(metadata["dataset_id"])
        if dataset_id not in by_id:
            by_id[dataset_id] = path
        elif "UCR_Anomaly_FullData" in str(path):
            by_id[dataset_id] = path

    return [by_id[key] for key in sorted(by_id)]


def load_category_metadata(path: Path | None) -> dict[str, str]:
    if path is None or not path.exists():
        return {}

    categories = pd.read_csv(path, sep=";")
    required_columns = {"name", "anomaly_type_2"}
    missing = required_columns - set(categories.columns)
    if missing:
        raise ValueError(f"Category metadata is missing columns: {sorted(missing)}")

    return dict(zip(categories["name"], categories["anomaly_type_2"]))


def read_ucr_series(path: Path) -> pd.DataFrame:
    values = np.loadtxt(path, dtype=float)
    index = pd.date_range("2026-01-01", periods=len(values), freq="s", name="time")
    return pd.DataFrame({"CH1": values}, index=index)


def make_labels(index: pd.Index, train_end: int, anomaly_start: int, anomaly_end: int) -> pd.DataFrame:
    labels = pd.DataFrame({"is_anomaly": 0, "is_train": 0}, index=index)
    labels.index.name = "time"
    labels.iloc[:train_end, labels.columns.get_loc("is_train")] = 1
    labels.iloc[anomaly_start : anomaly_end + 1, labels.columns.get_loc("is_anomaly")] = 1
    labels["anomaly_type"] = np.where(labels["is_anomaly"] == 1, "ucr_labeled_anomaly", "normal")
    return labels


def safe_name(dataset_id: int, dataset_name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", dataset_name)
    return f"ucr_{dataset_id:03d}_{cleaned}"


def safe_folder_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")
    return cleaned or "unknown"


def make_run_name(args: argparse.Namespace, order: tuple[int, int, int]) -> str:
    arima_name = "".join(str(value) for value in order)
    if args.score_mode == "pointwise":
        spacing = f"s{args.stride}"
    else:
        spacing = f"overlap{args.overlap_size}"
    return f"ucr_{args.score_mode}_w{args.window_size}_{spacing}_arima{arima_name}"


def configure_output_paths(args: argparse.Namespace, order: tuple[int, int, int]) -> None:
    run_name = args.run_name or make_run_name(args, order)
    run_output_dir = Path(args.run_output_dir) if args.run_output_dir else Path("results") / "benchmark" / run_name

    args.run_name = run_name
    args.run_output_dir = str(run_output_dir)
    args.sigma2_output_dir = str(run_output_dir / "sigma2_values")

    if args.figures_output_dir is None:
        args.figures_output_dir = str(Path("results") / "figures" / run_name)
    if args.summary_output is None:
        args.summary_output = str(run_output_dir / "ucr_all_summary.csv")
    if args.overall_summary_output is None:
        args.overall_summary_output = str(run_output_dir / "ucr_all_overall_summary.csv")
    if args.category_summary_output is None:
        args.category_summary_output = str(run_output_dir / "ucr_all_by_category_summary.csv")


def estimate_total_windows(
    files: list[Path],
    window_size: int,
    overlap_size: int,
    score_mode: str,
    stride: int,
) -> int:
    step = stride if score_mode == "pointwise" else window_size - overlap_size
    total = 0
    for path in files:
        n = len(np.loadtxt(path, dtype=float))
        if n < window_size:
            continue
        count = (n - window_size) // step + 1
        if score_mode == "pointwise" and (n - window_size) % step != 0:
            count += 1
        total += count
    return total


def process_one(
    path: Path,
    args: argparse.Namespace,
    order: tuple[int, int, int],
    category_by_file: dict[str, str],
    existing_metrics: pd.DataFrame | None = None,
) -> dict[str, float | int | str]:
    metadata = parse_ucr_metadata(path)
    dataset_id = int(metadata["dataset_id"])
    dataset_name = str(metadata["dataset_name"])
    stem = safe_name(dataset_id, dataset_name)
    category = category_by_file.get(path.name, "unknown")
    category_slug = safe_folder_name(category)

    figure_path = Path(args.figures_output_dir) / "overall" / f"{stem}_combined.png"
    category_figure_path = (
        Path(args.figures_output_dir)
        / "by_category"
        / category_slug
        / f"{stem}_combined.png"
    )
    sigma2_path = Path(args.sigma2_output_dir) / f"{stem}_sigma2_values.csv"

    if args.skip_existing and figure_path.exists() and existing_metrics is not None:
        existing = existing_metrics[existing_metrics["dataset_id"] == dataset_id]
        if not existing.empty:
            metrics = existing.iloc[0].to_dict()
            metrics.update({"status": "skipped"})
            return metrics

    if args.skip_existing and figure_path.exists() and existing_metrics is None:
        metrics = {
            "dataset_id": dataset_id,
            "dataset_name": dataset_name,
            "source_file": path.name,
            "anomaly_type_category": category,
            "status": "skipped_no_metrics",
        }
        return metrics

    df = read_ucr_series(path)
    labels = make_labels(
        df.index,
        train_end=int(metadata["train_end"]),
        anomaly_start=int(metadata["anomaly_start"]),
        anomaly_end=int(metadata["anomaly_end"]),
    )

    figure_path.parent.mkdir(parents=True, exist_ok=True)

    if args.score_mode == "pointwise":
        sigma2_df = sigma2_mod.compute_sigma2_pointwise(
            df,
            window_size=args.window_size,
            stride=args.stride,
            order=order,
        )
    else:
        sigma2_df = sigma2_mod.compute_sigma2(
            df,
            window_size=args.window_size,
            overlap_size=args.overlap_size,
            order=order,
        )

    if not args.no_save_sigma2:
        sigma2_path.parent.mkdir(parents=True, exist_ok=True)
        sigma2_df.to_csv(sigma2_path)

    predictions, threshold = eval_mod.compute_predictions(
        sigma2_df,
        labels,
        score_column=args.score_column,
        k=args.k,
    )
    metrics = eval_mod.compute_metrics(predictions)
    metrics.update(
        eval_mod.compute_ucr_metric(
            sigma2_df,
            score_column=args.score_column,
            anomaly_start=int(metadata["anomaly_start"]),
            anomaly_end=int(metadata["anomaly_end"]),
        )
    )
    metrics.update(
        {
            "dataset_id": dataset_id,
            "dataset_name": dataset_name,
            "source_file": path.name,
            "anomaly_type_category": category,
            "train_end": int(metadata["train_end"]),
            "anomaly_start": int(metadata["anomaly_start"]),
            "anomaly_end": int(metadata["anomaly_end"]),
            "threshold": threshold,
            "k": args.k,
            "window_size": args.window_size,
            "overlap_size": args.overlap_size,
            "score_mode": args.score_mode,
            "stride": args.stride,
            "order": str(order),
            "figure_path": str(figure_path),
            "category_figure_path": str(category_figure_path),
            "status": "processed",
        }
    )

    if not args.skip_figures:
        plot_mod.plot_combined(
            raw_df=df,
            predictions=predictions,
            labels=labels,
            title=f"UCR {dataset_id:03d} {dataset_name}",
            output_path=figure_path,
        )
        category_figure_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(figure_path, category_figure_path)

    return metrics


def load_existing_summary(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None

    summary = pd.read_csv(path)
    if "dataset_id" not in summary.columns:
        return None
    return summary


def summarize_metrics(metrics_df: pd.DataFrame) -> pd.DataFrame:
    if "status" in metrics_df.columns:
        metrics_df = metrics_df[metrics_df["status"] != "skipped_no_metrics"].copy()
    numeric_columns = [
        column
        for column in metrics_df.select_dtypes(include="number").columns
        if column
        not in {
            "dataset_id",
            "train_end",
            "anomaly_start",
            "anomaly_end",
            "ucr_prediction_index",
            "ucr_correct_start",
            "ucr_correct_end",
        }
    ]
    summary = metrics_df[numeric_columns].mean(numeric_only=True).to_dict()
    summary["n_datasets"] = int(len(metrics_df))
    summary["n_processed"] = int((metrics_df.get("status", "") == "processed").sum())
    summary["n_skipped"] = int((metrics_df.get("status", "") == "skipped").sum())
    return pd.DataFrame([summary])


def summarize_metrics_by_category(metrics_df: pd.DataFrame) -> pd.DataFrame:
    if "anomaly_type_category" not in metrics_df.columns:
        return pd.DataFrame()

    category_rows = []
    for category, group in metrics_df.groupby("anomaly_type_category", dropna=False):
        row = summarize_metrics(group).iloc[0].to_dict()
        row["anomaly_type_category"] = category
        category_rows.append(row)

    if not category_rows:
        return pd.DataFrame()
    return pd.DataFrame(category_rows).sort_values("anomaly_type_category")


def write_summaries(metrics: list[dict[str, float | int | str]], args: argparse.Namespace) -> None:
    metrics_df = pd.DataFrame(metrics)

    summary_path = Path(args.summary_output)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(summary_path, index=False)

    overall_path = Path(args.overall_summary_output)
    overall_path.parent.mkdir(parents=True, exist_ok=True)
    summarize_metrics(metrics_df).to_csv(overall_path, index=False)

    category_summary = summarize_metrics_by_category(metrics_df)
    if not category_summary.empty:
        category_path = Path(args.category_summary_output)
        category_path.parent.mkdir(parents=True, exist_ok=True)
        category_summary.to_csv(category_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run sigma2 workflow on all UCR anomaly datasets.")
    parser.add_argument("--extract-dir", default="data/raw/ucr/extracted")
    parser.add_argument("--run-name", help="Optional run folder name for parameter comparisons.")
    parser.add_argument("--run-output-dir", help="Optional benchmark output folder for this run.")
    parser.add_argument("--figures-output-dir", help="Optional figures output folder for this run.")
    parser.add_argument("--summary-output", help="Optional per-dataset summary CSV path.")
    parser.add_argument("--overall-summary-output", help="Optional overall summary CSV path.")
    parser.add_argument("--category-summary-output", help="Optional category summary CSV path.")
    parser.add_argument("--category-metadata", default=str(DEFAULT_CATEGORY_METADATA))
    parser.add_argument("--window-size", type=int, default=300)
    parser.add_argument("--overlap-size", type=int, default=0)
    parser.add_argument("--score-mode", choices=["window", "pointwise"], default="pointwise")
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--order", nargs=3, type=int, default=[1, 0, 1])
    parser.add_argument("--score-column", default="sigma2_mean")
    parser.add_argument("--k", type=float, default=3.0)
    parser.add_argument("--limit", type=int, help="Optional maximum number of datasets to process.")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--skip-figures", action="store_true", help="Compute performance only; do not save figures.")
    parser.add_argument("--no-save-sigma2", action="store_true", help="Do not save per-dataset sigma2 output CSVs.")
    parser.add_argument("--estimate-only", action="store_true")
    args = parser.parse_args()

    order = tuple(args.order)
    configure_output_paths(args, order)

    files = find_unique_ucr_files(Path(args.extract_dir))
    if args.limit:
        files = files[: args.limit]

    total_windows = estimate_total_windows(
        files,
        args.window_size,
        args.overlap_size,
        args.score_mode,
        args.stride,
    )
    print(f"Found {len(files)} unique UCR datasets.")
    print(f"Estimated ARIMA windows: {total_windows}")
    print(f"Run name: {args.run_name}")
    print(f"Benchmark output: {args.run_output_dir}")
    if not args.no_save_sigma2:
        print(f"Sigma2 output: {args.sigma2_output_dir}")
    if not args.skip_figures:
        print(f"Figures output: {args.figures_output_dir}")

    if args.estimate_only:
        return

    category_by_file = load_category_metadata(Path(args.category_metadata) if args.category_metadata else None)
    if category_by_file:
        print(f"Loaded anomaly categories for {len(category_by_file)} UCR files.")
    else:
        print("No anomaly category metadata loaded.")

    existing_metrics = load_existing_summary(Path(args.summary_output)) if args.skip_existing else None
    all_metrics = []
    for i, path in enumerate(tqdm(files, desc="Datasets", unit="dataset"), start=1):
        metadata = parse_ucr_metadata(path)
        tqdm.write(f"[{i}/{len(files)}] UCR {int(metadata['dataset_id']):03d} {metadata['dataset_name']}")
        metrics = process_one(path, args, order, category_by_file, existing_metrics)
        all_metrics.append(metrics)
        write_summaries(all_metrics, args)

    print(f"Saved summary to {args.summary_output}")
    print(f"Saved overall summary to {args.overall_summary_output}")
    print(f"Saved category summary to {args.category_summary_output}")
    if all_metrics and "ucr_correct" in all_metrics[0]:
        ucr_accuracy = pd.DataFrame(all_metrics)["ucr_correct"].mean()
        print(f"UCR accuracy: {ucr_accuracy:.4f}")


if __name__ == "__main__":
    main()
