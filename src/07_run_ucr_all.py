"""Run the sigma2 workflow for all unique UCR anomaly datasets.

For each dataset this script saves:
  - standardized series CSV
  - standardized labels CSV
  - sigma2 values CSV
  - sigma2 predictions CSV
  - metrics CSV
  - one combined figure with raw data and innovation variance estimate

Default outputs:
  - data/benchmark/ucr_all/
  - results/benchmark/ucr_all/
  - results/figures/ucr_all/
"""

from __future__ import annotations

import argparse
import importlib.util
import re
from pathlib import Path

import numpy as np
import pandas as pd


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
sigma2_mod.tqdm = lambda iterable, **kwargs: iterable


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


def estimate_total_windows(files: list[Path], window_size: int, overlap_size: int) -> int:
    step = window_size - overlap_size
    total = 0
    for path in files:
        n = len(np.loadtxt(path, dtype=float))
        total += (n - window_size) // step + 1 if n >= window_size else 0
    return total


def process_one(
    path: Path,
    args: argparse.Namespace,
    order: tuple[int, int, int],
) -> dict[str, float | int | str]:
    metadata = parse_ucr_metadata(path)
    dataset_id = int(metadata["dataset_id"])
    dataset_name = str(metadata["dataset_name"])
    stem = safe_name(dataset_id, dataset_name)

    series_path = Path(args.data_output_dir) / f"{stem}_series.csv"
    labels_path = Path(args.data_output_dir) / f"{stem}_labels.csv"
    sigma2_path = Path(args.results_output_dir) / f"{stem}_sigma2_values.csv"
    predictions_path = Path(args.results_output_dir) / f"{stem}_sigma2_predictions.csv"
    metrics_path = Path(args.results_output_dir) / f"{stem}_sigma2_metrics.csv"
    figure_path = Path(args.figures_output_dir) / f"{stem}_combined.png"

    if args.skip_existing and figure_path.exists() and metrics_path.exists():
        metrics = pd.read_csv(metrics_path).iloc[0].to_dict()
        metrics.update({"dataset_id": dataset_id, "dataset_name": dataset_name, "status": "skipped"})
        return metrics

    df = read_ucr_series(path)
    labels = make_labels(
        df.index,
        train_end=int(metadata["train_end"]),
        anomaly_start=int(metadata["anomaly_start"]),
        anomaly_end=int(metadata["anomaly_end"]),
    )

    series_path.parent.mkdir(parents=True, exist_ok=True)
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    sigma2_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    figure_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(series_path)
    labels.to_csv(labels_path)

    sigma2_df = sigma2_mod.compute_sigma2(
        df,
        window_size=args.window_size,
        overlap_size=args.overlap_size,
        order=order,
    )
    sigma2_df.to_csv(sigma2_path)

    predictions, threshold = eval_mod.compute_predictions(
        sigma2_df,
        labels,
        score_column=args.score_column,
        k=args.k,
    )
    metrics = eval_mod.compute_metrics(predictions)
    metrics.update(
        {
            "dataset_id": dataset_id,
            "dataset_name": dataset_name,
            "train_end": int(metadata["train_end"]),
            "anomaly_start": int(metadata["anomaly_start"]),
            "anomaly_end": int(metadata["anomaly_end"]),
            "threshold": threshold,
            "k": args.k,
            "window_size": args.window_size,
            "overlap_size": args.overlap_size,
            "order": str(order),
            "status": "processed",
        }
    )

    predictions.to_csv(predictions_path)
    pd.DataFrame([metrics]).to_csv(metrics_path, index=False)

    plot_mod.plot_combined(
        raw_df=df,
        predictions=predictions,
        labels=labels,
        title=f"UCR {dataset_id:03d} {dataset_name}",
        output_path=figure_path,
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Run sigma2 workflow on all UCR anomaly datasets.")
    parser.add_argument("--extract-dir", default="data/raw/ucr/extracted")
    parser.add_argument("--data-output-dir", default="data/benchmark/ucr_all")
    parser.add_argument("--results-output-dir", default="results/benchmark/ucr_all")
    parser.add_argument("--figures-output-dir", default="results/figures/ucr_all")
    parser.add_argument("--summary-output", default="results/benchmark/ucr_all_summary.csv")
    parser.add_argument("--window-size", type=int, default=300)
    parser.add_argument("--overlap-size", type=int, default=0)
    parser.add_argument("--order", nargs=3, type=int, default=[1, 0, 1])
    parser.add_argument("--score-column", default="sigma2_mean")
    parser.add_argument("--k", type=float, default=3.0)
    parser.add_argument("--limit", type=int, help="Optional maximum number of datasets to process.")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--estimate-only", action="store_true")
    args = parser.parse_args()

    files = find_unique_ucr_files(Path(args.extract_dir))
    if args.limit:
        files = files[: args.limit]

    total_windows = estimate_total_windows(files, args.window_size, args.overlap_size)
    print(f"Found {len(files)} unique UCR datasets.")
    print(f"Estimated ARIMA windows: {total_windows}")

    if args.estimate_only:
        return

    order = tuple(args.order)
    all_metrics = []
    for i, path in enumerate(files, start=1):
        metadata = parse_ucr_metadata(path)
        print(f"[{i}/{len(files)}] UCR {int(metadata['dataset_id']):03d} {metadata['dataset_name']}")
        metrics = process_one(path, args, order)
        all_metrics.append(metrics)

        summary_path = Path(args.summary_output)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(all_metrics).to_csv(summary_path, index=False)

    print(f"Saved summary to {args.summary_output}")


if __name__ == "__main__":
    main()
