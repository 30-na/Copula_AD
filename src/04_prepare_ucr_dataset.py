"""Download and prepare all UCR anomaly datasets for the sigma2 workflow.

This script:
  1. downloads the raw UCR zip file into data/,
  2. downloads the anomaly category CSV from the benchmark repo,
  3. extracts the raw .txt files,
  4. converts each UCR file into one data CSV and one labels CSV,
  5. groups prepared datasets by anomaly category.

Prepared output structure:
  - data/raw/ucr/UCR_TimeSeriesAnomalyDatasets2021.zip
  - data/raw/ucr/anomaly_types.csv
  - data/raw/ucr/extracted/
  - data/ucr/prepared/by_category/<category>/<dataset>/timeseries.csv
  - data/ucr/prepared/by_category/<category>/<dataset>/labels.csv
"""

from __future__ import annotations

import argparse
import re
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_UCR_URL = (
    "https://www.cs.ucr.edu/~eamonn/time_series_data_2018/"
    "UCR_TimeSeriesAnomalyDatasets2021.zip"
)
DEFAULT_CATEGORY_URL = (
    "https://gitlab.com/dlr-dw/is-it-worth-it-benchmark/-/raw/main/"
    "data/anomaly_types.csv"
)


def download_file(url: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        print(f"Using existing download: {output_path}")
        return

    print(f"Downloading {url}")
    urllib.request.urlretrieve(url, output_path)
    print(f"Saved download to {output_path}")


def extract_zip(zip_path: Path, extract_dir: Path) -> None:
    extract_dir.mkdir(parents=True, exist_ok=True)
    marker = extract_dir / ".extracted"
    if marker.exists():
        print(f"Using existing extracted folder: {extract_dir}")
        return

    print(f"Extracting {zip_path}")
    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(extract_dir)
    marker.write_text("done", encoding="utf-8")


def find_ucr_files(extract_dir: Path) -> list[Path]:
    files = sorted(extract_dir.rglob("*UCR_Anomaly*.txt"))
    if not files:
        raise FileNotFoundError(f"No UCR anomaly .txt files found under {extract_dir}")
    return files


def parse_ucr_metadata(path: Path) -> dict[str, int | str]:
    match = re.match(r"^(\d+)_UCR_Anomaly_(.+)_(\d+)_(\d+)_(\d+)\.txt$", path.name)
    if not match:
        raise ValueError(f"Could not parse UCR metadata from file name: {path.name}")

    dataset_id, dataset_name, train_end, anomaly_start, anomaly_end = match.groups()
    return {
        "dataset_id": int(dataset_id),
        "dataset_name": dataset_name,
        "train_end": int(train_end),
        "anomaly_start": int(anomaly_start),
        "anomaly_end": int(anomaly_end),
    }


def read_ucr_series(path: Path) -> pd.DataFrame:
    values = np.loadtxt(path, dtype=float)
    index = pd.date_range("2026-01-01", periods=len(values), freq="s", name="time")
    return pd.DataFrame({"value": values}, index=index)


def make_labels(
    index: pd.Index,
    anomaly_start: int,
    anomaly_end: int,
    anomaly_category: str,
) -> pd.DataFrame:
    labels = pd.DataFrame({"is_anomaly": 0}, index=index)
    labels.index.name = "time"

    start = max(anomaly_start, 0)
    end = min(anomaly_end, len(labels) - 1)

    labels.iloc[start : end + 1, labels.columns.get_loc("is_anomaly")] = 1
    labels["anomaly_type"] = np.where(labels["is_anomaly"] == 1, anomaly_category, "normal")
    return labels


def read_category_map(path: Path) -> dict[str, str]:
    categories = pd.read_csv(path, sep=";")
    required_columns = {"name", "anomaly_type_2"}
    missing = required_columns - set(categories.columns)
    if missing:
        raise ValueError(f"Category CSV is missing columns: {sorted(missing)}")
    return dict(zip(categories["name"], categories["anomaly_type_2"]))


def safe_folder_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")
    return cleaned or "unknown"


def dataset_folder_name(path: Path, metadata: dict[str, int | str]) -> str:
    dataset_id = int(metadata["dataset_id"])
    dataset_name = safe_folder_name(str(metadata["dataset_name"]))
    return f"ucr_{dataset_id:03d}_{dataset_name}"


def prepare_one_dataset(
    source_path: Path,
    prepared_dir: Path,
    category_by_file: dict[str, str],
) -> tuple[str, Path]:
    metadata = parse_ucr_metadata(source_path)
    category = category_by_file.get(source_path.name, "unknown")
    dataset_dir = (
        prepared_dir
        / "by_category"
        / safe_folder_name(category)
        / dataset_folder_name(source_path, metadata)
    )
    dataset_dir.mkdir(parents=True, exist_ok=True)

    df = read_ucr_series(source_path)
    labels = make_labels(
        df.index,
        anomaly_start=int(metadata["anomaly_start"]),
        anomaly_end=int(metadata["anomaly_end"]),
        anomaly_category=category,
    )

    timeseries_path = dataset_dir / "timeseries.csv"
    labels_path = dataset_dir / "labels.csv"
    df.to_csv(timeseries_path)
    labels.to_csv(labels_path)
    return category, dataset_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and prepare all UCR anomaly datasets.")
    parser.add_argument("--ucr-url", default=DEFAULT_UCR_URL)
    parser.add_argument("--category-url", default=DEFAULT_CATEGORY_URL)
    parser.add_argument("--zip-path", default="data/raw/ucr/UCR_TimeSeriesAnomalyDatasets2021.zip")
    parser.add_argument("--category-csv", default="data/raw/ucr/anomaly_types.csv")
    parser.add_argument("--extract-dir", default="data/raw/ucr/extracted")
    parser.add_argument("--prepared-dir", default="data/ucr/prepared")
    parser.add_argument("--limit", type=int, help="Optional maximum number of datasets to prepare.")
    args = parser.parse_args()

    zip_path = Path(args.zip_path)
    category_csv = Path(args.category_csv)
    extract_dir = Path(args.extract_dir)
    prepared_dir = Path(args.prepared_dir)

    download_file(args.ucr_url, zip_path)
    download_file(args.category_url, category_csv)
    extract_zip(zip_path, extract_dir)

    files = find_ucr_files(extract_dir)
    if args.limit is not None:
        files = files[: args.limit]

    category_by_file = read_category_map(category_csv)
    prepared_counts: dict[str, int] = {}

    for source_path in files:
        category, dataset_dir = prepare_one_dataset(source_path, prepared_dir, category_by_file)
        prepared_counts[category] = prepared_counts.get(category, 0) + 1
        print(f"Prepared {source_path.name} -> {dataset_dir}")

    print(f"Prepared {len(files)} UCR datasets in {prepared_dir}")
    for category in sorted(prepared_counts):
        print(f"{category}: {prepared_counts[category]}")


if __name__ == "__main__":
    main()
