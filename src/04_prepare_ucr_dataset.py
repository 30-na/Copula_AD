"""Download and prepare one UCR anomaly benchmark series.

UCR anomaly files encode the labeled anomaly interval in the file name:

    ..._<train_end>_<anomaly_start>_<anomaly_end>.txt

This script converts one selected UCR series into the same format used by the
simulated data scripts:

Outputs:
  - data/benchmark/ucr/ucr_series.csv
  - data/benchmark/ucr/ucr_labels.csv
  - results/figures/ucr_series.png
"""

from __future__ import annotations

import argparse
import os
import re
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".matplotlib"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_UCR_URL = (
    "https://www.cs.ucr.edu/~eamonn/time_series_data_2018/"
    "UCR_TimeSeriesAnomalyDatasets2021.zip"
)


def download_file(url: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        print(f"Using existing download: {output_path}")
        return

    print(f"Downloading {url}")
    urllib.request.urlretrieve(url, output_path)
    print(f"Saved archive to {output_path}")


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


def parse_ucr_metadata(path: Path) -> tuple[int, int, int]:
    match = re.search(r"_(\d+)_(\d+)_(\d+)\.txt$", path.name)
    if not match:
        raise ValueError(f"Could not parse UCR metadata from file name: {path.name}")
    train_end, anomaly_start, anomaly_end = (int(value) for value in match.groups())
    return train_end, anomaly_start, anomaly_end


def read_ucr_series(path: Path) -> pd.DataFrame:
    values = np.loadtxt(path, dtype=float)
    index = pd.date_range("2026-01-01", periods=len(values), freq="s", name="time")
    return pd.DataFrame({"CH1": values}, index=index)


def make_labels(index: pd.Index, train_end: int, anomaly_start: int, anomaly_end: int) -> pd.DataFrame:
    labels = pd.DataFrame({"is_anomaly": 0, "is_train": 0}, index=index)
    labels.index.name = "time"

    start = max(anomaly_start, 0)
    end = min(anomaly_end, len(labels) - 1)
    labels.iloc[start : end + 1, labels.columns.get_loc("is_anomaly")] = 1
    labels.iloc[:train_end, labels.columns.get_loc("is_train")] = 1
    labels["anomaly_type"] = np.where(labels["is_anomaly"] == 1, "ucr_labeled_anomaly", "normal")
    return labels


def plot_series(df: pd.DataFrame, labels: pd.DataFrame, output_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df.index, df["CH1"], color="black", linewidth=0.8)

    anomaly_times = labels.index[labels["is_anomaly"] == 1]
    if len(anomaly_times) > 0:
        ax.axvspan(anomaly_times.min(), anomaly_times.max(), color="red", alpha=0.15)

    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("CH1")
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def select_file(files: list[Path], dataset_name: str | None, file_index: int) -> Path:
    if dataset_name:
        matches = [path for path in files if dataset_name.lower() in path.name.lower()]
        if not matches:
            raise ValueError(f"No UCR file name contains: {dataset_name}")
        return matches[0]

    if file_index < 0 or file_index >= len(files):
        raise ValueError(f"file_index must be between 0 and {len(files) - 1}")
    return files[file_index]


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare one UCR anomaly benchmark dataset.")
    parser.add_argument("--url", default=DEFAULT_UCR_URL)
    parser.add_argument("--zip-path", default="data/raw/ucr/UCR_TimeSeriesAnomalyDatasets2021.zip")
    parser.add_argument("--extract-dir", default="data/raw/ucr/extracted")
    parser.add_argument("--dataset-name", help="Optional text to match in the UCR file name.")
    parser.add_argument("--file-index", type=int, default=0)
    parser.add_argument("--output", default="data/benchmark/ucr/ucr_series.csv")
    parser.add_argument("--labels-output", default="data/benchmark/ucr/ucr_labels.csv")
    parser.add_argument("--plot", default="results/figures/ucr_series.png")
    args = parser.parse_args()

    zip_path = Path(args.zip_path)
    extract_dir = Path(args.extract_dir)
    download_file(args.url, zip_path)
    extract_zip(zip_path, extract_dir)

    files = find_ucr_files(extract_dir)
    selected_file = select_file(files, args.dataset_name, args.file_index)
    train_end, anomaly_start, anomaly_end = parse_ucr_metadata(selected_file)

    df = read_ucr_series(selected_file)
    labels = make_labels(df.index, train_end, anomaly_start, anomaly_end)

    output_path = Path(args.output)
    labels_path = Path(args.labels_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path)
    labels.to_csv(labels_path)

    title = f"UCR Benchmark: {selected_file.stem}"
    plot_series(df, labels, Path(args.plot), title)

    print(f"Selected UCR file: {selected_file}")
    print(f"train_end={train_end}, anomaly_start={anomaly_start}, anomaly_end={anomaly_end}")
    print(f"Saved benchmark series to {output_path}")
    print(f"Saved labels to {labels_path}")
    print(f"Saved plot to {args.plot}")


if __name__ == "__main__":
    main()
