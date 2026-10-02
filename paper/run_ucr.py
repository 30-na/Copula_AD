"""Apply Algorithm 1 to real data: the UCR anomaly benchmark, one category
at a time (default: "noise"). Same plain algorithm as run_simulation.py --
no train-A/train-B split, no period estimation, no seasonal handling --
just: fit the reference model on the labeled anomaly-free training region,
then score the rest in fixed windows (algorithm.run_algorithm).

Each UCR file's clean/anomaly split point ("train_end") is not stored in
timeseries.csv/labels.csv -- it's only in the original raw filename
(<idx>_UCR_Anomaly_<name>_<train_end>_<anomaly_start>_<anomaly_end>.txt),
so it's recovered here by matching the prepared folder's <idx> back to
that raw file.

No raw-variance baseline here (that comparison was for the simulation
demo) -- just the raw series and the innovation variance ratio.

Outputs, under --output-dir (default results/ucr/<category>):
  - evaluation_table.csv  one row per dataset: order, sigma0^2, detection
                          rate, false-alarm rate, etc.
  - figures/<dataset>.png two panels per dataset: (a) raw series with the
                          true anomaly shaded, (b) innovation variance
                          ratio with two-sided control limits at alpha
"""

from __future__ import annotations

import argparse
import re
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from algorithm import run_algorithm

COLOR_INK = "#0b0b0b"
COLOR_MUTED = "#6b6a65"
COLOR_LINE = "#1a1a1a"
COLOR_LIMIT = "#8a8a8a"
COLOR_ANOMALY_SPAN = "#c9c7c0"

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
        "font.size": 10.5,
        "axes.labelsize": 10.5,
        "axes.titlesize": 10.5,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "legend.fontsize": 9.5,
        "axes.linewidth": 0.8,
    }
)


def _style_axis(ax, hide_xticks: bool = False) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(COLOR_MUTED)
    ax.spines["bottom"].set_color(COLOR_MUTED)
    ax.tick_params(colors=COLOR_MUTED)
    if hide_xticks:
        ax.set_xticks([])


def find_train_end(raw_dir: Path, dataset_folder_name: str) -> int:
    """Recover train_end from the original raw filename.

    Args:
        raw_dir: directory holding the raw <idx>_UCR_Anomaly_...txt files.
        dataset_folder_name: prepared folder name, e.g. "ucr_027_DISTORTEDInternalBleeding16".

    Returns:
        train_end (int): the first index of the raw filename's three
        trailing numbers (train_end, anomaly_start, anomaly_end).
    """
    m = re.match(r"ucr_(\d+)_", dataset_folder_name)
    if not m:
        raise ValueError(f"Could not parse an index from {dataset_folder_name!r}.")
    idx = m.group(1)
    matches = list(raw_dir.glob(f"{idx}_UCR_Anomaly_*.txt"))
    if not matches:
        raise FileNotFoundError(f"No raw file found for index {idx} in {raw_dir}.")
    m2 = re.search(r"_(\d+)_(\d+)_(\d+)\.txt$", matches[0].name)
    if not m2:
        raise ValueError(f"Could not parse train_end from {matches[0].name!r}.")
    return int(m2.group(1))


def f_critical_values(window_size: int, reference_df: int, alpha: float) -> tuple[float, float]:
    return (
        float(stats.f.ppf(alpha / 2, window_size, reference_df)),
        float(stats.f.ppf(1 - alpha / 2, window_size, reference_df)),
    )


def plot_dataset_figure(
    test_values: np.ndarray,
    test_is_anomaly: np.ndarray,
    results: pd.DataFrame,
    model: dict,
    window_size: int,
    alpha: float,
    dataset_name: str,
    output_path: Path,
) -> None:
    """Two panels: (a) raw series with the true anomaly shaded, (b) the
    innovation variance ratio with its own two-sided control limits.
    """
    x_series = np.arange(len(test_values))
    anomaly_positions = np.where(test_is_anomaly == 1)[0]
    anomaly_start_pos = int(anomaly_positions.min()) if len(anomaly_positions) else None
    anomaly_end_pos = int(anomaly_positions.max()) if len(anomaly_positions) else None
    x = results["window_start"] + window_size // 2

    fig, axes = plt.subplots(2, 1, figsize=(8, 4.7), height_ratios=(1, 1.1))

    ax = axes[0]
    _style_axis(ax, hide_xticks=True)
    ax.plot(x_series, test_values, color=COLOR_INK, linewidth=0.8)
    if anomaly_start_pos is not None:
        ax.axvspan(anomaly_start_pos, anomaly_end_pos, color=COLOR_ANOMALY_SPAN, alpha=0.6, linewidth=0, label="Anomaly period")
    y_min, y_max = ax.get_ylim()
    ax.set_ylim(y_min, y_max + 0.18 * (y_max - y_min))
    ax.set_ylabel("Value")
    ax.set_title(f"(a) {dataset_name}", loc="left", fontsize=10.5, color=COLOR_INK)
    ax.legend(loc="upper right", fontsize=9, frameon=True, facecolor="white", edgecolor="none")

    ax = axes[1]
    _style_axis(ax, hide_xticks=True)
    if anomaly_start_pos is not None:
        ax.axvspan(anomaly_start_pos, anomaly_end_pos, color=COLOR_ANOMALY_SPAN, alpha=0.6, linewidth=0)
    ax.axhline(1.0, color=COLOR_MUTED, linewidth=1.0, linestyle=":")
    lower, upper = f_critical_values(window_size, model["reference_df"], alpha)
    ax.axhline(lower, color=COLOR_LIMIT, linewidth=1.1, linestyle="--", label=f"control limits (alpha={alpha})")
    ax.axhline(upper, color=COLOR_LIMIT, linewidth=1.1, linestyle="--")
    significant = results["p_value"] < alpha
    ratio = results["variance_ratio"]
    ax.plot(x, ratio, color=COLOR_LINE, linewidth=1.3, zorder=3, label="Innovation variance ratio")
    ax.scatter(x[significant], ratio[significant], s=18, color=COLOR_LINE, zorder=4)
    ax.scatter(x[~significant], ratio[~significant], s=18, facecolors="white", edgecolors=COLOR_LINE, linewidths=1.0, zorder=4)
    ax.set_ylabel("Variance Ratio")
    ax.set_xlabel("Time")
    ax.set_title("(b) Innovation Variance Ratio", loc="left", fontsize=10.5, color=COLOR_INK)
    ax.legend(loc="upper right", fontsize=8.5, frameon=False)

    fig.tight_layout()
    fig.subplots_adjust(right=1.0 - fig.subplotpars.left)  # center the plot area
    output_path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(3):
        try:
            fig.savefig(output_path, dpi=220)
            break
        except OSError:
            if attempt == 2:
                raise
            time.sleep(0.5)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--category", default="noise", help="UCR anomaly category subfolder to run.")
    parser.add_argument("--data-dir", default="data/ucr_prepared")
    parser.add_argument("--raw-dir", default="data/ucr_raw")
    parser.add_argument("--window-size", type=int, default=100)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--max-p", type=int, default=3)
    parser.add_argument("--max-q", type=int, default=3)
    parser.add_argument("--limit", type=int, default=None, help="Only run the first N datasets (for quick testing).")
    parser.add_argument("--output-dir", default=None, help="Defaults to results/ucr/<category>.")
    args = parser.parse_args()

    data_dir = Path(args.data_dir) / args.category
    raw_dir = Path(args.raw_dir)
    output_dir = Path(args.output_dir) if args.output_dir else Path("results/ucr") / args.category
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_dirs = sorted(p for p in data_dir.iterdir() if p.is_dir())
    if args.limit is not None:
        dataset_dirs = dataset_dirs[: args.limit]

    rows = []
    for dataset_dir in dataset_dirs:
        name = dataset_dir.name
        print(f"--- {name} ---")
        try:
            train_end = find_train_end(raw_dir, name)
            ts = pd.read_csv(dataset_dir / "timeseries.csv")
            labels = pd.read_csv(dataset_dir / "labels.csv")
            values = ts["value"].to_numpy(dtype=float)
            is_anomaly = labels["is_anomaly"].to_numpy(dtype=int)

            if is_anomaly[:train_end].sum() > 0:
                print(f"  skipped: training segment (0:{train_end}) is not anomaly-free")
                rows.append({"dataset": name, "status": "skipped_dirty_training"})
                continue

            train_values = values[:train_end]
            test_values = values[train_end:]
            test_is_anomaly = is_anomaly[train_end:]

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                results, model, _ = run_algorithm(
                    train_values, test_values, args.window_size, args.alpha, args.max_p, args.max_q
                )

            actual_anomaly = []
            for _, row in results.iterrows():
                start, end = int(row["window_start"]), int(row["window_end"])
                actual_anomaly.append(int(test_is_anomaly[start:end].any()))
            results["actual_anomaly"] = actual_anomaly

            n_true = int((results["actual_anomaly"] == 1).sum())
            n_normal = int((results["actual_anomaly"] == 0).sum())
            n_detected = int(((results["actual_anomaly"] == 1) & (results["p_value"] < args.alpha)).sum())
            n_false_alarms = int(((results["actual_anomaly"] == 0) & (results["p_value"] < args.alpha)).sum())

            plot_dataset_figure(
                test_values, test_is_anomaly, results, model, args.window_size, args.alpha,
                name, output_dir / "figures" / f"{name}.png",
            )

            rows.append(
                {
                    "dataset": name, "status": "ok", "n_total": len(values), "train_end": train_end,
                    "order": str(model["order"]), "reference_sigma2": model["reference_sigma2"],
                    "n_windows": len(results), "n_true_anomaly_windows": n_true, "n_normal_windows": n_normal,
                    "n_detected": n_detected, "n_false_alarms": n_false_alarms,
                    "detection_rate": n_detected / n_true if n_true else float("nan"),
                    "false_alarm_rate": n_false_alarms / n_normal if n_normal else float("nan"),
                }
            )
            print(f"  order={model['order']}, detected {n_detected}/{n_true}, false alarms {n_false_alarms}/{n_normal}")
        except Exception as exc:
            print(f"  failed: {exc}")
            rows.append({"dataset": name, "status": f"failed: {exc}"})

    table = pd.DataFrame(rows)
    for attempt in range(3):
        try:
            table.to_csv(output_dir / "evaluation_table.csv", index=False)
            break
        except PermissionError:
            if attempt == 2:
                raise
            time.sleep(0.5)

    ok = table[table["status"] == "ok"] if "status" in table else pd.DataFrame()
    if len(ok):
        print(f"\n{len(ok)}/{len(table)} datasets ran successfully")
        print(f"Overall detection rate: {ok['n_detected'].sum()}/{ok['n_true_anomaly_windows'].sum()}")
        print(f"Overall false alarm rate: {ok['n_false_alarms'].sum()}/{ok['n_normal_windows'].sum()}")
    print(f"Results saved under {output_dir}")


if __name__ == "__main__":
    main()
