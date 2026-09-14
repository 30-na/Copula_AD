"""Run fixed-dynamics ARIMA variance detection on prepared UCR datasets.

For each dataset, the UCR ``train_end`` filename field defines the clean
training region.  That region is split chronologically into train A and
train B.  Train A selects and fits a non-seasonal ARIMA(p, d, q) model.
The selected dynamics are then held fixed while a concentrated innovation
variance is computed independently in every train-B and test window.

The variance-ratio statistic is R_w = sigma2_w / sigma2_0.  Both an
approximate parametric F upper-tail p-value and a train-B empirical-null
upper-tail p-value are reported for every test window.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
import re
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.arima.model import ARIMA
from tqdm import tqdm

os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".matplotlib"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sigma2_workflow_helper import estimate_period_autocorr, read_labels, read_timeseries


UCR_NAME_PATTERN = re.compile(
    r"^(?P<id>\d+)_UCR_Anomaly_.+_(?P<train_end>\d+)_"
    r"(?P<anomaly_start>\d+)_(?P<anomaly_end>\d+)\.txt$"
)


@dataclass
class FixedArimaModel:
    order: tuple[int, int, int]
    seasonal_order: tuple[int, int, int, int]
    parameter_names: list[str]
    fixed_parameters: np.ndarray
    reference_sigma2: float
    reference_df: int
    aic: float


def find_prepared_datasets(prepared_dir: Path) -> list[dict[str, str | Path]]:
    datasets: list[dict[str, str | Path]] = []
    for timeseries_path in sorted(prepared_dir.glob("*/*/timeseries.csv")):
        labels_path = timeseries_path.parent / "labels.csv"
        if labels_path.exists():
            datasets.append(
                {
                    "category": timeseries_path.parent.parent.name,
                    "dataset_name": timeseries_path.parent.name,
                    "timeseries_path": timeseries_path,
                    "labels_path": labels_path,
                }
            )
    if not datasets:
        raise FileNotFoundError(f"No prepared UCR datasets found under {prepared_dir}")
    return datasets


def raw_metadata_by_id(raw_dir: Path) -> dict[int, dict[str, int | Path]]:
    metadata: dict[int, dict[str, int | Path]] = {}
    for path in raw_dir.rglob("*UCR_Anomaly*.txt"):
        match = UCR_NAME_PATTERN.match(path.name)
        if match is None:
            continue
        dataset_id = int(match.group("id"))
        metadata[dataset_id] = {
            "raw_path": path,
            "train_end": int(match.group("train_end")),
            "anomaly_start": int(match.group("anomaly_start")),
            "anomaly_end": int(match.group("anomaly_end")),
        }
    return metadata


def dataset_id_from_folder(dataset_name: str) -> int:
    match = re.match(r"^ucr_(\d+)_", dataset_name)
    if match is None:
        raise ValueError(f"Cannot recover UCR dataset ID from {dataset_name!r}")
    return int(match.group(1))


def split_train_regions(
    series: pd.Series,
    train_end: int,
    train_a_fraction: float,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    # UCR filenames use train_end as the first index after the clean prefix.
    if not 2 <= train_end < len(series):
        raise ValueError(f"train_end={train_end} is invalid for a series of length {len(series)}")
    split = int(round(train_end * train_a_fraction))
    split = min(max(split, 1), train_end - 1)
    return series.iloc[:split], series.iloc[split:train_end], series.iloc[train_end:]


def fit_auto_arma(
    train_a: pd.Series,
    d: int,
    max_p: int,
    max_q: int,
    seasonal_period: int = 0,
    max_P: int = 0,
    max_Q: int = 0,
) -> FixedArimaModel:
    best_result = None
    best_order = None
    failures: list[str] = []

    seasonal_candidates = [(0, 0, 0, 0)]
    if seasonal_period > 1:
        seasonal_candidates = [
            (P, 0, Q, seasonal_period)
            for P in range(max_P + 1)
            for Q in range(max_Q + 1)
        ]

    for p in range(max_p + 1):
        for q in range(max_q + 1):
            for seasonal_order in seasonal_candidates:
                order = (p, d, q)
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", ConvergenceWarning)
                        warnings.simplefilter("ignore", UserWarning)
                        result = ARIMA(
                            train_a.to_numpy(dtype=float),
                            order=order,
                            seasonal_order=seasonal_order,
                        ).fit()
                    if np.isfinite(result.aic) and (
                        best_result is None or result.aic < best_result.aic
                    ):
                        best_result = result
                        best_order = (order, seasonal_order)
                except Exception as exc:
                    failures.append(f"{order}{seasonal_order}: {exc}")

    if best_result is None or best_order is None:
        detail = failures[-1] if failures else "no finite AIC values"
        raise RuntimeError(f"All ARMA candidates failed ({detail})")

    names = list(best_result.param_names)
    if "sigma2" not in names:
        raise RuntimeError("Selected ARIMA result has no sigma2 parameter")
    sigma_index = names.index("sigma2")
    params = np.asarray(best_result.params, dtype=float)
    reference_sigma2 = float(params[sigma_index])
    fixed_names = [name for name in names if name != "sigma2"]
    fixed_parameters = np.delete(params, sigma_index)
    reference_df = max(1, int(best_result.nobs) - len(params))

    selected_order, selected_seasonal_order = best_order
    return FixedArimaModel(
        order=selected_order,
        seasonal_order=selected_seasonal_order,
        parameter_names=fixed_names,
        fixed_parameters=fixed_parameters,
        reference_sigma2=reference_sigma2,
        reference_df=reference_df,
        aic=float(best_result.aic),
    )


def trailing_windows(
    series: pd.Series,
    window_size: int,
    stride: int,
) -> list[pd.Series]:
    if window_size <= 0 or stride <= 0:
        raise ValueError("window_size and stride must be positive")
    if len(series) < window_size:
        return []
    ends = list(range(window_size, len(series) + 1, stride))
    if ends[-1] != len(series):
        ends.append(len(series))
    return [series.iloc[end - window_size : end] for end in ends]


def fixed_window_scale(window: pd.Series, fitted: FixedArimaModel) -> tuple[float, int]:
    """Concentrate scale with ARMA dynamics fixed; perform no optimization."""
    model = ARIMA(
        window.to_numpy(dtype=float),
        order=fitted.order,
        seasonal_order=fitted.seasonal_order,
        concentrate_scale=True,
    )
    if list(model.param_names) != fitted.parameter_names:
        raise RuntimeError(
            f"Fixed parameter mismatch: expected {fitted.parameter_names}, got {model.param_names}"
        )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = model.filter(fitted.fixed_parameters)
    # With concentrate_scale=True, statsmodels obtains scale from the Kalman
    # prediction errors while keeping all supplied dynamics fixed.
    sigma2 = float(result.scale)
    burn = int(result.filter_results.loglikelihood_burn)
    effective_n = max(1, int(result.nobs) - burn)
    return sigma2, effective_n


def score_windows(
    series: pd.Series,
    fitted: FixedArimaModel,
    window_size: int,
    stride: int,
) -> pd.DataFrame:
    rows = []
    for window in trailing_windows(series, window_size, stride):
        try:
            sigma2, window_df = fixed_window_scale(window, fitted)
            ratio = sigma2 / fitted.reference_sigma2
            parametric_p = float(stats.f.sf(ratio, window_df, fitted.reference_df))
        except Exception:
            sigma2, window_df, ratio, parametric_p = np.nan, 0, np.nan, np.nan
        rows.append(
            {
                "window_start": window.index[0],
                "window_end": window.index[-1],
                "window_midpoint": window.index[len(window) // 2],
                "window_df": window_df,
                "sigma2_w": sigma2,
                "reference_sigma2": fitted.reference_sigma2,
                "variance_ratio": ratio,
                "parametric_p_value": parametric_p,
            }
        )
    return pd.DataFrame(rows)


def add_empirical_p_values(
    scored: pd.DataFrame,
    calibration_ratios: np.ndarray,
) -> pd.DataFrame:
    output = scored.copy()
    clean = calibration_ratios[np.isfinite(calibration_ratios)]
    if len(clean) == 0:
        output["empirical_p_value"] = np.nan
        return output
    output["empirical_p_value"] = [
        (1.0 + float(np.sum(clean >= ratio))) / (len(clean) + 1.0)
        if np.isfinite(ratio)
        else np.nan
        for ratio in output["variance_ratio"].to_numpy(dtype=float)
    ]
    return output


def calibration_metrics(p_values: pd.Series) -> dict[str, float | int]:
    clean = p_values.dropna().to_numpy(dtype=float)
    if len(clean) == 0:
        return {
            "n_train_b_windows": 0,
            "ks_statistic": np.nan,
            "ks_p_value": np.nan,
            "proportion_p_lt_0_10": np.nan,
            "proportion_p_lt_0_05": np.nan,
            "proportion_p_lt_0_01": np.nan,
        }
    ks = stats.kstest(clean, "uniform")
    return {
        "n_train_b_windows": len(clean),
        "ks_statistic": float(ks.statistic),
        "ks_p_value": float(ks.pvalue),
        "proportion_p_lt_0_10": float(np.mean(clean < 0.10)),
        "proportion_p_lt_0_05": float(np.mean(clean < 0.05)),
        "proportion_p_lt_0_01": float(np.mean(clean < 0.01)),
    }


def add_test_labels(test_results: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    output = test_results.copy()
    anomaly_times = labels.index[labels["is_anomaly"].astype(int) == 1]
    output["actual_anomaly"] = [
        int(((anomaly_times >= row.window_start) & (anomaly_times <= row.window_end)).any())
        for row in output.itertuples()
    ]
    return output


def plot_dataset(
    train_b_results: pd.DataFrame,
    test_results: pd.DataFrame,
    labels: pd.DataFrame,
    title: str,
    output_path: Path,
) -> None:
    p_values = train_b_results["parametric_p_value"].dropna().sort_values().to_numpy()
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    ax = axes[0, 0]
    ax.plot(test_results["window_midpoint"], test_results["variance_ratio"], marker="o", ms=3)
    anomaly_times = labels.index[labels["is_anomaly"].astype(int) == 1]
    if len(anomaly_times):
        ax.axvspan(anomaly_times.min(), anomaly_times.max(), color="red", alpha=0.15)
    ax.axhline(1.0, color="gray", linestyle="--")
    ax.set(title="Test variance ratios", ylabel="R_w")
    ax.grid(alpha=0.25)

    ax = axes[0, 1]
    ax.plot(test_results["window_midpoint"], test_results["parametric_p_value"], label="parametric")
    ax.plot(test_results["window_midpoint"], test_results["empirical_p_value"], label="empirical")
    ax.axhline(0.05, color="red", linestyle="--", label="0.05")
    ax.set(title="Test upper-tail p-values", ylabel="p-value", ylim=(-0.02, 1.02))
    ax.legend()
    ax.grid(alpha=0.25)

    ax = axes[1, 0]
    ax.hist(p_values, bins=np.linspace(0, 1, 11), edgecolor="black")
    ax.axhline(len(p_values) / 10 if len(p_values) else 0, color="red", linestyle="--")
    ax.set(title="Train-B parametric p-value histogram", xlabel="p-value", ylabel="count")

    ax = axes[1, 1]
    if len(p_values):
        theoretical = (np.arange(1, len(p_values) + 1) - 0.5) / len(p_values)
        ax.scatter(theoretical, p_values, s=22)
        ax.plot([0, 1], [0, 1], color="red", linestyle="--")
    ax.set(title="Uniform(0,1) Q-Q plot", xlabel="theoretical quantile", ylabel="observed quantile")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.25)

    fig.suptitle(title)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def process_one(
    info: dict[str, str | Path],
    metadata: dict[str, int | Path],
    args: argparse.Namespace,
    run_dir: Path,
) -> dict[str, str | int | float]:
    category = str(info["category"])
    dataset_name = str(info["dataset_name"])
    df = read_timeseries(Path(info["timeseries_path"]))
    labels = read_labels(Path(info["labels_path"]))
    if labels is None or "is_anomaly" not in labels:
        raise ValueError(f"Missing anomaly labels for {dataset_name}")
    series = df.iloc[:, 0].dropna()
    train_end = int(metadata["train_end"])
    train_a, train_b, test = split_train_regions(series, train_end, args.train_a_fraction)
    if len(train_a) < args.window_size or len(train_b) < args.window_size:
        raise ValueError(
            f"{dataset_name}: train A/B must each contain at least {args.window_size} points"
        )

    seasonal_period = 0
    if args.seasonal:
        seasonal_period = args.seasonal_period or (
            estimate_period_autocorr(train_a, max_period=args.max_seasonal_period) or 0
        )
    fitted = fit_auto_arma(
        train_a,
        args.d,
        args.max_p,
        args.max_q,
        seasonal_period=seasonal_period,
        max_P=args.max_P,
        max_Q=args.max_Q,
    )
    train_b_results = score_windows(train_b, fitted, args.window_size, args.stride)
    calibration_ratios = train_b_results["variance_ratio"].to_numpy(dtype=float)
    train_b_results = add_empirical_p_values(train_b_results, calibration_ratios)
    test_results = score_windows(test, fitted, args.window_size, args.stride)
    test_results = add_empirical_p_values(test_results, calibration_ratios)
    test_results = add_test_labels(test_results, labels)
    calibration = calibration_metrics(train_b_results["parametric_p_value"])

    output_dir = run_dir / category / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)
    train_b_results.to_csv(output_dir / "train_b_calibration_windows.csv", index=False)
    test_results.to_csv(output_dir / "test_window_results.csv", index=False)

    model_record = {
        "dataset_name": dataset_name,
        "train_end": train_end,
        "train_a_size": len(train_a),
        "train_b_size": len(train_b),
        "test_size": len(test),
        "p": fitted.order[0],
        "d": fitted.order[1],
        "q": fitted.order[2],
        "seasonal_P": fitted.seasonal_order[0],
        "seasonal_D": fitted.seasonal_order[1],
        "seasonal_Q": fitted.seasonal_order[2],
        "seasonal_period": fitted.seasonal_order[3],
        "seasonal": fitted.seasonal_order[3] > 0,
        "aic": fitted.aic,
        "reference_sigma2": fitted.reference_sigma2,
        "reference_df": fitted.reference_df,
        "fixed_parameter_names": fitted.parameter_names,
        "fixed_parameters": fitted.fixed_parameters.tolist(),
        "parametric_f_is_approximate": True,
    }
    (output_dir / "fixed_model.json").write_text(
        json.dumps(model_record, indent=2), encoding="utf-8"
    )
    pd.DataFrame([calibration]).to_csv(output_dir / "calibration_summary.csv", index=False)
    if not args.skip_plot:
        plot_dataset(
            train_b_results,
            test_results,
            labels,
            f"{dataset_name}: fixed ARMA({fitted.order[0]},{fitted.order[2]}) calibration and detection",
            run_dir / "figures" / f"{dataset_name}_calibration_and_detection.png",
        )

    return {
        "category": category,
        "dataset_name": dataset_name,
        "p": fitted.order[0],
        "d": fitted.order[1],
        "q": fitted.order[2],
        "seasonal_P": fitted.seasonal_order[0],
        "seasonal_D": fitted.seasonal_order[1],
        "seasonal_Q": fitted.seasonal_order[2],
        "seasonal_period": fitted.seasonal_order[3],
        "reference_sigma2": fitted.reference_sigma2,
        **calibration,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fixed-dynamics ARIMA variance detection and calibration on UCR datasets."
    )
    parser.add_argument("--prepared-dir", default="data/ucr_prepared")
    parser.add_argument("--raw-dir", default="data/ucr_raw")
    parser.add_argument("--run-output-dir", default="results/ucr_fixed_arima")
    parser.add_argument(
        "--category",
        help="Process only one prepared UCR category, for example 'noise'.",
    )
    parser.add_argument("--train-a-fraction", type=float, default=0.5)
    parser.add_argument("--window-size", type=int, default=100)
    parser.add_argument("--stride", type=int, default=100)
    parser.add_argument("--d", type=int, default=1, choices=[0, 1, 2])
    parser.add_argument("--max-p", type=int, default=5)
    parser.add_argument("--max-q", type=int, default=5)
    parser.add_argument("--seasonal", action="store_true")
    parser.add_argument("--seasonal-period", type=int, help="Fixed m; otherwise estimate it from train A.")
    parser.add_argument("--max-seasonal-period", type=int, default=200)
    parser.add_argument("--max-P", type=int, default=1)
    parser.add_argument("--max-Q", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--skip-plot", action="store_true")
    args = parser.parse_args()
    if not 0 < args.train_a_fraction < 1:
        parser.error("--train-a-fraction must be strictly between 0 and 1")
    if args.workers < 1:
        parser.error("--workers must be at least 1")

    datasets = find_prepared_datasets(Path(args.prepared_dir))
    if args.category is not None:
        datasets = [item for item in datasets if item["category"] == args.category]
        if not datasets:
            parser.error(f"No prepared datasets found for category {args.category!r}")
    if args.limit is not None:
        datasets = datasets[: args.limit]
    metadata_map = raw_metadata_by_id(Path(args.raw_dir))
    run_dir = Path(args.run_output_dir)
    tasks = []
    for info in datasets:
        dataset_id = dataset_id_from_folder(str(info["dataset_name"]))
        if dataset_id not in metadata_map:
            raise FileNotFoundError(
                f"No raw UCR filename metadata found for dataset {dataset_id:03d} under {args.raw_dir}"
            )
        tasks.append((info, metadata_map[dataset_id]))

    summaries = []
    if args.workers == 1:
        for info, metadata in tqdm(tasks, desc="Fixed ARIMA UCR", unit="dataset"):
            summaries.append(process_one(info, metadata, args, run_dir))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = [
                executor.submit(process_one, info, metadata, args, run_dir)
                for info, metadata in tasks
            ]
            for future in tqdm(
                as_completed(futures), total=len(futures), desc="Fixed ARIMA UCR", unit="dataset"
            ):
                summaries.append(future.result())

    run_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summaries).to_csv(run_dir / "overall_calibration_summary.csv", index=False)
    print(f"Processed {len(summaries)} datasets; results saved under {run_dir}")
    print("Parametric F p-values are approximate because dynamics are estimated on train A.")


if __name__ == "__main__":
    main()
