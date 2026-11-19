"""Test fixed-model UCR train-B standardized innovations for normality."""

from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.arima.model import ARIMA
from tqdm import tqdm

os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".matplotlib"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sigma2_workflow_helper import read_timeseries


workflow = importlib.import_module("06_run_ucr_fixed_arima_workflow")


def load_fixed_model(path: Path) -> tuple[dict, workflow.FixedArimaModel]:
    record = json.loads(path.read_text(encoding="utf-8"))
    seasonal_order = (
        int(record.get("seasonal_P", 0)),
        int(record.get("seasonal_D", 0)),
        int(record.get("seasonal_Q", 0)),
        int(record.get("seasonal_period", 0)),
    )
    fitted = workflow.FixedArimaModel(
        order=(int(record["p"]), int(record["d"]), int(record["q"])),
        seasonal_order=seasonal_order,
        parameter_names=list(record["fixed_parameter_names"]),
        fixed_parameters=np.asarray(record["fixed_parameters"], dtype=float),
        reference_sigma2=float(record["reference_sigma2"]),
        reference_df=int(record["reference_df"]),
        aic=float(record["aic"]),
    )
    return record, fitted


def standardized_innovations(series: pd.Series, fitted: workflow.FixedArimaModel) -> np.ndarray:
    model = ARIMA(
        series.to_numpy(dtype=float),
        order=fitted.order,
        seasonal_order=fitted.seasonal_order,
        concentrate_scale=True,
    )
    if list(model.param_names) != fitted.parameter_names:
        raise RuntimeError(
            f"Parameter mismatch: expected {fitted.parameter_names}, got {model.param_names}"
        )
    result = model.filter(fitted.fixed_parameters)
    errors = np.asarray(result.filter_results.standardized_forecasts_error[0], dtype=float)
    burn = int(result.filter_results.loglikelihood_burn)
    return errors[burn:][np.isfinite(errors[burn:])]


def diagnostic_statistics(errors: np.ndarray) -> dict[str, float | int]:
    n = len(errors)
    if n < 8:
        raise ValueError("At least eight finite innovations are required")
    jb = stats.jarque_bera(errors)
    dagostino = stats.normaltest(errors)
    # scipy warns that Shapiro p-values are inaccurate above 5000 observations.
    shapiro_sample = errors if n <= 5000 else errors[:5000]
    shapiro = stats.shapiro(shapiro_sample)
    anderson = stats.anderson(errors, dist="norm")
    ad_5_index = int(np.argmin(np.abs(np.asarray(anderson.significance_level) - 5.0)))
    lag = min(20, max(1, n // 5))
    ljung_box = acorr_ljungbox(errors, lags=[lag], return_df=True).iloc[0]
    return {
        "n_innovations": n,
        "mean": float(np.mean(errors)),
        "std": float(np.std(errors, ddof=1)),
        "skewness": float(stats.skew(errors, bias=False)),
        "excess_kurtosis": float(stats.kurtosis(errors, fisher=True, bias=False)),
        "jarque_bera_statistic": float(jb.statistic),
        "jarque_bera_p_value": float(jb.pvalue),
        "dagostino_k2_statistic": float(dagostino.statistic),
        "dagostino_k2_p_value": float(dagostino.pvalue),
        "shapiro_statistic": float(shapiro.statistic),
        "shapiro_p_value": float(shapiro.pvalue),
        "shapiro_n": len(shapiro_sample),
        "anderson_darling_statistic": float(anderson.statistic),
        "anderson_darling_5pct_critical": float(anderson.critical_values[ad_5_index]),
        "anderson_darling_reject_0_05": int(
            anderson.statistic > anderson.critical_values[ad_5_index]
        ),
        "ljung_box_lag": lag,
        "ljung_box_statistic": float(ljung_box["lb_stat"]),
        "ljung_box_p_value": float(ljung_box["lb_pvalue"]),
    }


def plot_diagnostics(errors: np.ndarray, title: str, output_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes[0, 0].plot(errors, linewidth=0.65)
    axes[0, 0].axhline(0, color="black", linestyle="--", linewidth=0.8)
    axes[0, 0].set_title("Standardized innovations on clean train B")
    axes[0, 0].set_ylabel("standardized innovation")

    x = np.linspace(-5, 5, 500)
    axes[0, 1].hist(errors, bins=40, density=True, alpha=0.7, edgecolor="black")
    axes[0, 1].plot(x, stats.norm.pdf(x), color="red", label="N(0,1)")
    axes[0, 1].set_title("Histogram against standard normal")
    axes[0, 1].legend()

    stats.probplot(errors, dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title("Normal Q-Q plot")

    sorted_errors = np.sort(errors)
    empirical = (np.arange(1, len(errors) + 1) - 0.5) / len(errors)
    axes[1, 1].plot(sorted_errors, stats.norm.cdf(sorted_errors), label="N(0,1) CDF")
    axes[1, 1].plot(sorted_errors, empirical, label="empirical CDF")
    axes[1, 1].set_title("Empirical versus standard-normal CDF")
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.25)

    fig.suptitle(title)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-dir", default="data/ucr_prepared")
    parser.add_argument("--raw-dir", default="data/ucr_raw")
    parser.add_argument("--model-results-dir", default="results/ucr_fixed_arima_noise_d0_pq5")
    parser.add_argument("--category", default="noise")
    parser.add_argument("--output-dir", default="results/ucr_noise_innovation_normality")
    args = parser.parse_args()

    prepared = {
        str(item["dataset_name"]): item
        for item in workflow.find_prepared_datasets(Path(args.prepared_dir))
        if item["category"] == args.category
    }
    raw_metadata = workflow.raw_metadata_by_id(Path(args.raw_dir))
    model_root = Path(args.model_results_dir) / args.category
    output_dir = Path(args.output_dir)
    rows = []

    model_paths = sorted(model_root.glob("*/fixed_model.json"))
    if not model_paths:
        raise FileNotFoundError(f"No fixed_model.json files found under {model_root}")
    for model_path in tqdm(model_paths, desc="Innovation normality", unit="dataset"):
        dataset_name = model_path.parent.name
        if dataset_name not in prepared:
            raise FileNotFoundError(f"No prepared dataset found for {dataset_name}")
        record, fitted = load_fixed_model(model_path)
        dataset_id = workflow.dataset_id_from_folder(dataset_name)
        metadata = raw_metadata[dataset_id]
        series = read_timeseries(Path(prepared[dataset_name]["timeseries_path"])).iloc[:, 0].dropna()
        _, train_b, _ = workflow.split_train_regions(
            series, int(metadata["train_end"]), 0.5
        )
        errors = standardized_innovations(train_b, fitted)
        statistics = diagnostic_statistics(errors)
        rows.append(
            {
                "category": args.category,
                "dataset_name": dataset_name,
                "p": fitted.order[0],
                "d": fitted.order[1],
                "q": fitted.order[2],
                **statistics,
            }
        )
        innovations_dir = output_dir / "innovations"
        innovations_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"standardized_innovation": errors}).to_csv(
            innovations_dir / f"{dataset_name}_standardized_innovations.csv",
            index=False,
        )
        plot_diagnostics(
            errors,
            f"{dataset_name}: fixed ARIMA train-B innovation diagnostics",
            output_dir / "figures" / f"{dataset_name}_innovation_normality.png",
        )

    results = pd.DataFrame(rows).sort_values("dataset_name")
    output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_dir / "innovation_normality_summary.csv", index=False)
    alpha = 0.05
    aggregate = pd.DataFrame(
        [
            {
                "n_datasets": len(results),
                "jarque_bera_rejections_0_05": int((results["jarque_bera_p_value"] < alpha).sum()),
                "dagostino_rejections_0_05": int((results["dagostino_k2_p_value"] < alpha).sum()),
                "shapiro_rejections_0_05": int((results["shapiro_p_value"] < alpha).sum()),
                "anderson_darling_rejections_0_05": int(results["anderson_darling_reject_0_05"].sum()),
                "ljung_box_rejections_0_05": int((results["ljung_box_p_value"] < alpha).sum()),
                "median_abs_skewness": float(results["skewness"].abs().median()),
                "median_excess_kurtosis": float(results["excess_kurtosis"].median()),
            }
        ]
    )
    aggregate.to_csv(output_dir / "aggregate_normality_summary.csv", index=False)
    print(aggregate.to_string(index=False))
    print(f"Results saved under {output_dir}")


if __name__ == "__main__":
    main()
