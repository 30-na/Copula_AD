"""Apply the fixed-dynamics ARIMA variance workflow to simulated data."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

import pandas as pd

from sigma2_workflow_helper import read_labels, read_timeseries


fixed = importlib.import_module("06_run_ucr_fixed_arima_workflow")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fixed-dynamics ARIMA calibration and detection on simulated data."
    )
    parser.add_argument("--input", default="data/simulation/simulated_with_anomaly.csv")
    parser.add_argument("--labels", default="data/simulation/simulated_anomaly_labels.csv")
    parser.add_argument("--run-output-dir", default="results/simulation_fixed_arima_d1")
    parser.add_argument("--train-a-fraction", type=float, default=0.5)
    parser.add_argument("--window-size", type=int, default=100)
    parser.add_argument("--stride", type=int, default=100)
    parser.add_argument("--d", type=int, default=1, choices=[0, 1, 2])
    parser.add_argument("--max-p", type=int, default=1)
    parser.add_argument("--max-q", type=int, default=1)
    parser.add_argument("--seasonal", action="store_true")
    parser.add_argument("--seasonal-period", type=int)
    parser.add_argument("--max-seasonal-period", type=int, default=50)
    parser.add_argument("--max-P", type=int, default=1)
    parser.add_argument("--max-Q", type=int, default=1)
    parser.add_argument("--skip-plot", action="store_true")
    args = parser.parse_args()
    if not 0 < args.train_a_fraction < 1:
        parser.error("--train-a-fraction must be strictly between 0 and 1")

    df = read_timeseries(Path(args.input))
    labels = read_labels(Path(args.labels))
    if labels is None or "is_anomaly" not in labels:
        raise ValueError("The label file must contain an is_anomaly column")
    anomaly_flags = labels["is_anomaly"].to_numpy(dtype=int)
    if anomaly_flags.sum() == 0:
        raise ValueError("The simulation labels contain no anomaly")

    train_end = int(anomaly_flags.argmax())
    series = df.iloc[:, 0].dropna()
    train_a, train_b, test = fixed.split_train_regions(
        series, train_end, args.train_a_fraction
    )
    seasonal_period = 0
    if args.seasonal:
        seasonal_period = args.seasonal_period or (
            fixed.estimate_period_autocorr(
                train_a, max_period=args.max_seasonal_period
            )
            or 0
        )
    fitted = fixed.fit_auto_arma(
        train_a,
        args.d,
        args.max_p,
        args.max_q,
        seasonal_period=seasonal_period,
        max_P=args.max_P,
        max_Q=args.max_Q,
    )

    train_b_results = fixed.score_windows(
        train_b, fitted, args.window_size, args.stride
    )
    calibration_ratios = train_b_results["variance_ratio"].to_numpy(dtype=float)
    train_b_results = fixed.add_empirical_p_values(
        train_b_results, calibration_ratios
    )
    test_results = fixed.score_windows(test, fitted, args.window_size, args.stride)
    test_results = fixed.add_empirical_p_values(test_results, calibration_ratios)
    test_results = fixed.add_test_labels(test_results, labels)
    calibration = fixed.calibration_metrics(train_b_results["parametric_p_value"])

    output_dir = Path(args.run_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_b_results.to_csv(output_dir / "train_b_calibration_windows.csv", index=False)
    test_results.to_csv(output_dir / "test_window_results.csv", index=False)
    pd.DataFrame([calibration]).to_csv(output_dir / "calibration_summary.csv", index=False)

    model_record = {
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
    if not args.skip_plot:
        fixed.plot_dataset(
            train_b_results,
            test_results,
            labels,
            f"Simulation: fixed ARIMA{fitted.order} calibration and detection",
            output_dir / "simulation_calibration_and_detection.png",
        )

    anomaly_windows = test_results[test_results["actual_anomaly"] == 1]
    normal_windows = test_results[test_results["actual_anomaly"] == 0]
    summary = {
        **model_record,
        **calibration,
        "n_test_windows": len(test_results),
        "n_anomaly_windows": len(anomaly_windows),
        "parametric_anomaly_detected_p_lt_0_05": bool(
            (anomaly_windows["parametric_p_value"] < 0.05).any()
        ),
        "empirical_anomaly_detected_p_lt_0_05": bool(
            (anomaly_windows["empirical_p_value"] < 0.05).any()
        ),
        "parametric_false_positive_rate": float(
            (normal_windows["parametric_p_value"] < 0.05).mean()
        ),
        "empirical_false_positive_rate": float(
            (normal_windows["empirical_p_value"] < 0.05).mean()
        ),
    }
    pd.DataFrame([summary]).to_csv(output_dir / "overall_summary.csv", index=False)
    print(pd.Series(summary).to_string())
    print(f"Results saved under {output_dir}")


if __name__ == "__main__":
    main()
