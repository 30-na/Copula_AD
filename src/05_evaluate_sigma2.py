"""Evaluate sigma2 anomaly scores against known labels.

This script turns sigma2 values into binary anomaly predictions using a simple
training-based threshold:

    threshold = mean(training scores) + k * std(training scores)

Outputs:
  - results/benchmark/ucr_single/sigma2_metrics.csv
  - results/benchmark/ucr_single/sigma2_predictions.csv
  - results/benchmark/ucr_single/figures/sigma2_evaluation.png
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".matplotlib"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_sigma2(path: Path, score_column: str) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index.name = "time"
    for column in ["window_start", "window_end"]:
        if column in df.columns:
            df[column] = pd.to_datetime(df[column])
    if "is_point_score" in df.columns:
        df["is_point_score"] = df["is_point_score"].astype(bool)
    if score_column not in df.columns:
        raise ValueError(f"Score column '{score_column}' not found in {path}")
    return df


def read_labels(path: Path) -> pd.DataFrame:
    labels = pd.read_csv(path, index_col=0, parse_dates=True)
    labels.index.name = "time"
    if "is_anomaly" not in labels.columns:
        raise ValueError(f"Label file must contain an 'is_anomaly' column: {path}")
    return labels


def align_labels_to_scores(labels: pd.DataFrame, score_index: pd.Index) -> pd.DataFrame:
    aligned = labels.reindex(score_index)
    if aligned["is_anomaly"].isna().any():
        aligned = labels.reindex(score_index, method="nearest")
    aligned["is_anomaly"] = aligned["is_anomaly"].astype(int)
    if "is_train" in aligned.columns:
        aligned["is_train"] = aligned["is_train"].fillna(0).astype(int)
    return aligned


def labels_by_window_overlap(sigma2_df: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    if "is_point_score" in sigma2_df.columns and sigma2_df["is_point_score"].fillna(False).all():
        return align_labels_to_scores(labels, sigma2_df.index)

    if "window_start" not in sigma2_df.columns or "window_end" not in sigma2_df.columns:
        return align_labels_to_scores(labels, sigma2_df.index)

    anomaly_times = labels.index[labels["is_anomaly"] == 1]
    if len(anomaly_times) == 0:
        aligned = pd.DataFrame({"is_anomaly": 0}, index=sigma2_df.index)
    else:
        anomaly_start = anomaly_times.min()
        anomaly_end = anomaly_times.max()
        overlaps = (
            (sigma2_df["window_start"] <= anomaly_end)
            & (sigma2_df["window_end"] >= anomaly_start)
        )
        aligned = pd.DataFrame({"is_anomaly": overlaps.astype(int)}, index=sigma2_df.index)

    if "is_train" in labels.columns:
        train_times = labels.index[labels["is_train"] == 1]
        if len(train_times) > 0:
            train_end = train_times.max()
            aligned["is_train"] = (sigma2_df["window_end"] <= train_end).astype(int)

    return aligned


def get_training_mask(aligned_labels: pd.DataFrame) -> pd.Series:
    if "is_train" in aligned_labels.columns and aligned_labels["is_train"].sum() > 0:
        return aligned_labels["is_train"] == 1

    anomaly_times = aligned_labels.index[aligned_labels["is_anomaly"] == 1]
    if len(anomaly_times) == 0:
        return aligned_labels["is_anomaly"] == 0
    return aligned_labels.index < anomaly_times.min()


def compute_predictions(
    sigma2_df: pd.DataFrame,
    labels: pd.DataFrame,
    score_column: str,
    k: float,
) -> tuple[pd.DataFrame, float]:
    aligned_labels = labels_by_window_overlap(sigma2_df, labels)
    train_mask = get_training_mask(aligned_labels)
    train_scores = sigma2_df.loc[train_mask, score_column].dropna()

    if train_scores.empty:
        raise ValueError("No training scores found for threshold calculation.")

    threshold = float(train_scores.mean() + k * train_scores.std(ddof=0))
    predictions = pd.DataFrame(index=sigma2_df.index)
    predictions["score"] = sigma2_df[score_column]
    for column in sigma2_df.columns:
        if (
            column in {"residual_mean", "innovation_mean", "predicted_mean", "is_point_score"}
            or column.endswith("_residual")
            or column.endswith("_innovation")
            or column.endswith("_predicted")
            or column.endswith("_window_count")
        ):
            predictions[column] = sigma2_df[column]
    predictions["is_anomaly"] = aligned_labels["is_anomaly"].astype(int)
    predictions["is_predicted"] = (predictions["score"] > threshold).astype(int)
    predictions["threshold"] = threshold
    return predictions, threshold


def compute_metrics(predictions: pd.DataFrame) -> dict[str, float | int | str]:
    valid = predictions["score"].notna()
    valid_predictions = predictions.loc[valid]
    y_true = valid_predictions["is_anomaly"].to_numpy(dtype=int)
    y_pred = valid_predictions["is_predicted"].to_numpy(dtype=int)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    auroc = np.nan
    if len(np.unique(y_true)) > 1:
        auroc = float(roc_auc_score(y_true, valid_predictions["score"].to_numpy(dtype=float)))

    anomaly_times = valid_predictions.index[valid_predictions["is_anomaly"] == 1]
    predicted_times = valid_predictions.index[valid_predictions["is_predicted"] == 1]
    detection_delay_seconds = np.nan

    if len(anomaly_times) > 0 and len(predicted_times) > 0:
        anomaly_start = anomaly_times.min()
        anomaly_end = anomaly_times.max()
        valid_detections = predicted_times[
            (predicted_times >= anomaly_start) & (predicted_times <= anomaly_end)
        ]
        if len(valid_detections) > 0:
            detection_delay_seconds = (valid_detections.min() - anomaly_start).total_seconds()

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc_roc": auroc,
        "detection_delay_seconds": detection_delay_seconds,
    }


def compute_ucr_metric(
    sigma2_df: pd.DataFrame,
    score_column: str,
    anomaly_start: int,
    anomaly_end: int,
) -> dict[str, float | int | str]:
    """Compute the official UCR one-location scoring rule.

    The UCR anomaly benchmark asks each method to return one integer location P.
    A dataset is scored as correct when P falls inside the relaxed interval:

        min(begin - L, begin - 100) < P < max(end + L, end + 100)

    where L = end - begin + 1. For windowed scores, we use the center of the
    highest-scoring window as P.
    """
    if score_column not in sigma2_df.columns:
        raise ValueError(f"Score column '{score_column}' not found in sigma2 data.")

    scores = sigma2_df[score_column].dropna()
    if scores.empty:
        return {
            "ucr_prediction_index": np.nan,
            "ucr_prediction_score": np.nan,
            "ucr_correct_start": np.nan,
            "ucr_correct_end": np.nan,
            "ucr_correct": 0,
        }

    best_label = scores.idxmax()
    best_row = sigma2_df.loc[best_label]
    best_score = float(best_row[score_column])

    if "is_point_score" in sigma2_df.columns and bool(best_row["is_point_score"]):
        prediction_index = int(sigma2_df.index.get_loc(best_label))
    elif "window_start" in sigma2_df.columns and "window_end" in sigma2_df.columns:
        window_start = pd.Timestamp(best_row["window_start"])
        window_end = pd.Timestamp(best_row["window_end"])
        if window_start == window_end:
            prediction_index = int(sigma2_df.index.get_loc(best_label))
        else:
            prediction_time = window_start + (window_end - window_start) / 2
            prediction_index = int(round((prediction_time - sigma2_df.index[0]).total_seconds()))
    else:
        prediction_index = int(sigma2_df.index.get_loc(best_label))

    anomaly_length = anomaly_end - anomaly_start + 1
    correct_start = min(anomaly_start - anomaly_length, anomaly_start - 100)
    correct_end = max(anomaly_end + anomaly_length, anomaly_end + 100)
    is_correct = int(correct_start < prediction_index < correct_end)

    return {
        "ucr_prediction_index": prediction_index,
        "ucr_prediction_score": best_score,
        "ucr_correct_start": correct_start,
        "ucr_correct_end": correct_end,
        "ucr_correct": is_correct,
    }


def plot_evaluation(predictions: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(
        predictions.index,
        predictions["score"],
        marker="o",
        markersize=3,
        linewidth=0.8,
        color="black",
        label="sigma2 score",
    )
    ax.axhline(
        predictions["threshold"].iloc[0],
        color="red",
        linestyle="--",
        linewidth=1,
        label="threshold",
    )

    anomaly_times = predictions.index[predictions["is_anomaly"] == 1]
    if len(anomaly_times) > 0:
        ax.axvspan(anomaly_times.min(), anomaly_times.max(), color="red", alpha=0.12, label="true anomaly")

    predicted = predictions[predictions["is_predicted"] == 1]
    ax.scatter(
        predicted.index,
        predicted["score"],
        color="red",
        marker="x",
        s=45,
        label="predicted anomaly",
    )

    ax.set_title("Sigma2 Anomaly Detection Evaluation")
    ax.set_xlabel("Time")
    ax.set_ylabel("sigma2 score")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="upper right")
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate sigma2 anomaly scores.")
    parser.add_argument("--sigma2", default="results/benchmark/ucr_single/sigma2_values.csv")
    parser.add_argument("--labels", default="results/benchmark/ucr_single/ucr_labels.csv")
    parser.add_argument("--score-column", default="sigma2_mean")
    parser.add_argument("--k", type=float, default=3.0)
    parser.add_argument("--metrics-output", default="results/benchmark/ucr_single/sigma2_metrics.csv")
    parser.add_argument("--predictions-output", default="results/benchmark/ucr_single/sigma2_predictions.csv")
    parser.add_argument("--plot", default="results/benchmark/ucr_single/figures/sigma2_evaluation.png")
    args = parser.parse_args()

    sigma2_df = read_sigma2(Path(args.sigma2), args.score_column)
    labels = read_labels(Path(args.labels))
    predictions, threshold = compute_predictions(sigma2_df, labels, args.score_column, args.k)
    metrics = compute_metrics(predictions)
    metrics.update({"threshold": threshold, "k": args.k, "score_column": args.score_column})

    metrics_path = Path(args.metrics_output)
    predictions_path = Path(args.predictions_output)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_path.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame([metrics]).to_csv(metrics_path, index=False)
    predictions.to_csv(predictions_path)
    plot_evaluation(predictions, Path(args.plot))

    print(f"Saved metrics to {metrics_path}")
    print(f"Saved predictions to {predictions_path}")
    print(f"Saved evaluation plot to {args.plot}")
    print(pd.DataFrame([metrics]).to_string(index=False))


if __name__ == "__main__":
    main()
