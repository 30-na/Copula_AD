"""Apply Algorithm 1 to a real seizure recording (mouse intracranial EEG).
Same core method as everywhere else (algorithm.run_algorithm) -- no
changes to the algorithm itself, just new data.

The raw recording is huge (a full day, ~172M samples/channel at 2000 Hz,
~1.4GB per file), so this script never loads the whole file. It reads only
a small, bounded excerpt directly from the binary file via seek + a sized
read: --n-train samples immediately before a seizure's onset (--seizure-index,
0-indexed; this file has 7 annotated seizures), and --n-test samples
starting AT that onset.

Every run is identified by its CONFIG (sample rate + pre-seizure extraction
length + window size). All 7 seizures for the same config share one folder,
e.g. data/mice/1000hz_pre5s/ and results/mice/1000hz_pre5s/, with one file
per seizure inside (timeseries_seizure1.csv, ..., figures/summary_seizure7.png)
-- so a single config's data, window results and figures for all seizures
are always together in one place.

Binary format (see data/mice/README.docx): 4 channels, int16, interleaved
per sample (ch1, ch2, ch3, ch4, ch1, ch2, ch3, ch4, ...), 2000 Hz.

Unlike the simulation/UCR figures (which only show the test segment), the
plot here shows the WHOLE extracted excerpt (training + test) in panel (a),
since the training portion is short enough (600 samples) to be worth
showing for context.

Outputs, under data/mice/<config>/ and results/mice/<config>/:
  - timeseries_seizureN.csv, labels_seizureN.csv   the extracted excerpt
                                  (n_train + n_test samples), saved once
                                  per seizure, reused on rerun
  - window_results_seizureN.csv    one row per test window
  - figures/summary_seizureN.png   two panels: (a) the full excerpt with the
                                  seizure shaded, (b) innovation variance
                                  ratio with control limits (test windows only)
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import decimate

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from algorithm import run_algorithm

N_CHANNELS = 4
SAMPLE_RATE_HZ = 2000
BYTES_PER_SAMPLE = 2  # int16

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


def f_critical_values(window_size: int, reference_df: int, alpha: float) -> tuple[float, float]:
    return (
        float(stats.f.ppf(alpha / 2, window_size, reference_df)),
        float(stats.f.ppf(1 - alpha / 2, window_size, reference_df)),
    )


def list_seizure_starts(annotation_path: Path) -> list[float]:
    """Args: path to a mice *.txt annotation file.
    Returns: every "Seizure starts" event's time, in seconds from the
    start of the recording (the file's "Time From Start" column), in
    chronological order.
    """
    df = pd.read_csv(annotation_path, sep="\t", skiprows=4)
    df.columns = [c.strip() for c in df.columns]
    starts = df[df["Annotation"].str.strip() == "Seizure starts"]
    if len(starts) == 0:
        raise ValueError(f"No 'Seizure starts' event found in {annotation_path}.")
    return sorted(float(t) for t in starts["Time From Start"])


def read_channel_excerpt(dat_path: Path, channel: int, start_sample: int, n_samples: int) -> np.ndarray:
    """Read one channel's values for a bounded sample range, without
    loading the rest of the (huge) binary file.

    Args:
        dat_path: path to the raw *_allCh.dat file.
        channel: 0-indexed channel to extract (0 = ch1).
        start_sample: first sample index (per-channel) to read.
        n_samples: how many samples (per-channel) to read.

    Returns:
        1D float array of length n_samples.
    """
    frame_bytes = N_CHANNELS * BYTES_PER_SAMPLE
    with open(dat_path, "rb") as f:
        f.seek(start_sample * frame_bytes)
        raw = f.read(n_samples * frame_bytes)
    frames = np.frombuffer(raw, dtype="<i2").reshape(-1, N_CHANNELS)
    return frames[:, channel].astype(float)


def prepare_sample(
    dat_path: Path, annotation_path: Path, channel: int, seizure_index: int, extract_n_train: int,
    extract_n_test: int, downsample_factor: int, sample_dir: Path,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract (or, if already saved, just reload) the excerpt: extract_n_train
    raw-rate samples immediately before the onset of seizure #seizure_index
    (0-indexed, chronological), then extract_n_test raw-rate samples
    starting at that onset -- then, if downsample_factor > 1, low-pass
    filter and decimate the whole excerpt down to
    SAMPLE_RATE_HZ / downsample_factor.

    Returns:
        (values, is_anomaly): both length (extract_n_train + extract_n_test)
        // downsample_factor. is_anomaly is 1 for every sample from
        seizure onset on, 0 before.
    """
    expected_len = (extract_n_train + extract_n_test) // downsample_factor
    seizure_number = seizure_index + 1
    tag = f"seizure{seizure_number}" + (f"_ch{channel + 1}" if channel != 0 else "")
    ts_path = sample_dir / f"timeseries_{tag}.csv"
    labels_path = sample_dir / f"labels_{tag}.csv"
    if ts_path.exists() and labels_path.exists():
        values = pd.read_csv(ts_path)["value"].to_numpy(dtype=float)
        is_anomaly = pd.read_csv(labels_path)["is_anomaly"].to_numpy(dtype=int)
        if len(values) == expected_len:
            print(f"Reusing existing sample at {sample_dir} ({len(values)} points).")
            return values, is_anomaly
        print("Existing sample length doesn't match the requested extraction/downsample settings; re-extracting.")

    seizure_starts = list_seizure_starts(annotation_path)
    if seizure_index >= len(seizure_starts):
        raise ValueError(f"seizure_index={seizure_index} but only {len(seizure_starts)} seizures are annotated.")
    seizure_start_s = seizure_starts[seizure_index]
    seizure_start_sample = int(round(seizure_start_s * SAMPLE_RATE_HZ))
    excerpt_start = seizure_start_sample - extract_n_train
    if excerpt_start < 0:
        raise ValueError(f"Not enough data before the seizure: need {extract_n_train} samples, only {seizure_start_sample} available.")

    values = read_channel_excerpt(dat_path, channel, excerpt_start, extract_n_train + extract_n_test)
    n_train_ds = extract_n_train // downsample_factor
    if downsample_factor > 1:
        values = decimate(values, downsample_factor, zero_phase=True)
    is_anomaly = np.zeros(len(values), dtype=int)
    is_anomaly[n_train_ds:] = 1  # the whole test portion is inside the seizure

    sample_dir.mkdir(parents=True, exist_ok=True)
    index = pd.RangeIndex(len(values), name="t")
    pd.DataFrame({"value": values}, index=index).to_csv(ts_path)
    pd.DataFrame({"is_anomaly": is_anomaly}, index=index).to_csv(labels_path)
    effective_hz = SAMPLE_RATE_HZ / downsample_factor
    print(f"Extracted {len(values)} samples at {effective_hz:.0f} Hz (seizure onset at sample {n_train_ds}) -> {sample_dir}")
    return values, is_anomaly


def plot_summary(
    values: np.ndarray, is_anomaly: np.ndarray, n_train: int, results: pd.DataFrame, model: dict,
    window_size: int, alpha: float, seizure_number: int, output_path: Path,
) -> None:
    """(a) the FULL excerpt (training + test), seizure shaded. (b) the
    innovation variance ratio for the test windows, with control limits.
    """
    anomaly_positions = np.where(is_anomaly == 1)[0]
    anomaly_span = (int(anomaly_positions.min()), int(anomaly_positions.max())) if len(anomaly_positions) else None

    fig, axes = plt.subplots(2, 1, figsize=(8, 4.7), height_ratios=(1, 1.1))

    ax = axes[0]
    _style_axis(ax, hide_xticks=True)
    ax.plot(np.arange(len(values)), values, color=COLOR_INK, linewidth=0.7)
    if anomaly_span is not None:
        ax.axvspan(*anomaly_span, color=COLOR_ANOMALY_SPAN, alpha=0.6, linewidth=0, label="Seizure")
    ax.axvline(n_train, color=COLOR_MUTED, linewidth=1.0, linestyle=":")
    y_min, y_max = ax.get_ylim()
    ax.set_ylim(y_min, y_max + 0.18 * (y_max - y_min))
    ax.set_ylabel("Value")
    ax.set_title(f"(a) Full Excerpt (training + test) — Seizure {seizure_number}", loc="left", fontsize=10.5, color=COLOR_INK)
    ax.legend(loc="upper right", fontsize=9, frameon=True, facecolor="white", edgecolor="none")

    ax = axes[1]
    _style_axis(ax, hide_xticks=True)
    if anomaly_span is not None:
        ax.axvspan(*anomaly_span, color=COLOR_ANOMALY_SPAN, alpha=0.6, linewidth=0)
    ax.axhline(1.0, color=COLOR_MUTED, linewidth=1.0, linestyle=":")
    lower, upper = f_critical_values(window_size, model["reference_df"], alpha)
    ax.axhline(lower, color=COLOR_LIMIT, linewidth=1.1, linestyle="--", label=f"control limits (alpha={alpha})")
    ax.axhline(upper, color=COLOR_LIMIT, linewidth=1.1, linestyle="--")
    x = n_train + results["window_start"] + window_size // 2  # offset into the FULL excerpt's x-axis
    significant = results["p_value"] < alpha
    ratio = results["variance_ratio"]
    ax.plot(x, ratio, color=COLOR_LINE, linewidth=1.2, zorder=3, label="Innovation variance ratio")
    ax.scatter(x[significant], ratio[significant], s=18, color=COLOR_LINE, zorder=4)
    ax.scatter(x[~significant], ratio[~significant], s=18, facecolors="white", edgecolors=COLOR_LINE, linewidths=1.0, zorder=4)
    ax.set_xlim(0, len(values))
    y_min, y_max = ax.get_ylim()
    ax.set_ylim(y_min, y_max + 0.22 * (y_max - y_min))
    ax.set_ylabel("Variance Ratio")
    ax.set_xlabel("Sample")
    ax.set_title("(b) Innovation Variance Ratio", loc="left", fontsize=10.5, color=COLOR_INK)
    ax.legend(loc="upper right", fontsize=8.5, frameon=False)

    fig.tight_layout()
    fig.subplots_adjust(right=1.0 - fig.subplotpars.left)
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
    parser.add_argument("--dat-file", default="data/mice/AC75a-5_DOB_072519_TS_2020-03-23_17_30_04_allCh.dat")
    parser.add_argument("--annotation-file", default="data/mice/AC75a-5_DOB 072519_TS_2020-03-23_17_30_04.txt")
    parser.add_argument("--channel", type=int, default=0, help="0-indexed (0 = ch1, ipsilateral hippocampus).")
    parser.add_argument("--seizure-index", type=int, default=0, help="0-indexed, chronological (this file has 7 annotated seizures). Determines the seizureN filenames inside the config folder.")
    parser.add_argument("--downsample-factor", type=int, default=2, help="Decimation factor applied after extraction, e.g. 2 turns the raw 2000 Hz recording into 1000 Hz.")
    parser.add_argument("--n-train", type=int, default=None, help="Reference-model training samples, AT THE DOWNSAMPLED RATE. Default: 600 / downsample-factor (same real-world duration as the 2000 Hz run).")
    parser.add_argument("--n-test", type=int, default=None, help="Test samples, AT THE DOWNSAMPLED RATE. Default: fills the rest of the extracted excerpt.")
    parser.add_argument("--extract-n-train", type=int, default=10000, help="RAW-RATE samples read before seizure onset when extracting (before downsampling).")
    parser.add_argument("--extract-n-test", type=int, default=10000, help="RAW-RATE samples read starting at seizure onset when extracting (before downsampling).")
    parser.add_argument("--window-size", type=int, default=None, help="Test window size, AT THE DOWNSAMPLED RATE. Default: 100 / downsample-factor.")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--max-p", type=int, default=3)
    parser.add_argument("--max-q", type=int, default=3)
    parser.add_argument("--sample-dir", default=None, help="Default: data/mice/<config>, config = e.g. 1000hz_pre5s or 200hz_pre5s_win200. Shared by all 7 seizures.")
    parser.add_argument("--output-dir", default=None, help="Default: results/mice/<config>. Shared by all 7 seizures.")
    args = parser.parse_args()

    if args.n_train is None:
        args.n_train = 600 // args.downsample_factor
    if args.window_size is None:
        args.window_size = 100 // args.downsample_factor
    pre_seconds = args.extract_n_train / SAMPLE_RATE_HZ
    config_name = f"{SAMPLE_RATE_HZ // args.downsample_factor}hz_pre{pre_seconds:g}s"
    default_window_size = 100 // args.downsample_factor
    if args.window_size != default_window_size:
        config_name += f"_win{args.window_size}"
    default_n_train = 600 // args.downsample_factor
    if args.n_train != default_n_train:
        config_name += f"_ntrain{args.n_train}"
    if args.sample_dir is None:
        args.sample_dir = f"data/mice/{config_name}"
    if args.output_dir is None:
        args.output_dir = f"results/mice/{config_name}"

    sample_dir = Path(args.sample_dir)
    values, is_anomaly = prepare_sample(
        Path(args.dat_file), Path(args.annotation_file), args.channel, args.seizure_index,
        args.extract_n_train, args.extract_n_test, args.downsample_factor, sample_dir,
    )

    if args.n_test is None:
        args.n_test = len(values) - args.n_train

    if args.n_train > len(values) - args.downsample_factor:
        raise ValueError(f"--n-train ({args.n_train}) leaves no room before the seizure onset in the {len(values)}-sample excerpt.")
    n_total = args.n_train + args.n_test
    if n_total > len(values):
        raise ValueError(f"--n-train + --n-test ({n_total}) exceeds the extracted sample ({len(values)}).")

    train_values = values[: args.n_train]
    test_values = values[args.n_train : n_total]
    test_is_anomaly = is_anomaly[args.n_train : n_total]
    print(f"Training: {len(train_values)} samples, test: {len(test_values)} samples")

    results, model, _ = run_algorithm(train_values, test_values, args.window_size, args.alpha, args.max_p, args.max_q)
    print(f"Selected order (p, d, q) = {model['order']}, AIC = {model['aic']:.2f}")
    print(f"Reference sigma0^2 = {model['reference_sigma2']:.4f}")

    actual_anomaly = []
    for _, row in results.iterrows():
        start, end = int(row["window_start"]), int(row["window_end"])
        actual_anomaly.append(int(test_is_anomaly[start:end].any()))
    results["actual_anomaly"] = actual_anomaly

    seizure_number = args.seizure_index + 1
    tag = f"seizure{seizure_number}" + (f"_ch{args.channel + 1}" if args.channel != 0 else "")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_dir / f"window_results_{tag}.csv", index=False)

    n_true = int((results["actual_anomaly"] == 1).sum())
    n_normal = int((results["actual_anomaly"] == 0).sum())
    n_detected = int(((results["actual_anomaly"] == 1) & (results["p_value"] < args.alpha)).sum())
    n_false_alarms = int(((results["actual_anomaly"] == 0) & (results["p_value"] < args.alpha)).sum())

    plot_summary(
        values[:n_total], is_anomaly[:n_total], args.n_train, results, model,
        args.window_size, args.alpha, seizure_number, output_dir / "figures" / f"summary_{tag}.png",
    )

    print(f"Test windows: {len(results)} total, {n_true} overlap the seizure")
    print(f"Detected {n_detected}/{n_true}, false alarms {n_false_alarms}/{n_normal}")
    print(f"Results saved under {output_dir}")


if __name__ == "__main__":
    main()
