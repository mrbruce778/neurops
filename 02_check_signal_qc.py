from pathlib import Path
from collections import defaultdict
import warnings

import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import welch


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(r"E:\neurops")

CLASS_FOLDERS = {
    "ec_alzheimer": "Alzheimer",
    "ec_control": "Control",
    "ec_ftd": "FTD",
}

OUTPUT_DIR = PROJECT_ROOT / "qc_results"
PLOT_DIR = OUTPUT_DIR / "plots"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

EXPECTED_SFREQ = 500.0
EXPECTED_CHANNELS = [
    "Fp1", "Fp2",
    "F3", "F4",
    "C3", "C4",
    "P3", "P4",
    "O1", "O2",
    "F7", "F8",
    "T3", "T4",
    "T5", "T6",
    "Fz", "Cz", "Pz",
]

# Frequency range used for PSD inspection
PSD_MIN_FREQ = 0.5
PSD_MAX_FREQ = 80.0

# EEG bands for descriptive QC only
BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}

# Numerical QC thresholds.
# These are FLAGS, not automatic rejection criteria.
FLAT_STD_UV = 0.1
EXTREME_STD_UV = 200.0
EXTREME_PEAK_UV = 1000.0

# How many representative plots per diagnostic group
N_REPRESENTATIVE = 2

# Duration of raw EEG displayed in trace plots
TRACE_SECONDS = 10.0


# ============================================================
# FILE DISCOVERY
# ============================================================

def find_set_files():
    records = []

    for folder_name, label in CLASS_FOLDERS.items():

        folder = PROJECT_ROOT / folder_name

        if not folder.exists():
            print(f"[SKIP] Folder not found: {folder}")
            continue

        for file_path in sorted(folder.rglob("*.set")):

            subject_id = None

            for parent in file_path.parents:
                if parent.name.lower().startswith("sub-"):
                    subject_id = parent.name
                    break

            if subject_id is None:
                subject_id = file_path.stem

            records.append({
                "class": label,
                "subject_id": subject_id,
                "path": file_path,
            })

    return records


# ============================================================
# PSD FUNCTIONS
# ============================================================

def calculate_psd(data, sfreq):
    """
    Calculate Welch PSD.

    data shape:
        channels x samples
    """

    nperseg = min(
        int(sfreq * 4),
        data.shape[1],
    )

    freqs, psd = welch(
        data,
        fs=sfreq,
        nperseg=nperseg,
        axis=1,
    )

    return freqs, psd


def band_power(freqs, psd, low, high):
    """
    Mean PSD inside a frequency band.
    """

    mask = (
        (freqs >= low)
        & (freqs < high)
    )

    if not np.any(mask):
        return np.nan

    return float(
        np.mean(psd[:, mask])
    )


def line_noise_ratio(freqs, psd, line_freq=50.0):
    """
    Simple descriptive estimate of a narrow 50-Hz peak.

    Compare PSD around 50 Hz with nearby frequencies.

    This is only a QC flag.
    """

    signal_mask = (
        (freqs >= line_freq - 0.5)
        & (freqs <= line_freq + 0.5)
    )

    nearby_mask = (
        (
            (freqs >= line_freq - 5)
            & (freqs <= line_freq - 2)
        )
        |
        (
            (freqs >= line_freq + 2)
            & (freqs <= line_freq + 5)
        )
    )

    if (
        not np.any(signal_mask)
        or not np.any(nearby_mask)
    ):
        return np.nan

    line_power = np.mean(
        psd[:, signal_mask]
    )

    nearby_power = np.mean(
        psd[:, nearby_mask]
    )

    if nearby_power <= 0:
        return np.nan

    return float(
        line_power / nearby_power
    )


# ============================================================
# SUBJECT QC
# ============================================================

def inspect_subject(record):

    path = record["path"]

    raw = mne.io.read_raw_eeglab(
        path,
        preload=True,
        verbose="ERROR",
    )

    raw.pick("eeg")

    sfreq = float(raw.info["sfreq"])
    channels = list(raw.ch_names)

    # MNE stores EEG internally in volts.
    data_v = raw.get_data()

    # Convert to microvolts for interpretable QC.
    data_uv = data_v * 1e6

    n_channels, n_samples = data_uv.shape

    duration_sec = n_samples / sfreq

    # --------------------------------------------------------
    # Invalid numerical values
    # --------------------------------------------------------

    nan_count = int(
        np.isnan(data_uv).sum()
    )

    inf_count = int(
        np.isinf(data_uv).sum()
    )

    # --------------------------------------------------------
    # Channel statistics
    # --------------------------------------------------------

    channel_mean = np.nanmean(
        data_uv,
        axis=1,
    )

    channel_std = np.nanstd(
        data_uv,
        axis=1,
    )

    channel_ptp = np.ptp(
        data_uv,
        axis=1,
    )

    channel_abs_peak = np.nanmax(
        np.abs(data_uv),
        axis=1,
    )

    # --------------------------------------------------------
    # Flag suspicious channels
    # --------------------------------------------------------

    flat_channels = [
        channels[i]
        for i, value in enumerate(channel_std)
        if value < FLAT_STD_UV
    ]

    high_std_channels = [
        channels[i]
        for i, value in enumerate(channel_std)
        if value > EXTREME_STD_UV
    ]

    extreme_peak_channels = [
        channels[i]
        for i, value in enumerate(channel_abs_peak)
        if value > EXTREME_PEAK_UV
    ]

    # --------------------------------------------------------
    # PSD
    # --------------------------------------------------------

    freqs, psd = calculate_psd(
        data_uv,
        sfreq,
    )

    psd_mask = (
        (freqs >= PSD_MIN_FREQ)
        & (freqs <= PSD_MAX_FREQ)
    )

    qc_freqs = freqs[psd_mask]
    qc_psd = psd[:, psd_mask]

    # --------------------------------------------------------
    # EEG band powers
    # --------------------------------------------------------

    band_results = {}

    for band_name, (low, high) in BANDS.items():

        band_results[band_name] = band_power(
            freqs,
            psd,
            low,
            high,
        )

    # --------------------------------------------------------
    # 50-Hz line-noise estimate
    # --------------------------------------------------------

    noise_ratio = line_noise_ratio(
        freqs,
        psd,
        50.0,
    )

    # --------------------------------------------------------
    # Channel-layout validation
    # --------------------------------------------------------

    correct_channels = (
        channels == EXPECTED_CHANNELS
    )

    correct_sfreq = np.isclose(
        sfreq,
        EXPECTED_SFREQ,
    )

    # --------------------------------------------------------
    # Overall QC flag
    # --------------------------------------------------------

    flags = []

    if nan_count > 0:
        flags.append("NaN")

    if inf_count > 0:
        flags.append("Inf")

    if flat_channels:
        flags.append("flat_channel")

    if high_std_channels:
        flags.append("high_std")

    if extreme_peak_channels:
        flags.append("extreme_peak")

    if not correct_channels:
        flags.append("channel_layout")

    if not correct_sfreq:
        flags.append("sampling_rate")

    if len(flags) == 0:
        qc_status = "PASS"
    else:
        qc_status = "REVIEW"

    result = {
        "subject_id": record["subject_id"],
        "class": record["class"],
        "filepath": str(path),

        "sfreq": sfreq,
        "n_channels": n_channels,
        "n_samples": n_samples,
        "duration_sec": duration_sec,
        "duration_min": duration_sec / 60,

        "nan_count": nan_count,
        "inf_count": inf_count,

        "mean_uv": float(
            np.nanmean(channel_mean)
        ),

        "median_channel_std_uv": float(
            np.nanmedian(channel_std)
        ),

        "max_channel_std_uv": float(
            np.nanmax(channel_std)
        ),

        "max_abs_peak_uv": float(
            np.nanmax(channel_abs_peak)
        ),

        "max_ptp_uv": float(
            np.nanmax(channel_ptp)
        ),

        "flat_channels": ",".join(
            flat_channels
        ),

        "high_std_channels": ",".join(
            high_std_channels
        ),

        "extreme_peak_channels": ",".join(
            extreme_peak_channels
        ),

        "line_noise_50hz_ratio": noise_ratio,

        "correct_channel_layout": correct_channels,
        "correct_sampling_rate": correct_sfreq,

        "qc_status": qc_status,
        "qc_flags": ",".join(flags),
    }

    for band_name, value in band_results.items():
        result[f"{band_name}_power"] = value

    extra = {
        "data_uv": data_uv,
        "channels": channels,
        "sfreq": sfreq,
        "freqs": qc_freqs,
        "psd": qc_psd,
    }

    return result, extra


# ============================================================
# PLOTTING
# ============================================================

def save_trace_plot(
    subject_id,
    label,
    data_uv,
    channels,
    sfreq,
):

    samples = min(
        int(TRACE_SECONDS * sfreq),
        data_uv.shape[1],
    )

    data = data_uv[:, :samples]

    times = (
        np.arange(samples)
        / sfreq
    )

    # Robust spacing so one extreme channel does not
    # completely destroy the visualization.
    scale = np.nanmedian(
        np.std(data, axis=1)
    )

    if (
        not np.isfinite(scale)
        or scale <= 0
    ):
        scale = 50.0

    spacing = scale * 6

    fig, ax = plt.subplots(
        figsize=(14, 10)
    )

    offsets = (
        np.arange(len(channels))[::-1]
        * spacing
    )

    for i, channel in enumerate(channels):

        ax.plot(
            times,
            data[i] + offsets[i],
            linewidth=0.6,
        )

    ax.set_yticks(offsets)
    ax.set_yticklabels(channels)

    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("EEG channels")

    ax.set_title(
        f"{subject_id} | {label} | "
        f"First {times[-1]:.1f} s"
    )

    fig.tight_layout()

    filename = (
        PLOT_DIR
        / f"{subject_id}_{label}_trace.png"
    )

    fig.savefig(
        filename,
        dpi=150,
    )

    plt.close(fig)


def save_psd_plot(
    subject_id,
    label,
    freqs,
    psd,
):

    # Mean across channels
    mean_psd = np.mean(
        psd,
        axis=0,
    )

    # Avoid log10(0)
    mean_psd = np.maximum(
        mean_psd,
        np.finfo(float).tiny,
    )

    psd_db = (
        10
        * np.log10(mean_psd)
    )

    fig, ax = plt.subplots(
        figsize=(12, 6)
    )

    ax.plot(
        freqs,
        psd_db,
        linewidth=1,
    )

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel(
        "Power spectral density (dB)"
    )

    ax.set_title(
        f"{subject_id} | {label} | Mean PSD"
    )

    ax.set_xlim(
        PSD_MIN_FREQ,
        PSD_MAX_FREQ,
    )

    ax.axvline(
        50,
        linestyle="--",
        linewidth=1,
    )

    fig.tight_layout()

    filename = (
        PLOT_DIR
        / f"{subject_id}_{label}_psd.png"
    )

    fig.savefig(
        filename,
        dpi=150,
    )

    plt.close(fig)


# ============================================================
# DATASET-LEVEL SUMMARY
# ============================================================

def print_summary(df):

    print("\n")
    print("=" * 70)
    print("EEG SIGNAL QC SUMMARY")
    print("=" * 70)

    print(
        f"\nSubjects checked: {len(df)}"
    )

    print("\nQC status:")

    print(
        df["qc_status"]
        .value_counts()
        .to_string()
    )

    print("\nBy class:")

    summary = (
        df.groupby("class")
        .agg(
            subjects=("subject_id", "count"),
            median_std_uv=(
                "median_channel_std_uv",
                "median",
            ),
            median_peak_uv=(
                "max_abs_peak_uv",
                "median",
            ),
            median_50hz_ratio=(
                "line_noise_50hz_ratio",
                "median",
            ),
        )
    )

    print(summary)

    print("\nNumerical problems:")

    print(
        f"  Subjects containing NaN: "
        f"{(df['nan_count'] > 0).sum()}"
    )

    print(
        f"  Subjects containing Inf: "
        f"{(df['inf_count'] > 0).sum()}"
    )

    print(
        f"  Subjects with flat channels: "
        f"{(df['flat_channels'] != '').sum()}"
    )

    print(
        f"  Subjects with extreme peaks: "
        f"{(df['extreme_peak_channels'] != '').sum()}"
    )

    print(
        f"  Subjects with high channel STD: "
        f"{(df['high_std_channels'] != '').sum()}"
    )

    print("\nAmplitude distribution:")

    for column in [
        "median_channel_std_uv",
        "max_abs_peak_uv",
        "max_ptp_uv",
    ]:

        values = df[column]

        print(
            f"\n  {column}"
        )

        print(
            f"    min    = {values.min():.3f}"
        )

        print(
            f"    median = {values.median():.3f}"
        )

        print(
            f"    mean   = {values.mean():.3f}"
        )

        print(
            f"    max    = {values.max():.3f}"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    warnings.filterwarnings(
        "ignore",
        category=RuntimeWarning,
    )

    print("=" * 70)
    print("EEG SIGNAL QUALITY CONTROL")
    print("=" * 70)

    records = find_set_files()

    print(
        f"\nFound {len(records)} EEG recordings.\n"
    )

    if not records:
        raise RuntimeError(
            "No .set files were found."
        )

    results = []

    # Keep temporary plotting data for selected subjects.
    plot_candidates = defaultdict(list)

    failures = []

    for i, record in enumerate(
        records,
        start=1,
    ):

        print(
            f"[{i:03d}/{len(records):03d}] "
            f"{record['subject_id']} | "
            f"{record['class']}"
        )

        try:

            result, extra = inspect_subject(
                record
            )

            results.append(result)

            print(
                f"    {result['qc_status']} | "
                f"STD={result['median_channel_std_uv']:.2f} uV | "
                f"Peak={result['max_abs_peak_uv']:.2f} uV | "
                f"50Hz ratio="
                f"{result['line_noise_50hz_ratio']:.2f}"
            )

            # Store a small number of subjects from each
            # class for representative plots.
            if (
                len(
                    plot_candidates[
                        record["class"]
                    ]
                )
                < N_REPRESENTATIVE
            ):

                plot_candidates[
                    record["class"]
                ].append(
                    (
                        result,
                        extra,
                    )
                )

        except Exception as exc:

            print(
                f"    ERROR: {exc}"
            )

            failures.append({
                "subject_id":
                    record["subject_id"],

                "class":
                    record["class"],

                "filepath":
                    str(record["path"]),

                "error":
                    str(exc),
            })

    if not results:
        raise RuntimeError(
            "No recordings could be inspected."
        )

    # ========================================================
    # SAVE QC TABLE
    # ========================================================

    df = pd.DataFrame(results)

    qc_csv = (
        OUTPUT_DIR
        / "eeg_signal_qc.csv"
    )

    df.to_csv(
        qc_csv,
        index=False,
    )

    # ========================================================
    # SAVE FAILURES
    # ========================================================

    if failures:

        failure_df = pd.DataFrame(
            failures
        )

        failure_df.to_csv(
            OUTPUT_DIR
            / "qc_failures.csv",
            index=False,
        )

    # ========================================================
    # REPRESENTATIVE PLOTS
    # ========================================================

    print("\nGenerating representative plots...")

    for label, subjects in (
        plot_candidates.items()
    ):

        for result, extra in subjects:

            save_trace_plot(
                result["subject_id"],
                label,
                extra["data_uv"],
                extra["channels"],
                extra["sfreq"],
            )

            save_psd_plot(
                result["subject_id"],
                label,
                extra["freqs"],
                extra["psd"],
            )

    # ========================================================
    # SUMMARY
    # ========================================================

    print_summary(df)

    print("\n")
    print("=" * 70)
    print("QC COMPLETE")
    print("=" * 70)

    print(
        f"\nQC table:\n  {qc_csv}"
    )

    print(
        f"\nPlots:\n  {PLOT_DIR}"
    )

    print(
        "\nIMPORTANT: REVIEW flags do not automatically "
        "mean that a recording should be removed."
    )

    print(
        "This script did not modify the EEG recordings."
    )


if __name__ == "__main__":
    main()