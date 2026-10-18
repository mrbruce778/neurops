from pathlib import Path
import json

import mne
import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(r"E:\neurops")

MANIFEST_PATH = (
    PROJECT_ROOT
    / "prepared"
    / "subject_manifest.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "windows"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

WINDOW_MANIFEST_PATH = (
    OUTPUT_DIR
    / "window_manifest.csv"
)

SUMMARY_PATH = (
    OUTPUT_DIR
    / "window_summary.txt"
)

CONFIG_PATH = (
    OUTPUT_DIR
    / "window_config.json"
)


# ------------------------------------------------------------
# Window configuration
# ------------------------------------------------------------

WINDOW_SECONDS = 4.0
OVERLAP = 0.50

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


# ============================================================
# LOAD SUBJECT
# ============================================================

def load_subject(filepath):

    raw = mne.io.read_raw_eeglab(
        filepath,
        preload=True,
        verbose="ERROR",
    )

    raw.pick("eeg")

    sfreq = float(
        raw.info["sfreq"]
    )

    channels = list(
        raw.ch_names
    )

    if not np.isclose(
        sfreq,
        EXPECTED_SFREQ,
    ):
        raise RuntimeError(
            f"Unexpected sampling rate: {sfreq}"
        )

    if channels != EXPECTED_CHANNELS:
        raise RuntimeError(
            "Unexpected EEG channel layout."
        )

    # MNE EEG data are internally stored in volts.
    data = raw.get_data()

    # Convert to float32 to reduce storage/memory usage.
    data = data.astype(
        np.float32,
        copy=False,
    )

    return data, sfreq


# ============================================================
# WINDOW CREATION
# ============================================================

def create_windows(
    data,
    sfreq,
):

    window_samples = int(
        round(
            WINDOW_SECONDS
            * sfreq
        )
    )

    step_samples = int(
        round(
            window_samples
            * (1.0 - OVERLAP)
        )
    )

    if step_samples <= 0:
        raise ValueError(
            "Window step must be greater than zero."
        )

    n_channels, n_samples = (
        data.shape
    )

    if n_samples < window_samples:

        return (
            np.empty(
                (
                    0,
                    n_channels,
                    window_samples,
                ),
                dtype=np.float32,
            ),
            np.empty(
                0,
                dtype=np.int64,
            ),
        )

    starts = np.arange(
        0,
        n_samples - window_samples + 1,
        step_samples,
        dtype=np.int64,
    )

    windows = np.empty(
        (
            len(starts),
            n_channels,
            window_samples,
        ),
        dtype=np.float32,
    )

    for i, start in enumerate(
        starts
    ):

        end = (
            start
            + window_samples
        )

        windows[i] = (
            data[:, start:end]
        )

    return windows, starts


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("EEG WINDOW CREATION")
    print("=" * 70)

    print(
        f"\nWindow length: "
        f"{WINDOW_SECONDS:.1f} seconds"
    )

    print(
        f"Overlap: "
        f"{OVERLAP * 100:.0f}%"
    )

    print(
        f"Sampling rate: "
        f"{EXPECTED_SFREQ:.1f} Hz"
    )

    window_samples = int(
        WINDOW_SECONDS
        * EXPECTED_SFREQ
    )

    step_samples = int(
        window_samples
        * (1 - OVERLAP)
    )

    print(
        f"Samples/window: "
        f"{window_samples}"
    )

    print(
        f"Step: "
        f"{step_samples} samples "
        f"({step_samples / EXPECTED_SFREQ:.2f} sec)"
    )

    print(
        "\nIMPORTANT:"
        "\n  No filtering is performed."
        "\n  No normalization is performed."
        "\n  No train/test split is performed."
    )

    # --------------------------------------------------------
    # Load master manifest
    # --------------------------------------------------------

    if not MANIFEST_PATH.exists():

        raise FileNotFoundError(
            f"Manifest not found:\n"
            f"{MANIFEST_PATH}"
        )

    manifest = pd.read_csv(
        MANIFEST_PATH
    )

    print(
        f"\nSubjects in manifest: "
        f"{len(manifest)}"
    )

    # --------------------------------------------------------
    # Process every subject
    # --------------------------------------------------------

    subject_records = []

    failures = []

    total_windows = 0

    print("\n")
    print("=" * 70)
    print("CREATING WINDOWS")
    print("=" * 70)

    for index, row in manifest.iterrows():

        subject_id = str(
            row["subject_id"]
        )

        class_name = str(
            row["class_name"]
        )

        label = int(
            row["label"]
        )

        filepath = Path(
            row["filepath"]
        )

        print(
            f"[{index + 1:03d}/"
            f"{len(manifest):03d}] "
            f"{subject_id} | "
            f"{class_name}"
        )

        try:

            # ------------------------------------------------
            # Load EEG
            # ------------------------------------------------

            data, sfreq = (
                load_subject(
                    filepath
                )
            )

            # ------------------------------------------------
            # Create windows
            # ------------------------------------------------

            windows, starts = (
                create_windows(
                    data,
                    sfreq,
                )
            )

            if len(windows) == 0:

                raise RuntimeError(
                    "Recording is shorter "
                    "than one complete window."
                )

            ends = (
                starts
                + windows.shape[-1]
            )

            # ------------------------------------------------
            # Safety check
            # ------------------------------------------------

            expected_shape = (
                len(EXPECTED_CHANNELS),
                window_samples,
            )

            if windows.shape[1:] != (
                expected_shape
            ):

                raise RuntimeError(
                    f"Unexpected window shape: "
                    f"{windows.shape}"
                )

            if not np.isfinite(
                windows
            ).all():

                raise RuntimeError(
                    "NaN or Inf detected "
                    "inside windows."
                )

            # ------------------------------------------------
            # Save one file per subject
            # ------------------------------------------------

            output_file = (
                OUTPUT_DIR
                / f"{subject_id}.npz"
            )

            np.savez_compressed(
                output_file,

                eeg=windows,

                starts=starts,

                ends=ends,

                subject_id=np.array(
                    subject_id
                ),

                class_name=np.array(
                    class_name
                ),

                label=np.array(
                    label,
                    dtype=np.int64,
                ),

                sfreq=np.array(
                    sfreq,
                    dtype=np.float32,
                ),

                window_seconds=np.array(
                    WINDOW_SECONDS,
                    dtype=np.float32,
                ),

                overlap=np.array(
                    OVERLAP,
                    dtype=np.float32,
                ),

                channels=np.array(
                    EXPECTED_CHANNELS
                ),
            )

            n_windows = int(
                len(windows)
            )

            total_windows += (
                n_windows
            )

            duration_min = (
                data.shape[1]
                / sfreq
                / 60
            )

            subject_records.append(
                {
                    "subject_id":
                        subject_id,

                    "class_name":
                        class_name,

                    "label":
                        label,

                    "source_file":
                        str(filepath),

                    "window_file":
                        str(
                            output_file.resolve()
                        ),

                    "n_windows":
                        n_windows,

                    "n_channels":
                        windows.shape[1],

                    "samples_per_window":
                        windows.shape[2],

                    "window_seconds":
                        WINDOW_SECONDS,

                    "overlap":
                        OVERLAP,

                    "step_seconds":
                        step_samples
                        / sfreq,

                    "sfreq":
                        sfreq,

                    "recording_minutes":
                        duration_min,
                }
            )

            print(
                f"    {n_windows} windows | "
                f"shape={windows.shape}"
            )

            # Free large arrays before next subject.
            del data
            del windows

        except Exception as exc:

            print(
                f"    ERROR: {exc}"
            )

            failures.append(
                {
                    "subject_id":
                        subject_id,

                    "class_name":
                        class_name,

                    "filepath":
                        str(filepath),

                    "error":
                        str(exc),
                }
            )

    # --------------------------------------------------------
    # Fail if any subject failed
    # --------------------------------------------------------

    if failures:

        failure_df = pd.DataFrame(
            failures
        )

        failure_path = (
            OUTPUT_DIR
            / "window_failures.csv"
        )

        failure_df.to_csv(
            failure_path,
            index=False,
        )

        raise RuntimeError(
            f"\n{len(failures)} subject(s) "
            f"failed during window creation.\n"
            f"See:\n{failure_path}"
        )

    # --------------------------------------------------------
    # Create window manifest
    # --------------------------------------------------------

    window_manifest = (
        pd.DataFrame(
            subject_records
        )
    )

    window_manifest.to_csv(
        WINDOW_MANIFEST_PATH,
        index=False,
    )

    # --------------------------------------------------------
    # Save configuration
    # --------------------------------------------------------

    config = {
        "window_seconds":
            WINDOW_SECONDS,

        "overlap":
            OVERLAP,

        "sampling_rate_hz":
            EXPECTED_SFREQ,

        "samples_per_window":
            window_samples,

        "step_samples":
            step_samples,

        "step_seconds":
            step_samples
            / EXPECTED_SFREQ,

        "n_channels":
            len(EXPECTED_CHANNELS),

        "channels":
            EXPECTED_CHANNELS,

        "normalization":
            "none",

        "filtering":
            "none",

        "split_assignment":
            "none",
    }

    with CONFIG_PATH.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            config,
            f,
            indent=4,
        )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("WINDOW DATASET SUMMARY")
    print("=" * 70)

    print(
        f"\nSubjects processed: "
        f"{len(window_manifest)}"
    )

    print(
        f"Total windows: "
        f"{total_windows}"
    )

    print(
        "\nWindows by class:"
    )

    class_summary = (
        window_manifest
        .groupby(
            "class_name"
        )
        .agg(
            subjects=(
                "subject_id",
                "count",
            ),
            windows=(
                "n_windows",
                "sum",
            ),
            min_windows=(
                "n_windows",
                "min",
            ),
            median_windows=(
                "n_windows",
                "median",
            ),
            max_windows=(
                "n_windows",
                "max",
            ),
        )
    )

    print(
        class_summary.to_string()
    )

    print(
        "\nWindows per subject:"
    )

    print(
        f"  Minimum: "
        f"{window_manifest['n_windows'].min()}"
    )

    print(
        f"  Maximum: "
        f"{window_manifest['n_windows'].max()}"
    )

    print(
        f"  Mean:    "
        f"{window_manifest['n_windows'].mean():.2f}"
    )

    print(
        f"  Median:  "
        f"{window_manifest['n_windows'].median():.2f}"
    )

    # --------------------------------------------------------
    # Save text summary
    # --------------------------------------------------------

    with SUMMARY_PATH.open(
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "EEG WINDOW DATASET SUMMARY\n"
        )

        f.write(
            "=" * 70
            + "\n\n"
        )

        f.write(
            f"Subjects: "
            f"{len(window_manifest)}\n"
        )

        f.write(
            f"Total windows: "
            f"{total_windows}\n"
        )

        f.write(
            f"Window length: "
            f"{WINDOW_SECONDS} seconds\n"
        )

        f.write(
            f"Sampling rate: "
            f"{EXPECTED_SFREQ} Hz\n"
        )

        f.write(
            f"Samples/window: "
            f"{window_samples}\n"
        )

        f.write(
            f"Channels: "
            f"{len(EXPECTED_CHANNELS)}\n"
        )

        f.write(
            f"Overlap: "
            f"{OVERLAP * 100:.0f}%\n"
        )

        f.write(
            f"Step: "
            f"{step_samples / EXPECTED_SFREQ:.2f} seconds\n"
        )

        f.write(
            "\nWindows by class:\n"
        )

        f.write(
            class_summary.to_string()
        )

        f.write(
            "\n\nWindows per subject:\n"
        )

        f.write(
            f"Minimum: "
            f"{window_manifest['n_windows'].min()}\n"
        )

        f.write(
            f"Maximum: "
            f"{window_manifest['n_windows'].max()}\n"
        )

        f.write(
            f"Mean: "
            f"{window_manifest['n_windows'].mean():.2f}\n"
        )

        f.write(
            f"Median: "
            f"{window_manifest['n_windows'].median():.2f}\n"
        )

    # --------------------------------------------------------
    # Done
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("WINDOW CREATION COMPLETE")
    print("=" * 70)

    print(
        f"\nWindow directory:\n"
        f"  {OUTPUT_DIR}"
    )

    print(
        f"\nManifest:\n"
        f"  {WINDOW_MANIFEST_PATH}"
    )

    print(
        f"\nConfiguration:\n"
        f"  {CONFIG_PATH}"
    )

    print(
        "\nNo normalization was applied."
    )

    print(
        "No train/validation/test split "
        "was applied."
    )


if __name__ == "__main__":
    main()