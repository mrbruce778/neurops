from pathlib import Path
from collections import Counter

import mne
import pandas as pd
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(r"E:\neurops")

CLASS_FOLDERS = {
    "ec_alzheimer": {
        "class_name": "Alzheimer",
        "label": 0,
    },
    "ec_control": {
        "class_name": "Control",
        "label": 1,
    },
    "ec_ftd": {
        "class_name": "FTD",
        "label": 2,
    },
}

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

OUTPUT_DIR = PROJECT_ROOT / "prepared"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MANIFEST_PATH = OUTPUT_DIR / "subject_manifest.csv"


# ============================================================
# SUBJECT DISCOVERY
# ============================================================

def discover_subjects():

    subjects = []

    print("=" * 70)
    print("DISCOVERING EEG SUBJECTS")
    print("=" * 70)

    for folder_name, class_info in CLASS_FOLDERS.items():

        folder = PROJECT_ROOT / folder_name

        if not folder.exists():

            print(
                f"[WARNING] Folder does not exist: {folder}"
            )

            continue

        set_files = sorted(
            folder.rglob("*.set")
        )

        print(
            f"\n{class_info['class_name']}: "
            f"{len(set_files)} recording(s)"
        )

        for path in set_files:

            subject_id = None

            for parent in path.parents:

                if parent.name.lower().startswith("sub-"):

                    subject_id = parent.name
                    break

            if subject_id is None:

                raise RuntimeError(
                    f"Could not determine subject ID for:\n"
                    f"{path}"
                )

            subjects.append(
                {
                    "subject_id": subject_id,
                    "class_name":
                        class_info["class_name"],
                    "label":
                        class_info["label"],
                    "filepath":
                        str(path.resolve()),
                }
            )

    return subjects


# ============================================================
# VALIDATE ONE SUBJECT
# ============================================================

def validate_subject(subject):

    path = Path(subject["filepath"])

    raw = mne.io.read_raw_eeglab(
        path,
        preload=False,
        verbose="ERROR",
    )

    raw.pick("eeg")

    sfreq = float(
        raw.info["sfreq"]
    )

    channels = list(
        raw.ch_names
    )

    n_channels = len(channels)

    n_samples = int(
        raw.n_times
    )

    duration_seconds = (
        n_samples / sfreq
    )

    duration_minutes = (
        duration_seconds / 60
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    sampling_rate_ok = np.isclose(
        sfreq,
        EXPECTED_SFREQ,
    )

    channel_count_ok = (
        n_channels
        == len(EXPECTED_CHANNELS)
    )

    channel_order_ok = (
        channels
        == EXPECTED_CHANNELS
    )

    missing_channels = [
        ch
        for ch in EXPECTED_CHANNELS
        if ch not in channels
    ]

    unexpected_channels = [
        ch
        for ch in channels
        if ch not in EXPECTED_CHANNELS
    ]

    valid = (
        sampling_rate_ok
        and channel_count_ok
        and channel_order_ok
    )

    result = dict(subject)

    result.update(
        {
            "sfreq": sfreq,
            "n_channels": n_channels,
            "n_samples": n_samples,
            "duration_seconds":
                duration_seconds,
            "duration_minutes":
                duration_minutes,
            "sampling_rate_ok":
                sampling_rate_ok,
            "channel_count_ok":
                channel_count_ok,
            "channel_order_ok":
                channel_order_ok,
            "missing_channels":
                ",".join(
                    missing_channels
                ),
            "unexpected_channels":
                ",".join(
                    unexpected_channels
                ),
            "valid":
                valid,
        }
    )

    return result


# ============================================================
# DUPLICATE CHECKING
# ============================================================

def check_duplicates(df):

    duplicate_ids = (
        df["subject_id"]
        .duplicated(
            keep=False
        )
    )

    if duplicate_ids.any():

        print("\n[ERROR] Duplicate subject IDs detected:")

        print(
            df.loc[
                duplicate_ids,
                [
                    "subject_id",
                    "class_name",
                    "filepath",
                ],
            ].to_string(
                index=False
            )
        )

        raise RuntimeError(
            "Duplicate subject IDs detected."
        )

    duplicate_files = (
        df["filepath"]
        .duplicated(
            keep=False
        )
    )

    if duplicate_files.any():

        print("\n[ERROR] Duplicate file paths detected:")

        print(
            df.loc[
                duplicate_files,
                [
                    "subject_id",
                    "filepath",
                ],
            ].to_string(
                index=False
            )
        )

        raise RuntimeError(
            "Duplicate EEG files detected."
        )


# ============================================================
# PRINT SUMMARY
# ============================================================

def print_summary(df):

    print("\n")
    print("=" * 70)
    print("PREPARED DATASET SUMMARY")
    print("=" * 70)

    print(
        f"\nTotal subjects: {len(df)}"
    )

    print("\nClass distribution:")

    class_summary = (
        df.groupby(
            [
                "class_name",
                "label",
            ]
        )
        .size()
        .reset_index(
            name="subjects"
        )
    )

    print(
        class_summary.to_string(
            index=False
        )
    )

    print("\nSampling frequencies:")

    print(
        df["sfreq"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print("\nChannel counts:")

    print(
        df["n_channels"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print("\nRecording duration:")

    print(
        f"  Minimum: "
        f"{df['duration_minutes'].min():.2f} min"
    )

    print(
        f"  Maximum: "
        f"{df['duration_minutes'].max():.2f} min"
    )

    print(
        f"  Mean:    "
        f"{df['duration_minutes'].mean():.2f} min"
    )

    print(
        f"  Median:  "
        f"{df['duration_minutes'].median():.2f} min"
    )

    print("\nValidation:")

    print(
        f"  Valid subjects: "
        f"{df['valid'].sum()}/{len(df)}"
    )

    print(
        f"  Correct sampling rate: "
        f"{df['sampling_rate_ok'].sum()}/{len(df)}"
    )

    print(
        f"  Correct channel count: "
        f"{df['channel_count_ok'].sum()}/{len(df)}"
    )

    print(
        f"  Correct channel order: "
        f"{df['channel_order_ok'].sum()}/{len(df)}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("EEG DATA PREPARATION")
    print("=" * 70)

    print(
        "\nThis script creates the master subject manifest."
    )

    print(
        "It does NOT modify EEG signals."
    )

    print(
        "It does NOT create train/test splits.\n"
    )

    # --------------------------------------------------------
    # Discover subjects
    # --------------------------------------------------------

    subjects = discover_subjects()

    if len(subjects) == 0:

        raise RuntimeError(
            "No EEG subjects were found."
        )

    print(
        f"\nTotal discovered subjects: "
        f"{len(subjects)}"
    )

    # --------------------------------------------------------
    # Validate subjects
    # --------------------------------------------------------

    validated = []

    print("\n")
    print("=" * 70)
    print("VALIDATING SUBJECTS")
    print("=" * 70)

    for index, subject in enumerate(
        subjects,
        start=1,
    ):

        print(
            f"[{index:03d}/{len(subjects):03d}] "
            f"{subject['subject_id']} | "
            f"{subject['class_name']}"
        )

        try:

            result = validate_subject(
                subject
            )

            validated.append(
                result
            )

            status = (
                "OK"
                if result["valid"]
                else "CHECK"
            )

            print(
                f"    {status} | "
                f"{result['sfreq']:.1f} Hz | "
                f"{result['n_channels']} channels | "
                f"{result['duration_minutes']:.2f} min"
            )

        except Exception as exc:

            print(
                f"    ERROR: {exc}"
            )

            raise

    # --------------------------------------------------------
    # Create dataframe
    # --------------------------------------------------------

    df = pd.DataFrame(
        validated
    )

    # Sort by numeric subject number when possible.
    def subject_number(subject_id):

        try:
            return int(
                subject_id.split("-")[1]
            )

        except Exception:
            return 999999

    df["_subject_number"] = (
        df["subject_id"]
        .apply(subject_number)
    )

    df = (
        df.sort_values(
            "_subject_number"
        )
        .drop(
            columns=[
                "_subject_number"
            ]
        )
        .reset_index(
            drop=True
        )
    )

    # --------------------------------------------------------
    # Duplicate checks
    # --------------------------------------------------------

    check_duplicates(df)

    # --------------------------------------------------------
    # Class consistency
    # --------------------------------------------------------

    label_mapping = (
        df[
            [
                "class_name",
                "label",
            ]
        ]
        .drop_duplicates()
        .sort_values(
            "label"
        )
    )

    print("\n")
    print("=" * 70)
    print("CLASS MAPPING")
    print("=" * 70)

    print(
        label_mapping.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Dataset validation
    # --------------------------------------------------------

    invalid = df[
        ~df["valid"]
    ]

    if len(invalid) > 0:

        print("\n")
        print("=" * 70)
        print("WARNING: INVALID SUBJECTS")
        print("=" * 70)

        print(
            invalid[
                [
                    "subject_id",
                    "class_name",
                    "sampling_rate_ok",
                    "channel_count_ok",
                    "channel_order_ok",
                    "missing_channels",
                    "unexpected_channels",
                ]
            ].to_string(
                index=False
            )
        )

        raise RuntimeError(
            "Dataset validation failed. "
            "Manifest was not saved."
        )

    # --------------------------------------------------------
    # Save master manifest
    # --------------------------------------------------------

    df.to_csv(
        MANIFEST_PATH,
        index=False,
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print_summary(df)

    print("\n")
    print("=" * 70)
    print("DATA PREPARATION COMPLETE")
    print("=" * 70)

    print(
        f"\nMaster manifest saved to:\n"
        f"  {MANIFEST_PATH}"
    )

    print(
        "\nNo EEG data were modified."
    )

    print(
        "No train/validation/test assignment "
        "has been made yet."
    )


if __name__ == "__main__":
    main()