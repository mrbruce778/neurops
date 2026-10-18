from pathlib import Path
import sys
import csv
from collections import Counter, defaultdict

import mne
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(r"E:\neurops")

# Add/remove folders here depending on what exists in your project.
#
# Format:
#     "folder_name": "class_label"
#
CLASS_FOLDERS = {
    "ec_alzheimer": "Alzheimer",
    "ec_control": "Control",
    "ec_ftd": "FTD",
}

OUTPUT_DIR = PROJECT_ROOT / "inspection_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MANIFEST_FILE = OUTPUT_DIR / "eeg_manifest.csv"
CHANNEL_FILE = OUTPUT_DIR / "channel_summary.txt"
SUMMARY_FILE = OUTPUT_DIR / "dataset_summary.txt"


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def find_set_files():
    """
    Find all EEGLAB .set files inside the configured class folders.
    """

    records = []

    for folder_name, label in CLASS_FOLDERS.items():

        class_dir = PROJECT_ROOT / folder_name

        if not class_dir.exists():
            print(f"[INFO] Folder not found, skipping: {class_dir}")
            continue

        files = sorted(class_dir.rglob("*.set"))

        print(
            f"[FOUND] {label:<12} "
            f"{len(files):>4} .set file(s) in {class_dir}"
        )

        for file_path in files:

            # Try to identify subject from directory structure.
            subject_id = None

            for parent in file_path.parents:
                if parent.name.lower().startswith("sub-"):
                    subject_id = parent.name
                    break

            if subject_id is None:
                subject_id = file_path.stem

            records.append(
                {
                    "label": label,
                    "subject_id": subject_id,
                    "path": file_path,
                }
            )

    return records


def load_eeg(file_path):
    """
    Load an EEGLAB .set file without loading all EEG samples
    into memory unless required.
    """

    raw = mne.io.read_raw_eeglab(
        file_path,
        preload=False,
        verbose="ERROR",
    )

    return raw


def inspect_record(record):
    """
    Extract useful metadata from one EEG recording.
    """

    file_path = record["path"]

    raw = load_eeg(file_path)

    sfreq = float(raw.info["sfreq"])

    n_channels = len(raw.ch_names)

    n_samples = raw.n_times

    duration_seconds = n_samples / sfreq

    duration_minutes = duration_seconds / 60

    channel_names = raw.ch_names

    channel_types = raw.get_channel_types()

    eeg_channel_count = channel_types.count("eeg")

    bad_channels = list(raw.info["bads"])

    annotations_count = len(raw.annotations)

    return {
        "class": record["label"],
        "subject_id": record["subject_id"],
        "filename": file_path.name,
        "filepath": str(file_path),
        "sampling_rate_hz": sfreq,
        "n_channels": n_channels,
        "n_eeg_channels": eeg_channel_count,
        "n_samples": n_samples,
        "duration_seconds": duration_seconds,
        "duration_minutes": duration_minutes,
        "n_annotations": annotations_count,
        "bad_channels": ";".join(bad_channels),
        "channel_names": channel_names,
        "channel_types": channel_types,
    }


# ============================================================
# MAIN INSPECTION
# ============================================================

def main():

    print("=" * 70)
    print("EEG DATASET INSPECTION")
    print("=" * 70)

    print(f"\nProject root: {PROJECT_ROOT}\n")

    # --------------------------------------------------------
    # 1. Find .set files
    # --------------------------------------------------------

    records = find_set_files()

    if not records:
        print("\nERROR: No .set files were found.")
        print("Check PROJECT_ROOT and CLASS_FOLDERS.")
        sys.exit(1)

    print(f"\nTotal .set files found: {len(records)}")

    # --------------------------------------------------------
    # 2. Inspect each file
    # --------------------------------------------------------

    inspected = []

    failed_files = []

    print("\n" + "=" * 70)
    print("READING EEG METADATA")
    print("=" * 70)

    for i, record in enumerate(records, start=1):

        print(
            f"[{i:03d}/{len(records):03d}] "
            f"{record['label']} | "
            f"{record['subject_id']} | "
            f"{record['path'].name}"
        )

        try:

            result = inspect_record(record)

            inspected.append(result)

            print(
                f"      Channels: {result['n_channels']} "
                f"(EEG={result['n_eeg_channels']}) | "
                f"Fs: {result['sampling_rate_hz']:.2f} Hz | "
                f"Duration: {result['duration_minutes']:.2f} min"
            )

        except Exception as exc:

            print(f"      [ERROR] {exc}")

            failed_files.append(
                {
                    "path": str(record["path"]),
                    "error": str(exc),
                }
            )

    if not inspected:
        print("\nNo files could be successfully loaded.")
        sys.exit(1)

    # --------------------------------------------------------
    # 3. Dataset statistics
    # --------------------------------------------------------

    class_counts = Counter(
        item["class"]
        for item in inspected
    )

    subjects_by_class = defaultdict(set)

    for item in inspected:
        subjects_by_class[item["class"]].add(
            item["subject_id"]
        )

    sampling_rates = Counter(
        item["sampling_rate_hz"]
        for item in inspected
    )

    channel_counts = Counter(
        item["n_channels"]
        for item in inspected
    )

    durations = np.array(
        [
            item["duration_minutes"]
            for item in inspected
        ],
        dtype=float,
    )

    # Channel layouts
    channel_layouts = Counter(
        tuple(item["channel_names"])
        for item in inspected
    )

    all_channels = Counter()

    for item in inspected:
        all_channels.update(item["channel_names"])

    # --------------------------------------------------------
    # 4. Print summary
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("DATASET SUMMARY")
    print("=" * 70)

    print(f"\nSuccessfully loaded files: {len(inspected)}")
    print(f"Failed files:              {len(failed_files)}")

    print("\nSubjects / recordings by class:")

    for label in sorted(class_counts):

        print(
            f"  {label:<15} "
            f"Subjects: {len(subjects_by_class[label]):>3} | "
            f"Recordings: {class_counts[label]:>3}"
        )

    total_subjects = len(
        {
            (item["class"], item["subject_id"])
            for item in inspected
        }
    )

    print(f"\nTotal subjects: {total_subjects}")

    print("\nSampling rates:")

    for rate, count in sorted(sampling_rates.items()):
        print(
            f"  {rate:.2f} Hz : "
            f"{count} recording(s)"
        )

    print("\nChannel counts:")

    for count, recordings in sorted(channel_counts.items()):
        print(
            f"  {count} channels : "
            f"{recordings} recording(s)"
        )

    print("\nRecording duration:")

    print(
        f"  Minimum : {durations.min():.2f} min"
    )

    print(
        f"  Maximum : {durations.max():.2f} min"
    )

    print(
        f"  Mean    : {durations.mean():.2f} min"
    )

    print(
        f"  Median  : {np.median(durations):.2f} min"
    )

    print(
        f"\nUnique channel layouts: {len(channel_layouts)}"
    )

    # --------------------------------------------------------
    # 5. Display first recording in detail
    # --------------------------------------------------------

    first = inspected[0]

    print("\n")
    print("=" * 70)
    print("EXAMPLE RECORDING")
    print("=" * 70)

    print(f"Subject:       {first['subject_id']}")
    print(f"Class:         {first['class']}")
    print(f"File:          {first['filename']}")
    print(
        f"Sampling rate: {first['sampling_rate_hz']:.2f} Hz"
    )
    print(f"Channels:      {first['n_channels']}")
    print(
        f"EEG channels:  {first['n_eeg_channels']}"
    )
    print(
        f"Duration:      {first['duration_minutes']:.2f} min"
    )

    print("\nChannel names:")

    print(", ".join(first["channel_names"]))

    print("\nChannel types:")

    for name, channel_type in zip(
        first["channel_names"],
        first["channel_types"],
    ):
        print(
            f"  {name:<15} {channel_type}"
        )

    # --------------------------------------------------------
    # 6. Save manifest CSV
    # --------------------------------------------------------

    csv_fields = [
        "class",
        "subject_id",
        "filename",
        "filepath",
        "sampling_rate_hz",
        "n_channels",
        "n_eeg_channels",
        "n_samples",
        "duration_seconds",
        "duration_minutes",
        "n_annotations",
        "bad_channels",
    ]

    with MANIFEST_FILE.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=csv_fields,
        )

        writer.writeheader()

        for item in inspected:

            row = {
                field: item[field]
                for field in csv_fields
            }

            writer.writerow(row)

    # --------------------------------------------------------
    # 7. Save channel summary
    # --------------------------------------------------------

    with CHANNEL_FILE.open(
        "w",
        encoding="utf-8",
    ) as f:

        f.write("EEG CHANNEL SUMMARY\n")
        f.write("=" * 70 + "\n\n")

        f.write(
            f"Unique channel layouts: "
            f"{len(channel_layouts)}\n\n"
        )

        for layout_number, (layout, count) in enumerate(
            channel_layouts.items(),
            start=1,
        ):

            f.write(
                f"Layout {layout_number} "
                f"({count} recordings)\n"
            )

            f.write(
                ", ".join(layout)
            )

            f.write("\n\n")

        f.write("\nCHANNEL OCCURRENCE\n")
        f.write("=" * 70 + "\n")

        for channel, count in sorted(
            all_channels.items()
        ):

            f.write(
                f"{channel:<15} "
                f"{count}/{len(inspected)} recordings\n"
            )

    # --------------------------------------------------------
    # 8. Save text summary
    # --------------------------------------------------------

    with SUMMARY_FILE.open(
        "w",
        encoding="utf-8",
    ) as f:

        f.write("EEG DATASET SUMMARY\n")
        f.write("=" * 70 + "\n\n")

        f.write(
            f"Successful recordings: {len(inspected)}\n"
        )

        f.write(
            f"Failed recordings: {len(failed_files)}\n"
        )

        f.write(
            f"Total subjects: {total_subjects}\n\n"
        )

        f.write("CLASS DISTRIBUTION\n")
        f.write("-" * 40 + "\n")

        for label in sorted(class_counts):

            f.write(
                f"{label}: "
                f"{len(subjects_by_class[label])} subjects, "
                f"{class_counts[label]} recordings\n"
            )

        f.write("\nSAMPLING RATES\n")
        f.write("-" * 40 + "\n")

        for rate, count in sorted(
            sampling_rates.items()
        ):

            f.write(
                f"{rate:.2f} Hz: "
                f"{count} recordings\n"
            )

        f.write("\nCHANNEL COUNTS\n")
        f.write("-" * 40 + "\n")

        for count, recordings in sorted(
            channel_counts.items()
        ):

            f.write(
                f"{count} channels: "
                f"{recordings} recordings\n"
            )

        f.write("\nDURATION\n")
        f.write("-" * 40 + "\n")

        f.write(
            f"Minimum: {durations.min():.2f} min\n"
        )

        f.write(
            f"Maximum: {durations.max():.2f} min\n"
        )

        f.write(
            f"Mean: {durations.mean():.2f} min\n"
        )

        f.write(
            f"Median: {np.median(durations):.2f} min\n"
        )

        f.write(
            f"\nUnique channel layouts: "
            f"{len(channel_layouts)}\n"
        )

        if failed_files:

            f.write("\nFAILED FILES\n")
            f.write("-" * 40 + "\n")

            for failure in failed_files:

                f.write(
                    f"\n{failure['path']}\n"
                )

                f.write(
                    f"Error: {failure['error']}\n"
                )

    # --------------------------------------------------------
    # Finished
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("INSPECTION COMPLETE")
    print("=" * 70)

    print("\nFiles created:")

    print(f"  {MANIFEST_FILE}")
    print(f"  {CHANNEL_FILE}")
    print(f"  {SUMMARY_FILE}")

    if failed_files:

        print(
            f"\nWARNING: {len(failed_files)} file(s) "
            "could not be loaded."
        )

    print(
        "\nNo EEG data were modified during this inspection."
    )


if __name__ == "__main__":
    main()