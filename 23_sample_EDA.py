"""
Standard EDA for the preprocessed EEG dataset.

Expected project folders:
    E:\neurops\ec_alzheimer
    E:\neurops\ec_control
    E:\neurops\ec_ftd

Outputs:
    E:\neurops\eda_preprocessed\
        eda_manifest.csv
        subject_eda_summary.csv
        class_eda_summary.csv
        subject_relative_bandpower.csv
        spectral_pca_coordinates.csv
        dataset_eda_summary.json
        EDA_REPORT.txt
        plots\*.png

This script performs EDA only. It does not modify the EEG files.
"""

from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mne

from scipy.signal import welch
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


PROJECT_ROOT = Path(r"E:\neurops")

CLASS_FOLDERS = {
    "Alzheimer": PROJECT_ROOT / "ec_alzheimer",
    "Control": PROJECT_ROOT / "ec_control",
    "FTD": PROJECT_ROOT / "ec_ftd",
}

OUTPUT_DIR = PROJECT_ROOT / "eda_preprocessed"
PLOTS_DIR = OUTPUT_DIR / "plots"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

EXPECTED_SFREQ = 500.0
EXPECTED_N_CHANNELS = 19

EXPECTED_CHANNELS = [
    "Fp1", "Fp2", "F3", "F4",
    "C3", "C4", "P3", "P4",
    "O1", "O2", "F7", "F8",
    "T3", "T4", "T5", "T6",
    "Fz", "Cz", "Pz",
]

BANDS = {
    "Delta": (0.5, 4.0),
    "Theta": (4.0, 8.0),
    "Alpha": (8.0, 13.0),
    "Beta": (13.0, 30.0),
    "Gamma": (30.0, 45.0),
}

PSD_FMIN = 0.5
PSD_FMAX = 45.0
NPERSEG_SECONDS = 2.0
OVERLAP_FRACTION = 0.50
EXAMPLE_SECONDS = 10.0
CORRELATION_SECONDS = 60.0
RANDOM_SEED = 42


def safe_subject_id(set_file: Path) -> str:
    for part in set_file.parts:
        if part.lower().startswith("sub-"):
            return part
    stem = set_file.stem
    return stem.split("_task-")[0] if "_task-" in stem else stem


def find_set_files():
    rows = []
    for class_name, folder in CLASS_FOLDERS.items():
        if not folder.exists():
            raise FileNotFoundError(f"Class folder not found: {folder}")
        files = sorted(folder.rglob("*.set"))
        print(f"{class_name:10s}: {len(files)} .set files")
        for file in files:
            rows.append(
                {
                    "class_name": class_name,
                    "subject_id": safe_subject_id(file),
                    "set_file": str(file),
                }
            )
    manifest = pd.DataFrame(rows)
    if manifest.empty:
        raise RuntimeError("No .set files were found.")
    return manifest


def load_raw(set_file: Path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw = mne.io.read_raw_eeglab(
            set_file,
            preload=True,
            verbose="ERROR",
        )
    return raw


def get_data_uv(raw):
    return raw.get_data().astype(np.float64, copy=False) * 1e6


def integrate(y, x, axis=-1):
    return np.trapezoid(y, x, axis=axis) if hasattr(np, "trapezoid") else np.trapz(y, x, axis=axis)


def compute_psd_and_relative_bandpower(data_uv, sfreq):
    nperseg = int(NPERSEG_SECONDS * sfreq)
    noverlap = int(nperseg * OVERLAP_FRACTION)

    freqs, psd = welch(
        data_uv,
        fs=sfreq,
        nperseg=nperseg,
        noverlap=noverlap,
        axis=-1,
        detrend="constant",
        scaling="density",
    )

    total_mask = (freqs >= PSD_FMIN) & (freqs <= PSD_FMAX)
    total_power = integrate(psd[:, total_mask], freqs[total_mask], axis=1)
    total_power = np.maximum(total_power, 1e-12)

    relative = {}
    for band, (low, high) in BANDS.items():
        mask = (freqs >= low) & (freqs < high)
        band_power = integrate(psd[:, mask], freqs[mask], axis=1)
        relative[band] = band_power / total_power

    keep = (freqs >= PSD_FMIN) & (freqs <= PSD_FMAX)
    return freqs[keep], psd[:, keep], relative


def save_plot(filename):
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / filename, dpi=220, bbox_inches="tight")
    plt.close()


def main():
    np.random.seed(RANDOM_SEED)

    print("=" * 72)
    print("PREPROCESSED EEG STANDARD EDA")
    print("=" * 72)

    manifest = find_set_files()
    manifest.to_csv(OUTPUT_DIR / "eda_manifest.csv", index=False)

    subject_rows = []
    bandpower_rows = []

    class_psd_sum = {}
    class_psd_count = {}
    class_corr_sum = {}
    class_corr_count = {}
    common_psd_freqs = None
    example_saved = set()

    for i, row in manifest.iterrows():
        class_name = row["class_name"]
        subject_id = row["subject_id"]
        set_file = Path(row["set_file"])

        print(f"[{i+1:02d}/{len(manifest)}] {class_name:10s} {subject_id}")

        try:
            raw = load_raw(set_file)
            sfreq = float(raw.info["sfreq"])
            ch_names = list(raw.ch_names)
            n_channels = len(ch_names)
            n_samples = int(raw.n_times)
            duration_sec = n_samples / sfreq
            data_uv = get_data_uv(raw)

            channel_means = np.mean(data_uv, axis=1)
            channel_stds = np.std(data_uv, axis=1)
            channel_rms = np.sqrt(np.mean(data_uv ** 2, axis=1))
            channel_ptp = np.ptp(data_uv, axis=1)

            freqs, psd, rel_bp = compute_psd_and_relative_bandpower(data_uv, sfreq)

            mean_psd = np.mean(psd, axis=0)
            if common_psd_freqs is None:
                common_psd_freqs = freqs.copy()

            if len(freqs) != len(common_psd_freqs) or not np.allclose(freqs, common_psd_freqs):
                mean_psd = np.interp(common_psd_freqs, freqs, mean_psd)

            class_psd_sum.setdefault(class_name, np.zeros_like(common_psd_freqs, dtype=np.float64))
            class_psd_count.setdefault(class_name, 0)
            class_psd_sum[class_name] += mean_psd
            class_psd_count[class_name] += 1

            bp_row = {"subject_id": subject_id, "class_name": class_name}
            for band in BANDS:
                values = rel_bp[band]
                bp_row[f"{band.lower()}_mean"] = float(np.mean(values))
                bp_row[f"{band.lower()}_median"] = float(np.median(values))
                for ch_i, ch in enumerate(ch_names):
                    bp_row[f"{ch}_{band}"] = float(values[ch_i])
            bandpower_rows.append(bp_row)

            corr_samples = min(n_samples, int(CORRELATION_SECONDS * sfreq))
            corr = np.corrcoef(data_uv[:, :corr_samples])

            class_corr_sum.setdefault(
                class_name,
                np.zeros((n_channels, n_channels), dtype=np.float64),
            )
            class_corr_count.setdefault(class_name, 0)
            class_corr_sum[class_name] += corr
            class_corr_count[class_name] += 1

            if class_name not in example_saved:
                seconds = min(EXAMPLE_SECONDS, duration_sec)
                samples = int(seconds * sfreq)
                example = data_uv[:, :samples]
                times = np.arange(samples) / sfreq
                robust_scale = np.median(np.std(example, axis=1))
                offset = max(robust_scale * 4, 50.0)

                plt.figure(figsize=(13, 8))
                for ch_i, ch in enumerate(ch_names):
                    plt.plot(
                        times,
                        example[ch_i] + ch_i * offset,
                        linewidth=0.6,
                    )

                plt.yticks(np.arange(n_channels) * offset, ch_names)
                plt.xlabel("Time (s)")
                plt.ylabel("EEG channels (offset for display)")
                plt.title(
                    f"Representative preprocessed EEG - "
                    f"{class_name} ({subject_id})"
                )
                save_plot(f"example_eeg_{class_name.lower()}.png")
                example_saved.add(class_name)

            subject_rows.append(
                {
                    "subject_id": subject_id,
                    "class_name": class_name,
                    "set_file": str(set_file),
                    "sfreq_hz": sfreq,
                    "n_channels": n_channels,
                    "n_samples": n_samples,
                    "duration_seconds": duration_sec,
                    "duration_minutes": duration_sec / 60.0,
                    "channel_order_match": ch_names == EXPECTED_CHANNELS,
                    "sfreq_match": np.isclose(sfreq, EXPECTED_SFREQ),
                    "channel_count_match": n_channels == EXPECTED_N_CHANNELS,
                    "mean_abs_channel_mean_uv": float(np.mean(np.abs(channel_means))),
                    "median_channel_std_uv": float(np.median(channel_stds)),
                    "mean_channel_std_uv": float(np.mean(channel_stds)),
                    "median_channel_rms_uv": float(np.median(channel_rms)),
                    "median_channel_ptp_uv": float(np.median(channel_ptp)),
                    "max_abs_uv": float(np.max(np.abs(data_uv))),
                    "nan_count": int(np.isnan(data_uv).sum()),
                    "inf_count": int(np.isinf(data_uv).sum()),
                }
            )

        except Exception as exc:
            print("   FAILED:", exc)
            subject_rows.append(
                {
                    "subject_id": subject_id,
                    "class_name": class_name,
                    "set_file": str(set_file),
                    "error": repr(exc),
                }
            )

    subject_df = pd.DataFrame(subject_rows)
    bandpower_df = pd.DataFrame(bandpower_rows)

    subject_df.to_csv(OUTPUT_DIR / "subject_eda_summary.csv", index=False)
    bandpower_df.to_csv(OUTPUT_DIR / "subject_relative_bandpower.csv", index=False)

    clean_subject_df = subject_df.copy()
    if "error" in clean_subject_df.columns:
        clean_subject_df = clean_subject_df[clean_subject_df["error"].isna()].copy()

    summary = {
        "n_subjects": int(len(clean_subject_df)),
        "class_counts": clean_subject_df["class_name"].value_counts().to_dict(),
        "sampling_rates": sorted(clean_subject_df["sfreq_hz"].dropna().unique().tolist()),
        "channel_counts": sorted(clean_subject_df["n_channels"].dropna().unique().tolist()),
        "duration_minutes": {
            "min": float(clean_subject_df["duration_minutes"].min()),
            "mean": float(clean_subject_df["duration_minutes"].mean()),
            "median": float(clean_subject_df["duration_minutes"].median()),
            "max": float(clean_subject_df["duration_minutes"].max()),
        },
    }

    with (OUTPUT_DIR / "dataset_eda_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

    class_summary = (
        clean_subject_df.groupby("class_name")
        .agg(
            n_subjects=("subject_id", "count"),
            duration_min_mean=("duration_minutes", "mean"),
            duration_min_median=("duration_minutes", "median"),
            signal_std_uv_mean=("median_channel_std_uv", "mean"),
            max_abs_uv_median=("max_abs_uv", "median"),
        )
        .reset_index()
    )

    class_summary.to_csv(
        OUTPUT_DIR / "class_eda_summary.csv",
        index=False,
    )

    classes = ["Alzheimer", "Control", "FTD"]
    band_names = list(BANDS.keys())

    # 1. Class distribution
    counts = clean_subject_df["class_name"].value_counts().reindex(classes)
    plt.figure(figsize=(7, 5))
    bars = plt.bar(counts.index, counts.values)
    plt.ylabel("Number of subjects")
    plt.title("Class distribution")
    for bar, value in zip(bars, counts.values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.5,
            str(int(value)),
            ha="center",
        )
    save_plot("01_class_distribution.png")

    # 2. Recording duration
    duration_data = [
        clean_subject_df.loc[
            clean_subject_df["class_name"] == cls,
            "duration_minutes",
        ].to_numpy()
        for cls in classes
    ]

    plt.figure(figsize=(8, 5))
    plt.boxplot(duration_data, tick_labels=classes, showmeans=True)
    plt.ylabel("Recording duration (minutes)")
    plt.title("Recording duration by class")
    save_plot("02_recording_duration_by_class.png")

    # 3. Signal variability
    std_data = [
        clean_subject_df.loc[
            clean_subject_df["class_name"] == cls,
            "median_channel_std_uv",
        ].to_numpy()
        for cls in classes
    ]

    plt.figure(figsize=(8, 5))
    plt.boxplot(std_data, tick_labels=classes, showmeans=True)
    plt.ylabel("Median channel SD (µV)")
    plt.title("Signal variability by class")
    save_plot("03_signal_std_by_class.png")

    # 4. Class-average PSD
    plt.figure(figsize=(9, 5))
    for cls in classes:
        if cls in class_psd_sum and class_psd_count.get(cls, 0) > 0:
            avg_psd = class_psd_sum[cls] / class_psd_count[cls]
            avg_psd_db = 10 * np.log10(np.maximum(avg_psd, 1e-20))
            plt.plot(
                common_psd_freqs,
                avg_psd_db,
                label=cls,
                linewidth=1.5,
            )

    plt.xlabel("Frequency (Hz)")
    plt.ylabel("PSD (dB/Hz)")
    plt.title("Class-average EEG power spectral density")
    plt.legend()
    save_plot("04_class_average_psd.png")

    # 5. Relative band power by class
    x = np.arange(len(band_names))
    plt.figure(figsize=(10, 5.5))

    for cls in classes:
        cls_df = bandpower_df[bandpower_df["class_name"] == cls]
        means = []
        sems = []

        for band in band_names:
            values = cls_df[f"{band.lower()}_mean"].to_numpy()
            means.append(np.mean(values))
            sems.append(
                np.std(values, ddof=1) / np.sqrt(max(len(values), 1))
            )

        plt.errorbar(
            x,
            means,
            yerr=sems,
            marker="o",
            capsize=3,
            label=cls,
        )

    plt.xticks(x, band_names)
    plt.ylabel("Mean relative power")
    plt.title("Relative EEG band power by class")
    plt.legend()
    save_plot("05_relative_bandpower_by_class.png")

    # 6. Class x band heatmap
    class_band = np.zeros((len(classes), len(band_names)))

    for i, cls in enumerate(classes):
        cls_df = bandpower_df[bandpower_df["class_name"] == cls]
        for j, band in enumerate(band_names):
            class_band[i, j] = cls_df[f"{band.lower()}_mean"].mean()

    plt.figure(figsize=(8, 4))
    im = plt.imshow(class_band, aspect="auto")
    plt.colorbar(im, label="Mean relative power")
    plt.xticks(np.arange(len(band_names)), band_names)
    plt.yticks(np.arange(len(classes)), classes)
    plt.title("Class-wise relative band-power profile")

    for i in range(len(classes)):
        for j in range(len(band_names)):
            plt.text(
                j,
                i,
                f"{class_band[i, j]:.3f}",
                ha="center",
                va="center",
            )

    save_plot("06_class_bandpower_heatmap.png")

    # 7. Channel x band heatmaps
    for cls in classes:
        cls_df = bandpower_df[bandpower_df["class_name"] == cls]
        matrix = np.zeros((len(EXPECTED_CHANNELS), len(band_names)))

        for i, ch in enumerate(EXPECTED_CHANNELS):
            for j, band in enumerate(band_names):
                column = f"{ch}_{band}"
                matrix[i, j] = (
                    cls_df[column].mean()
                    if column in cls_df.columns
                    else np.nan
                )

        plt.figure(figsize=(8, 8))
        im = plt.imshow(matrix, aspect="auto")
        plt.colorbar(im, label="Mean relative power")
        plt.xticks(np.arange(len(band_names)), band_names)
        plt.yticks(np.arange(len(EXPECTED_CHANNELS)), EXPECTED_CHANNELS)
        plt.title(f"Channel × frequency-band profile - {cls}")
        save_plot(f"07_channel_bandpower_{cls.lower()}.png")

    # 8. Class-average channel correlation
    for cls in classes:
        if cls not in class_corr_sum or class_corr_count.get(cls, 0) == 0:
            continue

        avg_corr = class_corr_sum[cls] / class_corr_count[cls]

        plt.figure(figsize=(8, 7))
        im = plt.imshow(avg_corr, vmin=-1, vmax=1, aspect="equal")
        plt.colorbar(im, label="Pearson correlation")
        plt.xticks(
            np.arange(len(EXPECTED_CHANNELS)),
            EXPECTED_CHANNELS,
            rotation=90,
        )
        plt.yticks(np.arange(len(EXPECTED_CHANNELS)), EXPECTED_CHANNELS)
        plt.title(f"Average channel correlation - {cls}")
        save_plot(f"08_channel_correlation_{cls.lower()}.png")

    # 9. PCA of 95D spectral profiles
    spectral_columns = [
        f"{ch}_{band}"
        for ch in EXPECTED_CHANNELS
        for band in band_names
        if f"{ch}_{band}" in bandpower_df.columns
    ]

    if len(spectral_columns) == 95:
        X = bandpower_df[spectral_columns].to_numpy(dtype=np.float64)
        X = StandardScaler().fit_transform(X)

        pca = PCA(
            n_components=2,
            random_state=RANDOM_SEED,
        )

        X_2d = pca.fit_transform(X)

        plt.figure(figsize=(7, 6))
        for cls in classes:
            mask = bandpower_df["class_name"].to_numpy() == cls
            plt.scatter(
                X_2d[mask, 0],
                X_2d[mask, 1],
                label=cls,
                alpha=0.8,
            )

        plt.xlabel(
            f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)"
        )
        plt.ylabel(
            f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)"
        )
        plt.title("PCA of subject-level spectral profiles")
        plt.legend()
        save_plot("09_spectral_pca.png")

        pd.DataFrame(
            {
                "subject_id": bandpower_df["subject_id"],
                "class_name": bandpower_df["class_name"],
                "PC1": X_2d[:, 0],
                "PC2": X_2d[:, 1],
            }
        ).to_csv(
            OUTPUT_DIR / "spectral_pca_coordinates.csv",
            index=False,
        )

    report_lines = [
        "PREPROCESSED EEG EDA REPORT",
        "=" * 50,
        "",
        f"Subjects analyzed: {len(clean_subject_df)}",
        f"Class counts: {summary['class_counts']}",
        f"Sampling rates: {summary['sampling_rates']}",
        f"Channel counts: {summary['channel_counts']}",
        "",
        "Recording duration (minutes):",
        json.dumps(summary["duration_minutes"], indent=2),
        "",
        "Generated tables:",
        "- eda_manifest.csv",
        "- subject_eda_summary.csv",
        "- class_eda_summary.csv",
        "- subject_relative_bandpower.csv",
        "- spectral_pca_coordinates.csv",
        "",
        "Generated plots:",
        str(PLOTS_DIR),
    ]

    (OUTPUT_DIR / "EDA_REPORT.txt").write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    print()
    print("=" * 72)
    print("EDA COMPLETE")
    print("=" * 72)
    print("Output directory:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()
