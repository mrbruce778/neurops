from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from scipy.signal import coherence
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
)


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(r"E:\neurops")

WINDOW_MANIFEST = (
    PROJECT_ROOT
    / "windows"
    / "window_manifest.csv"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "results_connectivity_baseline"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

SEED = 42
N_FOLDS = 5

SFREQ = 500.0

# Welch/coherence settings
NPERSEG = 1000
NOVERLAP = 500

BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}

CHANNEL_NAMES = [
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

N_CHANNELS = len(CHANNEL_NAMES)


# ============================================================
# CHANNEL PAIRS
# ============================================================

CHANNEL_PAIRS = [
    (i, j)
    for i in range(N_CHANNELS)
    for j in range(i + 1, N_CHANNELS)
]

print(
    f"Unique channel pairs: "
    f"{len(CHANNEL_PAIRS)}"
)

# 19 choose 2 = 171


# ============================================================
# CONNECTIVITY FOR ONE WINDOW
# ============================================================

def calculate_window_connectivity(eeg_window):
    """
    eeg_window:
        [19, 2000]

    Returns:
        [5 bands, 171 channel pairs]

    Connectivity measure:
        magnitude-squared coherence
    """

    if eeg_window.shape != (19, 2000):
        raise ValueError(
            f"Expected (19, 2000), "
            f"got {eeg_window.shape}"
        )

    features = np.zeros(
        (
            len(BANDS),
            len(CHANNEL_PAIRS),
        ),
        dtype=np.float32,
    )

    for pair_idx, (ch1, ch2) in enumerate(
        CHANNEL_PAIRS
    ):

        frequencies, coh = coherence(
            eeg_window[ch1],
            eeg_window[ch2],
            fs=SFREQ,
            nperseg=NPERSEG,
            noverlap=NOVERLAP,
            detrend="constant",
        )

        for band_idx, (_, limits) in enumerate(
            BANDS.items()
        ):

            low, high = limits

            mask = (
                (frequencies >= low)
                & (frequencies < high)
            )

            if not np.any(mask):
                raise RuntimeError(
                    f"No frequencies found "
                    f"for band {low}-{high} Hz."
                )

            features[
                band_idx,
                pair_idx,
            ] = np.mean(
                coh[mask]
            )

    return features


# ============================================================
# SUBJECT FEATURES
# ============================================================

def calculate_subject_features(
    window_file,
    subject_id,
):
    """
    Calculate connectivity for every window.

    Then median aggregate across windows.

    Final feature vector:

        5 frequency bands
        x
        171 channel pairs

        = 855 features per subject.
    """

    window_file = Path(
        window_file
    )

    with np.load(
        window_file,
        allow_pickle=False,
    ) as data:

        eeg = data["eeg"].astype(
            np.float32,
            copy=False,
        )

        eeg = eeg.copy()

    if eeg.ndim != 3:
        raise RuntimeError(
            f"{subject_id}: expected "
            f"[windows, channels, samples], "
            f"got {eeg.shape}"
        )

    if eeg.shape[1:] != (19, 2000):
        raise RuntimeError(
            f"{subject_id}: unexpected "
            f"shape {eeg.shape}"
        )

    n_windows = eeg.shape[0]

    print(
        f"  {subject_id}: "
        f"{n_windows} windows"
    )

    window_features = []

    for window_idx in range(n_windows):

        features = (
            calculate_window_connectivity(
                eeg[window_idx]
            )
        )

        window_features.append(
            features
        )

    window_features = np.stack(
        window_features,
        axis=0,
    )

    # [windows, bands, pairs]

    # Robust subject-level aggregation.
    subject_features = np.median(
        window_features,
        axis=0,
    )

    # [bands, pairs]

    return subject_features.reshape(-1)


# ============================================================
# FEATURE NAMES
# ============================================================

def make_feature_names():

    names = []

    for band_name in BANDS:

        for ch1, ch2 in CHANNEL_PAIRS:

            names.append(
                f"{band_name}_"
                f"{CHANNEL_NAMES[ch1]}_"
                f"{CHANNEL_NAMES[ch2]}"
            )

    return names


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    y_true,
    y_pred,
):

    return {
        "accuracy":
            accuracy_score(
                y_true,
                y_pred,
            ),

        "balanced_accuracy":
            balanced_accuracy_score(
                y_true,
                y_pred,
            ),

        "macro_f1":
            f1_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            ),

        "mcc":
            matthews_corrcoef(
                y_true,
                y_pred,
            ),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print(
        "EEG FUNCTIONAL CONNECTIVITY BASELINE"
    )
    print("=" * 70)

    print(
        "\nConnectivity measure:"
        "\n  Magnitude-squared coherence"
    )

    print(
        "\nBands:"
    )

    for name, limits in BANDS.items():

        print(
            f"  {name:6s}: "
            f"{limits[0]:4.1f}"
            f"-"
            f"{limits[1]:4.1f} Hz"
        )

    print(
        f"\nChannels: "
        f"{N_CHANNELS}"
    )

    print(
        f"Channel pairs: "
        f"{len(CHANNEL_PAIRS)}"
    )

    print(
        f"Features per subject: "
        f"{len(CHANNEL_PAIRS) * len(BANDS)}"
    )

    # --------------------------------------------------------
    # LOAD MANIFEST
    # --------------------------------------------------------

    if not WINDOW_MANIFEST.exists():

        raise FileNotFoundError(
            f"Manifest not found:\n"
            f"{WINDOW_MANIFEST}"
        )

    manifest = pd.read_csv(
        WINDOW_MANIFEST
    )

    required_columns = {
        "subject_id",
        "class_name",
        "label",
        "window_file",
    }

    missing = (
        required_columns
        - set(manifest.columns)
    )

    if missing:

        raise RuntimeError(
            f"Missing columns: "
            f"{sorted(missing)}"
        )

    print(
        f"\nSubjects: "
        f"{len(manifest)}"
    )

    # --------------------------------------------------------
    # EXTRACT FEATURES
    # --------------------------------------------------------

    feature_names = (
        make_feature_names()
    )

    feature_file = (
        RESULTS_DIR
        / "subject_connectivity_features.csv"
    )

    # Resume-safe feature extraction.
    # Connectivity mathematics is NOT changed.

    if feature_file.exists():

        feature_df = pd.read_csv(
            feature_file
        )

        completed_subjects = set(
            feature_df[
                "subject_id"
            ].astype(str)
        )

        print(
            f"\nCheckpoint found: "
            f"{len(completed_subjects)}/"
            f"{len(manifest)} subjects completed."
        )

    else:

        feature_df = pd.DataFrame(
            columns=[
                "subject_id",
                "class_name",
                "label",
                *feature_names,
            ]
        )

        completed_subjects = set()

    manifest_subjects = set(
        manifest[
            "subject_id"
        ].astype(str)
    )

    if completed_subjects != manifest_subjects:

        print(
            "\nExtracting/resuming "
            "connectivity features..."
        )

        print(
            "This is the slow part."
        )

        for subject_number, row in (
            manifest.iterrows()
        ):

            subject_id = str(
                row["subject_id"]
            )

            print()

            print(
                f"[{subject_number + 1}/"
                f"{len(manifest)}] "
                f"{subject_id}"
            )

            # --------------------------------------------
            # SKIP SUBJECTS ALREADY SAVED
            # --------------------------------------------

            if subject_id in completed_subjects:

                print(
                    "  already completed -> skipping"
                )

                continue

            # --------------------------------------------
            # SAME CONNECTIVITY CALCULATION AS BEFORE
            # --------------------------------------------

            features = (
                calculate_subject_features(
                    row["window_file"],
                    subject_id,
                )
            )

            output_row = {
                "subject_id":
                    subject_id,

                "class_name":
                    row["class_name"],

                "label":
                    int(row["label"]),
            }

            for name, value in zip(
                feature_names,
                features,
            ):

                output_row[name] = (
                    float(value)
                )

            # --------------------------------------------
            # ADD FINISHED SUBJECT
            # --------------------------------------------

            feature_df = pd.concat(
                [
                    feature_df,
                    pd.DataFrame(
                        [output_row]
                    ),
                ],
                ignore_index=True,
            )

            # --------------------------------------------
            # SAVE IMMEDIATELY
            # --------------------------------------------

            feature_df.to_csv(
                feature_file,
                index=False,
            )

            completed_subjects.add(
                subject_id
            )

            print(
                f"  checkpoint saved "
                f"({len(completed_subjects)}/"
                f"{len(manifest)})"
            )

    else:

        print(
            "\nComplete connectivity feature "
            "cache found."
        )

        print(
            "No connectivity recalculation needed."
        )
    # --------------------------------------------------------
    # VERIFY SUBJECT ORDER
    # --------------------------------------------------------

    if set(
        feature_df["subject_id"]
    ) != set(
        manifest["subject_id"]
    ):

        raise RuntimeError(
            "Feature subjects do not match "
            "manifest subjects."
        )

    feature_df = (
        manifest[
            [
                "subject_id",
            ]
        ]
        .merge(
            feature_df,
            on="subject_id",
            how="left",
            validate="one_to_one",
        )
    )

    # After merge class/label come from feature table.
    if feature_df.isna().any().any():

        raise RuntimeError(
            "Missing values after feature merge."
        )

    # --------------------------------------------------------
    # FEATURE MATRIX
    # --------------------------------------------------------

    X = feature_df[
        feature_names
    ].to_numpy(
        dtype=np.float64
    )

    y = feature_df[
        "label"
    ].to_numpy(
        dtype=np.int64
    )

    subject_ids = feature_df[
        "subject_id"
    ].to_numpy()

    print(
        f"\nFeature matrix: "
        f"{X.shape}"
    )

    if not np.isfinite(X).all():

        raise RuntimeError(
            "Non-finite connectivity features."
        )

    # --------------------------------------------------------
    # SAME SUBJECT-LEVEL 5-FOLD CV
    # --------------------------------------------------------

    skf = StratifiedKFold(
        n_splits=N_FOLDS,
        shuffle=True,
        random_state=SEED,
    )

    fold_metrics = []
    oof_records = []

    print()
    print("=" * 70)
    print("5-FOLD CROSS-VALIDATION")
    print("=" * 70)

    for fold_number, (
        train_indices,
        test_indices,
    ) in enumerate(
        skf.split(X, y),
        start=1,
    ):

        print()
        print("-" * 70)
        print(
            f"FOLD {fold_number}"
        )
        print("-" * 70)

        X_train = X[
            train_indices
        ]

        y_train = y[
            train_indices
        ]

        X_test = X[
            test_indices
        ]

        y_test = y[
            test_indices
        ]

        # IMPORTANT:
        # scaler is fitted only inside
        # the training fold.
        model = Pipeline(
            [
                (
                    "scaler",
                    StandardScaler(),
                ),
                (
                    "svm",
                    SVC(
                        kernel="linear",
                        C=1.0,
                        class_weight="balanced",
                    ),
                ),
            ]
        )

        model.fit(
            X_train,
            y_train,
        )

        y_pred = model.predict(
            X_test
        )

        metrics = calculate_metrics(
            y_test,
            y_pred,
        )

        cm = confusion_matrix(
            y_test,
            y_pred,
            labels=[0, 1, 2],
        )

        metrics[
            "fold"
        ] = fold_number

        fold_metrics.append(
            metrics
        )

        print(
            f"Accuracy:          "
            f"{metrics['accuracy']:.4f}"
        )

        print(
            f"Balanced accuracy: "
            f"{metrics['balanced_accuracy']:.4f}"
        )

        print(
            f"Macro F1:          "
            f"{metrics['macro_f1']:.4f}"
        )

        print(
            f"MCC:               "
            f"{metrics['mcc']:.4f}"
        )

        print(
            "\nConfusion matrix:"
        )

        print(cm)

        for idx, true_label, pred_label in zip(
            test_indices,
            y_test,
            y_pred,
        ):

            oof_records.append(
                {
                    "subject_id":
                        subject_ids[idx],

                    "true_label":
                        int(true_label),

                    "predicted_label":
                        int(pred_label),

                    "fold":
                        fold_number,
                }
            )

    # --------------------------------------------------------
    # SAVE FOLD RESULTS
    # --------------------------------------------------------

    metrics_df = pd.DataFrame(
        fold_metrics
    )

    metrics_df.to_csv(
        RESULTS_DIR
        / "five_fold_metrics.csv",
        index=False,
    )

    oof_df = pd.DataFrame(
        oof_records
    )

    oof_df.to_csv(
        RESULTS_DIR
        / "oof_subject_predictions.csv",
        index=False,
    )

    # --------------------------------------------------------
    # FIVE-FOLD SUMMARY
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "CONNECTIVITY BASELINE "
        "5-FOLD SUMMARY"
    )
    print("=" * 70)

    for metric in [
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "mcc",
    ]:

        values = metrics_df[
            metric
        ].to_numpy()

        print(
            f"{metric:20s}: "
            f"{values.mean():.4f} "
            f"± "
            f"{values.std(ddof=1):.4f}"
        )

    # --------------------------------------------------------
    # POOLED OOF
    # --------------------------------------------------------

    pooled_metrics = calculate_metrics(
        oof_df[
            "true_label"
        ].to_numpy(),

        oof_df[
            "predicted_label"
        ].to_numpy(),
    )

    pooled_cm = confusion_matrix(
        oof_df[
            "true_label"
        ],

        oof_df[
            "predicted_label"
        ],

        labels=[0, 1, 2],
    )

    print()
    print("=" * 70)
    print(
        "POOLED OUT-OF-FOLD RESULTS"
    )
    print("=" * 70)

    print(
        f"Accuracy:          "
        f"{pooled_metrics['accuracy']:.4f}"
    )

    print(
        f"Balanced accuracy: "
        f"{pooled_metrics['balanced_accuracy']:.4f}"
    )

    print(
        f"Macro F1:          "
        f"{pooled_metrics['macro_f1']:.4f}"
    )

    print(
        f"MCC:               "
        f"{pooled_metrics['mcc']:.4f}"
    )

    print(
        "\nPooled confusion matrix:"
    )

    print(
        pooled_cm
    )

    print(
        f"\nResults saved under:\n"
        f"{RESULTS_DIR}"
    )


if __name__ == "__main__":

    warnings.filterwarnings(
        "ignore",
        category=RuntimeWarning,
    )

    main()