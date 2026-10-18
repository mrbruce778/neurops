import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

from scipy.signal import welch

from sklearn.model_selection import (
    StratifiedKFold,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(r"E:\neurops")

WINDOW_MANIFEST = (
    PROJECT_ROOT
    / "windows"
    / "window_manifest.csv"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "results_spectral_baseline"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# CONFIG
# ============================================================

SEED = 42

N_FOLDS = 5

SFREQ = 500.0

# Welch PSD parameters
NPERSEG = 1000       # 2 seconds at 500 Hz
NOVERLAP = 500       # 50% Welch overlap

# Frequency range used to calculate total EEG power
TOTAL_LOW = 0.5
TOTAL_HIGH = 45.0

BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}

CLASS_NAMES = {
    0: "Alzheimer",
    1: "Control",
    2: "FTD",
}


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed):

    random.seed(seed)

    np.random.seed(seed)


# ============================================================
# BAND POWER
# ============================================================

def integrate_band(
    frequencies,
    psd,
    low,
    high,
):

    mask = (
        (frequencies >= low)
        & (frequencies < high)
    )

    if mask.sum() < 2:

        raise RuntimeError(
            f"Not enough frequency bins for "
            f"{low}-{high} Hz"
        )

    return np.trapezoid(
        psd[..., mask],
        frequencies[mask],
        axis=-1,
    )


# ============================================================
# EXTRACT FEATURES FROM ONE SUBJECT
# ============================================================

def extract_subject_features(
    window_file,
):

    with np.load(
        window_file,
        allow_pickle=False,
    ) as data:

        eeg = data["eeg"].astype(
            np.float64,
            copy=False,
        )

    # Expected:
    #
    # [windows, channels, samples]
    #
    # [N, 19, 2000]

    if eeg.ndim != 3:

        raise RuntimeError(
            f"Unexpected EEG shape: "
            f"{eeg.shape}"
        )

    if eeg.shape[1] != 19:

        raise RuntimeError(
            f"Expected 19 channels, "
            f"got {eeg.shape[1]}"
        )

    # --------------------------------------------------------
    # Welch PSD
    #
    # scipy works along the final axis, so PSD shape becomes:
    #
    # [windows, channels, frequencies]
    # --------------------------------------------------------

    frequencies, psd = welch(
        eeg,
        fs=SFREQ,
        nperseg=NPERSEG,
        noverlap=NOVERLAP,
        axis=-1,
        detrend="constant",
        scaling="density",
    )

    # --------------------------------------------------------
    # Total 0.5-45 Hz power
    # --------------------------------------------------------

    total_power = integrate_band(
        frequencies,
        psd,
        TOTAL_LOW,
        TOTAL_HIGH,
    )

    total_power = np.maximum(
        total_power,
        np.finfo(np.float64).eps,
    )

    # --------------------------------------------------------
    # Relative band power
    # --------------------------------------------------------

    window_features = []

    feature_names = []

    for band_name, (
        low,
        high,
    ) in BANDS.items():

        absolute_power = integrate_band(
            frequencies,
            psd,
            low,
            high,
        )

        relative_power = (
            absolute_power
            / total_power
        )

        window_features.append(
            relative_power
        )

        for channel_index in range(19):

            feature_names.append(
                f"ch{channel_index + 1:02d}_"
                f"{band_name}_relative_power"
            )

    # Current list:
    #
    # 5 arrays, each [windows, 19]
    #
    # concatenate -> [windows, 95]

    window_features = np.concatenate(
        window_features,
        axis=1,
    )

    # --------------------------------------------------------
    # Subject-level aggregation
    #
    # We deliberately classify subjects rather than treating
    # thousands of correlated windows as independent patients.
    #
    # Median provides a robust summary across windows.
    # --------------------------------------------------------

    subject_features = np.median(
        window_features,
        axis=0,
    )

    if not np.isfinite(
        subject_features
    ).all():

        raise RuntimeError(
            f"Non-finite spectral features in "
            f"{window_file}"
        )

    return (
        subject_features,
        feature_names,
    )


# ============================================================
# BUILD SUBJECT FEATURE TABLE
# ============================================================

def build_feature_table(
    manifest,
):

    print("\nExtracting subject-level spectral features...")

    rows = []

    expected_feature_names = None

    for index, row in manifest.iterrows():

        subject_id = str(
            row["subject_id"]
        )

        label = int(
            row["label"]
        )

        class_name = str(
            row["class_name"]
        )

        window_file = Path(
            row["window_file"]
        )

        (
            features,
            feature_names,
        ) = extract_subject_features(
            window_file
        )

        if expected_feature_names is None:

            expected_feature_names = (
                feature_names
            )

        elif (
            feature_names
            != expected_feature_names
        ):

            raise RuntimeError(
                "Feature-name mismatch."
            )

        feature_row = {
            "subject_id": subject_id,
            "label": label,
            "class_name": class_name,
        }

        for name, value in zip(
            feature_names,
            features,
        ):

            feature_row[name] = float(
                value
            )

        rows.append(
            feature_row
        )

        print(
            f"\r  {index + 1:02d}/"
            f"{len(manifest)} subjects",
            end="",
            flush=True,
        )

    print()

    feature_df = pd.DataFrame(
        rows
    )

    return (
        feature_df,
        expected_feature_names,
    )


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    y_true,
    y_pred,
):

    return {

        "accuracy": accuracy_score(
            y_true,
            y_pred,
        ),

        "balanced_accuracy":
            balanced_accuracy_score(
                y_true,
                y_pred,
            ),

        "macro_precision":
            precision_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            ),

        "macro_recall":
            recall_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
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
# RUN ONE FOLD
# ============================================================

def run_fold(
    fold_number,
    train_indices,
    test_indices,
    feature_df,
    feature_names,
):

    print()
    print("=" * 70)
    print(
        f"SPECTRAL BASELINE - FOLD {fold_number}"
    )
    print("=" * 70)

    train_df = (
        feature_df
        .iloc[train_indices]
        .reset_index(drop=True)
    )

    test_df = (
        feature_df
        .iloc[test_indices]
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Leakage check
    # --------------------------------------------------------

    train_subjects = set(
        train_df["subject_id"]
    )

    test_subjects = set(
        test_df["subject_id"]
    )

    assert train_subjects.isdisjoint(
        test_subjects
    )

    print(
        f"\nTrain subjects: {len(train_df)}"
    )

    print(
        f"Test subjects:  {len(test_df)}"
    )

    print("\nClass distribution:")

    print(
        "  Train:",
        train_df[
            "class_name"
        ].value_counts().to_dict(),
    )

    print(
        "  Test: ",
        test_df[
            "class_name"
        ].value_counts().to_dict(),
    )

    # --------------------------------------------------------
    # Data
    # --------------------------------------------------------

    X_train = train_df[
        feature_names
    ].to_numpy(
        dtype=np.float64
    )

    y_train = train_df[
        "label"
    ].to_numpy()

    X_test = test_df[
        feature_names
    ].to_numpy(
        dtype=np.float64
    )

    y_test = test_df[
        "label"
    ].to_numpy()

    # --------------------------------------------------------
    # Pipeline
    #
    # StandardScaler is fitted ONLY to training subjects.
    #
    # SVC class_weight="balanced" compensates for the
    # unequal numbers of AD / Control / FTD subjects.
    #
    # We keep hyperparameters fixed rather than optimizing
    # them against the outer test folds.
    # --------------------------------------------------------

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
                    probability=False,
                    random_state=SEED,
                ),
            ),
        ]
    )

    model.fit(
        X_train,
        y_train,
    )

    predictions = model.predict(
        X_test
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    metrics = calculate_metrics(
        y_test,
        predictions,
    )

    cm = confusion_matrix(
        y_test,
        predictions,
        labels=[
            0,
            1,
            2,
        ],
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    fold_dir = (
        RESULTS_DIR
        / f"fold_{fold_number}"
    )

    fold_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    prediction_df = pd.DataFrame(
        {
            "subject_id":
                test_df[
                    "subject_id"
                ],

            "true_label":
                y_test,

            "predicted_label":
                predictions,
        }
    )

    prediction_df[
        "true_class"
    ] = prediction_df[
        "true_label"
    ].map(
        CLASS_NAMES
    )

    prediction_df[
        "predicted_class"
    ] = prediction_df[
        "predicted_label"
    ].map(
        CLASS_NAMES
    )

    prediction_df.to_csv(
        fold_dir
        / "subject_predictions.csv",
        index=False,
    )

    np.savetxt(
        fold_dir
        / "confusion_matrix.csv",
        cm,
        delimiter=",",
        fmt="%d",
    )

    with (
        fold_dir
        / "metrics.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metrics,
            f,
            indent=4,
        )

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    print("\nResults:")

    print(
        f"  Accuracy:          "
        f"{metrics['accuracy']:.4f}"
    )

    print(
        f"  Balanced accuracy: "
        f"{metrics['balanced_accuracy']:.4f}"
    )

    print(
        f"  Macro precision:   "
        f"{metrics['macro_precision']:.4f}"
    )

    print(
        f"  Macro recall:      "
        f"{metrics['macro_recall']:.4f}"
    )

    print(
        f"  Macro F1:          "
        f"{metrics['macro_f1']:.4f}"
    )

    print(
        f"  MCC:               "
        f"{metrics['mcc']:.4f}"
    )

    print(
        "\nConfusion matrix "
        "(rows=true, columns=predicted):"
    )

    print(cm)

    metrics[
        "fold"
    ] = fold_number

    return (
        metrics,
        prediction_df,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    group = (
        parser
        .add_mutually_exclusive_group(
            required=True
        )
    )

    group.add_argument(
        "--fold",
        type=int,
        choices=[
            1,
            2,
            3,
            4,
            5,
        ],
    )

    group.add_argument(
        "--all-folds",
        action="store_true",
    )

    args = parser.parse_args()

    set_seed(SEED)

    # --------------------------------------------------------
    # Manifest
    # --------------------------------------------------------

    if not WINDOW_MANIFEST.exists():

        raise FileNotFoundError(
            f"Manifest not found:\n"
            f"{WINDOW_MANIFEST}"
        )

    manifest = pd.read_csv(
        WINDOW_MANIFEST
    )

    print("=" * 70)

    print(
        "CLASSICAL EEG SPECTRAL BASELINE"
    )

    print("=" * 70)

    print(
        f"\nSubjects: {len(manifest)}"
    )

    print(
        "Classes:",
        manifest[
            "class_name"
        ].value_counts().to_dict(),
    )

    # --------------------------------------------------------
    # Extract features once
    # --------------------------------------------------------

    (
        feature_df,
        feature_names,
    ) = build_feature_table(
        manifest
    )

    print(
        f"\nFeatures per subject: "
        f"{len(feature_names)}"
    )

    print(
        "Feature type: "
        "median relative band power"
    )

    feature_df.to_csv(
        RESULTS_DIR
        / "subject_spectral_features.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Same outer CV construction as neural-network experiment
    # --------------------------------------------------------

    skf = StratifiedKFold(
        n_splits=N_FOLDS,
        shuffle=True,
        random_state=SEED,
    )

    indices = np.arange(
        len(feature_df)
    )

    labels = feature_df[
        "label"
    ].to_numpy()

    folds = list(
        skf.split(
            indices,
            labels,
        )
    )

    if args.fold is not None:

        requested_folds = [
            args.fold
        ]

    else:

        requested_folds = list(
            range(
                1,
                N_FOLDS + 1,
            )
        )

    all_metrics = []

    all_predictions = []

    # --------------------------------------------------------
    # Run
    # --------------------------------------------------------

    for fold_number in requested_folds:

        (
            train_indices,
            test_indices,
        ) = folds[
            fold_number - 1
        ]

        (
            metrics,
            predictions,
        ) = run_fold(
            fold_number,
            train_indices,
            test_indices,
            feature_df,
            feature_names,
        )

        all_metrics.append(
            metrics
        )

        predictions[
            "fold"
        ] = fold_number

        all_predictions.append(
            predictions
        )

    # --------------------------------------------------------
    # Save requested results
    # --------------------------------------------------------

    metrics_df = pd.DataFrame(
        all_metrics
    )

    if args.all_folds:

        metrics_df.to_csv(
            RESULTS_DIR
            / "five_fold_metrics.csv",
            index=False,
        )

        oof_predictions = pd.concat(
            all_predictions,
            ignore_index=True,
        )

        oof_predictions.to_csv(
            RESULTS_DIR
            / "oof_subject_predictions.csv",
            index=False,
        )

        # ----------------------------------------------------
        # Mean +/- SD
        # ----------------------------------------------------

        print()
        print("=" * 70)
        print("5-FOLD SUMMARY")
        print("=" * 70)

        for metric in [
            "accuracy",
            "balanced_accuracy",
            "macro_precision",
            "macro_recall",
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

        # ----------------------------------------------------
        # Pooled OOF metrics
        #
        # Every subject appears exactly once.
        # ----------------------------------------------------

        pooled_metrics = (
            calculate_metrics(
                oof_predictions[
                    "true_label"
                ].to_numpy(),

                oof_predictions[
                    "predicted_label"
                ].to_numpy(),
            )
        )

        pooled_cm = confusion_matrix(
            oof_predictions[
                "true_label"
            ],

            oof_predictions[
                "predicted_label"
            ],

            labels=[
                0,
                1,
                2,
            ],
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

        with (
            RESULTS_DIR
            / "pooled_oof_metrics.json"
        ).open(
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                pooled_metrics,
                f,
                indent=4,
            )

        np.savetxt(
            RESULTS_DIR
            / "pooled_oof_confusion_matrix.csv",
            pooled_cm,
            delimiter=",",
            fmt="%d",
        )

    else:

        fold_number = args.fold

        metrics_df.to_csv(
            RESULTS_DIR
            / f"fold_{fold_number}"
            / "fold_summary.csv",
            index=False,
        )

    print(
        f"\nResults saved under:\n"
        f"{RESULTS_DIR}"
    )


if __name__ == "__main__":
    main()