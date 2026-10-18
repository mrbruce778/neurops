from pathlib import Path
import json

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
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

PROJECT_ROOT = Path(r"E:\neurops")

SPECTRAL_FILE = (
    PROJECT_ROOT
    / "results_spectral_baseline"
    / "subject_spectral_features.csv"
)

CONNECTIVITY_FILE = (
    PROJECT_ROOT
    / "results_connectivity_baseline"
    / "subject_connectivity_features.csv"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "results_spectral_connectivity_fusion"
)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
N_FOLDS = 5
CLASS_NAMES = {0: "Alzheimer", 1: "Control", 2: "FTD"}
META_COLUMNS = {"subject_id", "label", "class_name"}


def calculate_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_precision": precision_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "macro_recall": recall_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "macro_f1": f1_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "mcc": matthews_corrcoef(y_true, y_pred),
    }


def add_class_recalls(metrics, cm):
    for label, name in CLASS_NAMES.items():
        total = cm[label].sum()
        recall = cm[label, label] / total if total else np.nan
        metrics[f"{name.lower()}_recall"] = float(recall)
    return metrics


def load_fused_features():
    if not SPECTRAL_FILE.exists():
        raise FileNotFoundError(f"Missing spectral file:\n{SPECTRAL_FILE}")

    if not CONNECTIVITY_FILE.exists():
        raise FileNotFoundError(f"Missing connectivity file:\n{CONNECTIVITY_FILE}")

    spectral_df = pd.read_csv(SPECTRAL_FILE)
    connectivity_df = pd.read_csv(CONNECTIVITY_FILE)

    for name, df in [("spectral", spectral_df), ("connectivity", connectivity_df)]:
        missing = META_COLUMNS - set(df.columns)
        if missing:
            raise RuntimeError(f"{name} file missing columns: {sorted(missing)}")

        if df["subject_id"].duplicated().any():
            raise RuntimeError(f"Duplicate subject_id found in {name} file.")

    spectral_df["subject_id"] = spectral_df["subject_id"].astype(str)
    connectivity_df["subject_id"] = connectivity_df["subject_id"].astype(str)

    if set(spectral_df["subject_id"]) != set(connectivity_df["subject_id"]):
        raise RuntimeError("Subject IDs do not match between spectral and connectivity files.")

    spectral_features = [c for c in spectral_df.columns if c not in META_COLUMNS]
    connectivity_features = [c for c in connectivity_df.columns if c not in META_COLUMNS]

    if len(spectral_features) != 95:
        raise RuntimeError(f"Expected 95 spectral features, found {len(spectral_features)}")

    if len(connectivity_features) != 855:
        raise RuntimeError(
            f"Expected 855 connectivity features, found {len(connectivity_features)}"
        )

    spectral_df = spectral_df.rename(
        columns={c: f"spectral__{c}" for c in spectral_features}
    )
    connectivity_df = connectivity_df.rename(
        columns={c: f"connectivity__{c}" for c in connectivity_features}
    )

    spectral_features = [f"spectral__{c}" for c in spectral_features]
    connectivity_features = [f"connectivity__{c}" for c in connectivity_features]

    # Keep spectral table on the left so subject order stays identical
    # to the original spectral baseline.
    merged_df = spectral_df.merge(
        connectivity_df,
        on="subject_id",
        how="left",
        sort=False,
        validate="one_to_one",
        suffixes=("_spectral", "_connectivity"),
    )

    if not np.array_equal(
        merged_df["label_spectral"].to_numpy(),
        merged_df["label_connectivity"].to_numpy(),
    ):
        raise RuntimeError("Label mismatch between spectral and connectivity tables.")

    if not np.array_equal(
        merged_df["class_name_spectral"].astype(str).to_numpy(),
        merged_df["class_name_connectivity"].astype(str).to_numpy(),
    ):
        raise RuntimeError("Class-name mismatch between spectral and connectivity tables.")

    merged_df = merged_df.rename(
        columns={
            "label_spectral": "label",
            "class_name_spectral": "class_name",
        }
    )
    merged_df = merged_df.drop(
        columns=["label_connectivity", "class_name_connectivity"]
    )

    fused_features = spectral_features + connectivity_features

    if len(fused_features) != 950:
        raise RuntimeError(f"Expected 950 fused features, found {len(fused_features)}")

    X = merged_df[fused_features].to_numpy(dtype=np.float64)
    if not np.isfinite(X).all():
        raise RuntimeError("Non-finite values found in fused feature matrix.")

    return merged_df, fused_features


def run_fold(fold_number, train_idx, test_idx, feature_df, feature_names):
    print("\n" + "-" * 70)
    print(f"FOLD {fold_number}")
    print("-" * 70)

    train_df = feature_df.iloc[train_idx].reset_index(drop=True)
    test_df = feature_df.iloc[test_idx].reset_index(drop=True)

    assert set(train_df["subject_id"]).isdisjoint(set(test_df["subject_id"]))

    X_train = train_df[feature_names].to_numpy(dtype=np.float64)
    y_train = train_df["label"].to_numpy()
    X_test = test_df[feature_names].to_numpy(dtype=np.float64)
    y_test = test_df["label"].to_numpy()

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "svm",
                SVC(
                    kernel="linear",
                    C=1.0,
                    class_weight="balanced",
                    random_state=SEED,
                ),
            ),
        ]
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = calculate_metrics(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])
    metrics = add_class_recalls(metrics, cm)
    metrics["fold"] = fold_number

    print(f"Accuracy:          {metrics['accuracy']:.4f}")
    print(f"Balanced accuracy: {metrics['balanced_accuracy']:.4f}")
    print(f"Macro F1:          {metrics['macro_f1']:.4f}")
    print(f"MCC:               {metrics['mcc']:.4f}")
    print(f"FTD recall:        {metrics['ftd_recall']:.4f}")
    print("\nConfusion matrix:")
    print(cm)

    fold_dir = RESULTS_DIR / f"fold_{fold_number}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    pred_df = pd.DataFrame(
        {
            "subject_id": test_df["subject_id"].to_numpy(),
            "true_label": y_test,
            "predicted_label": y_pred,
        }
    )
    pred_df["true_class"] = pred_df["true_label"].map(CLASS_NAMES)
    pred_df["predicted_class"] = pred_df["predicted_label"].map(CLASS_NAMES)
    pred_df.to_csv(fold_dir / "subject_predictions.csv", index=False)

    np.savetxt(
        fold_dir / "confusion_matrix.csv",
        cm,
        delimiter=",",
        fmt="%d",
    )

    with (fold_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                k: (None if isinstance(v, float) and np.isnan(v) else float(v))
                for k, v in metrics.items()
                if k != "fold"
            },
            f,
            indent=4,
        )

    return metrics, pred_df


def main():
    print("=" * 70)
    print("SPECTRAL + CONNECTIVITY FUSION BASELINE")
    print("=" * 70)

    feature_df, feature_names = load_fused_features()

    print(f"\nSubjects: {len(feature_df)}")
    print("Spectral features: 95")
    print("Connectivity features: 855")
    print(f"Total fused features: {len(feature_names)}")
    print("Classes:", feature_df["class_name"].value_counts().to_dict())

    feature_df.to_csv(
        RESULTS_DIR / "subject_fused_features.csv",
        index=False,
    )

    skf = StratifiedKFold(
        n_splits=N_FOLDS,
        shuffle=True,
        random_state=SEED,
    )

    indices = np.arange(len(feature_df))
    labels = feature_df["label"].to_numpy()

    all_metrics = []
    all_predictions = []

    print("\n" + "=" * 70)
    print("5-FOLD CROSS-VALIDATION")
    print("=" * 70)

    for fold_number, (train_idx, test_idx) in enumerate(
        skf.split(indices, labels),
        start=1,
    ):
        metrics, pred_df = run_fold(
            fold_number,
            train_idx,
            test_idx,
            feature_df,
            feature_names,
        )
        all_metrics.append(metrics)
        pred_df["fold"] = fold_number
        all_predictions.append(pred_df)

    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(
        RESULTS_DIR / "five_fold_metrics.csv",
        index=False,
    )

    oof_df = pd.concat(all_predictions, ignore_index=True)
    oof_df.to_csv(
        RESULTS_DIR / "oof_subject_predictions.csv",
        index=False,
    )

    print("\n" + "=" * 70)
    print("SPECTRAL + CONNECTIVITY 5-FOLD SUMMARY")
    print("=" * 70)

    for metric in [
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "mcc",
        "ftd_recall",
    ]:
        values = metrics_df[metric].to_numpy(dtype=np.float64)
        print(
            f"{metric:20s}: "
            f"{np.nanmean(values):.4f} ± {np.nanstd(values, ddof=1):.4f}"
        )

    pooled_metrics = calculate_metrics(
        oof_df["true_label"].to_numpy(),
        oof_df["predicted_label"].to_numpy(),
    )

    pooled_cm = confusion_matrix(
        oof_df["true_label"],
        oof_df["predicted_label"],
        labels=[0, 1, 2],
    )

    pooled_metrics = add_class_recalls(pooled_metrics, pooled_cm)

    print("\n" + "=" * 70)
    print("POOLED OUT-OF-FOLD RESULTS")
    print("=" * 70)
    print(f"Accuracy:          {pooled_metrics['accuracy']:.4f}")
    print(f"Balanced accuracy: {pooled_metrics['balanced_accuracy']:.4f}")
    print(f"Macro F1:          {pooled_metrics['macro_f1']:.4f}")
    print(f"MCC:               {pooled_metrics['mcc']:.4f}")
    print(f"AD recall:         {pooled_metrics['alzheimer_recall']:.4f}")
    print(f"Control recall:    {pooled_metrics['control_recall']:.4f}")
    print(f"FTD recall:        {pooled_metrics['ftd_recall']:.4f}")
    print("\nPooled confusion matrix:")
    print(pooled_cm)

    with (RESULTS_DIR / "pooled_oof_metrics.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(
            {k: float(v) for k, v in pooled_metrics.items()},
            f,
            indent=4,
        )

    np.savetxt(
        RESULTS_DIR / "pooled_oof_confusion_matrix.csv",
        pooled_cm,
        delimiter=",",
        fmt="%d",
    )

    print(f"\nResults saved under:\n{RESULTS_DIR}")


if __name__ == "__main__":
    main()
