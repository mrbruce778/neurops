from pathlib import Path
import importlib.util
import json
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import (
    StratifiedKFold,
    train_test_split,
)
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
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

CONNECTIVITY_FILE = (
    PROJECT_ROOT
    / "results_connectivity_baseline"
    / "subject_connectivity_features.csv"
)

MODEL_FILE = (
    PROJECT_ROOT
    / "src"
    / "16_connectivity_attention_model.py"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "results_connectivity_attention"
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
VAL_FRACTION = 0.20

BAG_SIZE = 32
SUBJECT_BATCH_SIZE = 4

MAX_EPOCHS = 60
PATIENCE = 10

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

NORMALIZATION_EPS = 1e-6

CLASS_NAMES = {
    0: "Alzheimer",
    1: "Control",
    2: "FTD",
}


# ============================================================
# MODEL IMPORT
# ============================================================

spec = importlib.util.spec_from_file_location(
    "connectivity_attention_model",
    MODEL_FILE,
)

model_module = (
    importlib.util.module_from_spec(
        spec
    )
)

spec.loader.exec_module(
    model_module
)

ConnectivityAwareSWATS = (
    model_module.ConnectivityAwareSWATS
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# CONNECTIVITY TABLE
# ============================================================

def load_connectivity_table():

    if not CONNECTIVITY_FILE.exists():
        raise FileNotFoundError(
            f"Connectivity file not found:\n"
            f"{CONNECTIVITY_FILE}"
        )

    df = pd.read_csv(
        CONNECTIVITY_FILE
    )

    metadata_columns = {
        "subject_id",
        "label",
        "class_name",
    }

    feature_columns = [
        column
        for column in df.columns
        if column not in metadata_columns
    ]

    if len(feature_columns) != 855:
        raise RuntimeError(
            "Expected 855 connectivity features, "
            f"found {len(feature_columns)}."
        )

    if df["subject_id"].duplicated().any():
        raise RuntimeError(
            "Duplicate subject_id in connectivity table."
        )

    lookup = {}

    for _, row in df.iterrows():

        subject_id = str(
            row["subject_id"]
        )

        values = row[
            feature_columns
        ].to_numpy(
            dtype=np.float32
        )

        if not np.isfinite(values).all():
            raise RuntimeError(
                f"Non-finite connectivity values "
                f"for {subject_id}."
            )

        # The baseline flattened [5 bands, 171 pairs].
        values = values.reshape(
            5,
            171,
        )

        lookup[subject_id] = values

    return lookup


# ============================================================
# DATASET
# ============================================================

class SubjectBagDataset(Dataset):

    def __init__(
        self,
        subject_df,
        connectivity_lookup,
        training=True,
    ):
        self.subject_df = (
            subject_df
            .reset_index(drop=True)
            .copy()
        )

        self.connectivity_lookup = (
            connectivity_lookup
        )

        self.training = training

        self.epoch = 0

        self.subject_windows = {}

        for _, row in self.subject_df.iterrows():

            subject_id = str(
                row["subject_id"]
            )

            window_file = Path(
                row["window_file"]
            )

            if not window_file.exists():
                raise FileNotFoundError(
                    f"Window file not found:\n"
                    f"{window_file}"
                )

            with np.load(
                window_file,
                allow_pickle=False,
            ) as data:

                eeg = data[
                    "eeg"
                ].astype(
                    np.float32,
                    copy=False,
                )

            if eeg.ndim != 3:
                raise RuntimeError(
                    f"{subject_id}: unexpected EEG "
                    f"shape {eeg.shape}"
                )

            if eeg.shape[1:] != (
                19,
                2000,
            ):
                raise RuntimeError(
                    f"{subject_id}: expected "
                    f"[N,19,2000], got {eeg.shape}"
                )

            self.subject_windows[
                subject_id
            ] = eeg

            if subject_id not in (
                self.connectivity_lookup
            ):
                raise RuntimeError(
                    f"No connectivity features "
                    f"for {subject_id}."
                )

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return len(
            self.subject_df
        )

    def normalize_windows(self, eeg):

        # volts -> microvolts
        eeg = eeg * 1e6

        mean = eeg.mean(
            axis=2,
            keepdims=True,
        )

        std = eeg.std(
            axis=2,
            keepdims=True,
        )

        eeg = (
            eeg - mean
        ) / (
            std + NORMALIZATION_EPS
        )

        return eeg.astype(
            np.float32,
            copy=False,
        )

    def __getitem__(self, index):

        row = self.subject_df.iloc[
            index
        ]

        subject_id = str(
            row["subject_id"]
        )

        label = int(
            row["label"]
        )

        eeg = self.subject_windows[
            subject_id
        ]

        n_windows = eeg.shape[0]

        if self.training:

            rng = np.random.default_rng(
                SEED
                + self.epoch * 10000
                + index
            )

            if n_windows >= BAG_SIZE:
                selected = rng.choice(
                    n_windows,
                    size=BAG_SIZE,
                    replace=False,
                )
            else:
                selected = rng.choice(
                    n_windows,
                    size=BAG_SIZE,
                    replace=True,
                )

        else:

            # Validation uses one deterministic bag per subject
            # during training for stable early stopping.
            if n_windows >= BAG_SIZE:
                selected = np.linspace(
                    0,
                    n_windows - 1,
                    BAG_SIZE,
                    dtype=int,
                )
            else:
                selected = np.arange(
                    BAG_SIZE
                ) % n_windows

        bag = eeg[
            selected
        ].copy()

        bag = self.normalize_windows(
            bag
        )

        connectivity = (
            self.connectivity_lookup[
                subject_id
            ]
        )

        return (
            torch.from_numpy(
                bag
            ),
            torch.from_numpy(
                connectivity.copy()
            ),
            torch.tensor(
                label,
                dtype=torch.long,
            ),
            subject_id,
        )


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
# TRAIN / VALIDATE
# ============================================================

def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
):

    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for (
        eeg,
        connectivity,
        labels,
        _,
    ) in loader:

        eeg = eeg.to(
            device,
            non_blocking=True,
        )

        connectivity = connectivity.to(
            device,
            non_blocking=True,
        )

        labels = labels.to(
            device,
            non_blocking=True,
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        logits, _, _, _ = model(
            eeg,
            connectivity,
        )

        loss = criterion(
            logits,
            labels,
        )

        loss.backward()

        optimizer.step()

        total_loss += (
            loss.item()
            * labels.size(0)
        )

        predictions = logits.argmax(
            dim=1
        )

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

    return (
        total_loss / total,
        correct / total,
    )


@torch.no_grad()
def evaluate_loader(
    model,
    loader,
    criterion,
    device,
):

    model.eval()

    total_loss = 0.0
    total = 0

    y_true = []
    y_pred = []

    for (
        eeg,
        connectivity,
        labels,
        _,
    ) in loader:

        eeg = eeg.to(
            device
        )

        connectivity = connectivity.to(
            device
        )

        labels = labels.to(
            device
        )

        logits, _, _, _ = model(
            eeg,
            connectivity,
        )

        loss = criterion(
            logits,
            labels,
        )

        total_loss += (
            loss.item()
            * labels.size(0)
        )

        total += labels.size(0)

        predictions = logits.argmax(
            dim=1
        )

        y_true.extend(
            labels.cpu().numpy()
        )

        y_pred.extend(
            predictions.cpu().numpy()
        )

    metrics = calculate_metrics(
        np.asarray(y_true),
        np.asarray(y_pred),
    )

    return (
        total_loss / total,
        metrics,
    )


# ============================================================
# FULL SUBJECT TEST EVALUATION
# ============================================================

@torch.no_grad()
def evaluate_full_subjects(
    model,
    subject_df,
    connectivity_lookup,
    device,
):

    model.eval()

    rows = []

    for _, row in (
        subject_df
        .reset_index(drop=True)
        .iterrows()
    ):

        subject_id = str(
            row["subject_id"]
        )

        label = int(
            row["label"]
        )

        window_file = Path(
            row["window_file"]
        )

        with np.load(
            window_file,
            allow_pickle=False,
        ) as data:

            eeg = data[
                "eeg"
            ].astype(
                np.float32,
                copy=False,
            )

        # volts -> microvolts, then per-window/per-channel z-score
        eeg = eeg * 1e6

        mean = eeg.mean(
            axis=2,
            keepdims=True,
        )

        std = eeg.std(
            axis=2,
            keepdims=True,
        )

        eeg = (
            eeg - mean
        ) / (
            std + NORMALIZATION_EPS
        )

        connectivity = (
            connectivity_lookup[
                subject_id
            ]
        )

        bag_probabilities = []
        bag_sizes = []

        n_windows = eeg.shape[0]

        for start in range(
            0,
            n_windows,
            BAG_SIZE,
        ):

            bag = eeg[
                start:
                start + BAG_SIZE
            ]

            original_size = (
                bag.shape[0]
            )

            if original_size < BAG_SIZE:

                needed = (
                    BAG_SIZE
                    - original_size
                )

                padding = eeg[
                    :needed
                ]

                bag = np.concatenate(
                    [
                        bag,
                        padding,
                    ],
                    axis=0,
                )

            eeg_tensor = (
                torch.from_numpy(
                    bag.astype(
                        np.float32,
                        copy=False,
                    )
                )
                .unsqueeze(0)
                .to(device)
            )

            conn_tensor = (
                torch.from_numpy(
                    connectivity.astype(
                        np.float32,
                        copy=False,
                    )
                )
                .unsqueeze(0)
                .to(device)
            )

            (
                logits,
                _,
                _,
                _,
            ) = model(
                eeg_tensor,
                conn_tensor,
            )

            probabilities = (
                torch.softmax(
                    logits,
                    dim=1,
                )[0]
                .cpu()
                .numpy()
            )

            bag_probabilities.append(
                probabilities
            )

            bag_sizes.append(
                original_size
            )

        bag_probabilities = np.stack(
            bag_probabilities,
            axis=0,
        )

        bag_sizes = np.asarray(
            bag_sizes,
            dtype=np.float64,
        )

        subject_probability = np.average(
            bag_probabilities,
            axis=0,
            weights=bag_sizes,
        )

        prediction = int(
            np.argmax(
                subject_probability
            )
        )

        rows.append(
            {
                "subject_id": subject_id,
                "true_label": label,
                "predicted_label": prediction,
                "prob_alzheimer":
                    float(
                        subject_probability[0]
                    ),
                "prob_control":
                    float(
                        subject_probability[1]
                    ),
                "prob_ftd":
                    float(
                        subject_probability[2]
                    ),
            }
        )

    prediction_df = pd.DataFrame(
        rows
    )

    metrics = calculate_metrics(
        prediction_df[
            "true_label"
        ].to_numpy(),

        prediction_df[
            "predicted_label"
        ].to_numpy(),
    )

    cm = confusion_matrix(
        prediction_df[
            "true_label"
        ],

        prediction_df[
            "predicted_label"
        ],

        labels=[
            0,
            1,
            2,
        ],
    )

    return (
        metrics,
        cm,
        prediction_df,
    )


# ============================================================
# ONE OUTER FOLD
# ============================================================

def run_fold(
    fold_number,
    development_df,
    test_df,
    connectivity_lookup,
    device,
):

    print()
    print("=" * 70)
    print(
        f"CONNECTIVITY-ATTENTION MODEL - FOLD "
        f"{fold_number}"
    )
    print("=" * 70)

    train_df, val_df = train_test_split(
        development_df,
        test_size=VAL_FRACTION,
        stratify=development_df[
            "label"
        ],
        random_state=(
            SEED + fold_number
        ),
    )

    train_df = train_df.reset_index(
        drop=True
    )

    val_df = val_df.reset_index(
        drop=True
    )

    test_df = test_df.reset_index(
        drop=True
    )

    print(
        f"Train subjects: {len(train_df)}"
    )
    print(
        f"Val subjects:   {len(val_df)}"
    )
    print(
        f"Test subjects:  {len(test_df)}"
    )

    train_dataset = SubjectBagDataset(
        train_df,
        connectivity_lookup,
        training=True,
    )

    val_dataset = SubjectBagDataset(
        val_df,
        connectivity_lookup,
        training=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=SUBJECT_BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=SUBJECT_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    model = ConnectivityAwareSWATS(
        n_classes=3
    ).to(
        device
    )

    print(
        "Trainable parameters:",
        sum(
            p.numel()
            for p in model.parameters()
            if p.requires_grad
        ),
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    criterion = nn.CrossEntropyLoss()

    fold_dir = (
        RESULTS_DIR
        / f"fold_{fold_number}"
    )

    fold_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint_file = (
        fold_dir
        / "best_model.pt"
    )

    best_val_f1 = -1.0
    best_epoch = 0
    epochs_without_improvement = 0

    history = []

    for epoch in range(
        1,
        MAX_EPOCHS + 1,
    ):

        train_dataset.set_epoch(
            epoch
        )

        (
            train_loss,
            train_acc,
        ) = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
        )

        (
            val_loss,
            val_metrics,
        ) = evaluate_loader(
            model,
            val_loader,
            criterion,
            device,
        )

        val_f1 = (
            val_metrics[
                "macro_f1"
            ]
        )

        print(
            f"Epoch {epoch:03d} | "
            f"train loss {train_loss:.4f} | "
            f"train acc {train_acc:.4f} | "
            f"val loss {val_loss:.4f} | "
            f"val subject F1 {val_f1:.4f}"
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss":
                    train_loss,
                "train_accuracy":
                    train_acc,
                "val_loss":
                    val_loss,
                "val_macro_f1":
                    val_f1,
            }
        )

        if (
            val_f1
            > best_val_f1 + 1e-6
        ):

            best_val_f1 = val_f1
            best_epoch = epoch
            epochs_without_improvement = 0

            torch.save(
                {
                    "model_state_dict":
                        model.state_dict(),
                    "epoch":
                        epoch,
                    "val_macro_f1":
                        val_f1,
                },
                checkpoint_file,
            )

        else:

            epochs_without_improvement += 1

        if (
            epochs_without_improvement
            >= PATIENCE
        ):

            print(
                "\nEarly stopping."
            )
            break

    pd.DataFrame(
        history
    ).to_csv(
        fold_dir
        / "training_history.csv",
        index=False,
    )

    checkpoint = torch.load(
        checkpoint_file,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    print(
        f"\nBest epoch: {best_epoch}"
    )

    print(
        f"Best validation macro-F1: "
        f"{best_val_f1:.4f}"
    )

    (
        test_metrics,
        test_cm,
        prediction_df,
    ) = evaluate_full_subjects(
        model,
        test_df,
        connectivity_lookup,
        device,
    )

    print()
    print("=" * 70)
    print(
        f"FOLD {fold_number} TEST RESULTS"
    )
    print("=" * 70)

    print(
        f"Accuracy:          "
        f"{test_metrics['accuracy']:.4f}"
    )

    print(
        f"Balanced accuracy: "
        f"{test_metrics['balanced_accuracy']:.4f}"
    )

    print(
        f"Macro F1:          "
        f"{test_metrics['macro_f1']:.4f}"
    )

    print(
        f"MCC:               "
        f"{test_metrics['mcc']:.4f}"
    )

    print(
        "\nConfusion matrix:"
    )
    print(
        test_cm
    )

    prediction_df[
        "fold"
    ] = fold_number

    prediction_df.to_csv(
        fold_dir
        / "subject_predictions.csv",
        index=False,
    )

    np.savetxt(
        fold_dir
        / "confusion_matrix.csv",
        test_cm,
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
            test_metrics,
            f,
            indent=4,
        )

    test_metrics[
        "fold"
    ] = fold_number

    return (
        test_metrics,
        prediction_df,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    set_seed(SEED)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 70)
    print(
        "CONNECTIVITY-AWARE SUBJECT-LEVEL EEG MODEL"
    )
    print("=" * 70)

    print(
        "Device:",
        device,
    )

    manifest = pd.read_csv(
        WINDOW_MANIFEST
    )

    connectivity_lookup = (
        load_connectivity_table()
    )

    manifest_subjects = set(
        manifest[
            "subject_id"
        ].astype(str)
    )

    connectivity_subjects = set(
        connectivity_lookup.keys()
    )

    if (
        manifest_subjects
        != connectivity_subjects
    ):

        raise RuntimeError(
            "Subject mismatch between "
            "window manifest and connectivity table."
        )

    print(
        f"Subjects: {len(manifest)}"
    )

    print(
        "Classes:",
        manifest[
            "class_name"
        ].value_counts().to_dict(),
    )

    skf = StratifiedKFold(
        n_splits=N_FOLDS,
        shuffle=True,
        random_state=SEED,
    )

    indices = np.arange(
        len(manifest)
    )

    labels = manifest[
        "label"
    ].to_numpy()

    folds = list(
        skf.split(
            indices,
            labels,
        )
    )

    all_metrics = []
    all_predictions = []

    for fold_number, (
        development_indices,
        test_indices,
    ) in enumerate(
        folds,
        start=1,
    ):

        development_df = (
            manifest
            .iloc[
                development_indices
            ]
            .reset_index(drop=True)
        )

        test_df = (
            manifest
            .iloc[
                test_indices
            ]
            .reset_index(drop=True)
        )

        metrics, predictions = (
            run_fold(
                fold_number,
                development_df,
                test_df,
                connectivity_lookup,
                device,
            )
        )

        all_metrics.append(
            metrics
        )

        all_predictions.append(
            predictions
        )

    metrics_df = pd.DataFrame(
        all_metrics
    )

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

    print()
    print("=" * 70)
    print(
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
        ].to_numpy(
            dtype=np.float64
        )

        print(
            f"{metric:20s}: "
            f"{values.mean():.4f} "
            f"± "
            f"{values.std(ddof=1):.4f}"
        )

    pooled_metrics = calculate_metrics(
        oof_predictions[
            "true_label"
        ].to_numpy(),

        oof_predictions[
            "predicted_label"
        ].to_numpy(),
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

    ftd_recall = (
        pooled_cm[2, 2]
        / pooled_cm[2].sum()
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
        f"FTD recall:        "
        f"{ftd_recall:.4f}"
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

        payload = dict(
            pooled_metrics
        )

        payload[
            "ftd_recall"
        ] = float(
            ftd_recall
        )

        json.dump(
            payload,
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

    print(
        f"\nResults saved under:\n"
        f"{RESULTS_DIR}"
    )


if __name__ == "__main__":
    main()
