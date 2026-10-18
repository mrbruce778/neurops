import argparse

import importlib.util

import json

import random

from pathlib import Path

import numpy as np

import pandas as pd

import torch

import torch.nn as nn

from torch.utils.data import Dataset, DataLoader, Sampler

from sklearn.model_selection import StratifiedKFold, train_test_split

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

SRC_DIR = PROJECT_ROOT / "src"

WINDOW_MANIFEST = PROJECT_ROOT / "windows" / "window_manifest.csv"

RESULTS_DIR = PROJECT_ROOT / "results_model_v2"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================

# LOAD MODEL FROM 10_model_v2.py

# ============================================================

MODEL_FILE = SRC_DIR / "10_model_v2.py"

spec = importlib.util.spec_from_file_location(

    "eeg_model",

    MODEL_FILE,

)

if spec is None or spec.loader is None:

    raise ImportError(

        f"Could not load model file:\n{MODEL_FILE}"

    )

model_module = importlib.util.module_from_spec(spec)

spec.loader.exec_module(model_module)

EEGS3FNetV2 = model_module.EEGS3FNetV2

# ============================================================

# CONFIGURATION

# ============================================================

SEED = 42

N_FOLDS = 5

VALIDATION_FRACTION = 0.20

BATCH_SIZE = 64

MAX_EPOCHS = 60

PATIENCE = 10

LEARNING_RATE = 1e-3

WEIGHT_DECAY = 1e-4

NUM_WORKERS = 0

N_CLASSES = 3

CLASS_NAMES = [

    "Alzheimer",

    "Control",

    "FTD",

]

LABEL_TO_CLASS = {

    0: "Alzheimer",

    1: "Control",

    2: "FTD",

}

# Equal number of windows contributed by each training subject

# during each epoch.

WINDOWS_PER_SUBJECT = 64

# Small value used to avoid division by zero during z-scoring.

NORMALIZATION_EPS = 1e-6

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

# EEG DATASET

# ============================================================

class EEGWindowDataset(Dataset):

    """

    Loads all windows belonging to the supplied subjects.

    Each returned EEG window has shape:

        [channels, samples]

        [19, 2000]

    Processing performed in __getitem__:

    1. Convert volts -> microvolts.

    2. Calculate mean/std independently for every channel

       within the current 4-second window.

    3. Z-score each channel.

    No statistics from other subjects/windows are used.

    """

    def __init__(self, subject_df):

        self.subject_df = subject_df.reset_index(drop=True)

        self.windows = []

        self.subject_to_indices = {}

        print(

            f"Loading {len(self.subject_df)} subjects into memory..."

        )

        global_index = 0

        for _, row in self.subject_df.iterrows():

            subject_id = str(row["subject_id"])

            label = int(row["label"])

            window_file = Path(row["window_file"])

            if not window_file.exists():

                raise FileNotFoundError(

                    f"Window file not found:\n{window_file}"

                )

            with np.load(

                window_file,

                allow_pickle=False,

            ) as data:

                eeg = data["eeg"].astype(

                    np.float32,

                    copy=False,

                )

            # Expected:

            # [n_windows, 19, 2000]

            if eeg.ndim != 3:

                raise RuntimeError(

                    f"{subject_id}: expected 3D EEG array, "

                    f"got shape {eeg.shape}"

                )

            if eeg.shape[1] != 19:

                raise RuntimeError(

                    f"{subject_id}: expected 19 channels, "

                    f"got {eeg.shape[1]}"

                )

            if eeg.shape[2] != 2000:

                raise RuntimeError(

                    f"{subject_id}: expected 2000 samples/window, "

                    f"got {eeg.shape[2]}"

                )

            subject_indices = []

            for window_index in range(eeg.shape[0]):

                self.windows.append(

                    (

                        eeg[window_index],

                        label,

                        subject_id,

                        window_index,

                    )

                )

                subject_indices.append(global_index)

                global_index += 1

            self.subject_to_indices[subject_id] = subject_indices

        print(

            f"Loaded {len(self.windows)} windows."

        )

    def __len__(self):

        return len(self.windows)

    def __getitem__(self, index):

        (

            eeg,

            label,

            subject_id,

            window_index,

        ) = self.windows[index]

        # ----------------------------------------------------

        # Fixed unit conversion:

        # volts -> microvolts

        # ----------------------------------------------------

        eeg = eeg * 1e6

        # ----------------------------------------------------

        # Per-window, per-channel z-score normalization

        #

        # EEG shape:

        # [19 channels, 2000 samples]

        #

        # Therefore axis=1 computes one mean/std for each

        # channel across its 2000 time samples.

        # ----------------------------------------------------

        mean = eeg.mean(

            axis=1,

            keepdims=True,

        )

        std = eeg.std(

            axis=1,

            keepdims=True,

        )

        eeg = (

            (eeg - mean)

            / (std + NORMALIZATION_EPS)

        )

        # Safety check.

        if not np.isfinite(eeg).all():

            raise RuntimeError(

                f"Non-finite values after normalization: "

                f"{subject_id}, window {window_index}"

            )

        x = torch.from_numpy(

            eeg.copy()

        ).float()

        y = torch.tensor(

            label,

            dtype=torch.long,

        )

        return (

            x,

            y,

            subject_id,

            window_index,

        )

# ============================================================

# SUBJECT-BALANCED TRAINING SAMPLER

# ============================================================

class SubjectBalancedSampler(Sampler):

    """

    Each training subject contributes exactly

    WINDOWS_PER_SUBJECT windows during each epoch.

    This prevents long recordings from dominating training.

    """

    def __init__(

        self,

        dataset,

        windows_per_subject,

        seed,

    ):

        self.dataset = dataset

        self.windows_per_subject = windows_per_subject

        self.seed = seed

        self.epoch = 0

        self.subject_ids = list(

            dataset.subject_to_indices.keys()

        )

    def set_epoch(self, epoch):

        self.epoch = epoch

    def __len__(self):

        return (

            len(self.subject_ids)

            * self.windows_per_subject

        )

    def __iter__(self):

        rng = np.random.default_rng(

            self.seed + self.epoch

        )

        sampled_indices = []

        for subject_id in self.subject_ids:

            available = self.dataset.subject_to_indices[

                subject_id

            ]

            # Only sample with replacement when a subject

            # contains fewer windows than requested.

            replace = (

                len(available)

                < self.windows_per_subject

            )

            selected = rng.choice(

                available,

                size=self.windows_per_subject,

                replace=replace,

            )

            sampled_indices.extend(

                selected.tolist()

            )

        # Mix windows from all subjects.

        rng.shuffle(sampled_indices)

        return iter(sampled_indices)

# ============================================================

# SUBJECT-LEVEL AGGREGATION

# ============================================================

def aggregate_subject_predictions(records):

    """

    Convert window-level probabilities into one prediction

    per subject by averaging probabilities across all windows.

    """

    df = pd.DataFrame(records)

    if df.empty:

        raise RuntimeError(

            "No prediction records were generated."

        )

    subject_rows = []

    for subject_id, group in df.groupby("subject_id"):

        true_labels = group[

            "true_label"

        ].unique()

        if len(true_labels) != 1:

            raise RuntimeError(

                f"Subject {subject_id} has multiple labels: "

                f"{true_labels}"

            )

        true_label = int(true_labels[0])

        mean_probs = np.array(

            [

                group["prob_0"].mean(),

                group["prob_1"].mean(),

                group["prob_2"].mean(),

            ],

            dtype=np.float64,

        )

        predicted_label = int(

            np.argmax(mean_probs)

        )

        subject_rows.append(

            {

                "subject_id": subject_id,

                "true_label": true_label,

                "predicted_label": predicted_label,

                "prob_0": float(mean_probs[0]),

                "prob_1": float(mean_probs[1]),

                "prob_2": float(mean_probs[2]),

                "n_windows": len(group),

            }

        )

    return pd.DataFrame(subject_rows)

# ============================================================

# METRICS

# ============================================================

def calculate_metrics(subject_predictions):

    y_true = subject_predictions[

        "true_label"

    ].to_numpy()

    y_pred = subject_predictions[

        "predicted_label"

    ].to_numpy()

    metrics = {

        "accuracy": accuracy_score(

            y_true,

            y_pred,

        ),

        "balanced_accuracy": balanced_accuracy_score(

            y_true,

            y_pred,

        ),

        "macro_f1": f1_score(

            y_true,

            y_pred,

            average="macro",

            zero_division=0,

        ),

        "mcc": matthews_corrcoef(

            y_true,

            y_pred,

        ),

    }

    cm = confusion_matrix(

        y_true,

        y_pred,

        labels=[0, 1, 2],

    )

    return metrics, cm

# ============================================================

# VALIDATION / TEST PREDICTION

# ============================================================

def predict_dataset(

    model,

    loader,

    device,

):

    model.eval()

    records = []

    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0

    total_samples = 0

    with torch.no_grad():

        for (

            x,

            y,

            subject_ids,

            window_indices,

        ) in loader:

            x = x.to(

                device,

                non_blocking=True,

            )

            y = y.to(

                device,

                non_blocking=True,

            )

            logits = model(x)

            loss = criterion(

                logits,

                y,

            )

            probabilities = torch.softmax(

                logits,

                dim=1,

            )

            batch_size = x.size(0)

            total_loss += (

                loss.item()

                * batch_size

            )

            total_samples += batch_size

            probabilities = (

                probabilities

                .cpu()

                .numpy()

            )

            labels = (

                y

                .cpu()

                .numpy()

            )

            for i in range(batch_size):

                records.append(

                    {

                        "subject_id": subject_ids[i],

                        "window_index": int(

                            window_indices[i]

                        ),

                        "true_label": int(

                            labels[i]

                        ),

                        "prob_0": float(

                            probabilities[i, 0]

                        ),

                        "prob_1": float(

                            probabilities[i, 1]

                        ),

                        "prob_2": float(

                            probabilities[i, 2]

                        ),

                    }

                )

    average_loss = (

        total_loss

        / max(total_samples, 1)

    )

    return average_loss, records

# ============================================================

# TRAIN ONE EPOCH

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

    total_correct = 0

    total_samples = 0

    for (

        x,

        y,

        _,

        _,

    ) in loader:

        x = x.to(

            device,

            non_blocking=True,

        )

        y = y.to(

            device,

            non_blocking=True,

        )

        optimizer.zero_grad(

            set_to_none=True

        )

        logits = model(x)

        loss = criterion(

            logits,

            y,

        )

        loss.backward()

        optimizer.step()

        batch_size = x.size(0)

        total_loss += (

            loss.item()

            * batch_size

        )

        predictions = torch.argmax(

            logits,

            dim=1,

        )

        total_correct += (

            predictions == y

        ).sum().item()

        total_samples += batch_size

    average_loss = (

        total_loss

        / max(total_samples, 1)

    )

    accuracy = (

        total_correct

        / max(total_samples, 1)

    )

    return average_loss, accuracy

# ============================================================

# SAVE SUBJECT SPLIT

# ============================================================

def save_split(

    fold_dir,

    train_df,

    val_df,

    test_df,

):

    split_rows = []

    for split_name, df in [

        ("train", train_df),

        ("validation", val_df),

        ("test", test_df),

    ]:

        for _, row in df.iterrows():

            split_rows.append(

                {

                    "subject_id": row["subject_id"],

                    "class_name": row["class_name"],

                    "label": int(row["label"]),

                    "split": split_name,

                }

            )

    pd.DataFrame(

        split_rows

    ).to_csv(

        fold_dir / "subject_split.csv",

        index=False,

    )

# ============================================================

# PRINT CLASS DISTRIBUTION

# ============================================================

def print_class_distribution(

    name,

    df,

):

    counts = (

        df["class_name"]

        .value_counts()

        .to_dict()

    )

    print(

        f"  {name}: {counts}"

    )

# ============================================================

# RUN ONE CV FOLD

# ============================================================

def run_fold(

    fold_number,

    train_val_indices,

    test_indices,

    manifest,

    device,

):

    print()

    print("=" * 70)

    print(f"FOLD {fold_number}")

    print("=" * 70)

    fold_dir = (

        RESULTS_DIR

        / f"fold_{fold_number}"

    )

    fold_dir.mkdir(

        parents=True,

        exist_ok=True,

    )

    # ========================================================

    # OUTER SPLIT

    # ========================================================

    development_df = (

        manifest

        .iloc[train_val_indices]

        .reset_index(drop=True)

    )

    test_df = (

        manifest

        .iloc[test_indices]

        .reset_index(drop=True)

    )

    # ========================================================

    # INNER TRAIN / VALIDATION SPLIT

    # ========================================================

    inner_indices = np.arange(

        len(development_df)

    )

    train_indices, val_indices = train_test_split(

        inner_indices,

        test_size=VALIDATION_FRACTION,

        stratify=development_df[

            "label"

        ],

        random_state=(

            SEED

            + fold_number

        ),

    )

    train_df = (

        development_df

        .iloc[train_indices]

        .reset_index(drop=True)

    )

    val_df = (

        development_df

        .iloc[val_indices]

        .reset_index(drop=True)

    )

    # ========================================================

    # LEAKAGE CHECKS

    # ========================================================

    train_subjects = set(

        train_df["subject_id"]

    )

    val_subjects = set(

        val_df["subject_id"]

    )

    test_subjects = set(

        test_df["subject_id"]

    )

    assert train_subjects.isdisjoint(

        val_subjects

    )

    assert train_subjects.isdisjoint(

        test_subjects

    )

    assert val_subjects.isdisjoint(

        test_subjects

    )

    total_unique_subjects = (

        train_subjects

        | val_subjects

        | test_subjects

    )

    assert len(total_unique_subjects) == len(

        manifest

    )

    print(

        f"\nTrain subjects: {len(train_df)}"

    )

    print(

        f"Validation subjects: {len(val_df)}"

    )

    print(

        f"Test subjects: {len(test_df)}"

    )

    print("\nClass distribution:")

    print_class_distribution(

        "Train",

        train_df,

    )

    print_class_distribution(

        "Validation",

        val_df,

    )

    print_class_distribution(

        "Test",

        test_df,

    )

    save_split(

        fold_dir,

        train_df,

        val_df,

        test_df,

    )

    # ========================================================

    # DATASETS

    # ========================================================

    train_dataset = EEGWindowDataset(

        train_df

    )

    val_dataset = EEGWindowDataset(

        val_df

    )

    test_dataset = EEGWindowDataset(

        test_df

    )

    # ========================================================

    # VERIFY NORMALIZATION

    # ========================================================

    example_x, _, example_subject, _ = (

        train_dataset[0]

    )

    channel_means = (

        example_x

        .mean(dim=1)

        .numpy()

    )

    channel_stds = (

        example_x

        .std(dim=1, unbiased=False)

        .numpy()

    )

    print("\nNormalization check:")

    print(

        f"  Example subject: {example_subject}"

    )

    print(

        "  Mean absolute channel mean: "

        f"{np.abs(channel_means).mean():.6f}"

    )

    print(

        "  Mean channel std: "

        f"{channel_stds.mean():.6f}"

    )

    # After z-scoring these should be approximately:

    #

    # mean absolute channel mean \~ 0

    # mean channel std          \~ 1

    if (

        np.abs(channel_means).mean()

        > 1e-3

    ):

        raise RuntimeError(

            "Normalization verification failed: "

            "channel means are not approximately zero."

        )

    if not np.isclose(

        channel_stds.mean(),

        1.0,

        atol=1e-3,

    ):

        raise RuntimeError(

            "Normalization verification failed: "

            "channel standard deviation is not approximately 1."

        )

    # ========================================================

    # TRAINING SAMPLER

    # ========================================================

    train_sampler = SubjectBalancedSampler(

        train_dataset,

        windows_per_subject=(

            WINDOWS_PER_SUBJECT

        ),

        seed=(

            SEED

            + fold_number

        ),

    )

    # ========================================================

    # DATA LOADERS

    # ========================================================

    use_pin_memory = (

        device.type == "cuda"

    )

    train_loader = DataLoader(

        train_dataset,

        batch_size=BATCH_SIZE,

        sampler=train_sampler,

        num_workers=NUM_WORKERS,

        pin_memory=use_pin_memory,

    )

    val_loader = DataLoader(

        val_dataset,

        batch_size=BATCH_SIZE,

        shuffle=False,

        num_workers=NUM_WORKERS,

        pin_memory=use_pin_memory,

    )

    test_loader = DataLoader(

        test_dataset,

        batch_size=BATCH_SIZE,

        shuffle=False,

        num_workers=NUM_WORKERS,

        pin_memory=use_pin_memory,

    )

    # ========================================================

    # MODEL

    # ========================================================

    model = EEGS3FNetV2(

        n_channels=19,

        n_classes=N_CLASSES,

        feature_dim=128,

        sfreq=500.0,

    ).to(device)

    total_parameters = sum(

        p.numel()

        for p in model.parameters()

    )

    trainable_parameters = sum(

        p.numel()

        for p in model.parameters()

        if p.requires_grad

    )

    print(

        f"\nModel parameters: {total_parameters:,}"

    )

    print(

        f"Trainable parameters: "

        f"{trainable_parameters:,}"

    )

    optimizer = torch.optim.AdamW(

        model.parameters(),

        lr=LEARNING_RATE,

        weight_decay=WEIGHT_DECAY,

    )

    criterion = nn.CrossEntropyLoss()

    # ========================================================

    # EARLY STOPPING

    # ========================================================

    best_val_f1 = -1.0

    epochs_without_improvement = 0

    best_model_path = (

        fold_dir

        / "best_model.pt"

    )

    history = []

    print("\nTraining...")

    print("-" * 70)

    # ========================================================

    # TRAINING LOOP

    # ========================================================

    for epoch in range(

        1,

        MAX_EPOCHS + 1,

    ):

        train_sampler.set_epoch(epoch)

        (

            train_loss,

            train_accuracy,

        ) = train_one_epoch(

            model,

            train_loader,

            optimizer,

            criterion,

            device,

        )

        (

            val_loss,

            val_records,

        ) = predict_dataset(

            model,

            val_loader,

            device,

        )

        val_subject_predictions = (

            aggregate_subject_predictions(

                val_records

            )

        )

        (

            val_metrics,

            _,

        ) = calculate_metrics(

            val_subject_predictions

        )

        val_f1 = val_metrics[

            "macro_f1"

        ]

        print(

            f"Epoch {epoch:03d} | "

            f"train loss {train_loss:.4f} | "

            f"train acc {train_accuracy:.4f} | "

            f"val loss {val_loss:.4f} | "

            f"val subject F1 {val_f1:.4f}"

        )

        history.append(

            {

                "epoch": epoch,

                "train_loss":

                    train_loss,

                "train_window_accuracy":

                    train_accuracy,

                "val_loss":

                    val_loss,

                "val_subject_accuracy":

                    val_metrics[

                        "accuracy"

                    ],

                "val_subject_balanced_accuracy":

                    val_metrics[

                        "balanced_accuracy"

                    ],

                "val_subject_macro_f1":

                    val_f1,

                "val_subject_mcc":

                    val_metrics[

                        "mcc"

                    ],

            }

        )

        # ====================================================

        # CHECKPOINT SELECTION

        # ====================================================

        if val_f1 > (

            best_val_f1

            + 1e-6

        ):

            best_val_f1 = val_f1

            epochs_without_improvement = 0

            torch.save(

                {

                    "fold":

                        fold_number,

                    "epoch":

                        epoch,

                    "model_state_dict":

                        model.state_dict(),

                    "optimizer_state_dict":

                        optimizer.state_dict(),

                    "best_val_macro_f1":

                        best_val_f1,

                    "normalization":

                        "per_window_per_channel_zscore",

                    "config":

                        {

                            "seed":

                                SEED,

                            "batch_size":

                                BATCH_SIZE,

                            "learning_rate":

                                LEARNING_RATE,

                            "weight_decay":

                                WEIGHT_DECAY,

                            "windows_per_subject":

                                WINDOWS_PER_SUBJECT,

                            "max_epochs":

                                MAX_EPOCHS,

                            "patience":

                                PATIENCE,

                            "validation_fraction":

                                VALIDATION_FRACTION,

                        },

                },

                best_model_path,

            )

        else:

            epochs_without_improvement += 1

        if (

            epochs_without_improvement

            >= PATIENCE

        ):

            print("\nEarly stopping.")

            break

    # ========================================================

    # SAVE TRAINING HISTORY

    # ========================================================

    pd.DataFrame(

        history

    ).to_csv(

        fold_dir

        / "training_history.csv",

        index=False,

    )

    # ========================================================

    # LOAD BEST CHECKPOINT

    # ========================================================

    if not best_model_path.exists():

        raise RuntimeError(

            "Training finished without producing "

            "a model checkpoint."

        )

    checkpoint = torch.load(

        best_model_path,

        map_location=device,

        weights_only=False,

    )

    model.load_state_dict(

        checkpoint[

            "model_state_dict"

        ]

    )

    print(

        f"\nBest epoch: "

        f"{checkpoint['epoch']}"

    )

    print(

        f"Best validation macro-F1: "

        f"{checkpoint['best_val_macro_f1']:.4f}"

    )

    # ========================================================

    # FINAL TEST

    # ========================================================

    (

        test_loss,

        test_records,

    ) = predict_dataset(

        model,

        test_loader,

        device,

    )

    # --------------------------------------------------------

    # Window-level predictions

    # --------------------------------------------------------

    window_predictions = pd.DataFrame(

        test_records

    )

    window_predictions.to_csv(

        fold_dir

        / "test_window_predictions.csv",

        index=False,

    )

    # --------------------------------------------------------

    # Subject-level predictions

    # --------------------------------------------------------

    subject_predictions = (

        aggregate_subject_predictions(

            test_records

        )

    )

    subject_predictions[

        "true_class"

    ] = subject_predictions[

        "true_label"

    ].map(

        LABEL_TO_CLASS

    )

    subject_predictions[

        "predicted_class"

    ] = subject_predictions[

        "predicted_label"

    ].map(

        LABEL_TO_CLASS

    )

    subject_predictions.to_csv(

        fold_dir

        / "test_subject_predictions.csv",

        index=False,

    )

    # ========================================================

    # TEST METRICS

    # ========================================================

    (

        test_metrics,

        cm,

    ) = calculate_metrics(

        subject_predictions

    )

    test_metrics[

        "test_loss"

    ] = test_loss

    test_metrics[

        "best_epoch"

    ] = int(

        checkpoint["epoch"]

    )

    test_metrics[

        "best_val_macro_f1"

    ] = float(

        checkpoint[

            "best_val_macro_f1"

        ]

    )

    test_metrics[

        "normalization"

    ] = (

        "per_window_per_channel_zscore"

    )

    with (

        fold_dir

        / "test_metrics.json"

    ).open(

        "w",

        encoding="utf-8",

    ) as f:

        json.dump(

            test_metrics,

            f,

            indent=4,

        )

    np.savetxt(

        fold_dir

        / "confusion_matrix.csv",

        cm,

        delimiter=",",

        fmt="%d",

    )

    # ========================================================

    # REPORT

    # ========================================================

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

        "\nConfusion matrix "

        "(rows=true, columns=predicted):"

    )

    print(cm)

    # ========================================================

    # CLEANUP

    # ========================================================

    del train_dataset

    del val_dataset

    del test_dataset

    del train_loader

    del val_loader

    del test_loader

    del model

    if torch.cuda.is_available():

        torch.cuda.empty_cache()

    return test_metrics

# ============================================================

# MAIN

# ============================================================

def main():

    parser = argparse.ArgumentParser(

        description=(

            "Subject-level stratified 5-fold "

            "cross-validation for EEG classification."

        )

    )

    group = parser.add_mutually_exclusive_group(

        required=True

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

        help=(

            "Run one outer CV fold only."

        ),

    )

    group.add_argument(

        "--all-folds",

        action="store_true",

        help=(

            "Run all five outer CV folds."

        ),

    )

    args = parser.parse_args()

    # ========================================================

    # REPRODUCIBILITY

    # ========================================================

    set_seed(SEED)

    # ========================================================

    # DEVICE

    # ========================================================

    device = torch.device(

        "cuda"

        if torch.cuda.is_available()

        else "cpu"

    )

    print("=" * 70)

    print(

        "EEG S3F-NET V2 - 5-FOLD SUBJECT-LEVEL TRAINING"

    )

    print("=" * 70)

    print(

        f"\nDevice: {device}"

    )

    if torch.cuda.is_available():

        print(

            "GPU:",

            torch.cuda.get_device_name(0),

        )

        print(

            "PyTorch CUDA:",

            torch.version.cuda,

        )

    # ========================================================

    # LOAD MANIFEST

    # ========================================================

    if not WINDOW_MANIFEST.exists():

        raise FileNotFoundError(

            f"Window manifest not found:\n"

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

        "n_windows",

    }

    missing_columns = (

        required_columns

        - set(manifest.columns)

    )

    if missing_columns:

        raise RuntimeError(

            "Window manifest is missing columns: "

            f"{sorted(missing_columns)}"

        )

    print(

        f"\nSubjects: {len(manifest)}"

    )

    print(

        "Total available windows: "

        f"{manifest['n_windows'].sum()}"

    )

    # ========================================================

    # MANIFEST SAFETY CHECKS

    # ========================================================

    if manifest[

        "subject_id"

    ].duplicated().any():

        duplicates = manifest.loc[

            manifest[

                "subject_id"

            ].duplicated(

                keep=False

            ),

            "subject_id",

        ].tolist()

        raise RuntimeError(

            "Duplicate subject IDs detected: "

            f"{duplicates}"

        )

    if len(manifest) != 88:

        print(

            "\nWARNING: Expected 88 subjects, "

            f"found {len(manifest)}."

        )

    observed_labels = set(

        manifest["label"].unique()

    )

    expected_labels = {

        0,

        1,

        2,

    }

    if observed_labels != expected_labels:

        raise RuntimeError(

            "Unexpected label set. "

            f"Expected {expected_labels}, "

            f"found {observed_labels}."

        )

    # ========================================================

    # OUTER STRATIFIED 5-FOLD CV

    # ========================================================

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

    # ========================================================

    # SELECT FOLDS

    # ========================================================

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

    # ========================================================

    # RUN

    # ========================================================

    for fold_number in requested_folds:

        (

            train_val_indices,

            test_indices,

        ) = folds[

            fold_number - 1

        ]

        metrics = run_fold(

            fold_number,

            train_val_indices,

            test_indices,

            manifest,

            device,

        )

        metrics["fold"] = fold_number

        all_metrics.append(

            metrics

        )

    # ========================================================

    # SAVE SUMMARY

    # ========================================================

    metrics_df = pd.DataFrame(

        all_metrics

    )

    if args.all_folds:

        output_path = (

            RESULTS_DIR

            / "five_fold_metrics.csv"

        )

    else:

        output_path = (

            RESULTS_DIR

            / f"fold_{args.fold}"

            / "fold_summary.csv"

        )

    metrics_df.to_csv(

        output_path,

        index=False,

    )

    # ========================================================

    # FIVE-FOLD SUMMARY

    # ========================================================

    if args.all_folds:

        print()

        print("=" * 70)

        print("5-FOLD SUMMARY")

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

    print(

        f"\nResults saved under:\n"

        f"{RESULTS_DIR}"

    )

if __name__ == "__main__":

    main()
