import argparse
import importlib.util
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
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

RESULTS_DIR = PROJECT_ROOT / "results_swats"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD SWATS MODEL
# ============================================================

MODEL_FILE = SRC_DIR / "12_swats_model.py"

spec = importlib.util.spec_from_file_location(
    "swats_model",
    MODEL_FILE,
)

if spec is None or spec.loader is None:
    raise ImportError(f"Could not load model:\n{MODEL_FILE}")

model_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model_module)

SWATSNet = model_module.SWATSNet


# ============================================================
# CONFIG
# ============================================================

SEED = 42

N_FOLDS = 5
VALIDATION_FRACTION = 0.20

# Number of subjects processed simultaneously.
# Each subject contains BAG_SIZE windows.
SUBJECT_BATCH_SIZE = 4

BAG_SIZE = 32

MAX_EPOCHS = 60
PATIENCE = 10

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

NUM_WORKERS = 0
N_CLASSES = 3

NORMALIZATION_EPS = 1e-6

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
# NORMALIZATION
# ============================================================

def normalize_windows(eeg):
    """
    eeg:
        [K, 19, 2000]

    Same normalization as previous experiments:
    volts -> microvolts
    per-window, per-channel z-score.
    """

    eeg = eeg.astype(np.float32, copy=True)

    eeg *= 1e6

    mean = eeg.mean(
        axis=2,
        keepdims=True,
    )

    std = eeg.std(
        axis=2,
        keepdims=True,
    )

    eeg = (
        (eeg - mean)
        / (std + NORMALIZATION_EPS)
    )

    if not np.isfinite(eeg).all():
        raise RuntimeError(
            "Non-finite values after normalization."
        )

    return eeg


# ============================================================
# TRAINING DATASET
# ============================================================

class TrainingSubjectBagDataset(Dataset):
    """
    One dataset item = ONE SUBJECT.

    Every time a subject is requested, BAG_SIZE windows
    are sampled from that subject.

    Output:
        x = [K, 19, 2000]
        y = one subject label
    """

    def __init__(
        self,
        subject_df,
        bag_size,
        seed,
    ):

        self.subject_df = (
            subject_df
            .reset_index(drop=True)
            .copy()
        )

        self.bag_size = bag_size
        self.seed = seed

        self.epoch = 0

        self.subject_data = []

        print(
            f"Loading {len(self.subject_df)} "
            f"training subjects..."
        )

        for _, row in self.subject_df.iterrows():

            subject_id = str(
                row["subject_id"]
            )

            label = int(
                row["label"]
            )

            window_file = Path(
                row["window_file"]
            )

            if not window_file.exists():
                raise FileNotFoundError(
                    f"Missing window file:\n"
                    f"{window_file}"
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
                    f"{subject_id}: expected 3D EEG, "
                    f"got {eeg.shape}"
                )

            if eeg.shape[1:] != (19, 2000):
                raise RuntimeError(
                    f"{subject_id}: expected "
                    f"[N,19,2000], got {eeg.shape}"
                )

            self.subject_data.append(
                {
                    "subject_id": subject_id,
                    "label": label,
                    "eeg": eeg,
                }
            )

        print("Training subjects loaded.")

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return len(self.subject_data)

    def __getitem__(self, index):

        item = self.subject_data[index]

        eeg = item["eeg"]

        n_windows = eeg.shape[0]

        # Deterministic given:
        # seed + epoch + subject index
        rng = np.random.default_rng(
            self.seed
            + self.epoch * 10000
            + index
        )

        replace = (
            n_windows < self.bag_size
        )

        selected = rng.choice(
            n_windows,
            size=self.bag_size,
            replace=replace,
        )

        bag = eeg[selected]

        bag = normalize_windows(bag)

        x = torch.from_numpy(
            bag
        ).float()

        y = torch.tensor(
            item["label"],
            dtype=torch.long,
        )

        return (
            x,
            y,
            item["subject_id"],
        )


# ============================================================
# DETERMINISTIC EVALUATION DATASET
# ============================================================

class EvaluationSubjectBagDataset(Dataset):
    """
    Creates deterministic, non-random bags for validation/test.

    Every subject's windows are divided sequentially into
    BAG_SIZE groups.

    If the final group is smaller than BAG_SIZE, deterministic
    wrap-around padding from the beginning of the recording is
    used.

    This means every original window is represented in
    evaluation at least once.

    Each returned item is ONE BAG, but bags are later
    aggregated into ONE SUBJECT prediction.
    """

    def __init__(
        self,
        subject_df,
        bag_size,
    ):

        self.subject_df = (
            subject_df
            .reset_index(drop=True)
            .copy()
        )

        self.bag_size = bag_size

        self.bags = []

        print(
            f"Preparing deterministic evaluation bags "
            f"for {len(self.subject_df)} subjects..."
        )

        for _, row in self.subject_df.iterrows():

            subject_id = str(
                row["subject_id"]
            )

            label = int(
                row["label"]
            )

            window_file = Path(
                row["window_file"]
            )

            if not window_file.exists():
                raise FileNotFoundError(
                    f"Missing window file:\n"
                    f"{window_file}"
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
                    f"{subject_id}: expected 3D EEG, "
                    f"got {eeg.shape}"
                )

            if eeg.shape[1:] != (19, 2000):
                raise RuntimeError(
                    f"{subject_id}: unexpected shape "
                    f"{eeg.shape}"
                )

            n_windows = eeg.shape[0]

            if n_windows == 0:
                raise RuntimeError(
                    f"{subject_id}: no EEG windows."
                )

            bag_number = 0

            for start in range(
                0,
                n_windows,
                bag_size,
            ):

                stop = min(
                    start + bag_size,
                    n_windows,
                )

                indices = np.arange(
                    start,
                    stop,
                )

                # Deterministic padding.
                if len(indices) < bag_size:

                    needed = (
                        bag_size
                        - len(indices)
                    )

                    padding = (
                        np.arange(needed)
                        % n_windows
                    )

                    indices = np.concatenate(
                        [indices, padding]
                    )

                bag = eeg[indices]

                self.bags.append(
                    {
                        "subject_id": subject_id,
                        "label": label,
                        "bag_number": bag_number,
                        "eeg": bag,
                        "n_original_windows": (
                            stop - start
                        ),
                    }
                )

                bag_number += 1

        print(
            f"Prepared {len(self.bags)} "
            f"evaluation bags."
        )

    def __len__(self):
        return len(self.bags)

    def __getitem__(self, index):

        item = self.bags[index]

        bag = normalize_windows(
            item["eeg"]
        )

        x = torch.from_numpy(
            bag
        ).float()

        y = torch.tensor(
            item["label"],
            dtype=torch.long,
        )

        return (
            x,
            y,
            item["subject_id"],
            item["bag_number"],
            item["n_original_windows"],
        )


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

        "balanced_accuracy":
            balanced_accuracy_score(
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
# AGGREGATE BAG PREDICTIONS -> SUBJECT
# ============================================================

def aggregate_bag_predictions(records):

    df = pd.DataFrame(records)

    if df.empty:
        raise RuntimeError(
            "No prediction records generated."
        )

    subject_rows = []

    for subject_id, group in df.groupby(
        "subject_id"
    ):

        labels = group[
            "true_label"
        ].unique()

        if len(labels) != 1:
            raise RuntimeError(
                f"{subject_id} has multiple labels."
            )

        true_label = int(
            labels[0]
        )

        # Weighted by number of ORIGINAL windows
        # represented by each bag.
        weights = group[
            "n_original_windows"
        ].to_numpy(
            dtype=np.float64
        )

        probs = group[
            [
                "prob_0",
                "prob_1",
                "prob_2",
            ]
        ].to_numpy(
            dtype=np.float64
        )

        mean_probs = np.average(
            probs,
            axis=0,
            weights=weights,
        )

        predicted_label = int(
            np.argmax(mean_probs)
        )

        subject_rows.append(
            {
                "subject_id":
                    subject_id,

                "true_label":
                    true_label,

                "predicted_label":
                    predicted_label,

                "prob_0":
                    float(mean_probs[0]),

                "prob_1":
                    float(mean_probs[1]),

                "prob_2":
                    float(mean_probs[2]),

                "n_bags":
                    len(group),

                "n_windows":
                    int(weights.sum()),
            }
        )

    return pd.DataFrame(
        subject_rows
    )


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
    total_subjects = 0

    for (
        x,
        y,
        _,
    ) in loader:

        # x:
        # [B subjects, K windows, 19, 2000]

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

        total_subjects += batch_size

    return (
        total_loss
        / max(total_subjects, 1),

        total_correct
        / max(total_subjects, 1),
    )


# ============================================================
# EVALUATE
# ============================================================

def evaluate(
    model,
    loader,
    device,
):

    model.eval()

    records = []

    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_bags = 0

    with torch.no_grad():

        for (
            x,
            y,
            subject_ids,
            bag_numbers,
            n_original_windows,
        ) in loader:

            x = x.to(
                device,
                non_blocking=True,
            )

            y = y.to(
                device,
                non_blocking=True,
            )

            logits, attention = model(
                x,
                return_attention=True,
            )

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

            total_bags += batch_size

            probabilities = (
                probabilities
                .cpu()
                .numpy()
            )

            attention = (
                attention
                .cpu()
                .numpy()
            )

            labels = (
                y
                .cpu()
                .numpy()
            )

            if torch.is_tensor(
                bag_numbers
            ):
                bag_numbers = (
                    bag_numbers
                    .cpu()
                    .numpy()
                )

            if torch.is_tensor(
                n_original_windows
            ):
                n_original_windows = (
                    n_original_windows
                    .cpu()
                    .numpy()
                )

            for i in range(batch_size):

                records.append(
                    {
                        "subject_id":
                            subject_ids[i],

                        "bag_number":
                            int(
                                bag_numbers[i]
                            ),

                        "true_label":
                            int(
                                labels[i]
                            ),

                        "prob_0":
                            float(
                                probabilities[i, 0]
                            ),

                        "prob_1":
                            float(
                                probabilities[i, 1]
                            ),

                        "prob_2":
                            float(
                                probabilities[i, 2]
                            ),

                        "n_original_windows":
                            int(
                                n_original_windows[i]
                            ),

                        "attention_max":
                            float(
                                attention[i].max()
                            ),

                        "attention_min":
                            float(
                                attention[i].min()
                            ),

                        "attention_entropy":
                            float(
                                -np.sum(
                                    attention[i]
                                    * np.log(
                                        attention[i]
                                        + 1e-12
                                    )
                                )
                            ),
                    }
                )

    average_loss = (
        total_loss
        / max(total_bags, 1)
    )

    subject_predictions = (
        aggregate_bag_predictions(
            records
        )
    )

    metrics, cm = calculate_metrics(
        subject_predictions
    )

    return (
        average_loss,
        records,
        subject_predictions,
        metrics,
        cm,
    )


# ============================================================
# CLASS DISTRIBUTION
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
# SAVE SPLIT
# ============================================================

def save_split(
    fold_dir,
    train_df,
    val_df,
    test_df,
):

    rows = []

    for split_name, df in [
        ("train", train_df),
        ("validation", val_df),
        ("test", test_df),
    ]:

        for _, row in df.iterrows():

            rows.append(
                {
                    "subject_id":
                        row["subject_id"],

                    "class_name":
                        row["class_name"],

                    "label":
                        int(row["label"]),

                    "split":
                        split_name,
                }
            )

    pd.DataFrame(
        rows
    ).to_csv(
        fold_dir
        / "subject_split.csv",
        index=False,
    )


# ============================================================
# RUN FOLD
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
    print(f"SWATS FOLD {fold_number}")
    print("=" * 70)

    fold_dir = (
        RESULTS_DIR
        / f"fold_{fold_number}"
    )

    fold_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # OUTER SPLIT
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # INNER TRAIN/VAL SPLIT
    # --------------------------------------------------------

    inner_indices = np.arange(
        len(development_df)
    )

    train_indices, val_indices = (
        train_test_split(
            inner_indices,

            test_size=
                VALIDATION_FRACTION,

            stratify=
                development_df[
                    "label"
                ],

            random_state=
                SEED
                + fold_number,
        )
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

    # --------------------------------------------------------
    # LEAKAGE CHECK
    # --------------------------------------------------------

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

    assert len(
        train_subjects
        | val_subjects
        | test_subjects
    ) == len(manifest)

    print(
        f"\nTrain subjects: "
        f"{len(train_df)}"
    )

    print(
        f"Validation subjects: "
        f"{len(val_df)}"
    )

    print(
        f"Test subjects: "
        f"{len(test_df)}"
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

    # --------------------------------------------------------
    # DATASETS
    # --------------------------------------------------------

    train_dataset = (
        TrainingSubjectBagDataset(
            train_df,
            BAG_SIZE,
            seed=(
                SEED
                + fold_number
            ),
        )
    )

    val_dataset = (
        EvaluationSubjectBagDataset(
            val_df,
            BAG_SIZE,
        )
    )

    test_dataset = (
        EvaluationSubjectBagDataset(
            test_df,
            BAG_SIZE,
        )
    )

    # --------------------------------------------------------
    # NORMALIZATION SANITY CHECK
    # --------------------------------------------------------

    example_x, _, example_subject = (
        train_dataset[0]
    )

    channel_means = (
        example_x
        .mean(dim=2)
        .numpy()
    )

    channel_stds = (
        example_x
        .std(
            dim=2,
            unbiased=False,
        )
        .numpy()
    )

    print("\nNormalization check:")
    print(
        f"  Example subject: "
        f"{example_subject}"
    )

    print(
        "  Mean absolute channel mean: "
        f"{np.abs(channel_means).mean():.6f}"
    )

    print(
        "  Mean channel std: "
        f"{channel_stds.mean():.6f}"
    )

    if (
        np.abs(channel_means).mean()
        > 1e-3
    ):
        raise RuntimeError(
            "Normalization mean check failed."
        )

    if not np.isclose(
        channel_stds.mean(),
        1.0,
        atol=1e-3,
    ):
        raise RuntimeError(
            "Normalization std check failed."
        )

    # --------------------------------------------------------
    # LOADERS
    # --------------------------------------------------------

    pin_memory = (
        device.type == "cuda"
    )

    train_generator = torch.Generator()
    train_generator.manual_seed(
        SEED + fold_number
    )

    train_loader = DataLoader(
        train_dataset,

        batch_size=
            SUBJECT_BATCH_SIZE,

        shuffle=True,

        generator=
            train_generator,

        num_workers=
            NUM_WORKERS,

        pin_memory=
            pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,

        batch_size=
            SUBJECT_BATCH_SIZE,

        shuffle=False,

        num_workers=
            NUM_WORKERS,

        pin_memory=
            pin_memory,
    )

    test_loader = DataLoader(
        test_dataset,

        batch_size=
            SUBJECT_BATCH_SIZE,

        shuffle=False,

        num_workers=
            NUM_WORKERS,

        pin_memory=
            pin_memory,
    )

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    model = SWATSNet(
        n_channels=19,
        n_classes=N_CLASSES,
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
        f"\nModel parameters: "
        f"{total_parameters:,}"
    )

    print(
        f"Trainable parameters: "
        f"{trainable_parameters:,}"
    )

    # --------------------------------------------------------
    # OPTIMIZATION
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # First SWATS experiment:
    # keep ordinary CE to avoid introducing another
    # experimental change simultaneously.
    criterion = nn.CrossEntropyLoss()

    best_val_f1 = -1.0
    epochs_without_improvement = 0

    best_model_path = (
        fold_dir
        / "best_model.pt"
    )

    history = []

    print("\nTraining...")
    print("-" * 70)

    # --------------------------------------------------------
    # TRAINING LOOP
    # --------------------------------------------------------

    for epoch in range(
        1,
        MAX_EPOCHS + 1,
    ):

        train_dataset.set_epoch(
            epoch
        )

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
            _,
            val_subject_predictions,
            val_metrics,
            _,
        ) = evaluate(
            model,
            val_loader,
            device,
        )

        val_f1 = val_metrics[
            "macro_f1"
        ]

        print(
            f"Epoch {epoch:03d} | "
            f"train loss {train_loss:.4f} | "
            f"train subject acc "
            f"{train_accuracy:.4f} | "
            f"val loss {val_loss:.4f} | "
            f"val subject F1 {val_f1:.4f}"
        )

        history.append(
            {
                "epoch":
                    epoch,

                "train_loss":
                    train_loss,

                "train_subject_accuracy":
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

        # ----------------------------------------------------
        # CHECKPOINT
        # ----------------------------------------------------

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

                    "model":
                        "SWATS-Net",

                    "bag_size":
                        BAG_SIZE,

                    "normalization":
                        "per_window_per_channel_zscore",

                    "config": {
                        "seed":
                            SEED,

                        "subject_batch_size":
                            SUBJECT_BATCH_SIZE,

                        "bag_size":
                            BAG_SIZE,

                        "learning_rate":
                            LEARNING_RATE,

                        "weight_decay":
                            WEIGHT_DECAY,

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

    # --------------------------------------------------------
    # HISTORY
    # --------------------------------------------------------

    pd.DataFrame(
        history
    ).to_csv(
        fold_dir
        / "training_history.csv",
        index=False,
    )

    # --------------------------------------------------------
    # LOAD BEST MODEL
    # --------------------------------------------------------

    if not best_model_path.exists():
        raise RuntimeError(
            "No checkpoint produced."
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
        "Best validation macro-F1: "
        f"{checkpoint['best_val_macro_f1']:.4f}"
    )

    # --------------------------------------------------------
    # FINAL TEST
    # --------------------------------------------------------

    (
        test_loss,
        test_bag_records,
        subject_predictions,
        test_metrics,
        cm,
    ) = evaluate(
        model,
        test_loader,
        device,
    )

    # Save bag-level information.
    pd.DataFrame(
        test_bag_records
    ).to_csv(
        fold_dir
        / "test_bag_predictions.csv",
        index=False,
    )

    # Save subject predictions.
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
        "bag_size"
    ] = BAG_SIZE

    test_metrics[
        "model"
    ] = "SWATS-Net"

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

    # --------------------------------------------------------
    # REPORT
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(
        f"SWATS FOLD {fold_number} "
        f"TEST RESULTS"
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

    # --------------------------------------------------------
    # CLEANUP
    # --------------------------------------------------------

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
            "SWATS-Net subject-level "
            "window-attention EEG classification."
        )
    )

    group = (
        parser
        .add_mutually_exclusive_group(
            required=True
        )
    )

    group.add_argument(
        "--fold",
        type=int,
        choices=[1, 2, 3, 4, 5],
    )

    group.add_argument(
        "--all-folds",
        action="store_true",
    )

    args = parser.parse_args()

    set_seed(SEED)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 70)
    print(
        "SWATS-NET SUBJECT-LEVEL "
        "WINDOW-ATTENTION TRAINING"
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

    # --------------------------------------------------------
    # MANIFEST
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
        "n_windows",
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

    if manifest[
        "subject_id"
    ].duplicated().any():

        raise RuntimeError(
            "Duplicate subject IDs detected."
        )

    observed_labels = set(
        manifest["label"].unique()
    )

    if observed_labels != {0, 1, 2}:
        raise RuntimeError(
            f"Unexpected labels: "
            f"{observed_labels}"
        )

    print(
        f"\nSubjects: "
        f"{len(manifest)}"
    )

    print(
        "Total available windows: "
        f"{manifest['n_windows'].sum()}"
    )

    print(
        f"SWATS bag size: "
        f"{BAG_SIZE}"
    )

    # --------------------------------------------------------
    # OUTER CV
    # --------------------------------------------------------

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

        metrics["fold"] = (
            fold_number
        )

        all_metrics.append(
            metrics
        )

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

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    if args.all_folds:

        print()
        print("=" * 70)
        print("SWATS 5-FOLD SUMMARY")
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