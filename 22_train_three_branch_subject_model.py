from pathlib import Path
import importlib.util
import json
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, matthews_corrcoef, confusion_matrix

PROJECT_ROOT = Path(r"E:\neurops")
WINDOW_MANIFEST = PROJECT_ROOT / "windows" / "window_manifest.csv"
SPECTRAL_FILE = PROJECT_ROOT / "results_spectral_baseline" / "subject_spectral_features.csv"
CONNECTIVITY_FILE = PROJECT_ROOT / "results_connectivity_baseline" / "subject_connectivity_features.csv"
MODEL_FILE = PROJECT_ROOT / "src" / "21_three_branch_subject_model.py"
RESULTS_DIR = PROJECT_ROOT / "results_three_branch_subject_model"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
N_FOLDS = 5
VAL_FRACTION = 0.20
TRAIN_WINDOWS_PER_SUBJECT = 64
MAX_EPOCHS = 80
PATIENCE = 15
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 5e-4
NORMALIZATION_EPS = 1e-6
TEMPORAL_EVAL_CHUNK = 64

spec = importlib.util.spec_from_file_location("three_branch_subject_model", MODEL_FILE)
model_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model_module)
ThreeBranchSubjectModel = model_module.ThreeBranchSubjectModel


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_feature_columns(df):
    meta = {"subject_id", "label", "class_name"}
    return [c for c in df.columns if c not in meta]


def load_feature_tables():
    sdf = pd.read_csv(SPECTRAL_FILE)
    cdf = pd.read_csv(CONNECTIVITY_FILE)
    sdf["subject_id"] = sdf["subject_id"].astype(str)
    cdf["subject_id"] = cdf["subject_id"].astype(str)
    scols = get_feature_columns(sdf)
    ccols = get_feature_columns(cdf)
    if len(scols) != 95:
        raise RuntimeError(f"Expected 95 spectral features, found {len(scols)}")
    if len(ccols) != 855:
        raise RuntimeError(f"Expected 855 connectivity features, found {len(ccols)}")
    return sdf.set_index("subject_id"), cdf.set_index("subject_id"), scols, ccols


def load_subject_windows(path):
    with np.load(path, allow_pickle=False) as data:
        eeg = data["eeg"].astype(np.float32, copy=False)
    if eeg.ndim != 3 or eeg.shape[1:] != (19, 2000):
        raise RuntimeError(f"Unexpected EEG shape: {eeg.shape}")
    return eeg


def normalize_windows(eeg):
    eeg = eeg * 1e6
    mean = eeg.mean(axis=2, keepdims=True)
    std = eeg.std(axis=2, keepdims=True)
    return ((eeg - mean) / (std + NORMALIZATION_EPS)).astype(np.float32, copy=False)


def sample_training_windows(eeg, n_windows, seed):
    rng = np.random.default_rng(seed)
    total = eeg.shape[0]
    idx = rng.choice(total, size=n_windows, replace=(total < n_windows))
    return normalize_windows(eeg[idx].copy())


def calculate_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
    }


def fit_scalers(train_df, sdf, cdf, scols, ccols):
    ids = train_df["subject_id"].astype(str).tolist()
    ss = StandardScaler().fit(sdf.loc[ids, scols].to_numpy(dtype=np.float32))
    cs = StandardScaler().fit(cdf.loc[ids, ccols].to_numpy(dtype=np.float32))
    return ss, cs


def get_scaled_features(subject_id, sdf, cdf, scols, ccols, ss, cs):
    s = sdf.loc[subject_id, scols].to_numpy(dtype=np.float32).reshape(1, -1)
    c = cdf.loc[subject_id, ccols].to_numpy(dtype=np.float32).reshape(1, -1)
    return ss.transform(s).astype(np.float32)[0], cs.transform(c).astype(np.float32)[0]


def make_class_weights(train_df, device):
    counts = train_df["label"].value_counts().sort_index().to_numpy()
    weights = len(train_df) / (3.0 * counts)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def train_one_epoch(model, train_df, sdf, cdf, scols, ccols, ss, cs, optimizer, criterion, device, epoch):
    model.train()
    order = np.arange(len(train_df))
    rng = np.random.default_rng(SEED + epoch)
    rng.shuffle(order)
    total_loss = 0.0
    correct = 0

    for idx in order:
        row = train_df.iloc[idx]
        sid = str(row["subject_id"])
        label = int(row["label"])
        eeg = load_subject_windows(Path(row["window_file"]))
        bag = sample_training_windows(eeg, TRAIN_WINDOWS_PER_SUBJECT, SEED + epoch * 10000 + idx)
        s, c = get_scaled_features(sid, sdf, cdf, scols, ccols, ss, cs)

        eeg_t = torch.from_numpy(bag).unsqueeze(0).to(device)
        s_t = torch.from_numpy(s).unsqueeze(0).to(device)
        c_t = torch.from_numpy(c).unsqueeze(0).to(device)
        y_t = torch.tensor([label], dtype=torch.long, device=device)

        optimizer.zero_grad(set_to_none=True)
        logits, _, _, _ = model(eeg_t, s_t, c_t)
        loss = criterion(logits, y_t)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        total_loss += loss.item()
        correct += int(logits.argmax(dim=1).item() == label)

    return total_loss / len(train_df), correct / len(train_df)


@torch.no_grad()
def get_full_temporal_embedding(model, eeg, device):
    model.eval()
    eeg = normalize_windows(eeg)
    chunks = []
    for start in range(0, eeg.shape[0], TEMPORAL_EVAL_CHUNK):
        x = torch.from_numpy(eeg[start:start + TEMPORAL_EVAL_CHUNK]).to(device)
        chunks.append(model.temporal_encoder(x))
    return torch.cat(chunks, dim=0).mean(dim=0, keepdim=True)


@torch.no_grad()
def evaluate_subjects(model, subject_df, sdf, cdf, scols, ccols, ss, cs, device):
    model.eval()
    rows = []
    for _, row in subject_df.reset_index(drop=True).iterrows():
        sid = str(row["subject_id"])
        true_label = int(row["label"])
        eeg = load_subject_windows(Path(row["window_file"]))
        temporal = get_full_temporal_embedding(model, eeg, device)
        s, c = get_scaled_features(sid, sdf, cdf, scols, ccols, ss, cs)
        s_t = torch.from_numpy(s).unsqueeze(0).to(device)
        c_t = torch.from_numpy(c).unsqueeze(0).to(device)
        logits, _, _ = model.fuse_from_embeddings(temporal, s_t, c_t)
        p = torch.softmax(logits, dim=1)[0].cpu().numpy()
        pred = int(np.argmax(p))
        rows.append({
            "subject_id": sid,
            "true_label": true_label,
            "predicted_label": pred,
            "prob_alzheimer": float(p[0]),
            "prob_control": float(p[1]),
            "prob_ftd": float(p[2]),
        })

    pdf = pd.DataFrame(rows)
    yt = pdf["true_label"].to_numpy()
    yp = pdf["predicted_label"].to_numpy()
    metrics = calculate_metrics(yt, yp)
    cm = confusion_matrix(yt, yp, labels=[0, 1, 2])
    return metrics, cm, pdf


def run_fold(fold_number, development_df, test_df, sdf, cdf, scols, ccols, device):
    print("\n" + "=" * 72)
    print(f"THREE-BRANCH SUBJECT MODEL - FOLD {fold_number}")
    print("=" * 72)

    train_df, val_df = train_test_split(
        development_df,
        test_size=VAL_FRACTION,
        stratify=development_df["label"],
        random_state=SEED + fold_number,
    )
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    print(f"Train subjects: {len(train_df)}")
    print(f"Val subjects:   {len(val_df)}")
    print(f"Test subjects:  {len(test_df)}")

    ss, cs = fit_scalers(train_df, sdf, cdf, scols, ccols)
    model = ThreeBranchSubjectModel(n_classes=3).to(device)
    print("Trainable parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))

    class_weights = make_class_weights(train_df, device)
    print("Class weights:", class_weights.detach().cpu().numpy())

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    fold_dir = RESULTS_DIR / f"fold_{fold_number}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    ckpt = fold_dir / "best_model.pt"

    best_f1 = -1.0
    best_epoch = 0
    no_improve = 0
    history = []

    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_df, sdf, cdf, scols, ccols, ss, cs,
            optimizer, criterion, device, epoch
        )
        val_metrics, _, _ = evaluate_subjects(
            model, val_df, sdf, cdf, scols, ccols, ss, cs, device
        )
        val_f1 = val_metrics["macro_f1"]
        print(
            f"Epoch {epoch:03d} | train loss {train_loss:.4f} | "
            f"train acc {train_acc:.4f} | val subject F1 {val_f1:.4f}"
        )
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_accuracy": val_metrics["accuracy"],
            "val_balanced_accuracy": val_metrics["balanced_accuracy"],
            "val_macro_f1": val_f1,
            "val_mcc": val_metrics["mcc"],
        })

        if val_f1 > best_f1 + 1e-6:
            best_f1 = val_f1
            best_epoch = epoch
            no_improve = 0
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch, "val_macro_f1": val_f1}, ckpt)
        else:
            no_improve += 1

        if no_improve >= PATIENCE:
            print("\nEarly stopping.")
            break

    pd.DataFrame(history).to_csv(fold_dir / "training_history.csv", index=False)
    checkpoint = torch.load(ckpt, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    print(f"\nBest epoch: {best_epoch}")
    print(f"Best validation macro-F1: {best_f1:.4f}")

    test_metrics, test_cm, pred_df = evaluate_subjects(
        model, test_df, sdf, cdf, scols, ccols, ss, cs, device
    )

    print("\n" + "=" * 72)
    print(f"FOLD {fold_number} TEST RESULTS")
    print("=" * 72)
    print(f"Accuracy:          {test_metrics['accuracy']:.4f}")
    print(f"Balanced accuracy: {test_metrics['balanced_accuracy']:.4f}")
    print(f"Macro F1:          {test_metrics['macro_f1']:.4f}")
    print(f"MCC:               {test_metrics['mcc']:.4f}")
    print("\nConfusion matrix:")
    print(test_cm)

    pred_df["fold"] = fold_number
    pred_df.to_csv(fold_dir / "subject_predictions.csv", index=False)
    np.savetxt(fold_dir / "confusion_matrix.csv", test_cm, delimiter=",", fmt="%d")
    with (fold_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=4)

    result = dict(test_metrics)
    result["fold"] = fold_number
    return result, pred_df


def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 72)
    print("THREE-BRANCH SUBJECT-LEVEL EEG MODEL")
    print("=" * 72)
    print("Device:", device)

    manifest = pd.read_csv(WINDOW_MANIFEST)
    manifest["subject_id"] = manifest["subject_id"].astype(str)
    sdf, cdf, scols, ccols = load_feature_tables()

    subjects = set(manifest["subject_id"])
    if subjects != set(sdf.index):
        raise RuntimeError("Spectral subject IDs do not match the window manifest.")
    if subjects != set(cdf.index):
        raise RuntimeError("Connectivity subject IDs do not match the window manifest.")

    print("Subjects:", len(manifest))
    print("Classes:", manifest["class_name"].value_counts().to_dict())
    print("Spectral features:", len(scols))
    print("Connectivity features:", len(ccols))

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    indices = np.arange(len(manifest))
    labels = manifest["label"].to_numpy()
    all_metrics, all_predictions = [], []

    for fold_number, (dev_idx, test_idx) in enumerate(skf.split(indices, labels), start=1):
        development_df = manifest.iloc[dev_idx].reset_index(drop=True)
        test_df = manifest.iloc[test_idx].reset_index(drop=True)
        metrics, predictions = run_fold(
            fold_number, development_df, test_df, sdf, cdf, scols, ccols, device
        )
        all_metrics.append(metrics)
        all_predictions.append(predictions)

    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(RESULTS_DIR / "five_fold_metrics.csv", index=False)
    oof = pd.concat(all_predictions, ignore_index=True)
    oof.to_csv(RESULTS_DIR / "oof_subject_predictions.csv", index=False)

    print("\n" + "=" * 72)
    print("5-FOLD SUMMARY")
    print("=" * 72)
    for metric in ["accuracy", "balanced_accuracy", "macro_f1", "mcc"]:
        v = metrics_df[metric].to_numpy(dtype=np.float64)
        print(f"{metric:20s}: {v.mean():.4f} ± {v.std(ddof=1):.4f}")

    pooled = calculate_metrics(oof["true_label"].to_numpy(), oof["predicted_label"].to_numpy())
    cm = confusion_matrix(oof["true_label"], oof["predicted_label"], labels=[0, 1, 2])
    ftd_recall = float(cm[2, 2] / cm[2].sum())

    print("\n" + "=" * 72)
    print("POOLED OUT-OF-FOLD RESULTS")
    print("=" * 72)
    print(f"Accuracy:          {pooled['accuracy']:.4f}")
    print(f"Balanced accuracy: {pooled['balanced_accuracy']:.4f}")
    print(f"Macro F1:          {pooled['macro_f1']:.4f}")
    print(f"MCC:               {pooled['mcc']:.4f}")
    print(f"FTD recall:        {ftd_recall:.4f}")
    print("\nPooled confusion matrix:")
    print(cm)

    payload = dict(pooled)
    payload["ftd_recall"] = ftd_recall
    with (RESULTS_DIR / "pooled_oof_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4)
    np.savetxt(RESULTS_DIR / "pooled_oof_confusion_matrix.csv", cm, delimiter=",", fmt="%d")
    print("\nResults saved under:")
    print(RESULTS_DIR)


if __name__ == "__main__":
    main()
