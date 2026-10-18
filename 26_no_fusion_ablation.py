# No-fusion single-branch ablation; chronological temporal windows.
"""RA-TSFCNet experimental variants. Uses existing .set-derived feature cache.
Mamba mode uses a portable PyTorch *Mamba-inspired* selective SSM, NOT the
official mamba_ssm package or its optimized CUDA selective scan.
"""
"""
24_ra_tsfcnet_ad_vs_ftd.py

RA-TSFCNet — Reliability-Aware Temporal-Spectral Functional Connectivity Network
Full proposed-model implementation for the manually preprocessed E:\\neurops AD vs FTD dataset.

Validated dataset assumptions:
- manually preprocessed EEG is stored as EEGLAB .set files
- expected 19 EEG channels at 500 Hz
- continuous recordings are divided into non-overlapping 4-second windows
- classes used: ec_alzheimer and ec_ftd only
- ec_control is intentionally ignored
- subject-level cross-validation: epochs NEVER cross subjects/folds

Architecture
------------
Temporal:
    4-s EEG windows -> compact 1D CNN -> window attention -> subject embedding

Spectral:
    ALL 4-s windows -> Welch PSD -> relative delta/theta/alpha/beta/gamma
    per channel -> median across epochs -> 19*5 subject feature vector -> MLP

Functional connectivity:
    ALL 4-s windows -> frequency-specific correlation connectivity
    delta and alpha bands -> compact upper-triangle summary -> MLP

Reliability-aware fusion:
    each branch produces an auxiliary class prediction
    reliability = 1 - normalized predictive entropy
    branch embeddings are projected to a common space and fused using
    normalized reliability weights -> final MLP classifier

NOTE:
This is a practical implementation of the proposed architecture for today's
experiment. The exact reliability formula and FC feature-selection strategy
are implementation choices and can be refined for the thesis experiment.
"""

from pathlib import Path
import argparse, copy, json, random, warnings
import numpy as np
import pandas as pd
import mne
from scipy.signal import welch
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    matthews_corrcoef, confusion_matrix, classification_report
)

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

SEED = 42
LABELS = {"ec_alzheimer": 0, "ec_ftd": 1}
NAMES = {0: "AD", 1: "FTD"}
BANDS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}
FC_BANDS = {"delta": (1.0, 4.0), "alpha": (8.0, 13.0)}

def seed_all(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def subject_id(path):
    for part in path.parts:
        if part.lower().startswith("sub-"):
            return part
    return path.stem

def build_manifest(root):
    """Scan ONLY ec_alzheimer and ec_ftd. ec_control is never read."""
    rows = []
    for folder, label in LABELS.items():
        d = root / folder
        if not d.exists():
            raise RuntimeError(f"Missing class folder: {d}")
        for subject_dir in sorted(d.glob("sub-*")):
            eeg_dir = subject_dir / "eeg"
            files = sorted(eeg_dir.glob("*.set"))
            if not files:
                warnings.warn(f"No .set file found in {eeg_dir}")
                continue
            if len(files) > 1:
                raise RuntimeError(
                    f"{eeg_dir}: multiple .set files found: {[f.name for f in files]}. "
                    "Keep/identify the intended eyes-closed recording before training."
                )
            rows.append({
                "subject": subject_dir.name,
                "label": label,
                "diagnosis": NAMES[label],
                "file": str(files[0]),
            })
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No AD/FTD .set files found.")
    # IDs can theoretically overlap across class folders; cache key is class+subject.
    df["cache_id"] = df["diagnosis"] + "_" + df["subject"]
    if df["cache_id"].duplicated().any():
        raise RuntimeError("Duplicate diagnosis+subject entries found.")
    return df

def relative_bandpower(epoch_data, sfreq):
    """epoch_data: E,C,T in normalized arbitrary amplitude."""
    e, c, t = epoch_data.shape
    nperseg = min(t, int(2.0 * sfreq))
    freqs, psd = welch(
        epoch_data, fs=sfreq, axis=-1,
        nperseg=nperseg, noverlap=nperseg // 2,
        detrend="constant", scaling="density"
    )
    total_mask = (freqs >= 1.0) & (freqs < 45.0)
    total = np.trapezoid(psd[..., total_mask], freqs[total_mask], axis=-1)
    total = np.maximum(total, 1e-20)

    feats = []
    for lo, hi in BANDS.values():
        m = (freqs >= lo) & (freqs < hi)
        bp = np.trapezoid(psd[..., m], freqs[m], axis=-1)
        feats.append(bp / total)                  # E,C
    features = np.stack(feats, axis=-1)          # E,C,5
    return np.median(features, axis=0).reshape(-1).astype(np.float32)

def band_limited_epochs(x, sfreq, lo, hi):
    """Band-pass a COPY of E,C,T data using MNE filtering."""
    x64 = np.asarray(x, dtype=np.float64)
    y = mne.filter.filter_data(
        x64, sfreq=sfreq, l_freq=lo, h_freq=hi,
        method="fir", verbose="ERROR"
    )
    return y.astype(np.float32)

def fc_features(epoch_data, sfreq):
    """
    Frequency-specific Pearson FC in delta and alpha.
    For each epoch: channel correlation matrix.
    Then median connectivity across all epochs.
    Returns upper triangle for both bands.
    19 channels => 171 pairs x 2 bands = 342 initial FC features.
    """
    c = epoch_data.shape[1]
    iu = np.triu_indices(c, k=1)
    all_band = []

    for lo, hi in FC_BANDS.values():
        filtered = band_limited_epochs(epoch_data, sfreq, lo, hi)
        epoch_fc = []
        for ep in filtered:
            corr = np.corrcoef(ep)
            corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
            epoch_fc.append(corr[iu])
        all_band.append(np.median(np.stack(epoch_fc), axis=0))

    return np.concatenate(all_band).astype(np.float32)

def load_subject_features(path, expected_channels=19, window_sec=4.0):
    """Load an already-cleaned continuous EEGLAB recording and create 4-s windows."""
    raw = mne.io.read_raw_eeglab(path, preload=True, verbose="ERROR")
    picks = mne.pick_types(raw.info, eeg=True, meg=False, eog=False, ecg=False, stim=False, exclude=[])
    sf = float(raw.info["sfreq"])
    continuous = raw.get_data(picks=picks).astype(np.float32)  # C,T volts

    if continuous.shape[0] != expected_channels:
        raise RuntimeError(f"{path}: expected {expected_channels} EEG channels, got {continuous.shape[0]}")

    n_samp = int(round(window_sec * sf))
    n_win = continuous.shape[1] // n_samp
    if n_win < 1:
        raise RuntimeError(f"{path}: shorter than one {window_sec}-second window")
    continuous = continuous[:, :n_win*n_samp]
    x = continuous.reshape(continuous.shape[0], n_win, n_samp).transpose(1,0,2).copy()

    # volts -> microvolts
    x *= 1e6

    # Subject/channel normalization used by temporal and FC branches.
    mu = x.mean(axis=(0, 2), keepdims=True)
    sd = x.std(axis=(0, 2), keepdims=True)
    xn = (x - mu) / np.maximum(sd, 1e-6)

    # Stable whole-recording spectral representation.
    spectral = relative_bandpower(x, sf)
    # Frequency-specific whole-recording connectivity representation.
    fc = fc_features(xn, sf)

    duration_min = continuous.shape[1] / sf / 60.0
    return xn.astype(np.float32), spectral, fc, sf, duration_min

def prepare_cache(df, cache_dir):
    """
    Precompute spectral and FC features once.
    Temporal EEG is also cached as compressed .npz to avoid repeatedly loading FIF.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    print("\nPreparing/caching subject features...")
    for i, r in df.iterrows():
        target = cache_dir / f"{r.cache_id}.npz"
        if target.exists():
            print(f"[{i+1:02d}/{len(df):02d}] cached {r.subject}")
            continue
        print(f"[{i+1:02d}/{len(df):02d}] computing {r.diagnosis} {r.subject}")
        x, spec, fc, sf, duration_min = load_subject_features(Path(r.file))
        np.savez_compressed(
            target, eeg=x, spectral=spec, fc=fc,
            sfreq=np.float32(sf), duration_min=np.float32(duration_min), label=np.int64(r.label)
        )

def standardization_stats(df, cache_dir, key):
    """
    IMPORTANT: statistics are calculated from TRAINING SUBJECTS ONLY.
    Avoids feature-normalization leakage.
    """
    arrays = []
    for _, r in df.iterrows():
        z = np.load(cache_dir / f"{r.cache_id}.npz")
        arrays.append(z[key].astype(np.float32))
    a = np.stack(arrays)
    mu = a.mean(axis=0)
    sd = a.std(axis=0)
    sd = np.maximum(sd, 1e-6)
    return mu.astype(np.float32), sd.astype(np.float32)

class SubjectDataset(Dataset):
    def __init__(self, df, cache_dir, bag_size, train, spec_stats, fc_stats):
        self.df = df.reset_index(drop=True)
        self.cache_dir = cache_dir
        self.bag_size = bag_size
        self.train = train
        self.spec_mu, self.spec_sd = spec_stats
        self.fc_mu, self.fc_sd = fc_stats

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        r = self.df.iloc[idx]
        z = np.load(self.cache_dir / f"{r.cache_id}.npz")
        x = z["eeg"].astype(np.float32)
        n = len(x)

        if n >= self.bag_size:
            if self.train:
                ids = np.sort(np.random.choice(n, self.bag_size, replace=False))
            else:
                ids = np.linspace(0, n - 1, self.bag_size).round().astype(int)
        else:
            ids = np.resize(np.arange(n), self.bag_size)

        temporal = x[ids]
        spectral = (z["spectral"].astype(np.float32) - self.spec_mu) / self.spec_sd
        fc = (z["fc"].astype(np.float32) - self.fc_mu) / self.fc_sd

        return (
            torch.from_numpy(temporal),
            torch.from_numpy(spectral),
            torch.from_numpy(fc),
            torch.tensor(int(r.label), dtype=torch.long),
            r.subject,
        )

def collate(batch):
    return (
        torch.stack([b[0] for b in batch]),
        torch.stack([b[1] for b in batch]),
        torch.stack([b[2] for b in batch]),
        torch.stack([b[3] for b in batch]),
        [b[4] for b in batch],
    )

class SelectiveSSM(nn.Module):
    """Portable Mamba-inspired selective state-space block (NOT official Mamba)."""
    def __init__(self, dim=128, state_dim=16, conv_kernel=3):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.in_proj = nn.Linear(dim, dim * 2)
        self.conv = nn.Conv1d(dim, dim, conv_kernel, padding=conv_kernel-1, groups=dim)
        self.param_proj = nn.Linear(dim, 2 * state_dim + dim)
        self.A_log = nn.Parameter(torch.log(torch.arange(1, state_dim+1, dtype=torch.float32)).repeat(dim, 1))
        self.D = nn.Parameter(torch.ones(dim))
        self.out_proj = nn.Linear(dim, dim)
        self.state_dim = state_dim

    def forward(self, x):
        # x: [B, windows, features], preserve window order
        residual = x
        u, gate = self.in_proj(self.norm(x)).chunk(2, dim=-1)
        u = torch.nn.functional.silu(self.conv(u.transpose(1,2))[:, :, :x.shape[1]].transpose(1,2))
        params = self.param_proj(u)
        Bv, Cv, dt = torch.split(params, [self.state_dim, self.state_dim, u.shape[-1]], dim=-1)
        dt = torch.nn.functional.softplus(dt).clamp(max=2.0)
        A = -torch.exp(self.A_log).unsqueeze(0)
        h = torch.zeros(x.shape[0], x.shape[-1], self.state_dim, device=x.device, dtype=x.dtype)
        outputs = []
        for i in range(x.shape[1]):
            delta = dt[:,i,:].unsqueeze(-1)
            decay = torch.exp(delta * A)
            h = decay * h + delta * Bv[:,i,:].unsqueeze(1) * u[:,i,:].unsqueeze(-1)
            y = (h * Cv[:,i,:].unsqueeze(1)).sum(-1) + self.D * u[:,i,:]
            outputs.append(y)
        y = torch.stack(outputs, dim=1) * torch.nn.functional.silu(gate)
        return residual + self.out_proj(y)


class TemporalBranch(nn.Module):
    def __init__(self, emb=128, temporal_mode="attention", state_dim=16):
        super().__init__()
        self.temporal_mode = temporal_mode
        self.encoder = nn.Sequential(
            nn.Conv1d(19, 64, 15, stride=4, padding=7),
            nn.BatchNorm1d(64), nn.GELU(), nn.MaxPool1d(4),
            nn.Conv1d(64, 96, 9, stride=2, padding=4),
            nn.BatchNorm1d(96), nn.GELU(), nn.MaxPool1d(4),
            nn.Conv1d(96, emb, 7, stride=2, padding=3),
            nn.BatchNorm1d(emb), nn.GELU(), nn.AdaptiveAvgPool1d(1),
        )
        self.ssm = SelectiveSSM(emb, state_dim) if temporal_mode == "mamba" else None
        self.attn = nn.Sequential(nn.Linear(emb, 64), nn.Tanh(), nn.Linear(64, 1))

    def forward(self, x):
        b, e, c, t = x.shape
        z = self.encoder(x.reshape(b * e, c, t)).squeeze(-1).reshape(b, e, -1)
        if self.ssm is not None:
            z = self.ssm(z)
        a = torch.softmax(self.attn(z).squeeze(-1), dim=1)
        return torch.sum(z * a.unsqueeze(-1), dim=1), a

class MLPBranch(nn.Module):
    def __init__(self, in_dim, emb=128, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(0.30),
            nn.Linear(hidden, emb),
            nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)

class RATSFCNet(nn.Module):
    """Single-branch ablation: no fusion, no auxiliary classifiers."""
    def __init__(self, spec_dim=19*5, fc_dim=171*2, emb=128,
                 n_classes=2, temporal_mode="mamba", head_mode="original",
                 state_dim=16, branch="temporal"):
        super().__init__()
        self.branch = branch
        if branch == "temporal":
            self.encoder = TemporalBranch(emb, temporal_mode, state_dim)
        elif branch == "spectral":
            self.encoder = MLPBranch(spec_dim, emb, hidden=256)
        elif branch == "connectivity":
            self.encoder = MLPBranch(fc_dim, emb, hidden=384)
        else:
            raise ValueError(f"Unknown branch: {branch}")
        if head_mode == "wider":
            self.final = nn.Sequential(
                nn.Linear(emb, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(0.35),
                nn.Linear(128, 64), nn.GELU(), nn.Dropout(0.25),
                nn.Linear(64, n_classes))
        else:
            self.final = nn.Sequential(
                nn.Linear(emb, 64), nn.GELU(), nn.Dropout(0.35),
                nn.Linear(64, n_classes))

    def forward(self, temporal, spectral, fc):
        if self.branch == "temporal":
            embedding, attn = self.encoder(temporal)
        elif self.branch == "spectral":
            embedding, attn = self.encoder(spectral), None
        else:
            embedding, attn = self.encoder(fc), None
        logits = self.final(embedding)
        # One-hot branch indicators, NOT reliability weights.
        weights = logits.new_zeros((logits.shape[0], 3))
        weights[:, {"temporal": 0, "spectral": 1, "connectivity": 2}[self.branch]] = 1.0
        return logits, {}, weights, attn

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    ys, ps, subs, weights = [], [], [], []
    for xt, xs, xc, y, s in loader:
        logits, _, w, _ = model(
            xt.to(device), xs.to(device), xc.to(device)
        )
        pred = logits.argmax(1).cpu().numpy()
        ys.extend(y.numpy().tolist())
        ps.extend(pred.tolist())
        subs.extend(s)
        weights.extend(w.cpu().numpy().tolist())

    return {
        "accuracy": accuracy_score(ys, ps),
        "balanced_accuracy": balanced_accuracy_score(ys, ps),
        "macro_f1": f1_score(ys, ps, average="macro"),
        "mcc": matthews_corrcoef(ys, ps),
        "confusion_matrix": confusion_matrix(ys, ps, labels=[0,1]).tolist(),
        "report": classification_report(
            ys, ps, target_names=["AD", "FTD"], digits=4, zero_division=0
        ),
        "subjects": subs, "y_true": ys, "y_pred": ps, "weights": weights,
    }

def loaders(train, val, test, cache, args):
    spec_stats = standardization_stats(train, cache, "spectral")
    fc_stats = standardization_stats(train, cache, "fc")

    def ds(frame, is_train):
        return SubjectDataset(
            frame, cache, args.bag_size, is_train, spec_stats, fc_stats
        )

    tr = DataLoader(ds(train, True), batch_size=args.batch_size,
                    shuffle=True, num_workers=0, collate_fn=collate)
    va = DataLoader(ds(val, False), batch_size=args.batch_size,
                    shuffle=False, num_workers=0, collate_fn=collate)
    te = DataLoader(ds(test, False), batch_size=args.batch_size,
                    shuffle=False, num_workers=0, collate_fn=collate)
    return tr, va, te

def train_fold(train, val, test, cache, out, device, args, fold):
    tr, va, te = loaders(train, val, test, cache, args)
    model = RATSFCNet(temporal_mode=args.temporal_mode, head_mode=args.head_mode, state_dim=args.state_dim, branch=args.branch).to(device)

    counts = train.label.value_counts().to_dict()
    class_weights = torch.tensor(
        [len(train)/(2*counts.get(i,1)) for i in (0,1)],
        dtype=torch.float32, device=device
    )
    ce = nn.CrossEntropyLoss(weight=class_weights)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_f1 = -1.0
    best_state = None
    wait = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []

        for xt, xs, xc, y, _ in tr:
            xt, xs, xc, y = (
                xt.to(device), xs.to(device), xc.to(device), y.to(device)
            )
            opt.zero_grad(set_to_none=True)
            logits, aux, _, _ = model(xt, xs, xc)

            # Only the chosen branch and the final classifier are trained.
            loss = ce(logits, y)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(loss.item())

        vm = evaluate(model, va, device)
        print(
            f"Fold {fold} | Ep {epoch:03d} | "
            f"loss {np.mean(losses):.4f} | "
            f"val BA {vm['balanced_accuracy']:.4f} | "
            f"val F1 {vm['macro_f1']:.4f}"
        )

        if vm["macro_f1"] > best_f1 + 1e-5:
            best_f1 = vm["macro_f1"]
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= args.patience:
                print("Early stopping.")
                break

    model.load_state_dict(best_state)
    tm = evaluate(model, te, device)
    torch.save(model.state_dict(), out / f"fold_{fold}_best.pt")

    w = np.asarray(tm["weights"])
    pred_df = pd.DataFrame({
        "subject": tm["subjects"],
        "y_true": tm["y_true"],
        "y_pred": tm["y_pred"],
        "w_temporal": w[:,0],
        "w_spectral": w[:,1],
        "w_connectivity": w[:,2],
    })
    pred_df.to_csv(out / f"fold_{fold}_predictions.csv", index=False)

    print("\nTEST")
    print(tm["report"])
    print("Confusion matrix rows=true [AD,FTD], cols=pred:")
    print(np.asarray(tm["confusion_matrix"]))
    print(
        "Active branch indicators (not reliability):",
        dict(
            temporal=float(w[:,0].mean()),
            spectral=float(w[:,1].mean()),
            connectivity=float(w[:,2].mean()),
        )
    )

    return {
        "fold": fold,
        "accuracy": tm["accuracy"],
        "balanced_accuracy": tm["balanced_accuracy"],
        "macro_f1": tm["macro_f1"],
        "mcc": tm["mcc"],
        "branch_indicator_temporal": float(w[:,0].mean()),
        "branch_indicator_spectral": float(w[:,1].mean()),
        "branch_indicator_connectivity": float(w[:,2].mean()),
    }

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--project-root", type=Path,
                   default=Path(r"E:\neurops"))
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--bag-size", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--aux-weight", type=float, default=0.25)
    p.add_argument("--rebuild-cache", action="store_true")
    p.add_argument("--temporal-mode", choices=["attention", "mamba"], default="mamba")
    p.add_argument("--branch", choices=["temporal", "spectral", "connectivity"], required=True)
    p.add_argument("--head-mode", choices=["original", "wider"], default="original")
    p.add_argument("--state-dim", type=int, default=16)
    p.add_argument("--run-name", type=str, default="single_branch")
    args = p.parse_args()

    seed_all()
    root = args.project_root
    base = args.project_root / "results_ra_tsfcnet_ad_ftd_proposed"
    out = args.project_root / "results_no_fusion" / (args.run_name + "_" + args.branch)
    cache = base / "feature_cache"
    out.mkdir(parents=True, exist_ok=True)

    df = build_manifest(root)
    df.to_csv(out / "manifest.csv", index=False)

    print("=" * 78)
    print(f"RA-TSFCNet EXPERIMENT — branch={args.branch}, temporal={args.temporal_mode}, head={args.head_mode}")
    print("=" * 78)
    print(df.diagnosis.value_counts())
    print("ec_control: IGNORED")
    print(f"\nTotal subjects: {len(df)}")
    print(df.diagnosis.value_counts())

    if args.rebuild_cache and cache.exists():
        for f in cache.glob("*.npz"):
            f.unlink()

    prepare_cache(df, cache)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\nDevice:", device)
    if device.type == "cpu":
        print("CPU mode enabled. Feature caching/FC calculation may take time.")

    skf = StratifiedKFold(
        n_splits=args.folds, shuffle=True, random_state=SEED
    )
    fold_results = []

    for fold, (dev_idx, test_idx) in enumerate(
        skf.split(df, df.label), start=1
    ):
        dev = df.iloc[dev_idx].reset_index(drop=True)
        test = df.iloc[test_idx].reset_index(drop=True)

        tr_idx, va_idx = train_test_split(
            np.arange(len(dev)),
            test_size=0.20,
            stratify=dev.label,
            random_state=SEED + fold,
        )
        train = dev.iloc[tr_idx].reset_index(drop=True)
        val = dev.iloc[va_idx].reset_index(drop=True)

        print("\n" + "=" * 78)
        print(
            f"FOLD {fold} | train={len(train)} "
            f"val={len(val)} test={len(test)}"
        )
        print("Train:", train.diagnosis.value_counts().to_dict())
        print("Val:  ", val.diagnosis.value_counts().to_dict())
        print("Test: ", test.diagnosis.value_counts().to_dict())

        result = train_fold(
            train, val, test, cache, out, device, args, fold
        )
        fold_results.append(result)

    res = pd.DataFrame(fold_results)
    res.to_csv(out / "cv_metrics.csv", index=False)

    metrics = ["accuracy", "balanced_accuracy", "macro_f1", "mcc"]
    summary = {
        m: {
            "mean": float(res[m].mean()),
            "std": float(res[m].std(ddof=1)),
        }
        for m in metrics
    }
    (out / "cv_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print("\n" + "=" * 78)
    print("RA-TSFCNet 5-FOLD SUBJECT-LEVEL SUMMARY")
    print("=" * 78)
    for m in metrics:
        print(
            f"{m:20s}: "
            f"{summary[m]['mean']:.4f} ± {summary[m]['std']:.4f}"
        )
    print("\nResults:", out)

if __name__ == "__main__":
    main()
