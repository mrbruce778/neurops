from pathlib import Path
import importlib.util
import math

import numpy as np
import pandas as pd
import torch


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

DIAGNOSTIC_DIR = (
    PROJECT_ROOT
    / "results_connectivity_attention_diagnostics"
)

DIAGNOSTIC_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# CONFIG
# ============================================================

BAG_SIZE = 32
NORMALIZATION_EPS = 1e-6

CLASS_NAMES = {
    0: "Alzheimer",
    1: "Control",
    2: "FTD",
}

CHANNEL_NAMES = [
    "Fp1", "Fp2", "F3", "F4",
    "C3", "C4", "P3", "P4",
    "O1", "O2", "F7", "F8",
    "T3", "T4", "T5", "T6",
    "Fz", "Cz", "Pz",
]

CHANNEL_PAIRS = [
    (CHANNEL_NAMES[i], CHANNEL_NAMES[j])
    for i in range(len(CHANNEL_NAMES))
    for j in range(i + 1, len(CHANNEL_NAMES))
]

assert len(CHANNEL_PAIRS) == 171


# ============================================================
# IMPORT MODEL
# ============================================================

spec = importlib.util.spec_from_file_location(
    "connectivity_attention_model",
    MODEL_FILE,
)

model_module = importlib.util.module_from_spec(
    spec
)

spec.loader.exec_module(
    model_module
)

ConnectivityAwareSWATS = (
    model_module.ConnectivityAwareSWATS
)


# ============================================================
# HELPERS
# ============================================================

def load_connectivity_lookup():
    df = pd.read_csv(
        CONNECTIVITY_FILE
    )

    metadata_columns = {
        "subject_id",
        "label",
        "class_name",
    }

    feature_columns = [
        c for c in df.columns
        if c not in metadata_columns
    ]

    if len(feature_columns) != 855:
        raise RuntimeError(
            f"Expected 855 connectivity features, "
            f"found {len(feature_columns)}"
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

        values = values.reshape(
            5,
            171,
        )

        lookup[subject_id] = values

    return lookup


def normalize_windows(eeg):
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


def normalized_entropy(weights):
    """
    1.0  -> nearly uniform attention
    lower -> more concentrated attention
    """
    weights = np.asarray(
        weights,
        dtype=np.float64,
    )

    weights = np.clip(
        weights,
        1e-12,
        1.0,
    )

    entropy = -np.sum(
        weights * np.log(weights)
    )

    max_entropy = math.log(
        len(weights)
    )

    return float(
        entropy / max_entropy
    )


# ============================================================
# SUBJECT DIAGNOSTICS
# ============================================================

@torch.no_grad()
def diagnose_subject(
    model,
    window_file,
    connectivity,
    device,
):

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

    eeg = normalize_windows(
        eeg
    )

    n_windows = eeg.shape[0]

    bag_probs = []
    bag_sizes = []

    bag_activity_gate = []
    bag_connectivity_gate = []

    bag_pair_attention = []
    bag_pair_entropy = []

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
                bag
            )
            .unsqueeze(0)
            .to(device)
        )

        conn_tensor = (
            torch.from_numpy(
                connectivity
            )
            .unsqueeze(0)
            .to(device)
        )

        (
            logits,
            window_attention,
            pair_attention,
            gate_weights,
        ) = model(
            eeg_tensor,
            conn_tensor,
        )

        probs = torch.softmax(
            logits,
            dim=1,
        )[0].cpu().numpy()

        pair_weights = (
            pair_attention[0]
            .cpu()
            .numpy()
        )

        gate = (
            gate_weights[0]
            .cpu()
            .numpy()
        )

        bag_probs.append(
            probs
        )

        bag_sizes.append(
            original_size
        )

        bag_activity_gate.append(
            gate[0]
        )

        bag_connectivity_gate.append(
            gate[1]
        )

        bag_pair_attention.append(
            pair_weights
        )

        bag_pair_entropy.append(
            normalized_entropy(
                pair_weights
            )
        )

    weights = np.asarray(
        bag_sizes,
        dtype=np.float64,
    )

    probs = np.average(
        np.stack(
            bag_probs,
            axis=0,
        ),
        axis=0,
        weights=weights,
    )

    activity_gate = float(
        np.average(
            np.asarray(
                bag_activity_gate
            ),
            weights=weights,
        )
    )

    connectivity_gate = float(
        np.average(
            np.asarray(
                bag_connectivity_gate
            ),
            weights=weights,
        )
    )

    pair_attention = np.average(
        np.stack(
            bag_pair_attention,
            axis=0,
        ),
        axis=0,
        weights=weights,
    )

    pair_entropy = float(
        np.average(
            np.asarray(
                bag_pair_entropy
            ),
            weights=weights,
        )
    )

    top_indices = np.argsort(
        pair_attention
    )[::-1][:10]

    top_pairs = []

    for rank, pair_idx in enumerate(
        top_indices,
        start=1,
    ):
        ch1, ch2 = CHANNEL_PAIRS[
            pair_idx
        ]

        top_pairs.append(
            {
                "rank": rank,
                "pair_index": int(
                    pair_idx
                ),
                "channel_1": ch1,
                "channel_2": ch2,
                "attention_weight": float(
                    pair_attention[
                        pair_idx
                    ]
                ),
            }
        )

    prediction = int(
        np.argmax(
            probs
        )
    )

    return {
        "prediction": prediction,
        "prob_alzheimer": float(
            probs[0]
        ),
        "prob_control": float(
            probs[1]
        ),
        "prob_ftd": float(
            probs[2]
        ),
        "activity_gate": activity_gate,
        "connectivity_gate":
            connectivity_gate,
        "pair_attention_entropy":
            pair_entropy,
        "top_pairs": top_pairs,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 70)
    print(
        "CONNECTIVITY-ATTENTION DIAGNOSTICS"
    )
    print("=" * 70)
    print(
        "Device:",
        device,
    )

    manifest = pd.read_csv(
        WINDOW_MANIFEST
    )

    manifest[
        "subject_id"
    ] = manifest[
        "subject_id"
    ].astype(str)

    manifest_lookup = (
        manifest
        .set_index(
            "subject_id"
        )
    )

    connectivity_lookup = (
        load_connectivity_lookup()
    )

    all_subject_rows = []
    all_top_pair_rows = []

    for fold_number in range(
        1,
        6,
    ):

        fold_dir = (
            RESULTS_DIR
            / f"fold_{fold_number}"
        )

        checkpoint_file = (
            fold_dir
            / "best_model.pt"
        )

        prediction_file = (
            fold_dir
            / "subject_predictions.csv"
        )

        if not checkpoint_file.exists():
            raise FileNotFoundError(
                f"Missing checkpoint:\n"
                f"{checkpoint_file}"
            )

        if not prediction_file.exists():
            raise FileNotFoundError(
                f"Missing prediction file:\n"
                f"{prediction_file}"
            )

        saved_predictions = pd.read_csv(
            prediction_file
        )

        model = ConnectivityAwareSWATS(
            n_classes=3
        ).to(
            device
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

        model.eval()

        print()
        print(
            f"Fold {fold_number}"
        )

        for _, pred_row in (
            saved_predictions
            .iterrows()
        ):

            subject_id = str(
                pred_row[
                    "subject_id"
                ]
            )

            if subject_id not in (
                manifest_lookup.index
            ):
                raise RuntimeError(
                    f"{subject_id} not found "
                    f"in manifest."
                )

            manifest_row = (
                manifest_lookup.loc[
                    subject_id
                ]
            )

            true_label = int(
                manifest_row[
                    "label"
                ]
            )

            class_name = str(
                manifest_row[
                    "class_name"
                ]
            )

            window_file = Path(
                manifest_row[
                    "window_file"
                ]
            )

            result = diagnose_subject(
                model=model,
                window_file=window_file,
                connectivity=(
                    connectivity_lookup[
                        subject_id
                    ]
                ),
                device=device,
            )

            row = {
                "fold": fold_number,
                "subject_id":
                    subject_id,
                "true_label":
                    true_label,
                "true_class":
                    class_name,
                "predicted_label":
                    result[
                        "prediction"
                    ],
                "predicted_class":
                    CLASS_NAMES[
                        result[
                            "prediction"
                        ]
                    ],
                "correct":
                    int(
                        result[
                            "prediction"
                        ]
                        == true_label
                    ),
                "prob_alzheimer":
                    result[
                        "prob_alzheimer"
                    ],
                "prob_control":
                    result[
                        "prob_control"
                    ],
                "prob_ftd":
                    result[
                        "prob_ftd"
                    ],
                "activity_gate":
                    result[
                        "activity_gate"
                    ],
                "connectivity_gate":
                    result[
                        "connectivity_gate"
                    ],
                "pair_attention_entropy":
                    result[
                        "pair_attention_entropy"
                    ],
            }

            all_subject_rows.append(
                row
            )

            for pair_info in (
                result[
                    "top_pairs"
                ]
            ):

                pair_row = {
                    "fold":
                        fold_number,
                    "subject_id":
                        subject_id,
                    "true_class":
                        class_name,
                    **pair_info,
                }

                all_top_pair_rows.append(
                    pair_row
                )

    subject_df = pd.DataFrame(
        all_subject_rows
    )

    pair_df = pd.DataFrame(
        all_top_pair_rows
    )

    subject_file = (
        DIAGNOSTIC_DIR
        / "subject_diagnostics.csv"
    )

    pair_file = (
        DIAGNOSTIC_DIR
        / "top_connectivity_pairs.csv"
    )

    subject_df.to_csv(
        subject_file,
        index=False,
    )

    pair_df.to_csv(
        pair_file,
        index=False,
    )

    class_summary = (
        subject_df
        .groupby(
            "true_class"
        )
        .agg(
            n_subjects=(
                "subject_id",
                "count",
            ),
            accuracy=(
                "correct",
                "mean",
            ),
            mean_prob_alzheimer=(
                "prob_alzheimer",
                "mean",
            ),
            mean_prob_control=(
                "prob_control",
                "mean",
            ),
            mean_prob_ftd=(
                "prob_ftd",
                "mean",
            ),
            mean_activity_gate=(
                "activity_gate",
                "mean",
            ),
            mean_connectivity_gate=(
                "connectivity_gate",
                "mean",
            ),
            mean_pair_attention_entropy=(
                "pair_attention_entropy",
                "mean",
            ),
        )
        .reset_index()
    )

    class_summary_file = (
        DIAGNOSTIC_DIR
        / "class_summary.csv"
    )

    class_summary.to_csv(
        class_summary_file,
        index=False,
    )

    print()
    print("=" * 70)
    print(
        "CLASS-LEVEL DIAGNOSTIC SUMMARY"
    )
    print("=" * 70)

    print(
        class_summary.to_string(
            index=False
        )
    )

    print()
    print(
        "Interpretation guide:"
    )

    print(
        "  connectivity_gate near 0 "
        "-> connectivity mostly ignored"
    )

    print(
        "  connectivity_gate near 1 "
        "-> connectivity dominates"
    )

    print(
        "  pair_attention_entropy near 1 "
        "-> pair attention is nearly uniform"
    )

    print(
        "  much lower entropy "
        "-> attention concentrates on fewer pairs"
    )

    print()
    print(
        "Files saved under:"
    )

    print(
        DIAGNOSTIC_DIR
    )


if __name__ == "__main__":
    main()
