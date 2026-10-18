import torch
import torch.nn as nn
import torch.nn.functional as F


SFREQ = 500.0
BANDS = [
    (0.5, 4.0),
    (4.0, 8.0),
    (8.0, 13.0),
    (13.0, 30.0),
    (30.0, 45.0),
]


class TemporalEncoder(nn.Module):
    """
    Same temporal idea used in the earlier EEG model.
    Input:  [B*K, 19, 2000]
    Output: [B*K, 128]
    """
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv1d(19, 32, kernel_size=15, padding=7),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            nn.Conv1d(32, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            nn.MaxPool1d(4),
            nn.Dropout(0.20),

            nn.Conv1d(32, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Conv1d(64, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.MaxPool1d(4),
            nn.Dropout(0.25),

            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.AdaptiveAvgPool1d(1),
        )

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.30),
        )

    def forward(self, x):
        return self.head(self.net(x))


class SpectralEncoder(nn.Module):
    """
    Relative band-power encoder.
    Input:  [B*K, 19, 2000]
    Output: [B*K, 64]
    """
    def __init__(self):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(19 * 5, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.30),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.20),
        )

    def relative_band_power(self, x):
        x = x - x.mean(dim=-1, keepdim=True)

        spectrum = torch.fft.rfft(x, dim=-1)
        power = spectrum.abs().pow(2)

        n_samples = x.shape[-1]
        freqs = torch.fft.rfftfreq(
            n_samples,
            d=1.0 / SFREQ,
            device=x.device,
        )

        total_mask = (freqs >= 0.5) & (freqs < 45.0)
        total_power = power[..., total_mask].sum(dim=-1)
        total_power = total_power.clamp_min(
            torch.finfo(power.dtype).eps
        )

        band_features = []

        for low, high in BANDS:
            mask = (freqs >= low) & (freqs < high)
            band_power = power[..., mask].sum(dim=-1)
            relative = band_power / total_power
            band_features.append(relative)

        # [B*K, 19, 5]
        features = torch.stack(band_features, dim=-1)

        return features.flatten(start_dim=1)

    def forward(self, x):
        features = self.relative_band_power(x)
        return self.mlp(features)


class WindowAttention(nn.Module):
    """
    Attention across EEG windows.
    Input:  [B, K, D]
    Output: subject vector [B, D], attention [B, K]
    """
    def __init__(self, feature_dim):
        super().__init__()

        self.score = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        scores = self.score(x).squeeze(-1)
        weights = torch.softmax(scores, dim=1)
        subject_vector = torch.sum(
            x * weights.unsqueeze(-1),
            dim=1,
        )
        return subject_vector, weights


class ConnectivityPairAttention(nn.Module):
    """
    Learns which channel-pairs matter most.

    Connectivity input:
        [B, 5, 171]
        5 frequency bands x 171 unique channel pairs

    Each channel-pair is represented by its 5 band-wise
    coherence values, then attention is learned across pairs.
    """
    def __init__(self):
        super().__init__()

        self.pair_encoder = nn.Sequential(
            nn.Linear(5, 32),
            nn.ReLU(),
            nn.Dropout(0.15),

            nn.Linear(32, 64),
            nn.ReLU(),
        )

        self.pair_score = nn.Sequential(
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 1),
        )

    def forward(self, connectivity):
        # [B, 5, 171] -> [B, 171, 5]
        x = connectivity.transpose(1, 2)

        pair_embeddings = self.pair_encoder(x)

        scores = self.pair_score(
            pair_embeddings
        ).squeeze(-1)

        weights = torch.softmax(
            scores,
            dim=1,
        )

        connectivity_vector = torch.sum(
            pair_embeddings
            * weights.unsqueeze(-1),
            dim=1,
        )

        return connectivity_vector, weights


class ConnectivityAwareSWATS(nn.Module):
    """
    Subject-level Temporal + Spectral + Connectivity model.

    EEG:
        [B, K, 19, 2000]

    Connectivity:
        [B, 5, 171]

    Outputs:
        logits
        window attention
        pair attention
        modality gate
    """
    def __init__(self, n_classes=3):
        super().__init__()

        self.temporal_encoder = TemporalEncoder()
        self.spectral_encoder = SpectralEncoder()

        # Window representation: 128 + 64 = 192
        self.window_attention = WindowAttention(
            feature_dim=192
        )

        self.connectivity_encoder = (
            ConnectivityPairAttention()
        )

        # Project activity and connectivity to same dimension.
        self.activity_projection = nn.Sequential(
            nn.Linear(192, 128),
            nn.ReLU(),
            nn.Dropout(0.25),
        )

        self.connectivity_projection = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.25),
        )

        # Learn how much to trust EEG activity vs connectivity.
        self.modality_gate = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.40),
            nn.Linear(64, n_classes),
        )

    def forward(self, eeg, connectivity):
        b, k, c, t = eeg.shape

        flat_eeg = eeg.reshape(
            b * k,
            c,
            t,
        )

        temporal = self.temporal_encoder(
            flat_eeg
        )

        spectral = self.spectral_encoder(
            flat_eeg
        )

        window_features = torch.cat(
            [temporal, spectral],
            dim=1,
        )

        window_features = window_features.reshape(
            b,
            k,
            -1,
        )

        activity_subject, window_weights = (
            self.window_attention(
                window_features
            )
        )

        connectivity_subject, pair_weights = (
            self.connectivity_encoder(
                connectivity
            )
        )

        activity_128 = self.activity_projection(
            activity_subject
        )

        connectivity_128 = (
            self.connectivity_projection(
                connectivity_subject
            )
        )

        gate_input = torch.cat(
            [
                activity_128,
                connectivity_128,
            ],
            dim=1,
        )

        gate_weights = torch.softmax(
            self.modality_gate(
                gate_input
            ),
            dim=1,
        )

        fused = (
            gate_weights[:, 0:1]
            * activity_128
            +
            gate_weights[:, 1:2]
            * connectivity_128
        )

        logits = self.classifier(
            fused
        )

        return (
            logits,
            window_weights,
            pair_weights,
            gate_weights,
        )


if __name__ == "__main__":
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model = ConnectivityAwareSWATS().to(
        device
    )

    eeg = torch.randn(
        2,
        8,
        19,
        2000,
        device=device,
    )

    connectivity = torch.rand(
        2,
        5,
        171,
        device=device,
    )

    with torch.no_grad():
        (
            logits,
            window_attention,
            pair_attention,
            modality_gate,
        ) = model(
            eeg,
            connectivity,
        )

    print("Device:", device)
    print("Input EEG:", tuple(eeg.shape))
    print("Connectivity:", tuple(connectivity.shape))
    print("Output:", tuple(logits.shape))
    print(
        "Window attention:",
        tuple(window_attention.shape),
    )
    print(
        "Pair attention:",
        tuple(pair_attention.shape),
    )
    print(
        "Modality gate:",
        tuple(modality_gate.shape),
    )

    params = sum(
        p.numel()
        for p in model.parameters()
    )

    print("Parameters:", params)
    print(
        "Window attention sums:",
        window_attention.sum(dim=1),
    )
    print(
        "Pair attention sums:",
        pair_attention.sum(dim=1),
    )
    print(
        "Gate sums:",
        modality_gate.sum(dim=1),
    )
    print("Forward-pass test: PASS")
