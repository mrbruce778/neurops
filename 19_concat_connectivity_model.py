import torch
import torch.nn as nn


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
    Same temporal encoder used in the previous models.

    Input:
        [B*K, 19, 2000]

    Output:
        [B*K, 128]
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

    Input:
        [B*K, 19, 2000]

    Output:
        [B*K, 64]
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
        x = x - x.mean(
            dim=-1,
            keepdim=True,
        )

        spectrum = torch.fft.rfft(
            x,
            dim=-1,
        )

        power = spectrum.abs().pow(2)

        n_samples = x.shape[-1]

        freqs = torch.fft.rfftfreq(
            n_samples,
            d=1.0 / SFREQ,
            device=x.device,
        )

        total_mask = (
            (freqs >= 0.5)
            & (freqs < 45.0)
        )

        total_power = power[
            ...,
            total_mask,
        ].sum(
            dim=-1
        )

        total_power = total_power.clamp_min(
            torch.finfo(
                power.dtype
            ).eps
        )

        band_features = []

        for low, high in BANDS:
            mask = (
                (freqs >= low)
                & (freqs < high)
            )

            band_power = power[
                ...,
                mask,
            ].sum(
                dim=-1
            )

            relative_power = (
                band_power
                / total_power
            )

            band_features.append(
                relative_power
            )

        features = torch.stack(
            band_features,
            dim=-1,
        )

        return features.flatten(
            start_dim=1
        )

    def forward(self, x):
        features = self.relative_band_power(
            x
        )

        return self.mlp(
            features
        )


class WindowAttention(nn.Module):
    """
    Attention across EEG windows.

    Input:
        [B, K, 192]

    Output:
        subject vector [B, 192]
        attention weights [B, K]
    """

    def __init__(self, feature_dim=192):
        super().__init__()

        self.score = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        scores = self.score(
            x
        ).squeeze(
            -1
        )

        weights = torch.softmax(
            scores,
            dim=1,
        )

        subject_vector = torch.sum(
            x * weights.unsqueeze(-1),
            dim=1,
        )

        return (
            subject_vector,
            weights,
        )


class ConnectivityEncoder(nn.Module):
    """
    Simple subject-level connectivity encoder.

    IMPORTANT:
    No pair attention.
    No modality gate.

    Input:
        [B, 5, 171]

    Flattened:
        [B, 855]

    Output:
        [B, 64]
    """

    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Flatten(),

            nn.Linear(855, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.30),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.20),
        )

    def forward(self, connectivity):
        return self.net(
            connectivity
        )


class ConcatConnectivityModel(nn.Module):
    """
    Controlled fusion experiment.

    EEG pathway:
        Temporal 128D
        +
        Spectral 64D
        -> 192D/window
        -> Window attention
        -> 192D subject representation

    Connectivity pathway:
        855 coherence features
        -> 128
        -> 64D

    Fusion:
        concatenate 192D + 64D
        -> 256D
        -> classifier

    There is deliberately:
        NO modality gate
        NO connectivity pair attention
    """

    def __init__(
        self,
        n_classes=3,
    ):
        super().__init__()

        self.temporal_encoder = (
            TemporalEncoder()
        )

        self.spectral_encoder = (
            SpectralEncoder()
        )

        self.window_attention = (
            WindowAttention(
                feature_dim=192
            )
        )

        self.connectivity_encoder = (
            ConnectivityEncoder()
        )

        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.40),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.30),

            nn.Linear(64, n_classes),
        )

    def forward(
        self,
        eeg,
        connectivity,
    ):
        b, k, c, t = eeg.shape

        flat_eeg = eeg.reshape(
            b * k,
            c,
            t,
        )

        temporal = (
            self.temporal_encoder(
                flat_eeg
            )
        )

        spectral = (
            self.spectral_encoder(
                flat_eeg
            )
        )

        window_features = torch.cat(
            [
                temporal,
                spectral,
            ],
            dim=1,
        )

        window_features = (
            window_features.reshape(
                b,
                k,
                -1,
            )
        )

        (
            eeg_subject,
            window_weights,
        ) = self.window_attention(
            window_features
        )

        connectivity_subject = (
            self.connectivity_encoder(
                connectivity
            )
        )

        fused = torch.cat(
            [
                eeg_subject,
                connectivity_subject,
            ],
            dim=1,
        )

        logits = self.classifier(
            fused
        )

        return (
            logits,
            window_weights,
            connectivity_subject,
        )


if __name__ == "__main__":
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model = ConcatConnectivityModel(
        n_classes=3
    ).to(
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
            connectivity_embedding,
        ) = model(
            eeg,
            connectivity,
        )

    print("Device:", device)
    print(
        "EEG input:",
        tuple(
            eeg.shape
        ),
    )
    print(
        "Connectivity input:",
        tuple(
            connectivity.shape
        ),
    )
    print(
        "Logits:",
        tuple(
            logits.shape
        ),
    )
    print(
        "Window attention:",
        tuple(
            window_attention.shape
        ),
    )
    print(
        "Connectivity embedding:",
        tuple(
            connectivity_embedding.shape
        ),
    )

    print(
        "Window attention sums:",
        window_attention.sum(
            dim=1
        ),
    )

    parameter_count = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        "Trainable parameters:",
        parameter_count,
    )

    print(
        "Forward-pass test: PASS"
    )
