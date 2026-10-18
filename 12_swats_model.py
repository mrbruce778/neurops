"""
12_swats_model.py

SWATS-Net
Subject-Level Window-Attention Temporal-Spectral Network

Input:
    [B, K, 19, 2000]

    B = subjects per batch
    K = EEG windows sampled from each subject

Output:
    [B, 3]

    0 = Alzheimer
    1 = Control
    2 = FTD
"""

import torch
import torch.nn as nn


# ============================================================
# TEMPORAL ENCODER
# Kept consistent with our previous temporal branch.
# ============================================================

class TemporalEncoder(nn.Module):
    def __init__(self, n_channels=19, feature_dim=128):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(n_channels, 32, kernel_size=15, padding=7),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),

            nn.Conv1d(32, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),

            nn.MaxPool1d(4),
            nn.Dropout(0.20),

            nn.Conv1d(32, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),

            nn.Conv1d(64, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),

            nn.MaxPool1d(4),
            nn.Dropout(0.25),

            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),

            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool1d(1),
        )

        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.30),
        )

    def forward(self, x):
        x = self.features(x)
        return self.projection(x)


# ============================================================
# SPECTRAL ENCODER
# Explicit relative EEG band-power representation.
# ============================================================

class SpectralEncoder(nn.Module):
    def __init__(
        self,
        n_channels=19,
        sfreq=500.0,
        feature_dim=64,
    ):
        super().__init__()

        self.n_channels = n_channels
        self.sfreq = sfreq

        # Half-open intervals [low, high)
        self.bands = (
            (0.5, 4.0),    # delta
            (4.0, 8.0),    # theta
            (8.0, 13.0),   # alpha
            (13.0, 30.0),  # beta
            (30.0, 45.0),  # gamma
        )

        n_features = n_channels * len(self.bands)

        self.projection = nn.Sequential(
            nn.Linear(n_features, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.30),

            nn.Linear(128, feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.30),
        )

    def relative_band_power(self, x):
        """
        x: [N, C, T]

        Returns:
            [N, C, 5]
        """

        # Remove temporal DC component.
        x = x - x.mean(dim=-1, keepdim=True)

        n_samples = x.shape[-1]

        fft = torch.fft.rfft(x, dim=-1)
        power = fft.abs().pow(2)

        freqs = torch.fft.rfftfreq(
            n_samples,
            d=1.0 / self.sfreq,
            device=x.device,
        )

        total_mask = (freqs >= 0.5) & (freqs < 45.0)

        total_power = power[..., total_mask].sum(
            dim=-1,
            keepdim=True,
        )

        eps = torch.finfo(power.dtype).eps
        total_power = total_power.clamp_min(eps)

        band_features = []

        for low, high in self.bands:
            mask = (freqs >= low) & (freqs < high)

            band_power = power[..., mask].sum(
                dim=-1,
                keepdim=True,
            )

            relative_power = band_power / total_power
            band_features.append(relative_power)

        return torch.cat(band_features, dim=-1)

    def forward(self, x):
        band_power = self.relative_band_power(x)

        # [N, 19, 5] -> [N, 95]
        band_power = band_power.flatten(start_dim=1)

        return self.projection(band_power)


# ============================================================
# WINDOW ATTENTION
# Learns how important each EEG window is to the subject.
# ============================================================

class WindowAttention(nn.Module):
    def __init__(self, feature_dim=192, attention_dim=64):
        super().__init__()

        self.attention = nn.Sequential(
            nn.Linear(feature_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1),
        )

    def forward(self, x):
        """
        x: [B, K, D]

        Returns:
            subject_embedding: [B, D]
            attention_weights: [B, K]
        """

        scores = self.attention(x).squeeze(-1)

        weights = torch.softmax(scores, dim=1)

        subject_embedding = torch.sum(
            x * weights.unsqueeze(-1),
            dim=1,
        )

        return subject_embedding, weights


# ============================================================
# SWATS-NET
# ============================================================

class SWATSNet(nn.Module):
    def __init__(
        self,
        n_channels=19,
        n_classes=3,
        sfreq=500.0,
        temporal_dim=128,
        spectral_dim=64,
        attention_dim=64,
    ):
        super().__init__()

        self.temporal_encoder = TemporalEncoder(
            n_channels=n_channels,
            feature_dim=temporal_dim,
        )

        self.spectral_encoder = SpectralEncoder(
            n_channels=n_channels,
            sfreq=sfreq,
            feature_dim=spectral_dim,
        )

        fused_dim = temporal_dim + spectral_dim

        self.window_attention = WindowAttention(
            feature_dim=fused_dim,
            attention_dim=attention_dim,
        )

        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, 96),
            nn.ReLU(inplace=True),
            nn.Dropout(0.40),
            nn.Linear(96, n_classes),
        )

    def forward(self, x, return_attention=False):
        """
        x:
            [B, K, C, T]

        Example:
            [4 subjects, 32 windows, 19 channels, 2000 samples]
        """

        if x.ndim != 4:
            raise ValueError(
                "SWATSNet expects input [B, K, C, T], "
                f"but received {tuple(x.shape)}"
            )

        B, K, C, T = x.shape

        # ----------------------------------------------------
        # Treat windows as temporary batch items for encoders.
        # ----------------------------------------------------

        windows = x.reshape(B * K, C, T)

        temporal_features = self.temporal_encoder(windows)
        spectral_features = self.spectral_encoder(windows)

        # [B*K, 128] + [B*K, 64]
        window_features = torch.cat(
            [temporal_features, spectral_features],
            dim=1,
        )

        # Restore subject/window structure.
        window_features = window_features.reshape(
            B,
            K,
            -1,
        )

        # ----------------------------------------------------
        # Learn which windows matter for each subject.
        # ----------------------------------------------------

        subject_features, attention_weights = (
            self.window_attention(window_features)
        )

        # ONE prediction per subject.
        logits = self.classifier(subject_features)

        if return_attention:
            return logits, attention_weights

        return logits


# ============================================================
# FORWARD-PASS TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("SWATS-NET MODEL TEST")
    print("Subject-Level Window-Attention Temporal-Spectral Network")
    print("=" * 70)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"\nDevice: {device}")

    # Small test batch to avoid unnecessary GPU memory usage.
    B = 2
    K = 8
    C = 19
    T = 2000

    x = torch.randn(B, K, C, T).to(device)

    model = SWATSNet(
        n_channels=19,
        n_classes=3,
        sfreq=500.0,
    ).to(device)

    model.eval()

    with torch.no_grad():
        logits, attention = model(
            x,
            return_attention=True,
        )

    total_params = sum(
        p.numel() for p in model.parameters()
    )

    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(f"\nInput shape:     {tuple(x.shape)}")
    print(f"Output shape:    {tuple(logits.shape)}")
    print(f"Attention shape: {tuple(attention.shape)}")

    print(f"\nTotal parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    print("\nAttention sums:")
    print(attention.sum(dim=1).cpu())

    assert logits.shape == (B, 3)
    assert attention.shape == (B, K)

    assert torch.allclose(
        attention.sum(dim=1),
        torch.ones(B, device=device),
        atol=1e-5,
    )

    print("\nForward-pass test: PASS")