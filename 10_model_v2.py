import torch
import torch.nn as nn


# ============================================================
# TEMPORAL BRANCH
# ============================================================

class TemporalBranch(nn.Module):
    """
    Learns patterns directly from the EEG time signal.

    Input:
        [batch, 19, 2000]

    Output:
        [batch, 128]

    NOTE:
        This branch is intentionally unchanged from Model v1.
    """

    def __init__(self, n_channels=19, feature_dim=128):
        super().__init__()

        self.features = nn.Sequential(

            # Block 1
            nn.Conv1d(
                n_channels, 32,
                kernel_size=15,
                padding=7,
                bias=False
            ),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            nn.Conv1d(
                32, 32,
                kernel_size=7,
                padding=3,
                bias=False
            ),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            nn.MaxPool1d(4),
            nn.Dropout(0.20),

            # Block 2
            nn.Conv1d(
                32, 64,
                kernel_size=7,
                padding=3,
                bias=False
            ),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Conv1d(
                64, 64,
                kernel_size=5,
                padding=2,
                bias=False
            ),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.MaxPool1d(4),
            nn.Dropout(0.25),

            # Block 3
            nn.Conv1d(
                64, 128,
                kernel_size=5,
                padding=2,
                bias=False
            ),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Conv1d(
                128, 128,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.AdaptiveAvgPool1d(1),
        )

        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, feature_dim),
            nn.ReLU(),
            nn.Dropout(0.30),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.projection(x)
        return x


# ============================================================
# SPECTRAL BRANCH V2
# ============================================================

class SpectralBranchV2(nn.Module):
    """
    EEG-specific spectral branch based on relative band power.

    For each EEG channel, relative spectral power is calculated
    for five conventional frequency bands:

        Delta: 0.5 - 4 Hz
        Theta: 4   - 8 Hz
        Alpha: 8   - 13 Hz
        Beta:  13  - 30 Hz
        Gamma: 30  - 45 Hz

    Input:
        [batch, 19, 2000]

    Output:
        [batch, 128]

    With 19 channels and 5 bands:
        19 x 5 = 95 spectral features per EEG window.
    """

    def __init__(
        self,
        n_channels=19,
        feature_dim=128,
        sfreq=500.0,
    ):
        super().__init__()

        self.n_channels = n_channels
        self.sfreq = sfreq

        # Frequency bands in Hz.
        self.bands = (
            (0.5, 4.0),    # Delta
            (4.0, 8.0),    # Theta
            (8.0, 13.0),   # Alpha
            (13.0, 30.0),  # Beta
            (30.0, 45.0),  # Gamma
        )

        n_spectral_features = n_channels * len(self.bands)

        # Convert the 95 relative-band-power features
        # into a 128-dimensional learned representation.
        self.projection = nn.Sequential(
            nn.Linear(n_spectral_features, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.30),

            nn.Linear(128, feature_dim),
            nn.ReLU(),
            nn.Dropout(0.30),
        )

    def _relative_band_power(self, x):
        """
        Calculate relative FFT-based spectral power.

        x:
            [batch, channels, samples]

        returns:
            [batch, channels, 5]
        """

        # Remove the temporal mean of each channel.
        x = x - x.mean(dim=-1, keepdim=True)

        n_samples = x.shape[-1]

        # Real FFT.
        spectrum = torch.fft.rfft(
            x,
            dim=-1,
        )

        # Power spectrum.
        power = spectrum.abs().pow(2)

        # Frequency corresponding to each FFT bin.
        freqs = torch.fft.rfftfreq(
            n_samples,
            d=1.0 / self.sfreq,
            device=x.device,
        )

        # ----------------------------------------------------
        # Total power from 0.5 to 45 Hz
        # ----------------------------------------------------

        total_mask = (
            (freqs >= 0.5)
            & (freqs <= 45.0)
        )

        total_power = power[
            ...,
            total_mask
        ].sum(
            dim=-1,
            keepdim=True,
        )

        # Numerical stability.
        eps = torch.finfo(power.dtype).eps

        total_power = total_power.clamp_min(eps)

        # ----------------------------------------------------
        # Relative power for each EEG frequency band
        # ----------------------------------------------------

        relative_powers = []

        for low, high in self.bands:

            # Use a half-open interval [low, high)
            # so boundary bins are not counted twice.
            band_mask = (
                (freqs >= low)
                & (freqs < high)
            )

            band_power = power[
                ...,
                band_mask
            ].sum(
                dim=-1,
                keepdim=True,
            )

            relative_power = (
                band_power
                / total_power
            )

            relative_powers.append(
                relative_power
            )

        # [B, C, 5]
        features = torch.cat(
            relative_powers,
            dim=-1,
        )

        return features

    def forward(self, x):

        # [B, 19, 5]
        x = self._relative_band_power(x)

        # [B, 95]
        x = torch.flatten(
            x,
            start_dim=1,
        )

        # [B, 128]
        x = self.projection(x)

        return x


# ============================================================
# TEMPORAL-SPECTRAL FUSION MODEL V2
# ============================================================

class EEGS3FNetV2(nn.Module):
    """
    S3F-Net-inspired temporal-spectral EEG classifier.

    Temporal branch:
        unchanged 1D CNN from Model v1

    Spectral branch:
        relative EEG band-power representation

    Fusion:
        concatenation of temporal and spectral embeddings

    Output:
        3 classes:
            0 = Alzheimer
            1 = Control
            2 = FTD
    """

    def __init__(
        self,
        n_channels=19,
        n_classes=3,
        feature_dim=128,
        sfreq=500.0,
    ):
        super().__init__()

        # ----------------------------------------------------
        # Temporal representation
        # ----------------------------------------------------

        self.temporal_branch = TemporalBranch(
            n_channels=n_channels,
            feature_dim=feature_dim,
        )

        # ----------------------------------------------------
        # Spectral representation V2
        # ----------------------------------------------------

        self.spectral_branch = SpectralBranchV2(
            n_channels=n_channels,
            feature_dim=feature_dim,
            sfreq=sfreq,
        )

        # ----------------------------------------------------
        # Fusion classifier
        # ----------------------------------------------------

        self.classifier = nn.Sequential(

            nn.Linear(
                feature_dim * 2,
                128,
            ),

            nn.ReLU(),

            nn.Dropout(0.40),

            nn.Linear(
                128,
                n_classes,
            ),
        )

    def forward(self, x):

        temporal_features = self.temporal_branch(x)

        spectral_features = self.spectral_branch(x)

        fused_features = torch.cat(
            (
                temporal_features,
                spectral_features,
            ),
            dim=1,
        )

        logits = self.classifier(
            fused_features
        )

        return logits


# ============================================================
# QUICK MODEL TEST
# ============================================================

if __name__ == "__main__":

    # Fake batch:
    # 8 EEG windows
    # 19 channels
    # 2000 samples/window

    x = torch.randn(
        8,
        19,
        2000,
    )

    model = EEGS3FNetV2(
        n_channels=19,
        n_classes=3,
        feature_dim=128,
        sfreq=500.0,
    )

    with torch.no_grad():
        output = model(x)

    total_parameters = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable_parameters = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print("=" * 60)
    print("EEG S3F-NET V2 MODEL TEST")
    print("=" * 60)

    print(
        f"Input shape:  {tuple(x.shape)}"
    )

    print(
        f"Output shape: {tuple(output.shape)}"
    )

    print(
        f"Total parameters: {total_parameters:,}"
    )

    print(
        f"Trainable parameters: {trainable_parameters:,}"
    )

    assert output.shape == (8, 3)

    print("\nForward-pass test: PASS")