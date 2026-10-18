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
# SPECTRAL BRANCH
# ============================================================

class SpectralBranch(nn.Module):
    """
    Converts EEG from the time domain into the frequency
    domain using FFT, then learns frequency patterns.

    Input:
        [batch, 19, 2000]

    Output:
        [batch, 128]
    """

    def __init__(
        self,
        n_channels=19,
        feature_dim=128,
        sfreq=500.0,
        min_freq=0.5,
        max_freq=45.0,
    ):
        super().__init__()

        self.sfreq = sfreq
        self.min_freq = min_freq
        self.max_freq = max_freq

        self.features = nn.Sequential(

            # Spectral block 1
            nn.Conv1d(
                n_channels, 32,
                kernel_size=7,
                padding=3,
                bias=False
            ),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),

            # Spectral block 2
            nn.Conv1d(
                32, 64,
                kernel_size=5,
                padding=2,
                bias=False
            ),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            # Spectral block 3
            nn.Conv1d(
                64, 128,
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

        # Remove DC offset independently from each EEG channel.
        x = x - x.mean(
            dim=-1,
            keepdim=True
        )

        # ----------------------------------------------------
        # FFT
        # Time domain -> frequency domain
        # ----------------------------------------------------

        spectrum = torch.fft.rfft(
            x,
            dim=-1
        )

        # Magnitude of each frequency.
        magnitude = torch.abs(
            spectrum
        )

        # Frequencies represented by FFT bins.
        freqs = torch.fft.rfftfreq(
            x.shape[-1],
            d=1.0 / self.sfreq,
            device=x.device,
        )

        # ----------------------------------------------------
        # Keep EEG frequencies from 0.5 to 45 Hz.
        # ----------------------------------------------------

        mask = (
            (freqs >= self.min_freq)
            &
            (freqs <= self.max_freq)
        )

        magnitude = magnitude[
            ..., mask
        ]

        # Compress the large dynamic range.
        magnitude = torch.log1p(
            magnitude
        )

        # CNN learns patterns across frequency.
        x = self.features(
            magnitude
        )

        x = self.projection(
            x
        )

        return x


# ============================================================
# EEG S3F-STYLE FUSION NETWORK
# ============================================================

class EEGS3FNet(nn.Module):

    def __init__(
        self,
        n_channels=19,
        n_classes=3,
        feature_dim=128,
        sfreq=500.0,
    ):
        super().__init__()

        # Temporal representation
        self.temporal = TemporalBranch(
            n_channels=n_channels,
            feature_dim=feature_dim,
        )

        # Spectral representation
        self.spectral = SpectralBranch(
            n_channels=n_channels,
            feature_dim=feature_dim,
            sfreq=sfreq,
        )

        # 128 temporal + 128 spectral = 256
        self.classifier = nn.Sequential(

            nn.Linear(
                feature_dim * 2,
                128
            ),

            nn.ReLU(),

            nn.Dropout(0.40),

            nn.Linear(
                128,
                n_classes
            ),
        )

    def forward(self, x):

        temporal_features = self.temporal(
            x
        )

        spectral_features = self.spectral(
            x
        )

        # ----------------------------------------------------
        # Feature fusion
        # ----------------------------------------------------

        fused_features = torch.cat(
            (
                temporal_features,
                spectral_features
            ),
            dim=1
        )

        logits = self.classifier(
            fused_features
        )

        return logits


# ============================================================
# TEST MODEL
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("EEG TEMPORAL-SPECTRAL MODEL TEST")
    print("=" * 70)

    # --------------------------------------------------------
    # Select GPU automatically when available
    # --------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"\nPyTorch version: {torch.__version__}")
    print(f"Device: {device}")

    if torch.cuda.is_available():

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

        print(
            "CUDA version:",
            torch.version.cuda
        )

    # --------------------------------------------------------
    # Create network
    # --------------------------------------------------------

    model = EEGS3FNet(
        n_channels=19,
        n_classes=3,
        feature_dim=128,
        sfreq=500.0,
    )

    model = model.to(
        device
    )

    # Fake batch matching our real data:
    #
    # 8 EEG windows
    # 19 channels
    # 2000 samples
    #
    x = torch.randn(
        8,
        19,
        2000,
        device=device
    )

    model.eval()

    with torch.no_grad():

        temporal_features = (
            model.temporal(x)
        )

        spectral_features = (
            model.spectral(x)
        )

        output = model(x)

    print(
        "\nInput:",
        tuple(x.shape)
    )

    print(
        "Temporal features:",
        tuple(
            temporal_features.shape
        )
    )

    print(
        "Spectral features:",
        tuple(
            spectral_features.shape
        )
    )

    print(
        "Fused features:",
        (
            x.shape[0],
            temporal_features.shape[1]
            + spectral_features.shape[1]
        )
    )

    print(
        "Output:",
        tuple(output.shape)
    )

    # --------------------------------------------------------
    # Parameter count
    # --------------------------------------------------------

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
        f"\nTotal parameters: "
        f"{total_parameters:,}"
    )

    print(
        f"Trainable parameters: "
        f"{trainable_parameters:,}"
    )

    # --------------------------------------------------------
    # Tests
    # --------------------------------------------------------

    assert temporal_features.shape == (
        8,
        128
    )

    assert spectral_features.shape == (
        8,
        128
    )

    assert output.shape == (
        8,
        3
    )

    assert torch.isfinite(
        output
    ).all()

    print("\nForward-pass test: PASS")

    print(
        "\nThe model architecture is working."
    )