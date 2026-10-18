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
    """EEG temporal/spectral ablation network."""
    VALID_MODES = {"temporal", "spectral", "fusion"}
    def __init__(self, n_channels=19, n_classes=3, feature_dim=128, sfreq=500.0, mode="fusion"):
        super().__init__()
        mode = str(mode).lower()
        if mode not in self.VALID_MODES:
            raise ValueError(f"Unknown mode {mode!r}. Expected one of {sorted(self.VALID_MODES)}.")
        self.mode = mode
        self.temporal = TemporalBranch(n_channels, feature_dim) if mode in {"temporal", "fusion"} else None
        self.spectral = SpectralBranch(n_channels, feature_dim, sfreq) if mode in {"spectral", "fusion"} else None
        input_dim = feature_dim * 2 if mode == "fusion" else feature_dim
        self.classifier = nn.Sequential(nn.Linear(input_dim, 128), nn.ReLU(), nn.Dropout(0.40), nn.Linear(128, n_classes))
    def forward(self, x):
        if self.mode == "temporal": features = self.temporal(x)
        elif self.mode == "spectral": features = self.spectral(x)
        else: features = torch.cat((self.temporal(x), self.spectral(x)), dim=1)
        return self.classifier(features)


# ============================================================
# TEST MODEL
# ============================================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn(8, 19, 2000, device=device)
    for mode in ("temporal", "spectral", "fusion"):
        m = EEGS3FNet(mode=mode).to(device).eval()
        with torch.no_grad(): output = m(x)
        n_params = sum(p.numel() for p in m.parameters())
        assert output.shape == (8, 3) and torch.isfinite(output).all()
        print(f"{mode:8s} | output={tuple(output.shape)} | parameters={n_params:,} | PASS")
