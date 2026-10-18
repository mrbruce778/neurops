import torch
import torch.nn as nn


class TemporalWindowEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(19, 32, 15, padding=7), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32, 32, 7, padding=3), nn.BatchNorm1d(32), nn.ReLU(),
            nn.MaxPool1d(4), nn.Dropout(0.20),
            nn.Conv1d(32, 64, 7, padding=3), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 64, 5, padding=2), nn.BatchNorm1d(64), nn.ReLU(),
            nn.MaxPool1d(4), nn.Dropout(0.25),
            nn.Conv1d(64, 128, 5, padding=2), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, 128, 3, padding=1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.30),
        )

    def forward(self, x):
        return self.head(self.features(x))


class SpectralSubjectEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(95, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(64, 64),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class ConnectivitySubjectEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(855, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.30),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.20),
        )

    def forward(self, x):
        return self.net(x)


class ThreeBranchSubjectModel(nn.Module):
    def __init__(self, n_classes=3):
        super().__init__()
        self.temporal_encoder = TemporalWindowEncoder()
        self.spectral_encoder = SpectralSubjectEncoder()
        self.connectivity_encoder = ConnectivitySubjectEncoder()
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.40),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.30),
            nn.Linear(64, n_classes),
        )

    def encode_temporal_bag(self, eeg_bag):
        b, k, c, t = eeg_bag.shape
        x = eeg_bag.reshape(b * k, c, t)
        w = self.temporal_encoder(x).reshape(b, k, 128)
        return w.mean(dim=1)

    def fuse_from_embeddings(self, temporal_subject, spectral_features, connectivity_features):
        spectral_embedding = self.spectral_encoder(spectral_features)
        connectivity_embedding = self.connectivity_encoder(connectivity_features)
        fused = torch.cat([temporal_subject, spectral_embedding, connectivity_embedding], dim=1)
        logits = self.classifier(fused)
        return logits, spectral_embedding, connectivity_embedding

    def forward(self, eeg_bag, spectral_features, connectivity_features):
        temporal_subject = self.encode_temporal_bag(eeg_bag)
        logits, spectral_embedding, connectivity_embedding = self.fuse_from_embeddings(
            temporal_subject, spectral_features, connectivity_features
        )
        return logits, temporal_subject, spectral_embedding, connectivity_embedding


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ThreeBranchSubjectModel().to(device)
    eeg = torch.randn(2, 16, 19, 2000, device=device)
    spectral = torch.randn(2, 95, device=device)
    connectivity = torch.randn(2, 855, device=device)
    with torch.no_grad():
        logits, t, s, c = model(eeg, spectral, connectivity)
    print("Device:", device)
    print("Logits:", tuple(logits.shape))
    print("Temporal embedding:", tuple(t.shape))
    print("Spectral embedding:", tuple(s.shape))
    print("Connectivity embedding:", tuple(c.shape))
    print("Trainable parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))
    print("Forward-pass test: PASS")
