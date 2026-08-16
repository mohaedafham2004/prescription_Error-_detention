"""
src/models/cnn_model.py
========================
Lightweight CNN for single-character recognition (A–Z, a–z, 0–9).

Architecture
------------
  Input: 32×32 grayscale image  (1 channel)
  ├── Block 1: Conv(32, 3×3, ReLU) → BN → Conv(32, 3×3, ReLU) → BN → MaxPool(2×2) → Dropout(0.25)
  ├── Block 2: Conv(64, 3×3, ReLU) → BN → Conv(64, 3×3, ReLU) → BN → MaxPool(2×2) → Dropout(0.25)
  ├── Block 3: Conv(128, 3×3, ReLU) → BN → MaxPool(2×2) → Dropout(0.25)
  ├── Flatten → Dense(256, ReLU) → BN → Dropout(0.5)
  └── Dense(num_classes) → LogSoftmax

Design goals
------------
- Stays under ~1M parameters so it trains comfortably on CPU in reasonable time.
- num_classes is determined at runtime from the data folder — no hardcoding.
- BatchNorm after every conv layer stabilises CPU training (no GPU batch-norm quirks).
- Dropout prevents over-fitting on small character datasets.

Usage
-----
    from src.models.cnn_model import CharCNN, build_model

    model = build_model(num_classes=62)   # 26+26+10
    print(model)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CharCNN(nn.Module):
    """Lightweight Convolutional Neural Network for handwritten character recognition.

    Parameters
    ----------
    num_classes : int
        Number of output classes (determined from data/characters/ subfolders).
    input_channels : int
        Number of image channels — 1 for grayscale (default).
    """

    def __init__(self, num_classes: int, input_channels: int = 1):
        super().__init__()
        if num_classes < 2:
            raise ValueError(f"num_classes must be ≥ 2, got {num_classes}")

        self.num_classes = num_classes

        # ── Block 1 ──────────────────────────────────────────────────────────
        self.block1 = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 32×32 → 16×16
            nn.Dropout2d(p=0.25),
        )

        # ── Block 2 ──────────────────────────────────────────────────────────
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 16×16 → 8×8
            nn.Dropout2d(p=0.25),
        )

        # ── Block 3 ──────────────────────────────────────────────────────────
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 8×8 → 4×4
            nn.Dropout2d(p=0.25),
        )

        # ── Classifier ───────────────────────────────────────────────────────
        # After 3 max-pools: 32/2/2/2 = 4 → feature map is 4×4×128 = 2048
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, num_classes),
        )

        # Weight initialisation
        self._init_weights()

    def _init_weights(self):
        """Kaiming (He) init for conv layers, Xavier for linear layers."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : Tensor of shape (B, 1, 32, 32)

        Returns
        -------
        log_probs : Tensor of shape (B, num_classes)  — log-softmax output
        """
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.classifier(x)
        return F.log_softmax(x, dim=1)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Convenience method: returns class indices (not log-probs)."""
        with torch.no_grad():
            log_probs = self.forward(x)
            return torch.argmax(log_probs, dim=1)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Returns softmax probabilities (not log-probs)."""
        with torch.no_grad():
            log_probs = self.forward(x)
            return torch.exp(log_probs)

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def extra_repr(self) -> str:
        return (
            f"num_classes={self.num_classes}, "
            f"params={self.count_parameters():,}"
        )


# ─── Factory ─────────────────────────────────────────────────────────────────

def build_model(num_classes: int, input_channels: int = 1) -> CharCNN:
    """Instantiate a CharCNN and print a parameter summary.

    Parameters
    ----------
    num_classes    : Number of output classes (len of data/characters/ subfolders)
    input_channels : 1 for grayscale (default)

    Returns
    -------
    model : CharCNN instance (not yet moved to any device)
    """
    model = CharCNN(num_classes=num_classes, input_channels=input_channels)
    print(f"CharCNN | classes={num_classes} | "
          f"trainable params={model.count_parameters():,}")
    return model


# ─── Quick sanity-check ───────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    num_cls = int(sys.argv[1]) if len(sys.argv) > 1 else 62

    model = build_model(num_classes=num_cls)
    print(model)

    # Forward pass with a dummy batch
    dummy = torch.zeros(4, 1, 32, 32)          # batch of 4 grayscale 32×32 images
    out = model(dummy)
    print(f"\nInput  shape : {dummy.shape}")
    print(f"Output shape : {out.shape}          # (batch, num_classes) log-probs")
    print(f"Argmax preds : {model.predict(dummy)}")
    print("\n✓ CharCNN forward pass OK")
