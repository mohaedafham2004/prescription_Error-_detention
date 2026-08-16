"""
src/models/train_cnn.py
========================
Training script for the CharCNN character recognition model.

What this script does
---------------------
1. Scans data/characters/ for class subfolders (each subfolder name = a label).
2. Loads all images, applies augmentation for training, builds an 80/20 split.
3. Trains the CharCNN with NLL loss + Adam optimiser + cosine LR scheduler.
4. Checkpoints the model every epoch to models/cnn_character/.
5. Logs accuracy and loss per epoch and saves a training-curve plot.
6. Saves the label-to-index mapping for inference time.

All training runs on CPU — no GPU required.

Usage
-----
    python -m src.models.train_cnn

    # Override defaults:
    python -m src.models.train_cnn \
        --data-dir data/characters \
        --model-dir models/cnn_character \
        --epochs 30 \
        --batch-size 32 \
        --lr 0.001 \
        --img-size 32

    # Resume from a checkpoint:
    python -m src.models.train_cnn --resume models/cnn_character/best_model.pth
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from PIL import Image

# Allow running from project root
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.models.cnn_model import CharCNN, build_model


# ─── Dataset ─────────────────────────────────────────────────────────────────

class CharacterDataset(Dataset):
    """Loads character images from  data/characters/<label>/<img>.png

    Parameters
    ----------
    data_dir   : Root folder containing one subfolder per character class.
    transform  : torchvision transform pipeline applied to each image.
    label_map  : Optional pre-built {label: index} dict.  If None, built
                 automatically from subfolder names sorted alphabetically.
    """

    def __init__(self, data_dir: str,
                 transform=None,
                 label_map: Dict[str, int] | None = None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.samples: List[Tuple[Path, int]] = []

        # Discover class folders
        class_dirs = sorted(
            [d for d in self.data_dir.iterdir() if d.is_dir()]
        )
        if not class_dirs:
            raise RuntimeError(
                f"No class subfolders found in {data_dir}. "
                "Run segment_characters.py --mode manual first."
            )

        if label_map is None:
            # Build label map from folder names, sorted for reproducibility
            self.label_map: Dict[str, int] = {
                d.name: idx for idx, d in enumerate(class_dirs)
            }
        else:
            self.label_map = label_map

        # Collect (image_path, class_index) pairs
        valid_exts = {".png", ".jpg", ".jpeg", ".bmp"}
        for class_dir in class_dirs:
            label = class_dir.name
            if label not in self.label_map:
                print(f"  WARNING: folder '{label}' not in label_map — skipping.")
                continue
            idx = self.label_map[label]
            for img_path in class_dir.iterdir():
                if img_path.suffix.lower() in valid_exts:
                    self.samples.append((img_path, idx))

        if not self.samples:
            raise RuntimeError(
                f"No images found in {data_dir}. "
                "Add labeled images first using segment_characters.py."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        img_path, label_idx = self.samples[index]
        img = Image.open(img_path).convert("L")   # Grayscale
        if self.transform:
            img = self.transform(img)
        return img, label_idx

    @property
    def num_classes(self) -> int:
        return len(self.label_map)

    @property
    def index_to_label(self) -> Dict[int, str]:
        return {v: k for k, v in self.label_map.items()}

    def class_counts(self) -> Dict[str, int]:
        """Return per-class sample counts."""
        counts: Dict[str, int] = {k: 0 for k in self.label_map}
        idx_to_label = self.index_to_label
        for _, idx in self.samples:
            counts[idx_to_label[idx]] += 1
        return counts


# ─── Transform Pipelines ─────────────────────────────────────────────────────

def get_transforms(img_size: int = 32,
                   augment: bool = True) -> transforms.Compose:
    """Build a torchvision transform pipeline.

    Training augmentations
    ----------------------
    - Random affine: slight rotation (±10°), translation, shear
      → mimics natural handwriting variation
    - Random perspective: mild distortion
    - Gaussian blur: simulate ink bleed
    - Random erasing: simulate incomplete strokes
    """
    base = [
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),                          # → [0,1] float32
        transforms.Normalize(mean=[0.5], std=[0.5]),    # → [-1,1]
    ]

    if augment:
        aug = [
            transforms.RandomAffine(
                degrees=10,
                translate=(0.1, 0.1),
                shear=5,
                fill=255,           # white background for out-of-bounds pixels
            ),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.3, fill=255),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
        ]
        pipeline = aug + base + [
            transforms.RandomErasing(p=0.1, scale=(0.01, 0.05)),
        ]
    else:
        pipeline = base

    return transforms.Compose(pipeline)


# ─── Training Loop ────────────────────────────────────────────────────────────

class EarlyStopping:
    """Stop training if val loss does not improve for `patience` epochs."""

    def __init__(self, patience: int = 7, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.should_stop = False

    def step(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


def train_one_epoch(model: CharCNN,
                    loader: DataLoader,
                    criterion: nn.Module,
                    optimizer: optim.Optimizer,
                    device: torch.device) -> Tuple[float, float]:
    """One training epoch. Returns (avg_loss, accuracy)."""
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        log_probs = model(images)
        loss = criterion(log_probs, labels)
        loss.backward()
        # Gradient clipping for stability on CPU
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = log_probs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model: CharCNN,
             loader: DataLoader,
             criterion: nn.Module,
             device: torch.device) -> Tuple[float, float]:
    """Validation pass. Returns (avg_loss, accuracy)."""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        log_probs = model(images)
        loss = criterion(log_probs, labels)

        total_loss += loss.item() * images.size(0)
        preds = log_probs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)

    return total_loss / total, correct / total


# ─── Plotting ────────────────────────────────────────────────────────────────

def plot_training_curves(history: dict, save_path: str) -> None:
    """Save a loss + accuracy curve plot."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss
    axes[0].plot(history["train_loss"], label="Train Loss", color="#e74c3c")
    axes[0].plot(history["val_loss"],   label="Val Loss",   color="#3498db")
    axes[0].set_title("Loss per Epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("NLL Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Accuracy
    axes[1].plot(
        [a * 100 for a in history["train_acc"]],
        label="Train Acc", color="#e74c3c"
    )
    axes[1].plot(
        [a * 100 for a in history["val_acc"]],
        label="Val Acc", color="#3498db"
    )
    axes[1].set_title("Accuracy per Epoch")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.suptitle("CharCNN Training History", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Training curve saved → {save_path}")


# ─── Main Training Function ───────────────────────────────────────────────────

def train(data_dir: str = "data/characters",
          model_dir: str = "models/cnn_character",
          epochs: int = 30,
          batch_size: int = 32,
          lr: float = 1e-3,
          val_split: float = 0.2,
          img_size: int = 32,
          patience: int = 7,
          resume: str | None = None,
          seed: int = 42) -> None:
    """Full training pipeline."""

    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cpu")  # CPU only — no GPU on this machine
    print(f"\n{'='*60}")
    print(f"  CharCNN Training")
    print(f"{'='*60}")
    print(f"  Data dir   : {data_dir}")
    print(f"  Model dir  : {model_dir}")
    print(f"  Device     : {device}")
    print(f"  Epochs     : {epochs}  |  Batch size: {batch_size}  |  LR: {lr}")
    print(f"{'='*60}\n")

    os.makedirs(model_dir, exist_ok=True)

    # ── Dataset ──────────────────────────────────────────────────────────────
    print("  Loading dataset …")
    full_dataset = CharacterDataset(
        data_dir=data_dir,
        transform=get_transforms(img_size=img_size, augment=True),
    )
    print(f"  Total samples : {len(full_dataset)}")
    print(f"  Classes       : {full_dataset.num_classes}")

    # Print class distribution
    counts = full_dataset.class_counts()
    print(f"  Class distribution:")
    for label, count in sorted(counts.items()):
        print(f"    '{label}': {count}")

    # Save label map
    label_map_path = os.path.join(model_dir, "label_map.json")
    with open(label_map_path, "w") as f:
        json.dump(full_dataset.label_map, f, indent=2, sort_keys=True)
    print(f"\n  Label map saved → {label_map_path}")

    # 80 / 20 train-val split
    n_total = len(full_dataset)
    n_val = max(1, int(n_total * val_split))
    n_train = n_total - n_val

    train_ds, val_ds = random_split(
        full_dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(seed)
    )

    # Val set should use no augmentation
    val_ds.dataset = CharacterDataset(
        data_dir=data_dir,
        transform=get_transforms(img_size=img_size, augment=False),
        label_map=full_dataset.label_map,
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True, num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, num_workers=0, pin_memory=False)

    print(f"\n  Train samples : {n_train}")
    print(f"  Val samples   : {n_val}")
    print(f"  Train batches : {len(train_loader)}")
    print(f"  Val batches   : {len(val_loader)}\n")

    # ── Model ────────────────────────────────────────────────────────────────
    model = build_model(num_classes=full_dataset.num_classes)
    model = model.to(device)

    start_epoch = 0
    if resume:
        ckpt = torch.load(resume, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        start_epoch = ckpt.get("epoch", 0) + 1
        print(f"  Resumed from checkpoint: {resume} (epoch {start_epoch})\n")

    # ── Optimiser & Scheduler ────────────────────────────────────────────────
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr * 0.01
    )

    early_stop = EarlyStopping(patience=patience)

    # ── Training History ─────────────────────────────────────────────────────
    history = {
        "train_loss": [], "val_loss": [],
        "train_acc":  [], "val_acc":  [],
    }
    best_val_acc = 0.0

    print(f"  {'Epoch':>6}  {'Train Loss':>10}  {'Train Acc':>10}  "
          f"{'Val Loss':>10}  {'Val Acc':>10}  {'LR':>10}  {'Time':>8}")
    print("  " + "-" * 72)

    for epoch in range(start_epoch, start_epoch + epochs):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        elapsed = time.time() - t0

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        current_lr = scheduler.get_last_lr()[0]
        print(f"  {epoch + 1:>6}  {train_loss:>10.4f}  {train_acc*100:>9.2f}%  "
              f"{val_loss:>10.4f}  {val_acc*100:>9.2f}%  "
              f"{current_lr:>10.2e}  {elapsed:>7.1f}s")

        # ── Checkpoint every epoch ────────────────────────────────────────────
        ckpt_path = os.path.join(model_dir, f"checkpoint_epoch_{epoch + 1:03d}.pth")
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "val_acc": val_acc,
            "label_map": full_dataset.label_map,
            "num_classes": full_dataset.num_classes,
        }, ckpt_path)

        # ── Keep best model ───────────────────────────────────────────────────
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_path = os.path.join(model_dir, "best_model.pth")
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "val_acc": val_acc,
                "label_map": full_dataset.label_map,
                "num_classes": full_dataset.num_classes,
            }, best_path)
            print(f"  ★ New best model saved (val acc = {val_acc*100:.2f}%)")

        # ── Early stopping ────────────────────────────────────────────────────
        if early_stop.step(val_loss):
            print(f"\n  Early stopping triggered at epoch {epoch + 1} "
                  f"(no improvement for {patience} epochs).")
            break

    # ── Post-training ─────────────────────────────────────────────────────────
    print(f"\n  Training complete. Best val acc = {best_val_acc*100:.2f}%")

    # Save history JSON
    history_path = os.path.join(model_dir, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"  History saved  → {history_path}")

    # Save training curve plot
    curve_path = os.path.join(model_dir, "training_curves.png")
    plot_training_curves(history, curve_path)

    print(f"\n  Model directory contents:")
    for f_name in sorted(os.listdir(model_dir)):
        size = os.path.getsize(os.path.join(model_dir, f_name))
        print(f"    {f_name:<45} {size/1024:>8.1f} KB")


# ─── CLI Entry Point ──────────────────────────────────────────────────────────

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Train the CharCNN character recognition model."
    )
    parser.add_argument("--data-dir",   default="data/characters",
                        help="Root data/characters/ folder (default: data/characters)")
    parser.add_argument("--model-dir",  default="models/cnn_character",
                        help="Where to save checkpoints (default: models/cnn_character)")
    parser.add_argument("--epochs",     type=int, default=30,
                        help="Number of training epochs (default: 30)")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Mini-batch size (default: 32)")
    parser.add_argument("--lr",         type=float, default=1e-3,
                        help="Initial learning rate (default: 0.001)")
    parser.add_argument("--val-split",  type=float, default=0.2,
                        help="Fraction of data held out for validation (default: 0.2)")
    parser.add_argument("--img-size",   type=int, default=32,
                        help="Image size in pixels (default: 32)")
    parser.add_argument("--patience",   type=int, default=7,
                        help="Early stopping patience in epochs (default: 7)")
    parser.add_argument("--resume",     default=None,
                        help="Path to a checkpoint .pth file to resume from")
    parser.add_argument("--seed",       type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    train(
        data_dir=args.data_dir,
        model_dir=args.model_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        val_split=args.val_split,
        img_size=args.img_size,
        patience=args.patience,
        resume=args.resume,
        seed=args.seed,
    )
