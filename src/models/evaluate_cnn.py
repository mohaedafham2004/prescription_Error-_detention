"""
src/models/evaluate_cnn.py
===========================
Evaluation script for the trained CharCNN model.

Computes and saves
------------------
- Overall accuracy, precision, recall, F1 (macro + per-class)
- Confusion matrix (saved as both PNG and CSV)
- Per-class accuracy bar chart
- Full classification report (text file)
- Summary JSON for the Streamlit dashboard to load

Usage
-----
    # Evaluate the best checkpoint (default):
    python -m src.models.evaluate_cnn

    # Specific checkpoint and data:
    python -m src.models.evaluate_cnn \
        --checkpoint models/cnn_character/best_model.pth \
        --data-dir data/characters \
        --output-dir evaluation/cnn_eval

All outputs are saved to evaluation/cnn_eval/ by default.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader

# Allow running from project root
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.models.cnn_model import CharCNN
from src.models.train_cnn import CharacterDataset, get_transforms


# ─── Prediction Loop ─────────────────────────────────────────────────────────

@torch.no_grad()
def collect_predictions(model: CharCNN,
                         loader: DataLoader,
                         device: torch.device) -> tuple:
    """Run model on the full loader, collect all true labels and predictions.

    Returns
    -------
    (all_labels, all_preds) : Two lists of integer class indices.
    """
    model.eval()
    all_labels: List[int] = []
    all_preds:  List[int] = []

    for images, labels in loader:
        images = images.to(device)
        log_probs = model(images)
        preds = log_probs.argmax(dim=1).cpu().tolist()
        all_preds.extend(preds)
        all_labels.extend(labels.tolist())

    return all_labels, all_preds


# ─── Confusion Matrix Plot ────────────────────────────────────────────────────

def plot_confusion_matrix(cm: np.ndarray,
                           class_names: List[str],
                           save_path: str,
                           max_classes_for_ticks: int = 40) -> None:
    """Save a confusion matrix heatmap.

    For datasets with many classes the tick labels are suppressed to avoid clutter.
    """
    n = len(class_names)
    fig_size = max(8, n * 0.35)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))

    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax, shrink=0.7)

    if n <= max_classes_for_ticks:
        ax.set_xticks(np.arange(n))
        ax.set_yticks(np.arange(n))
        ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(class_names, fontsize=8)

        # Annotate cells with counts (only if ≤ 30 classes to stay readable)
        if n <= 30:
            thresh = cm.max() / 2.0
            for i in range(n):
                for j in range(n):
                    ax.text(j, i, str(cm[i, j]),
                            ha="center", va="center",
                            color="white" if cm[i, j] > thresh else "black",
                            fontsize=7)
    else:
        ax.set_xticks([])
        ax.set_yticks([])

    ax.set_xlabel("Predicted label", fontsize=11)
    ax.set_ylabel("True label",      fontsize=11)
    ax.set_title(f"Confusion Matrix  ({n} classes)", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Confusion matrix plot saved → {save_path}")


# ─── Per-class Accuracy Bar Chart ────────────────────────────────────────────

def plot_per_class_accuracy(class_names: List[str],
                             per_class_acc: List[float],
                             save_path: str) -> None:
    """Horizontal bar chart of per-class accuracy."""
    n = len(class_names)
    fig_height = max(6, n * 0.28)
    fig, ax = plt.subplots(figsize=(10, fig_height))

    colors = ["#2ecc71" if a >= 0.8 else "#e67e22" if a >= 0.5 else "#e74c3c"
              for a in per_class_acc]

    y_pos = np.arange(n)
    ax.barh(y_pos, [a * 100 for a in per_class_acc], color=colors, edgecolor="none")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(class_names, fontsize=8)
    ax.set_xlabel("Accuracy (%)")
    ax.set_title("Per-class Accuracy  (green ≥80%, orange ≥50%, red <50%)")
    ax.axvline(x=80, color="green",  linestyle="--", linewidth=0.8, alpha=0.5)
    ax.axvline(x=50, color="orange", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_xlim(0, 105)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Per-class accuracy chart saved → {save_path}")


# ─── Main Evaluation Function ─────────────────────────────────────────────────

def evaluate(checkpoint_path: str = "models/cnn_character/best_model.pth",
             data_dir: str = "data/characters",
             output_dir: str = "evaluation/cnn_eval",
             batch_size: int = 64,
             img_size: int = 32) -> Dict:
    """Load a checkpoint, run on the full validation split, save all results.

    Returns
    -------
    summary : Dict with overall accuracy, F1, and paths to saved files.
    """
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cpu")

    print(f"\n{'='*60}")
    print(f"  CharCNN Evaluation")
    print(f"{'='*60}")
    print(f"  Checkpoint : {checkpoint_path}")
    print(f"  Data dir   : {data_dir}")
    print(f"  Output dir : {output_dir}")
    print(f"{'='*60}\n")

    # ── Load checkpoint ───────────────────────────────────────────────────────
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            "Run train_cnn.py first."
        )

    ckpt = torch.load(checkpoint_path, map_location=device)
    label_map: Dict[str, int] = ckpt["label_map"]
    num_classes: int = ckpt["num_classes"]
    index_to_label = {v: k for k, v in label_map.items()}
    class_names = [index_to_label[i] for i in range(num_classes)]

    print(f"  Classes     : {num_classes}")
    print(f"  Epoch       : {ckpt.get('epoch', 'unknown')}")
    print(f"  Val acc (training): {ckpt.get('val_acc', 0)*100:.2f}%\n")

    # ── Build model ───────────────────────────────────────────────────────────
    model = CharCNN(num_classes=num_classes)
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device)

    # ── Build dataset (no augmentation) ──────────────────────────────────────
    print("  Loading dataset (no augmentation) …")
    dataset = CharacterDataset(
        data_dir=data_dir,
        transform=get_transforms(img_size=img_size, augment=False),
        label_map=label_map,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    print(f"  Total samples: {len(dataset)}\n")

    # ── Collect predictions ───────────────────────────────────────────────────
    print("  Running inference …")
    all_labels, all_preds = collect_predictions(model, loader, device)

    # ── Metrics ───────────────────────────────────────────────────────────────
    accuracy  = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    recall    = recall_score(all_labels, all_preds,    average="macro", zero_division=0)
    f1_macro  = f1_score(all_labels, all_preds,        average="macro", zero_division=0)

    print(f"  Overall Accuracy  : {accuracy*100:.2f}%")
    print(f"  Macro Precision   : {precision*100:.2f}%")
    print(f"  Macro Recall      : {recall*100:.2f}%")
    print(f"  Macro F1          : {f1_macro*100:.2f}%")

    # ── Per-class accuracy ────────────────────────────────────────────────────
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))
    per_class_acc = []
    for i in range(num_classes):
        row_sum = cm[i].sum()
        per_class_acc.append(float(cm[i, i] / row_sum) if row_sum > 0 else 0.0)

    # ── Classification report ─────────────────────────────────────────────────
    report_str = classification_report(
        all_labels, all_preds,
        target_names=class_names,
        zero_division=0,
    )
    report_path = os.path.join(output_dir, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write("CharCNN — Classification Report\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Checkpoint : {checkpoint_path}\n")
        f.write(f"Data dir   : {data_dir}\n")
        f.write(f"Samples    : {len(dataset)}\n\n")
        f.write(report_str)
    print(f"\n  Classification report saved → {report_path}")

    # ── Confusion matrix CSV ──────────────────────────────────────────────────
    import pandas as pd
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    cm_csv_path = os.path.join(output_dir, "confusion_matrix.csv")
    cm_df.to_csv(cm_csv_path)
    print(f"  Confusion matrix CSV saved → {cm_csv_path}")

    # ── Confusion matrix plot ─────────────────────────────────────────────────
    cm_img_path = os.path.join(output_dir, "confusion_matrix.png")
    plot_confusion_matrix(cm, class_names, cm_img_path)

    # ── Per-class accuracy chart ──────────────────────────────────────────────
    acc_chart_path = os.path.join(output_dir, "per_class_accuracy.png")
    plot_per_class_accuracy(class_names, per_class_acc, acc_chart_path)

    # ── Summary JSON (for the Streamlit dashboard) ────────────────────────────
    summary = {
        "checkpoint": str(checkpoint_path),
        "num_classes": num_classes,
        "total_samples": len(dataset),
        "accuracy": round(accuracy, 4),
        "macro_precision": round(precision, 4),
        "macro_recall": round(recall, 4),
        "macro_f1": round(f1_macro, 4),
        "per_class": {
            cls: {
                "accuracy": round(per_class_acc[i], 4),
            }
            for i, cls in enumerate(class_names)
        },
        "files": {
            "classification_report": report_path,
            "confusion_matrix_csv":  cm_csv_path,
            "confusion_matrix_img":  cm_img_path,
            "per_class_accuracy_img": acc_chart_path,
        }
    }
    summary_path = os.path.join(output_dir, "eval_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Eval summary JSON saved → {summary_path}")

    print(f"\n  ✓ Evaluation complete.  All results in: {output_dir}")
    return summary


# ─── CLI Entry Point ──────────────────────────────────────────────────────────

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate the trained CharCNN on the full dataset."
    )
    parser.add_argument("--checkpoint", "-c",
                        default="models/cnn_character/best_model.pth",
                        help="Path to the .pth checkpoint file.")
    parser.add_argument("--data-dir", "-d",
                        default="data/characters",
                        help="Root data/characters/ folder.")
    parser.add_argument("--output-dir", "-o",
                        default="evaluation/cnn_eval",
                        help="Where to save evaluation outputs.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--img-size",   type=int, default=32)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    evaluate(
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        img_size=args.img_size,
    )
