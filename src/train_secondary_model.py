"""
src/train_secondary_model.py
============================
Secondary prescription model training & evaluation pipeline.
- Performs image preprocessing & data augmentation on /data/nlp_test_image/
- Implements Leave-One-Out Cross Validation (LOOCV - 9 folds) to prevent false generalization claims
- Trains a secondary deep neural multi-task model (128x128 resolution for fast CPU training)
- Saves model weights to models/secondary_model/
- Logs training progression and fold metrics to reports/secondary_model_training_log.json
"""

import os
import sys
import json
import time
import random
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Set deterministic seeds
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

TARGET_IMAGE_SIZE = (128, 128)

# ══════════════════════════════════════════════════════════════════════════
# Preprocessing & Data Augmentation
# ══════════════════════════════════════════════════════════════════════════

def preprocess_image_pipeline(img_bgr: np.ndarray, target_size: Tuple[int, int] = TARGET_IMAGE_SIZE) -> np.ndarray:
    """Grayscale, CLAHE contrast enhancement, median denoise, adaptive threshold, deskew, and resize."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if img_bgr.ndim == 3 else img_bgr.copy()
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    denoised = cv2.medianBlur(enhanced, 3)
    binary = cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10
    )
    
    coords = np.column_stack(np.where(binary < 128))
    if len(coords) >= 30:
        rect = cv2.minAreaRect(coords)
        angle = rect[-1]
        if angle < -45:
            angle = 90 + angle
        elif angle > 45:
            angle = -(90 - angle)
        if abs(angle) > 0.5:
            h, w = binary.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            enhanced = cv2.warpAffine(enhanced, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    resized = cv2.resize(enhanced, target_size, interpolation=cv2.INTER_AREA)
    return resized


def augment_image(img_gray: np.ndarray) -> np.ndarray:
    """Data augmentation: slight rotation (±5°), brightness/contrast jitter, slight noise."""
    h, w = img_gray.shape[:2]
    angle = random.uniform(-5.0, 5.0)
    scale = random.uniform(0.95, 1.05)
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, scale)
    augmented = cv2.warpAffine(img_gray, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    
    alpha = random.uniform(0.85, 1.15)
    beta = random.uniform(-15, 15)
    augmented = np.clip(alpha * augmented.astype(np.float32) + beta, 0, 255).astype(np.uint8)
    return augmented


# ══════════════════════════════════════════════════════════════════════════
# Model Architecture
# ══════════════════════════════════════════════════════════════════════════

class PrescriptionFeatureExtractor(nn.Module):
    def __init__(self, in_channels: int = 1, base_channels: int = 16):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
            
            nn.AdaptiveAvgPool2d((1, 1))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.conv(x)
        return torch.flatten(feat, 1)


class SecondaryPrescriptionModel(nn.Module):
    def __init__(self, num_drug_classes: int = 48, feature_dim: int = 64):
        super().__init__()
        self.backbone = PrescriptionFeatureExtractor(in_channels=1, base_channels=16)
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, num_drug_classes)
        )
        self.count_head = nn.Sequential(
            nn.Linear(feature_dim, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = self.backbone(x)
        drug_logits = self.classifier(feat)
        med_count = self.count_head(feat)
        return drug_logits, med_count


# ══════════════════════════════════════════════════════════════════════════
# Dataset
# ══════════════════════════════════════════════════════════════════════════

def build_vocabulary(labels_dir: Path) -> Tuple[List[str], Dict[str, int]]:
    all_drugs = set()
    for jf in labels_dir.glob("*.json"):
        with open(jf, "r", encoding="utf-8") as f:
            data = json.load(f)
            for m in data.get("medications", []):
                all_drugs.add(m["drug"])
    vocab = sorted(list(all_drugs))
    drug_to_idx = {d: i for i, d in enumerate(vocab)}
    return vocab, drug_to_idx


class PrescriptionDataset(Dataset):
    def __init__(self, samples: List[Tuple[np.ndarray, List[str]]], drug_to_idx: Dict[str, int], is_train: bool = True, augment_factor: int = 8):
        self.items = []
        self.drug_to_idx = drug_to_idx
        num_classes = len(drug_to_idx)
        
        for img, drugs in samples:
            target = np.zeros(num_classes, dtype=np.float32)
            for d in drugs:
                if d in drug_to_idx:
                    target[drug_to_idx[d]] = 1.0
            count_target = float(len(drugs))
            
            self.items.append((img, target, count_target))
            if is_train:
                for _ in range(augment_factor):
                    aug_img = augment_image(img)
                    self.items.append((aug_img, target, count_target))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        img, target, count = self.items[idx]
        tensor_img = torch.from_numpy(img).float().unsqueeze(0) / 127.5 - 1.0
        return tensor_img, torch.from_numpy(target).float(), torch.tensor([count], dtype=torch.float32)


# ══════════════════════════════════════════════════════════════════════════
# Training & LOOCV Routine
# ══════════════════════════════════════════════════════════════════════════

def train_one_epoch(model, loader, optimizer, bce_criterion, mse_criterion, device):
    model.train()
    total_loss = 0.0
    for imgs, targets, counts in loader:
        imgs, targets, counts = imgs.to(device), targets.to(device), counts.to(device)
        optimizer.zero_grad()
        logits, pred_counts = model(imgs)
        loss = bce_criterion(logits, targets) + 0.1 * mse_criterion(pred_counts, counts)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate_model(model, loader, bce_criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for imgs, targets, counts in loader:
            imgs, targets = imgs.to(device), targets.to(device)
            logits, _ = model(imgs)
            loss = bce_criterion(logits, targets)
            total_loss += loss.item()
            probs = torch.sigmoid(logits).cpu().numpy()
            all_preds.append(probs)
            all_targets.append(targets.cpu().numpy())
            
    all_preds = np.concatenate(all_preds, axis=0) if all_preds else np.zeros((0,))
    all_targets = np.concatenate(all_targets, axis=0) if all_targets else np.zeros((0,))
    return (total_loss / len(loader)) if len(loader) > 0 else 0.0, all_preds, all_targets


def run_loocv_training():
    print("=" * 70, flush=True)
    print("🚀  Starting Secondary Model LOOCV Training & Data Augmentation", flush=True)
    print("=" * 70, flush=True)
    
    data_dir = _PROJECT_ROOT / "data" / "nlp_test_image"
    labels_dir = data_dir / "labels"
    processed_dir = data_dir / "processed"
    models_dir = _PROJECT_ROOT / "models" / "secondary_model"
    reports_dir = _PROJECT_ROOT / "reports"
    
    processed_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    vocab, drug_to_idx = build_vocabulary(labels_dir)
    print(f"📊  Vocabulary size: {len(vocab)} unique drugs across 9 images: {vocab}", flush=True)
    
    dataset_records = []
    image_names = sorted([f"test_image_{i}.jpeg" for i in range(1, 10)])
    
    print("\n📦  Preprocessing 9 prescription images...", flush=True)
    for img_name in image_names:
        img_path = data_dir / img_name
        label_path = labels_dir / f"{Path(img_name).stem}.json"
        
        raw_bgr = cv2.imread(str(img_path))
        if raw_bgr is None:
            raise FileNotFoundError(f"Missing {img_path}")
            
        preprocessed = preprocess_image_pipeline(raw_bgr, target_size=TARGET_IMAGE_SIZE)
        
        proc_out_path = processed_dir / f"proc_{img_name}"
        cv2.imwrite(str(proc_out_path), preprocessed)
        
        with open(label_path, "r", encoding="utf-8") as f:
            label_data = json.load(f)
            drugs = [m["drug"] for m in label_data.get("medications", [])]
            
        dataset_records.append((img_name, preprocessed, drugs))
        print(f"   [OK] {img_name} -> {proc_out_path.name} (Medications: {len(drugs)})", flush=True)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n⚙️  Training Device: {device}", flush=True)
    
    bce_criterion = nn.BCEWithLogitsLoss()
    mse_criterion = nn.MSELoss()
    
    loocv_results = []
    training_logs = {
        "architecture": "SecondaryPrescriptionModel (Fast CNN + Multi-Head Classifier)",
        "validation_strategy": "Leave-One-Out Cross-Validation (LOOCV - 9 Folds)",
        "total_images": 9,
        "input_size": list(TARGET_IMAGE_SIZE),
        "vocabulary": vocab,
        "folds": []
    }
    
    print("\n" + "─" * 70, flush=True)
    print("🔄  Executing Leave-One-Out Cross-Validation (9 Folds)", flush=True)
    print("─" * 70, flush=True)
    
    for fold_idx, (test_name, test_img, test_drugs) in enumerate(dataset_records):
        train_samples = [(img, drg) for (n, img, drg) in dataset_records if n != test_name]
        test_samples = [(test_img, test_drugs)]
        
        train_ds = PrescriptionDataset(train_samples, drug_to_idx, is_train=True, augment_factor=8)
        test_ds = PrescriptionDataset(test_samples, drug_to_idx, is_train=False)
        
        train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)
        
        model = SecondaryPrescriptionModel(num_drug_classes=len(vocab)).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
        
        fold_epoch_losses = []
        for epoch in range(1, 16):
            loss = train_one_epoch(model, train_loader, optimizer, bce_criterion, mse_criterion, device)
            fold_epoch_losses.append(round(loss, 4))
            
        val_loss, preds, targets = evaluate_model(model, test_loader, bce_criterion, device)
        
        pred_indices = np.where(preds[0] >= 0.35)[0] if len(preds) > 0 else []
        pred_drugs = [vocab[i] for i in pred_indices]
        
        gt_set = set(test_drugs)
        pred_set = set(pred_drugs)
        tp = len(gt_set & pred_set)
        fp = len(pred_set - gt_set)
        fn = len(gt_set - pred_set)
        
        prec = tp / (tp + fp) if (tp + fp) > 0 else (1.0 if len(gt_set) == 0 else 0.0)
        rec = tp / (tp + fn) if (tp + fn) > 0 else (1.0 if len(gt_set) == 0 else 0.0)
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        
        fold_record = {
            "fold": fold_idx + 1,
            "held_out_image": test_name,
            "ground_truth": test_drugs,
            "predicted": pred_drugs,
            "precision": round(prec, 3),
            "recall": round(rec, 3),
            "f1_score": round(f1, 3),
            "val_loss": round(val_loss, 4),
            "train_loss_history": fold_epoch_losses
        }
        
        training_logs["folds"].append(fold_record)
        loocv_results.append(fold_record)
        
        print(f"  Fold {fold_idx+1}/9 [{test_name:18s}] -> GT: {len(test_drugs)} | Pred: {len(pred_drugs)} | Precision: {prec:.1%} | Recall: {rec:.1%} | F1: {f1:.1%}", flush=True)
        
    print("\n🏆  Training final secondary production model on full augmented set...", flush=True)
    full_samples = [(img, drg) for (n, img, drg) in dataset_records]
    full_ds = PrescriptionDataset(full_samples, drug_to_idx, is_train=True, augment_factor=10)
    full_loader = DataLoader(full_ds, batch_size=8, shuffle=True)
    
    final_model = SecondaryPrescriptionModel(num_drug_classes=len(vocab)).to(device)
    final_opt = optim.AdamW(final_model.parameters(), lr=2e-3, weight_decay=1e-4)
    
    final_losses = []
    for epoch in range(1, 21):
        loss = train_one_epoch(final_model, full_loader, final_opt, bce_criterion, mse_criterion, device)
        final_losses.append(round(loss, 4))
        
    model_save_path = models_dir / "best_secondary_model.pt"
    torch.save(final_model.state_dict(), str(model_save_path))
    
    config_save_path = models_dir / "model_config.json"
    with open(config_save_path, "w", encoding="utf-8") as f:
        json.dump({
            "model_type": "SecondaryPrescriptionModel",
            "num_drug_classes": len(vocab),
            "vocabulary": vocab,
            "drug_to_idx": drug_to_idx,
            "input_size": list(TARGET_IMAGE_SIZE),
            "saved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ")
        }, f, indent=2)
        
    avg_prec = sum(f["precision"] for f in loocv_results) / len(loocv_results)
    avg_rec = sum(f["recall"] for f in loocv_results) / len(loocv_results)
    avg_f1 = sum(f["f1_score"] for f in loocv_results) / len(loocv_results)
    
    training_logs["aggregate_loocv_metrics"] = {
        "average_precision": round(avg_prec, 4),
        "average_recall": round(avg_rec, 4),
        "average_f1_score": round(avg_f1, 4)
    }
    training_logs["final_train_loss_history"] = final_losses
    
    log_save_path = reports_dir / "secondary_model_training_log.json"
    with open(log_save_path, "w", encoding="utf-8") as f:
        json.dump(training_logs, f, indent=2)
        
    print("\n" + "=" * 70, flush=True)
    print("📊  LOOCV VALIDATION SUMMARY (HONEST GENERALIZATION METRICS)", flush=True)
    print("=" * 70, flush=True)
    print(f"  Folds Evaluated     : {len(loocv_results)}", flush=True)
    print(f"  Mean LOOCV Precision: {avg_prec:.1%}", flush=True)
    print(f"  Mean LOOCV Recall   : {avg_rec:.1%}", flush=True)
    print(f"  Mean LOOCV F1-Score : {avg_f1:.1%}", flush=True)
    print(f"  Model Saved To      : {model_save_path}", flush=True)
    print(f"  Training Log Saved  : {log_save_path}", flush=True)
    print("=" * 70, flush=True)

if __name__ == "__main__":
    run_loocv_training()
