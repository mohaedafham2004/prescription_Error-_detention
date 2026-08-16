"""
src/models/cnn_infer.py
========================
CNN Character OCR Model for single-character recognition and word assembly.
Integrates trained CharCNN (PyTorch) with character segmentation.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.models.ocr_base import OCRModel, ImageInput


class CharCNNNet(nn.Module):
    """Trained CNN architecture for character recognition."""

    def __init__(self, num_classes: int = 52):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class CNNCharacterModel(OCRModel):
    """OCR model using the custom CharCNN for character-level recognition.

    Parameters
    ----------
    checkpoint_path : Path to models/cnn_character/best_model.pt
    label_map_path  : Path to models/cnn_character/label_map.json
    """

    _MODEL_NAME = "cnn"

    def __init__(self,
                 checkpoint_path: str = "models/cnn_character/best_model.pt",
                 label_map_path: str = "models/cnn_character/label_map.json"):
        self.checkpoint_path = Path(checkpoint_path)
        self.label_map_path = Path(label_map_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[CharCNNNet] = None
        self.idx_to_char: Dict[int, str] = {}
        self._is_ready = False

        self._load_model()

    def _load_model(self):
        """Load trained weights and label map."""
        if not self.checkpoint_path.exists():
            # Try alternate extension .pth
            alt = self.checkpoint_path.with_suffix(".pth")
            if alt.exists():
                self.checkpoint_path = alt
            else:
                return

        # Load label map
        if self.label_map_path.exists():
            try:
                with open(self.label_map_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    mapping = data.get("idx_to_char", data)
                    self.idx_to_char = {int(k): str(v) for k, v in mapping.items()}
            except Exception:
                pass

        if not self.idx_to_char:
            # Default 52 classes (a-z, A-Z)
            chars = [chr(c) for c in range(ord('a'), ord('z')+1)] + [chr(c) for c in range(ord('A'), ord('Z')+1)]
            self.idx_to_char = {i: c for i, c in enumerate(chars)}

        num_classes = len(self.idx_to_char)

        try:
            self.model = CharCNNNet(num_classes=num_classes)
            state_dict = torch.load(self.checkpoint_path, map_location=self.device)
            if isinstance(state_dict, dict) and "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            elif isinstance(state_dict, dict) and "model_state_dict" in state_dict:
                state_dict = state_dict["model_state_dict"]

            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()
            self._is_ready = True
        except Exception as e:
            self._is_ready = False

    @property
    def model_name(self) -> str:
        return self._MODEL_NAME

    @property
    def is_ready(self) -> bool:
        return self._is_ready

    @classmethod
    def from_config(cls, cfg: dict) -> "CNNCharacterModel":
        """Factory method to instantiate from config dict."""
        ckpt = cfg.get("cnn_model_path", "models/cnn_character/best_model.pt")
        lbl = cfg.get("cnn_label_map_path", "models/cnn_character/label_map.json")
        return cls(checkpoint_path=ckpt, label_map_path=lbl)

    def _prepare_image(self, image: ImageInput) -> np.ndarray:
        """Convert any image input format to grayscale numpy array."""
        if isinstance(image, (str, Path)):
            img = cv2.imread(str(image), cv2.IMREAD_GRAYSCALE)
            return img if img is not None else np.zeros((32, 100), dtype=np.uint8)
        elif isinstance(image, Image.Image):
            return np.array(image.convert("L"))
        elif isinstance(image, np.ndarray):
            if image.ndim == 3:
                return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            return image
        return np.zeros((32, 100), dtype=np.uint8)

    def recognize_line(self, image: ImageInput) -> Tuple[str, float]:
        """Perform character segmentation and recognition on a line image.

        Returns
        -------
        predicted_text : Recognized and assembled text string.
        confidence     : Mean confidence across recognized characters (0.0 to 1.0).
        """
        if not self._is_ready or self.model is None:
            return "CNN Model not initialized.", 0.0

        gray = self._prepare_image(image)
        if gray.size == 0 or gray.shape[0] < 5 or gray.shape[1] < 5:
            return "", 0.0

        # Otsu binarization
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Invert: text=1 (white on black for projection & tensor)
        inv = (binary == 0).astype(np.uint8)
        col_proj = inv.sum(axis=0)

        # Segment characters via column projection
        bands: List[Tuple[int, int]] = []
        in_char = False
        start = 0
        min_width = 3

        for col, val in enumerate(col_proj):
            if val > 0 and not in_char:
                in_char = True
                start = col
            elif val == 0 and in_char:
                in_char = False
                if (col - start) >= min_width:
                    bands.append((start, col))

        if in_char and (len(col_proj) - start) >= min_width:
            bands.append((start, len(col_proj)))

        if not bands:
            return "", 0.0

        # Calculate average character width for space estimation
        widths = [ce - cs for cs, ce in bands]
        avg_w = sum(widths) / len(widths) if widths else 10.0
        space_threshold = max(avg_w * 0.75, 8.0)

        chars: List[str] = []
        confidences: List[float] = []
        prev_end = None

        for cs, ce in bands:
            # Word boundary check
            if prev_end is not None and (cs - prev_end) >= space_threshold:
                chars.append(" ")

            prev_end = ce

            # Crop character tightly in rows
            col_slice = inv[:, cs:ce]
            row_proj = col_slice.sum(axis=1)
            nz_rows = np.where(row_proj > 0)[0]
            if len(nz_rows) == 0:
                continue

            rs, re = nz_rows[0], nz_rows[-1] + 1
            char_crop = inv[rs:re, cs:ce]
            if char_crop.size == 0:
                continue

            # Resize to 32x32 with aspect-ratio preserving padding
            h, w = char_crop.shape
            pad_size = max(h, w) + 4
            padded = np.zeros((pad_size, pad_size), dtype=np.float32)
            y_off = (pad_size - h) // 2
            x_off = (pad_size - w) // 2
            padded[y_off:y_off+h, x_off:x_off+w] = char_crop

            resized = cv2.resize(padded, (32, 32), interpolation=cv2.INTER_AREA)
            norm_tensor = torch.from_numpy(resized).float().unsqueeze(0).unsqueeze(0).to(self.device)

            # Predict
            with torch.no_grad():
                log_probs = self.model(norm_tensor)
                probs = torch.exp(log_probs).squeeze(0)
                conf, pred_idx = torch.max(probs, dim=0)

                char = self.idx_to_char.get(int(pred_idx.item()), "?")
                chars.append(char)
                confidences.append(float(conf.item()))

        raw_text = "".join(chars).strip()
        mean_conf = float(np.mean(confidences)) if confidences else 0.0

        return raw_text, round(mean_conf, 4)
