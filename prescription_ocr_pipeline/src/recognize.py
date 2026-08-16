"""
prescription_ocr_pipeline/src/recognize.py
===========================================
Recognition module: dual-model OCR pipeline.
- Primary : TrOCR (microsoft/trocr-base-handwritten) via HuggingFace
- Fallback : CharCNN (models/cnn_character/best_model.pt) – word-level via projection segmentation
- Hybrid   : Try TrOCR first; if confidence < threshold, blend with CNN

Each engine returns: {"text": str, "confidence": float, "engine": str}
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path
from typing import Optional, Dict, Any

import cv2
import numpy as np

# ── Project root on path ──────────────────────────────────────────────────
_PIPELINE_ROOT = Path(__file__).resolve().parents[1]
_ROOT = Path(__file__).resolve().parents[3]
for p in [str(_ROOT), str(_PIPELINE_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)


# ══════════════════════════════════════════════════════════════════════════
# TrOCR Engine
# ══════════════════════════════════════════════════════════════════════════
class TrOCREngine:
    """Microsoft TrOCR for handwritten text recognition."""

    MODEL_NAME = "microsoft/trocr-base-handwritten"
    _instance: Optional["TrOCREngine"] = None  # singleton

    def __init__(self):
        self._loaded = False
        self.processor = None
        self.model = None

    @classmethod
    def get(cls) -> "TrOCREngine":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _lazy_load(self):
        if self._loaded:
            return
        try:
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel
            import torch
            warnings.filterwarnings("ignore", category=FutureWarning)
            print(f"  [TrOCR] Loading {self.MODEL_NAME} …", flush=True)
            self.processor = TrOCRProcessor.from_pretrained(self.MODEL_NAME)
            self.model = VisionEncoderDecoderModel.from_pretrained(self.MODEL_NAME)
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model.to(self._device)
            self.model.eval()
            self._loaded = True
            print(f"  [TrOCR] Ready on {self._device}", flush=True)
        except Exception as exc:
            raise RuntimeError(f"TrOCR load failed: {exc}") from exc

    def recognize(self, binary_image: np.ndarray) -> Dict[str, Any]:
        """
        Run TrOCR on a preprocessed binary numpy image.
        Returns extracted text and a naive confidence proxy.
        """
        self._lazy_load()
        import torch
        from PIL import Image

        # TrOCR expects RGB PIL image
        if binary_image.ndim == 2:
            pil_img = Image.fromarray(binary_image).convert("RGB")
        else:
            pil_img = Image.fromarray(cv2.cvtColor(binary_image, cv2.COLOR_BGR2RGB))

        pixel_values = self.processor(images=pil_img, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(self._device)

        with torch.no_grad():
            outputs = self.model.generate(
                pixel_values,
                output_scores=True,
                return_dict_in_generate=True,
                max_new_tokens=256,
            )

        generated_ids = outputs.sequences
        text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        # Confidence proxy: mean token log-softmax score (clamped 0–1)
        confidence = 0.85  # default when scores not easily parsed
        if hasattr(outputs, "scores") and outputs.scores:
            import torch.nn.functional as F
            token_probs = [
                F.softmax(s, dim=-1).max(dim=-1).values.item()
                for s in outputs.scores
            ]
            if token_probs:
                confidence = float(np.clip(np.mean(token_probs), 0.0, 1.0))

        return {"text": text.strip(), "confidence": confidence, "engine": "trocr"}


# ══════════════════════════════════════════════════════════════════════════
# CNN Engine (CharCNN fallback)
# ══════════════════════════════════════════════════════════════════════════
class CNNEngine:
    """CharCNN wrapper using projection-based character segmentation."""

    _instance: Optional["CNNEngine"] = None

    def __init__(self, model_path: Optional[str] = None):
        self._loaded = False
        self._model_path = model_path or str(
            _ROOT / "models" / "cnn_character" / "best_model.pt"
        )
        self._infer = None

    @classmethod
    def get(cls, model_path: Optional[str] = None) -> "CNNEngine":
        if cls._instance is None:
            cls._instance = cls(model_path)
        return cls._instance

    def _lazy_load(self):
        if self._loaded:
            return
        try:
            from src.models.cnn_infer import CNNCharacterModel
            self._infer = CNNCharacterModel(checkpoint_path=self._model_path)
            self._loaded = True
        except Exception as exc:
            raise RuntimeError(f"CNN model load failed: {exc}") from exc

    def recognize(self, binary_image: np.ndarray) -> Dict[str, Any]:
        self._lazy_load()
        try:
            result = self._infer.predict_text(binary_image)
            text = result if isinstance(result, str) else str(result)
        except Exception as exc:
            text = ""
        return {"text": text.strip(), "confidence": 0.35, "engine": "cnn"}


# ══════════════════════════════════════════════════════════════════════════
# Hybrid / Main Entry Point
# ══════════════════════════════════════════════════════════════════════════
TROCR_CONFIDENCE_THRESHOLD = 0.45  # below this we try CNN blend


def recognize_prescription(
    binary_image: np.ndarray,
    mode: str = "hybrid",
    cnn_model_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run OCR recognition on a preprocessed prescription image.

    Parameters
    ----------
    binary_image     : Clean binary uint8 numpy array.
    mode             : "trocr" | "cnn" | "hybrid" (default)
    cnn_model_path   : Optional override for CNN model .pt path.

    Returns
    -------
    result : {
        "text"       : str    — extracted text (possibly multi-line),
        "confidence" : float  — 0–1 proxy confidence,
        "engine"     : str    — which engine produced the final result,
    }
    """
    if mode == "trocr":
        return TrOCREngine.get().recognize(binary_image)

    if mode == "cnn":
        return CNNEngine.get(cnn_model_path).recognize(binary_image)

    # ── hybrid ──────────────────────────────────────────────────────────
    trocr_result = TrOCREngine.get().recognize(binary_image)

    if trocr_result["confidence"] >= TROCR_CONFIDENCE_THRESHOLD:
        trocr_result["engine"] = "trocr_primary"
        return trocr_result

    # Low-confidence → try CNN and pick longer result (more text ≈ more info)
    try:
        cnn_result = CNNEngine.get(cnn_model_path).recognize(binary_image)
        trocr_text = trocr_result["text"]
        cnn_text = cnn_result["text"]
        if len(cnn_text) > len(trocr_text) * 1.5 and len(cnn_text) > 3:
            return {
                "text": cnn_text,
                "confidence": cnn_result["confidence"],
                "engine": "cnn_fallback",
            }
    except Exception:
        pass

    trocr_result["engine"] = "trocr_lowconf"
    return trocr_result
