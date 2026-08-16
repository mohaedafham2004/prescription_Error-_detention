"""
prescription_ocr_pipeline/src/preprocess.py
=============================================
Preprocessing module: grayscale, adaptive threshold, deskew,
noise removal, resize. Saves intermediate images to data/processed/.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

_PIPELINE_ROOT = Path(__file__).resolve().parents[2]
_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def load_image(image_path: str) -> np.ndarray:
    """Load BGR image, raise on failure."""
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    return img


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert BGR or RGBA to grayscale."""
    if image.ndim == 2:
        return image
    if image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def denoise(gray: np.ndarray) -> np.ndarray:
    """Median blur to reduce noise while preserving ink edges."""
    return cv2.medianBlur(gray, 3)


def adaptive_threshold(denoised: np.ndarray) -> np.ndarray:
    """Adaptive Gaussian threshold for uneven lighting compensation."""
    return cv2.adaptiveThreshold(
        denoised, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=31,
        C=10,
    )


def deskew(binary: np.ndarray) -> np.ndarray:
    """Correct skew using minAreaRect on non-zero pixels."""
    coords = np.column_stack(np.where(binary < 128))
    if len(coords) < 20:
        return binary  # too sparse to estimate skew

    rect = cv2.minAreaRect(coords)
    angle = rect[-1]

    if angle < -45:
        angle = 90 + angle
    elif angle > 45:
        angle = -(90 - angle)

    if abs(angle) < 0.5:  # not worth rotating
        return binary

    h, w = binary.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        binary, M, (w, h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return rotated


def morphological_open(binary: np.ndarray) -> np.ndarray:
    """Remove speckle noise via morphological opening."""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    return cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)


def enhance_contrast(gray: np.ndarray) -> np.ndarray:
    """CLAHE contrast enhancement for faint handwriting."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def preprocess_image(
    image_path: str,
    save_processed_dir: Optional[str] = None,
    verbose: bool = False,
) -> np.ndarray:
    """
    Full preprocessing pipeline: load → grayscale → enhance → denoise
    → adaptive threshold → deskew → morphological clean.

    Parameters
    ----------
    image_path       : Path to the input image.
    save_processed_dir : If set, saves intermediate stages here.
    verbose          : Print step timings.

    Returns
    -------
    processed : Clean binary uint8 ndarray (white text on black background).
    """
    img = load_image(image_path)
    gray = to_grayscale(img)
    enhanced = enhance_contrast(gray)
    denoised = denoise(enhanced)
    binary = adaptive_threshold(denoised)
    deskewed = deskew(binary)
    cleaned = morphological_open(deskewed)

    if verbose:
        print(f"  Preprocess: {Path(image_path).name} → shape={cleaned.shape}")

    if save_processed_dir:
        os.makedirs(save_processed_dir, exist_ok=True)
        stem = Path(image_path).stem
        cv2.imwrite(os.path.join(save_processed_dir, f"{stem}_gray.png"), gray)
        cv2.imwrite(os.path.join(save_processed_dir, f"{stem}_binary.png"), binary)
        cv2.imwrite(os.path.join(save_processed_dir, f"{stem}_deskewed.png"), deskewed)
        cv2.imwrite(os.path.join(save_processed_dir, f"{stem}_cleaned.png"), cleaned)

    return cleaned
