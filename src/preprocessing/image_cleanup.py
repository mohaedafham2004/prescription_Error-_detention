"""
src/preprocessing/image_cleanup.py
===================================
Cleans a raw prescription scan before segmentation and OCR.

TWO distinct cleanup modes
--------------------------

1. TrOCR / handwriting path  (default — light-touch)
   ─────────────────────────────────────────────────
   Original image
       → Grayscale
       → Light denoise  (Gaussian blur only — mild, preserves stroke edges)
       → Deskew
       → (returns a clean grayscale image — NO binarization)

   WHY NO BINARIZATION FOR TrOCR?
   TrOCR's Vision Transformer encoder was trained on natural grayscale/colour
   images of handwriting, not clean black-and-white binary scans. Aggressive
   adaptive thresholding:
     a) Destroys thin handwriting strokes (they become gaps in the binary mask)
     b) Amplifies salt-and-pepper noise in the thresholded regions
     c) Causes the model to hallucinate web-page boilerplate when the input
        is mostly white with few meaningful pixels — exactly the "Personal
        tools" / "Jump to navigation" hallucination pattern we identified in
        debug_output/02_preprocessed.png.

   The NLM denoising step was deliberately removed from the TrOCR path.
   NLM's neighbourhood-averaging rounds stroke edges, causing characters to
   blur into each other at the small crop sizes (30–64px height) fed into TrOCR.

2. CNN / character segmentation path  (opt-in, binary=True)
   ─────────────────────────────────────────────────────────
   The CNN character model DOES benefit from clean binary crops:
     • Consistent black-on-white contrast → cleaner character bounding boxes
     • Removes scan background gradients that confuse character classifiers
   Call clean_image(..., binary=True) for this path.

Usage
-----
    # TrOCR path (default — light-touch grayscale)
    gray = clean_image("data/raw/prescriptions/rx1.jpg")

    # CNN/segmentation path (binarized)
    binary = clean_image("data/raw/prescriptions/rx1.jpg", binary=True)

    # Debug: save before/after
    gray = clean_image("rx1.jpg", output_path="debug_output/02_preprocessed.png",
                        debug_side_by_side="debug_output/02b_side_by_side.png")

CLI
---
    python -m src.preprocessing.image_cleanup --input path/to/scan.jpg
    python -m src.preprocessing.image_cleanup --input scan.jpg --binary --show
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np


# ─── Individual Stage Functions ──────────────────────────────────────────────

def load_image(image_path: str) -> np.ndarray:
    """Load an image from disk.  Returns a BGR numpy array."""
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    return img


def to_grayscale(img: np.ndarray) -> np.ndarray:
    """Convert a BGR or BGRA image to grayscale.
    If the image is already single-channel, it is returned as-is."""
    if len(img.shape) == 2:
        return img  # already grayscale
    if img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def enhance_contrast_clahe(img: np.ndarray, clip_limit: float = 2.5, tile_size: int = 8) -> np.ndarray:
    """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to enhance faint handwriting strokes."""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    if len(img.shape) == 3:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    else:
        return clahe.apply(img)


def denoise_light(gray: np.ndarray, ksize: int = 3) -> np.ndarray:
    """Light single-pass Gaussian blur — preserves stroke detail.

    Used in the TrOCR path. Removes salt-and-pepper scan noise without
    blurring character edges.

    Parameters
    ----------
    gray  : Grayscale image
    ksize : Gaussian kernel size (must be odd). 3 = very mild.
            Increase to 5 only for extremely noisy scans.
    """
    k = ksize if ksize % 2 == 1 else ksize + 1
    return cv2.GaussianBlur(gray, (k, k), 0)


def denoise(gray: np.ndarray,
            gaussian_ksize: int = 3,
            nlm_h: float = 10.0,
            nlm_template_size: int = 7,
            nlm_search_size: int = 21) -> np.ndarray:
    """Two-pass denoising (for CNN/binary path only).

    Gaussian + Non-Local Means.  NLM smooths homogeneous regions aggressively,
    which helps binary thresholding produce clean edges on printed text.
    DO NOT use on TrOCR-bound images — NLM blurs thin handwriting strokes.

    Parameters
    ----------
    gray              : Grayscale image
    gaussian_ksize    : Kernel size for Gaussian blur (must be odd)
    nlm_h             : Filter strength for NLM (higher = more smoothing)
    nlm_template_size : Template patch size for NLM
    nlm_search_size   : Search window size for NLM
    """
    ksize = gaussian_ksize if gaussian_ksize % 2 == 1 else gaussian_ksize + 1
    blurred = cv2.GaussianBlur(gray, (ksize, ksize), 0)
    denoised = cv2.fastNlMeansDenoising(
        blurred,
        h=nlm_h,
        templateWindowSize=nlm_template_size,
        searchWindowSize=nlm_search_size,
    )
    return denoised


def deskew(gray: np.ndarray, angle_threshold: float = 45.0) -> np.ndarray:
    """Correct skew in a grayscale prescription image.

    Strategy:
    - Detect edges with Canny
    - Find lines with Hough Transform
    - Compute median angle across detected lines
    - Rotate image to correct that angle

    Parameters
    ----------
    gray            : Grayscale (possibly denoised) image
    angle_threshold : Ignore angles beyond ±this value (avoids 90° misreads)
    """
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)

    if lines is None or len(lines) == 0:
        return gray

    angles = []
    for line in lines:
        rho, theta = line[0]
        angle = np.degrees(theta) - 90.0
        if abs(angle) <= angle_threshold:
            angles.append(angle)

    if not angles:
        return gray

    skew_angle = float(np.median(angles))

    if abs(skew_angle) < 0.3 or abs(skew_angle) > angle_threshold:
        return gray

    h, w = gray.shape
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, skew_angle, 1.0)
    rotated = cv2.warpAffine(
        gray, M, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return rotated


def binarize(gray: np.ndarray,
             block_size: int = 31,
             C: int = 10) -> np.ndarray:
    """Adaptive Gaussian thresholding → clean black-on-white binary image.

    For the CNN / character segmentation path ONLY.  Not recommended for
    the TrOCR path — see module docstring for the full explanation.

    Parameters
    ----------
    gray       : Grayscale image (denoised, deskewed)
    block_size : Neighbourhood size for adaptive threshold (must be odd, >1)
    C          : Constant subtracted from mean (controls sensitivity).
                 Lower C (e.g. 5–7) preserves more thin strokes.
                 Higher C (e.g. 12–15) gives cleaner background but risks
                 dropping faint strokes.
    """
    bs = block_size if block_size % 2 == 1 else block_size + 1
    bs = max(bs, 3)
    return cv2.adaptiveThreshold(
        gray,
        maxValue=255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY,
        blockSize=bs,
        C=C,
    )


# ─── Main Entry Points ────────────────────────────────────────────────────────

def clean_image(image_path: str,
                output_path: str | None = None,
                binary: bool = False,
                clahe: bool = True,
                verbose: bool = False,
                debug_side_by_side: str | None = None) -> np.ndarray:
    """Full cleanup pipeline for prescription images.

    TrOCR path (binary=False, default)
    -----------------------------------
    load → grayscale → CLAHE enhancement → light_denoise → deskew → return grayscale

    CNN/segmentation path (binary=True)
    ------------------------------------
    load → grayscale → CLAHE → denoise (NLM) → deskew → binarize → return binary

    Parameters
    ----------
    image_path          : Path to the raw prescription image
    output_path         : If provided, saves the result image here
    binary              : If True, run the CNN/binarization path.
                          If False (default), run the light TrOCR path.
    clahe               : If True, apply CLAHE contrast equalization for faint handwriting
    verbose             : Print progress messages
    debug_side_by_side  : If provided, save an original vs processed comparison
                          to this path.

    Returns
    -------
    np.ndarray : Cleaned image. Grayscale (H, W) for TrOCR path,
                 binary (H, W) for CNN path.
    """
    def _log(msg):
        if verbose:
            print(f"  [image_cleanup] {msg}")

    _log(f"Loading: {image_path}  (mode={'binary/CNN' if binary else 'grayscale/TrOCR'})")
    img = load_image(image_path)
    original_gray = to_grayscale(img)   # kept for side-by-side debug

    _log("Grayscale …")
    gray = original_gray.copy()

    if clahe:
        _log("CLAHE contrast enhancement …")
        gray = enhance_contrast_clahe(gray)

    if binary:
        # ── CNN path: aggressive cleanup ──────────────────────────────────────
        _log("Denoising (Gaussian + NLM) …")
        denoised = denoise(gray)
        _log("Deskewing …")
        deskewed = deskew(denoised)
        _log("Binarizing (adaptive threshold) …")
        result = binarize(deskewed)
    else:
        # ── TrOCR path: light-touch — preserve stroke detail ──────────────────
        _log("Light denoise (Gaussian only) …")
        denoised = denoise_light(gray, ksize=3)
        _log("Deskewing …")
        result = deskew(denoised)

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        cv2.imwrite(str(output_path), result)
        _log(f"Saved → {output_path}")

    if debug_side_by_side:
        side = _make_side_by_side(
            original_gray, result,
            label_l="Original",
            label_r="Preprocessed (binary)" if binary else "Preprocessed (gray)",
        )
        os.makedirs(os.path.dirname(os.path.abspath(debug_side_by_side)), exist_ok=True)
        cv2.imwrite(str(debug_side_by_side), side)
        _log(f"Side-by-side debug image saved → {debug_side_by_side}")

    return result


def _make_side_by_side(left: np.ndarray, right: np.ndarray,
                        label_l: str = "Before", label_r: str = "After") -> np.ndarray:
    """Stack two same-height grayscale images side-by-side with labels."""
    h = max(left.shape[0], right.shape[0])

    def _resize_h(im, target_h):
        scale = target_h / im.shape[0]
        return cv2.resize(im, (int(im.shape[1] * scale), target_h),
                          interpolation=cv2.INTER_AREA)

    l = _resize_h(left, h)
    r = _resize_h(right, h)
    divider = np.full((h, 4), 128, dtype=np.uint8)
    combined = np.hstack([l, divider, r])
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(combined, label_l, (10, 28), font, 0.8, 0, 2)
    cv2.putText(combined, label_r, (l.shape[1] + 14, 28), font, 0.8, 0, 2)
    return combined


# ─── CLI Entry Point ──────────────────────────────────────────────────────────

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Clean a raw prescription image (deskew/denoise/optionally binarize).",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--input", "-i", required=True,
        help="Path to the raw input image (JPG/PNG/TIFF …)"
    )
    parser.add_argument(
        "--output", "-o", default=None,
        help="Path to save the cleaned image. "
             "Defaults to <input_stem>_cleaned.png in the same folder."
    )
    parser.add_argument(
        "--binary", action="store_true",
        help="Run the CNN/binarization path instead of the light TrOCR path."
    )
    parser.add_argument(
        "--show", action="store_true",
        help="Display original vs cleaned images side-by-side (requires a display)."
    )
    parser.add_argument(
        "--no-verbose", action="store_true",
        help="Suppress progress messages."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    verbose = not args.no_verbose

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    output_path = args.output or str(
        input_path.parent / f"{input_path.stem}_cleaned.png"
    )

    mode = "binary/CNN" if args.binary else "grayscale/TrOCR"
    print(f"\n{'='*55}")
    print(f"  Smart Prescription — Image Cleanup ({mode})")
    print(f"{'='*55}")
    print(f"  Input : {input_path}")
    print(f"  Output: {output_path}")
    print(f"{'='*55}\n")

    debug_sbs = None
    if args.show:
        debug_sbs = str(input_path.parent / f"{input_path.stem}_side_by_side.png")

    cleaned = clean_image(
        str(input_path),
        output_path=output_path,
        binary=args.binary,
        verbose=verbose,
        debug_side_by_side=debug_sbs,
    )

    print(f"\n  ✓ Done. Cleaned image saved to: {output_path}")
    print(f"  Image shape: {cleaned.shape}  (H × W)")

    if args.show and debug_sbs:
        print(f"  Side-by-side comparison saved to: {debug_sbs}")
        sbs = cv2.imread(debug_sbs)
        if sbs is not None:
            cv2.imshow("Prescription Cleanup", sbs)
            print("  Press any key to close the preview window …")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
