"""
src/preprocessing/segment_characters.py
=========================================
Splits a cleaned prescription image (or a line crop) into individual character
crops and optionally lets you label each crop interactively to build the CNN
training dataset.

Two modes
---------
1. AUTO mode   (--mode auto)
   Segments characters and saves them all to a single unlabeled dump folder.
   Use this when you will label later with external tools.

2. MANUAL mode (--mode manual)  ← primary mode for building data/characters/
   Segments each character, shows you a zoomed preview in an OpenCV window,
   and asks you to type the correct label on the terminal.
   Each labeled crop is saved to:   data/characters/<label>/<uuid>.png
   Press ENTER with no label to SKIP a crop.
   Type 'q' to QUIT and save progress so far.

Algorithm
---------
1. Compute vertical projection profile (dark pixels per column).
2. Find column-gaps → character bounding columns.
3. For each character column band, find the tight row bounding box.
4. Crop, resize to a standard size, and save.

Usage (CLI)
-----------
    # Manual labeling — this is how you build data/characters/
    python -m src.preprocessing.segment_characters \
        --input data/raw/prescriptions/rx1_cleaned.png \
        --mode manual \
        --char-dir data/characters

    # Auto dump (no labeling)
    python -m src.preprocessing.segment_characters \
        --input data/words_lines/images/rx1_line_00.png \
        --mode auto \
        --output unlabeled_chars/

    # If your input is still raw, add --auto-clean
    python -m src.preprocessing.segment_characters \
        --input data/raw/prescriptions/rx1.jpg \
        --mode manual \
        --char-dir data/characters \
        --auto-clean
"""

import argparse
import os
import sys
import uuid
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

# Allow running as module from project root
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.preprocessing.image_cleanup import (
    clean_image, load_image, to_grayscale, binarize
)

# ─── Constants ────────────────────────────────────────────────────────────────

# Standard size every character crop is resized to before saving
CHAR_IMG_SIZE = (32, 32)   # (width, height)

# OpenCV preview window size (zoomed, so you can actually see the character)
PREVIEW_SIZE = (256, 256)


# ─── Projection-based Segmentation ───────────────────────────────────────────

def compute_col_projection(binary: np.ndarray) -> np.ndarray:
    """Vertical projection profile: dark pixels per column."""
    inverted = (binary == 0).astype(np.uint8)
    return inverted.sum(axis=0).astype(np.float32)


def compute_row_projection(binary: np.ndarray) -> np.ndarray:
    """Horizontal projection profile: dark pixels per row."""
    inverted = (binary == 0).astype(np.uint8)
    return inverted.sum(axis=1).astype(np.float32)


def find_char_column_bands(profile: np.ndarray,
                           min_gap_width: int = 2,
                           min_char_width: int = 4,
                           density_threshold: float = 1.0) -> List[Tuple[int, int]]:
    """Find column ranges that contain character strokes.

    Parameters
    ----------
    profile           : Vertical projection (dark pixels per column)
    min_gap_width     : Minimum blank columns to count as a gap between chars
    min_char_width    : Discard bands narrower than this (noise)
    density_threshold : Columns with ≤ this many dark pixels are 'empty'

    Returns
    -------
    bands : List of (col_start, col_end) — exclusive end.
    """
    in_char = profile > density_threshold
    bands = []
    start = None
    gap_count = 0

    for i, active in enumerate(in_char):
        if active:
            if start is None:
                start = i
            gap_count = 0
        else:
            if start is not None:
                gap_count += 1
                if gap_count >= min_gap_width:
                    end = i - gap_count + 1
                    if (end - start) >= min_char_width:
                        bands.append((start, end))
                    start = None
                    gap_count = 0

    if start is not None:
        end = len(profile)
        if (end - start) >= min_char_width:
            bands.append((start, end))

    return bands


def tight_row_bounds(binary: np.ndarray,
                     col_start: int,
                     col_end: int,
                     row_padding: int = 2) -> Tuple[int, int]:
    """Find the tight row bounds of dark pixels in a column slice."""
    col_slice = binary[:, col_start:col_end]
    row_profile = compute_row_projection(col_slice)
    nonzero_rows = np.where(row_profile > 0)[0]
    if len(nonzero_rows) == 0:
        return 0, binary.shape[0]
    r_start = max(0, nonzero_rows[0] - row_padding)
    r_end = min(binary.shape[0], nonzero_rows[-1] + 1 + row_padding)
    return r_start, r_end


def extract_character_crops(
        binary: np.ndarray,
        min_gap_width: int = 2,
        min_char_width: int = 4,
        density_threshold: float = 1.0,
        col_padding: int = 2,
        row_padding: int = 2,
) -> List[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
    """Extract individual character crops from a binarized line/image.

    Returns
    -------
    results : List of (crop_image, (col_start, col_end, row_start, row_end))
              where coordinates are in the original image space.
    """
    col_profile = compute_col_projection(binary)
    col_bands = find_char_column_bands(
        col_profile,
        min_gap_width=min_gap_width,
        min_char_width=min_char_width,
        density_threshold=density_threshold,
    )

    W = binary.shape[1]
    results = []
    for cs, ce in col_bands:
        cs_pad = max(0, cs - col_padding)
        ce_pad = min(W, ce + col_padding)
        rs, re = tight_row_bounds(binary, cs_pad, ce_pad, row_padding=row_padding)
        crop = binary[rs:re, cs_pad:ce_pad]
        if crop.size == 0:
            continue
        results.append((crop, (cs_pad, ce_pad, rs, re)))

    return results


def resize_for_cnn(crop: np.ndarray,
                   size: Tuple[int, int] = CHAR_IMG_SIZE) -> np.ndarray:
    """Resize a character crop to a fixed square size for CNN training."""
    resized = cv2.resize(crop, size, interpolation=cv2.INTER_AREA)
    return resized


# ─── Saving Helpers ───────────────────────────────────────────────────────────

def save_labeled(crop: np.ndarray, label: str, char_dir: str) -> str:
    """Save a labeled crop to  char_dir/<label>/<uuid>.png .

    Parameters
    ----------
    crop     : Character image (will be resized to CHAR_IMG_SIZE)
    label    : Single character label (e.g. 'A', 'b', '3')
    char_dir : Root data/characters/ directory

    Returns
    -------
    saved_path : Absolute path of the saved file.
    """
    label_dir = os.path.join(char_dir, label)
    os.makedirs(label_dir, exist_ok=True)
    resized = resize_for_cnn(crop)
    filename = f"{uuid.uuid4().hex}.png"
    out_path = os.path.join(label_dir, filename)
    cv2.imwrite(out_path, resized)
    return os.path.abspath(out_path)


def save_unlabeled(crop: np.ndarray, output_dir: str, index: int) -> str:
    """Save an unlabeled crop to output_dir/char_<NNN>.png ."""
    os.makedirs(output_dir, exist_ok=True)
    resized = resize_for_cnn(crop)
    filename = f"char_{index:04d}.png"
    out_path = os.path.join(output_dir, filename)
    cv2.imwrite(out_path, resized)
    return os.path.abspath(out_path)


# ─── Interactive Labeling Helper ──────────────────────────────────────────────

def _show_char_preview(crop: np.ndarray,
                       index: int,
                       total: int,
                       window_name: str = "Character Preview") -> None:
    """Show the character crop zoomed in an OpenCV window."""
    zoomed = cv2.resize(crop, PREVIEW_SIZE, interpolation=cv2.INTER_NEAREST)
    # Add a border and index label
    bordered = cv2.copyMakeBorder(
        zoomed, 40, 10, 10, 10,
        cv2.BORDER_CONSTANT, value=200
    )
    label_text = f"Char {index + 1} / {total}   (ENTER=skip, q=quit)"
    cv2.putText(bordered, label_text, (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, 0, 1)
    cv2.imshow(window_name, bordered)
    cv2.waitKey(1)  # refresh


def interactive_label_session(crops_and_coords, char_dir: str) -> dict:
    """Walk through each crop, show a preview, and ask for a label.

    Returns
    -------
    stats : {'labeled': int, 'skipped': int, 'saved_paths': list}
    """
    labeled, skipped = 0, 0
    saved_paths = []
    total = len(crops_and_coords)
    window_name = "Character Preview"

    print("\n" + "─" * 55)
    print("  MANUAL LABELING MODE")
    print("  ─ Type the correct character label and press ENTER")
    print("  ─ Press ENTER with no input to SKIP this crop")
    print("  ─ Type 'q' and press ENTER to quit and save progress")
    print("─" * 55 + "\n")

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, PREVIEW_SIZE[0] + 20, PREVIEW_SIZE[1] + 60)

    for idx, (crop, coords) in enumerate(crops_and_coords):
        _show_char_preview(crop, idx, total, window_name)

        try:
            user_input = input(f"  [{idx + 1}/{total}] Label (col {coords[0]}–{coords[1]}): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Session interrupted — saving progress.")
            break

        if user_input.lower() == 'q':
            print("  Quit signal received — saving progress.")
            break
        elif user_input == '':
            print("  → Skipped.")
            skipped += 1
        elif len(user_input) == 1:
            path = save_labeled(crop, user_input, char_dir)
            saved_paths.append(path)
            labeled += 1
            print(f"  → Saved as label '{user_input}': {path}")
        else:
            print(f"  ⚠ Label must be a SINGLE character. Got '{user_input}' — skipping.")
            skipped += 1

    cv2.destroyAllWindows()

    print(f"\n  Session complete: {labeled} labeled, {skipped} skipped.")
    return {"labeled": labeled, "skipped": skipped, "saved_paths": saved_paths}


# ─── Auto Mode ────────────────────────────────────────────────────────────────

def auto_segment(binary: np.ndarray, output_dir: str,
                 verbose: bool = True) -> List[str]:
    """Segment all characters and dump them as unlabeled crops."""
    crops_and_coords = extract_character_crops(binary)
    saved = []
    for idx, (crop, _) in enumerate(crops_and_coords):
        path = save_unlabeled(crop, output_dir, idx)
        saved.append(path)
        if verbose:
            print(f"  [segment_characters] Saved unlabeled char {idx:04d}: {path}")
    return saved


# ─── Top-level Entry Point ────────────────────────────────────────────────────

def segment_characters(image_path: str,
                       mode: str = "manual",
                       char_dir: str = "data/characters",
                       output_dir: str = "unlabeled_chars",
                       auto_clean: bool = False,
                       verbose: bool = True) -> dict:
    """Segment characters from a prescription image.

    Parameters
    ----------
    image_path : Path to the input image (cleaned or raw)
    mode       : 'manual' (interactive labeling) or 'auto' (dump unlabeled)
    char_dir   : Root directory for labeled crops  (used in manual mode)
    output_dir : Output dir for unlabeled dumps  (used in auto mode)
    auto_clean : Run image_cleanup pipeline before segmenting
    verbose    : Print progress

    Returns
    -------
    result dict with keys: 'crops_found', 'labeled'/'saved', 'paths'
    """
    def _log(msg):
        if verbose:
            print(f"  [segment_characters] {msg}")

    if auto_clean:
        _log("Running image cleanup …")
        binary = clean_image(image_path, verbose=verbose)
    else:
        _log(f"Loading image: {image_path}")
        img = load_image(image_path)
        gray = to_grayscale(img)
        unique = np.unique(gray)
        binary = binarize(gray) if len(unique) > 2 else gray

    _log("Extracting character crops …")
    crops_and_coords = extract_character_crops(binary)
    _log(f"Found {len(crops_and_coords)} potential character(s).")

    if not crops_and_coords:
        print("  WARNING: No characters detected. "
              "Try --auto-clean or adjusting segmentation parameters.",
              file=sys.stderr)
        return {"crops_found": 0, "labeled": 0, "paths": []}

    if mode == "manual":
        stats = interactive_label_session(crops_and_coords, char_dir)
        return {
            "crops_found": len(crops_and_coords),
            "labeled": stats["labeled"],
            "skipped": stats["skipped"],
            "paths": stats["saved_paths"],
        }
    else:  # auto
        saved = auto_segment(binary, output_dir, verbose=verbose)
        return {
            "crops_found": len(crops_and_coords),
            "saved": len(saved),
            "paths": saved,
        }


# ─── CLI Entry Point ──────────────────────────────────────────────────────────

def _parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Segment a prescription image into character crops.\n"
            "Use --mode manual to interactively label and build data/characters/."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--input", "-i", required=True,
                        help="Path to the input image.")
    parser.add_argument("--mode", "-m", choices=["manual", "auto"],
                        default="manual",
                        help="'manual' = interactive labeling (builds CNN dataset)\n"
                             "'auto'   = dump unlabeled crops to --output dir")
    parser.add_argument("--char-dir", default="data/characters",
                        help="Root directory for labeled CNN data  (manual mode only).\n"
                             "Default: data/characters")
    parser.add_argument("--output", "-o", default="unlabeled_chars",
                        help="Output directory for unlabeled dumps  (auto mode only).")
    parser.add_argument("--auto-clean", action="store_true",
                        help="Run image_cleanup pipeline on the input before segmenting.")
    parser.add_argument("--no-verbose", action="store_true",
                        help="Suppress progress messages.")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    verbose = not args.no_verbose

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    print(f"\n{'='*55}")
    print(f"  Smart Prescription — Character Segmentation")
    print(f"{'='*55}")
    print(f"  Input : {input_path}")
    print(f"  Mode  : {args.mode.upper()}")
    if args.mode == "manual":
        print(f"  Chars → {args.char_dir}/")
    else:
        print(f"  Dump  → {args.output}/")
    print(f"{'='*55}\n")

    result = segment_characters(
        image_path=str(input_path),
        mode=args.mode,
        char_dir=args.char_dir,
        output_dir=args.output,
        auto_clean=args.auto_clean,
        verbose=verbose,
    )

    print(f"\n  ✓ Crops found  : {result['crops_found']}")
    if args.mode == "manual":
        print(f"  ✓ Labeled      : {result.get('labeled', 0)}")
        print(f"  ✓ Skipped      : {result.get('skipped', 0)}")
    else:
        print(f"  ✓ Saved        : {result.get('saved', 0)}")
    print()
