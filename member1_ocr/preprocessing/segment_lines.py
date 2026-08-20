"""
src/preprocessing/segment_lines.py
====================================
Splits a cleaned (binarized) prescription image into individual line images
and saves each crop to an output folder.

Algorithm
---------
1. Compute a horizontal projection profile (sum of white pixels per row).
2. Find runs of rows with low pixel density → these are the gaps between lines.
3. Extract the bounding box of each text-bearing row band.
4. Optionally pad each crop by a few pixels for cleaner TrOCR context.

Usage (CLI)
-----------
    python -m src.preprocessing.segment_lines \
        --input data/raw/prescriptions/rx1_cleaned.png \
        --output data/words_lines/images/ \
        --prefix rx1

    # This will save rx1_line_00.png, rx1_line_01.png … in the output folder.

    # If you have a raw (uncleaned) image, use --auto-clean to run image_cleanup first:
    python -m src.preprocessing.segment_lines \
        --input data/raw/prescriptions/rx1.jpg \
        --output data/words_lines/images/ \
        --prefix rx1 \
        --auto-clean \
        --show
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

# Allow running as module from project root
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.preprocessing.image_cleanup import clean_image, load_image, to_grayscale, binarize, denoise, deskew


# ─── Core Segmentation Logic ──────────────────────────────────────────────────

def compute_row_projection(binary: np.ndarray) -> np.ndarray:
    """Horizontal projection profile.

    For a binarized image where text pixels are BLACK (0) and background is
    WHITE (255), we count dark pixels per row.

    Returns
    -------
    profile : 1-D array of shape (H,) with dark-pixel count per row.
    """
    # Dark pixels are 0 in a binarized image; invert so text pixels = 1
    inverted = (binary == 0).astype(np.uint8)
    profile = inverted.sum(axis=1)  # sum across columns
    return profile.astype(np.float32)


def find_line_bands(profile: np.ndarray,
                    min_gap_height: int = 5,
                    min_line_height: int = 8,
                    density_threshold: float = 1.0) -> List[Tuple[int, int]]:
    """Identify row-ranges that contain text lines.

    Parameters
    ----------
    profile            : Horizontal projection (dark pixels per row)
    min_gap_height     : Minimum number of consecutive empty rows to count as
                         a gap between lines.
    min_line_height    : Discard bands shorter than this (noise / stray marks).
    density_threshold  : Rows with fewer dark pixels than this are 'empty'.

    Returns
    -------
    bands : List of (row_start, row_end) tuples (inclusive, exclusive).
    """
    in_text = profile > density_threshold
    bands = []
    start = None
    gap_count = 0

    for i, text in enumerate(in_text):
        if text:
            if start is None:
                start = i
            gap_count = 0
        else:
            if start is not None:
                gap_count += 1
                if gap_count >= min_gap_height:
                    end = i - gap_count + 1
                    if (end - start) >= min_line_height:
                        bands.append((start, end))
                    start = None
                    gap_count = 0

    # Close an open band at the bottom of the image
    if start is not None:
        end = len(profile)
        if (end - start) >= min_line_height:
            bands.append((start, end))

    return bands


def crop_line_images(binary: np.ndarray,
                     bands: List[Tuple[int, int]],
                     padding: int = 4) -> List[np.ndarray]:
    """Crop each line band from the binary image with optional padding.

    Parameters
    ----------
    binary  : Cleaned binary image (H, W)
    bands   : List of (row_start, row_end) tuples from find_line_bands()
    padding : Extra rows to include above and below each band

    Returns
    -------
    crops : List of numpy arrays, one per line.
    """
    H = binary.shape[0]
    crops = []
    for start, end in bands:
        r_start = max(0, start - padding)
        r_end = min(H, end + padding)
        crop = binary[r_start:r_end, :]
        crops.append(crop)
    return crops


def segment_lines(image_path: str,
                  output_dir: str,
                  prefix: str = "line",
                  padding: int = 4,
                  min_gap_height: int = 5,
                  min_line_height: int = 8,
                  density_threshold: float = 1.0,
                  auto_clean: bool = False,
                  verbose: bool = True) -> List[str]:
    """Full pipeline: load image → (optionally clean) → segment → save crops.

    Parameters
    ----------
    image_path        : Path to the input image (cleaned or raw if auto_clean=True)
    output_dir        : Directory where line crops will be saved
    prefix            : Filename prefix for saved crops  (e.g. "rx1_line_00.png")
    padding           : Pixel padding added to each line crop
    min_gap_height    : Min blank rows between lines to split on
    min_line_height   : Discard bands shorter than this
    density_threshold : Dark-pixel count threshold for 'empty' rows
    auto_clean        : Run image_cleanup pipeline first if True
    verbose           : Print progress

    Returns
    -------
    saved_paths : List of absolute paths to saved line-crop images.
    """
    def _log(msg):
        if verbose:
            print(f"  [segment_lines] {msg}")

    os.makedirs(output_dir, exist_ok=True)

    if auto_clean:
        _log("Running image cleanup …")
        binary = clean_image(image_path, verbose=verbose)
    else:
        _log(f"Loading pre-cleaned image: {image_path}")
        img = load_image(image_path)
        gray = to_grayscale(img)
        # Lightweight check: if the image isn't binary yet, binarize it
        unique_vals = np.unique(gray)
        if len(unique_vals) > 2:
            _log("Image appears not binarized — running binarize() …")
            binary = binarize(gray)
        else:
            binary = gray

    _log("Computing row projection profile …")
    profile = compute_row_projection(binary)

    _log("Finding text line bands …")
    bands = find_line_bands(
        profile,
        min_gap_height=min_gap_height,
        min_line_height=min_line_height,
        density_threshold=density_threshold,
    )
    _log(f"Found {len(bands)} line(s).")

    if not bands:
        print("  WARNING: No text lines detected. "
              "Try adjusting --min-gap or --density-threshold.", file=sys.stderr)
        return []

    _log("Cropping line images …")
    crops = crop_line_images(binary, bands, padding=padding)

    saved_paths = []
    for idx, crop in enumerate(crops):
        filename = f"{prefix}_line_{idx:02d}.png"
        out_path = os.path.join(output_dir, filename)
        cv2.imwrite(out_path, crop)
        saved_paths.append(os.path.abspath(out_path))
        _log(f"  Saved: {filename}  (H={crop.shape[0]}, W={crop.shape[1]})")

    _log(f"Done — {len(saved_paths)} line crops saved to: {output_dir}")
    return saved_paths


# ─── Visualization Helper ─────────────────────────────────────────────────────

def visualize_segments(binary: np.ndarray,
                       bands: List[Tuple[int, int]],
                       window_title: str = "Line Segmentation") -> None:
    """Draw colored horizontal bands over the image and show it."""
    vis = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    colors = [(0, 200, 0), (0, 0, 200), (200, 0, 0),
              (0, 200, 200), (200, 200, 0), (200, 0, 200)]
    for i, (start, end) in enumerate(bands):
        color = colors[i % len(colors)]
        cv2.rectangle(vis, (0, start), (vis.shape[1], end), color, 2)
        cv2.putText(vis, f"L{i}", (5, start + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    cv2.imshow(window_title, vis)
    print("  Press any key to close the visualisation window …")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ─── CLI Entry Point ──────────────────────────────────────────────────────────

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Segment a prescription image into individual line crops."
    )
    parser.add_argument("--input", "-i", required=True,
                        help="Path to input image (cleaned binarized PNG preferred).")
    parser.add_argument("--output", "-o", required=True,
                        help="Output directory for line crop images.")
    parser.add_argument("--prefix", "-p", default="line",
                        help="Filename prefix for saved crops (default: 'line').")
    parser.add_argument("--padding", type=int, default=4,
                        help="Pixel padding around each line crop (default: 4).")
    parser.add_argument("--min-gap", type=int, default=5,
                        help="Min blank rows between lines to split on (default: 5).")
    parser.add_argument("--min-line-height", type=int, default=8,
                        help="Discard bands shorter than this in pixels (default: 8).")
    parser.add_argument("--density-threshold", type=float, default=1.0,
                        help="Dark-pixel count to consider a row 'not empty' (default: 1.0).")
    parser.add_argument("--auto-clean", action="store_true",
                        help="Run image_cleanup pipeline before segmenting.")
    parser.add_argument("--show", action="store_true",
                        help="Display segmentation visualisation (requires a display).")
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
    print(f"  Smart Prescription — Line Segmentation")
    print(f"{'='*55}")
    print(f"  Input : {input_path}")
    print(f"  Output: {args.output}")
    print(f"{'='*55}\n")

    # Run segmentation
    if args.auto_clean:
        binary = clean_image(str(input_path), verbose=verbose)
    else:
        img = load_image(str(input_path))
        gray = to_grayscale(img)
        unique = np.unique(gray)
        binary = binarize(gray) if len(unique) > 2 else gray

    profile = compute_row_projection(binary)
    bands = find_line_bands(
        profile,
        min_gap_height=args.min_gap,
        min_line_height=args.min_line_height,
        density_threshold=args.density_threshold,
    )

    if args.show and bands:
        visualize_segments(binary, bands)

    saved = segment_lines(
        image_path=str(input_path),
        output_dir=args.output,
        prefix=args.prefix,
        padding=args.padding,
        min_gap_height=args.min_gap,
        min_line_height=args.min_line_height,
        density_threshold=args.density_threshold,
        auto_clean=args.auto_clean,
        verbose=verbose,
    )

    print(f"\n  ✓ {len(saved)} line crop(s) saved to: {args.output}")
    for p in saved:
        print(f"    → {p}")
