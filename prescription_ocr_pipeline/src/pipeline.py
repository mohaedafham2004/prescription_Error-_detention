"""
prescription_ocr_pipeline/src/pipeline.py
==========================================
Main orchestration script.

Usage:
    python prescription_ocr_pipeline/src/pipeline.py --input <image_or_dir>
    python prescription_ocr_pipeline/src/pipeline.py --input data/raw/prescriptions/ --evaluate
    python prescription_ocr_pipeline/src/pipeline.py --input data/raw/prescriptions/ --evaluate --report reports/

Arguments:
    --input    : Path to a single image file or a directory of images.
    --mode     : OCR mode — trocr | cnn | hybrid (default: hybrid)
    --evaluate : If set, compare against ground_truth.json and produce accuracy report.
    --gt       : Path to ground_truth.json (default: prescription_ocr_pipeline/data/ground_truth/ground_truth.json)
    --report   : Directory to save accuracy_report.json (default: reports/)
    --save-processed : If set, save intermediate preprocessing images to data/processed/
    --threshold: Fuzzy match threshold (default: 70)
    --verbose  : Print detailed progress.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# ── Paths ──────────────────────────────────────────────────────────────────
_PIPELINE_ROOT = Path(__file__).resolve().parents[1]
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
for p in [str(_PROJECT_ROOT), str(_PIPELINE_ROOT / "src")]:
    if p not in sys.path:
        sys.path.insert(0, p)

from preprocess import preprocess_image
from recognize import recognize_prescription
from postprocess import postprocess
from evaluate import evaluate_image, aggregate_report

# ── Supported image extensions ─────────────────────────────────────────────
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


def find_images(path: str) -> List[Path]:
    p = Path(path)
    if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
        return [p]
    if p.is_dir():
        imgs = sorted([f for f in p.iterdir() if f.suffix.lower() in IMAGE_EXTS])
        return imgs
    raise FileNotFoundError(f"No images found at: {path}")


def run_single(
    image_path: Path,
    mode: str,
    processed_dir: Optional[str],
    fuzzy_threshold: int,
    verbose: bool,
) -> Dict[str, Any]:
    """Full pipeline for one image. Returns postprocess() dict."""
    t0 = time.perf_counter()

    # 1. Preprocess
    binary = preprocess_image(
        str(image_path),
        save_processed_dir=processed_dir,
        verbose=verbose,
    )

    # 2. Recognise
    ocr_result = recognize_prescription(binary, mode=mode)

    if verbose:
        print(f"  OCR [{ocr_result['engine']}] conf={ocr_result['confidence']:.2f}: {ocr_result['text'][:80]!r}")

    # 3. Post-process
    structured = postprocess(ocr_result, fuzzy_threshold=fuzzy_threshold)
    structured["image"] = image_path.name
    structured["elapsed_s"] = round(time.perf_counter() - t0, 2)
    return structured


def main():
    parser = argparse.ArgumentParser(
        description="Prescription OCR Pipeline — preprocess + recognise + postprocess + evaluate"
    )
    parser.add_argument("--input", required=True, help="Image file or directory")
    parser.add_argument(
        "--mode", default="hybrid", choices=["trocr", "cnn", "hybrid"],
        help="OCR engine mode (default: hybrid)",
    )
    parser.add_argument(
        "--evaluate", action="store_true",
        help="Run accuracy evaluation against ground truth",
    )
    parser.add_argument(
        "--gt",
        default=str(_PIPELINE_ROOT / "data" / "ground_truth" / "ground_truth.json"),
        help="Path to ground_truth.json",
    )
    parser.add_argument(
        "--report",
        default=str(_PROJECT_ROOT / "reports"),
        help="Directory to save accuracy_report.json",
    )
    parser.add_argument(
        "--save-processed", action="store_true",
        help="Save intermediate preprocessed images",
    )
    parser.add_argument(
        "--threshold", type=int, default=70,
        help="Fuzzy match threshold 0-100 (default: 70)",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    images = find_images(args.input)
    print(f"\n[INFO]  Found {len(images)} image(s) to process.")

    processed_dir = (
        str(_PROJECT_ROOT / "data" / "processed")
        if args.save_processed else None
    )

    # ── Run pipeline ────────────────────────────────────────────────────────
    predictions: List[Dict[str, Any]] = []
    for img in images:
        print(f"\n-->  Processing: {img.name}")
        try:
            result = run_single(
                img, args.mode, processed_dir, args.threshold, args.verbose
            )
            predictions.append(result)
            print(f"   Medicines  : {result['medicines']}")
            print(f"   Dosage     : {result['dosage']}")
            print(f"   Frequency  : {result['frequency']}")
            print(f"   Duration   : {result['duration']}")
            print(f"   Engine     : {result['engine']}  (conf={result['confidence']:.2f})")
        except Exception as exc:
            print(f"   [ERROR]: {exc}")
            predictions.append({"image": img.name, "error": str(exc)})

    # ── Evaluate (optional) ─────────────────────────────────────────────────
    report: Dict[str, Any] = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "mode": args.mode,
        "fuzzy_threshold": args.threshold,
        "predictions": predictions,
    }

    if args.evaluate:
        print(f"\n[EVAL]  Loading ground truth from: {args.gt}")
        try:
            with open(args.gt, "r", encoding="utf-8") as f:
                gt_data: Dict[str, Any] = json.load(f)
        except FileNotFoundError:
            print("   [WARN] Ground truth file not found. Skipping evaluation.")
            gt_data = {}

        if gt_data:
            per_image: List[Dict[str, Any]] = []
            for pred in predictions:
                img_name = pred.get("image", "")
                if img_name in gt_data and "error" not in pred:
                    eval_result = evaluate_image(
                        pred,
                        gt_data[img_name],
                        fuzzy_threshold=args.threshold / 100.0,
                    )
                    per_image.append(eval_result)
                elif img_name not in gt_data:
                    print(f"   [INFO]  No ground truth for {img_name!r} - skipping eval.")

            summary = aggregate_report(per_image)

            report["per_image_eval"] = per_image
            report["aggregate"] = summary

            print("\n" + "=" * 60)
            print("[REPORT]  ACCURACY REPORT")
            print("=" * 60)
            print(f"  Images evaluated    : {summary.get('n_images', 0)}")
            print(f"  Medicine F1         : {summary.get('avg_medicine_f1', 0):.1%}")
            print(f"  Dosage accuracy     : {summary.get('avg_dosage_accuracy', 0):.1%}")
            print(f"  Frequency accuracy  : {summary.get('avg_frequency_accuracy', 0):.1%}")
            print(f"  Duration accuracy   : {summary.get('avg_duration_accuracy', 0):.1%}")
            print(f"  Overall field score : {summary.get('overall_field_score', 0):.1%}")
            print("=" * 60)

    # ── Save report ─────────────────────────────────────────────────────────
    os.makedirs(args.report, exist_ok=True)
    report_path = Path(args.report) / "accuracy_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n[DONE]  Report saved to: {report_path}")


if __name__ == "__main__":
    main()
