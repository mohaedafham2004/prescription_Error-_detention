"""
scripts/test_full_pipeline.py
==============================
Quick end-to-end smoke test for the full prescription analysis pipeline.

Usage
-----
    # Uses the first image found in data/raw/prescriptions/
    python scripts/test_full_pipeline.py

    # Test on a specific image:
    python scripts/test_full_pipeline.py --image data/raw/prescriptions/rx1.jpg

    # Verbose mode (shows per-stage timing):
    python scripts/test_full_pipeline.py --image rx1.jpg --verbose

What a PASSING run looks like
------------------------------
    ✅ OCR          : 3 lines extracted, mean confidence 0.72
    ✅ NER          : MEDICINE, DOSAGE, FREQUENCY found
    ✅ Error check  : 2 issue(s) flagged
    Full text preview:
        Amoxicillin 500mg twice daily for 7 days
        ...

What a FAILING run looks like
------------------------------
    ❌ error: "OCR stage failed: ..."   → check config.yaml paths
    ❌ NER not available                → train NER model first (ok for now)
    ❌ All entities empty               → OCR produced no text (check image)
"""

import argparse
import json
import os
import sys
from pathlib import Path

# ── Project root ──────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Ensure Windows stdout handles UTF-8 emojis
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.pipeline.full_pipeline import run_full_pipeline
from src.utils.config_loader import load_config


# ── ANSI helpers ─────────────────────────────────────────────────────────────
_G = "\033[92m"; _Y = "\033[93m"; _R = "\033[91m"; _B = "\033[94m"
_BOLD = "\033[1m"; _RESET = "\033[0m"

SEV_COLOR = {"HIGH": _R, "MEDIUM": _Y, "LOW": _B, "INFO": "\033[90m"}


def _find_sample_image() -> str | None:
    for pattern in ["data/raw/prescriptions/*.jpg",
                    "data/raw/prescriptions/*.png",
                    "data/words_lines/images/*.png"]:
        matches = sorted((_ROOT / pattern.split("/")[0]).glob(
            "/".join(pattern.split("/")[1:])
        ))
        if matches:
            return str(matches[0])
    return None


def main():
    parser = argparse.ArgumentParser(description="End-to-end pipeline smoke test.")
    parser.add_argument("--image",   "-i", default=None,
                        help="Path to a prescription image. "
                             "Auto-detected from data/ if not provided.")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print per-stage timing.")
    parser.add_argument("--json",    action="store_true",
                        help="Dump the full result dict as JSON.")
    parser.add_argument("--debug",   "-d", action="store_true",
                        help=(
                            "Save intermediate pipeline images to debug_output/:\n"
                            "  01_original.png, 02_preprocessed.png,\n"
                            "  02b_side_by_side.png, 02c_line_bands_overlay.png,\n"
                            "  03_line_00.png … (one per segmented line)\n"
                            "Use these to confirm preprocessing and segmentation quality."
                        ))
    parser.add_argument("--debug-dir", default="debug_output",
                        help="Output folder for debug images (default: debug_output/).")
    args = parser.parse_args()

    # ── Find image ────────────────────────────────────────────────────────────
    image_path = args.image or _find_sample_image()
    if not image_path:
        print(f"{_R}❌ No image found. Provide one with --image or place a JPG/PNG "
              f"in data/raw/prescriptions/{_RESET}")
        sys.exit(1)

    if not Path(image_path).exists():
        print(f"{_R}❌ Image not found: {image_path}{_RESET}")
        sys.exit(1)

    # ── Show config ───────────────────────────────────────────────────────────
    cfg = load_config()
    print(f"\n{_BOLD}{'='*62}{_RESET}")
    print(f"{_BOLD}  Smart Prescription — Pipeline Smoke Test{_RESET}")
    print(f"{'='*62}")
    print(f"  Image           : {image_path}")
    print(f"  Active OCR      : {cfg.get('active_ocr_model', 'trocr')}")
    print(f"  TrOCR path      : {cfg.get('trocr_model_path')}")
    print(f"  Use pretrained  : {cfg.get('trocr_use_pretrained')}")
    print(f"  Active NER      : {cfg.get('active_ner_model', 'spacy')}")
    print(f"  NER model path  : {cfg.get('ner_model_path')}")
    print(f"  NER enabled     : {cfg.get('ner_enabled')}")
    if args.debug:
        print(f"  Debug output    : {os.path.abspath(args.debug_dir)}/")
    print(f"{'='*62}\n")

    # ── Run pipeline ──────────────────────────────────────────────────────────
    debug_dir = args.debug_dir if args.debug else None
    print("  Running full pipeline …")
    result = run_full_pipeline(
        image_path,
        verbose=args.verbose,
        debug_dir=debug_dir,
    )


    # ── Top-level error check ─────────────────────────────────────────────────
    if result.get("error"):
        print(f"\n  {_R}❌ Pipeline error: {result['error']}{_RESET}")
        print(f"\n  Possible fixes:")
        print(f"    • Check config.yaml — is trocr_model_path correct?")
        print(f"    • If model not downloaded yet, set trocr_use_pretrained: true")
        sys.exit(1)

    # ── Results ───────────────────────────────────────────────────────────────
    print(f"\n{_BOLD}{'─'*62}{_RESET}")
    print(f"{_BOLD}  Results{_RESET}")
    print(f"{'─'*62}")

    # OCR
    n    = result["n_lines"]
    conf = result["mean_ocr_confidence"]
    low  = result["low_confidence_lines"]
    ocr_ok = n > 0
    icon = f"{_G}✅{_RESET}" if ocr_ok else f"{_R}❌{_RESET}"
    print(f"  {icon} OCR   : {n} line(s), mean conf={conf:.2f}"
          + (f", low-conf lines: {low}" if low else ""))

    # NER
    ner_ok = result["ner_available"]
    icon = f"{_G}✅{_RESET}" if ner_ok else f"{_Y}⚠ {_RESET}"
    ents = result["entities"]
    found = [lbl for lbl, vals in ents.items() if vals]
    missing = result.get("missing_fields",
                [lbl for lbl, vals in ents.items() if not vals])
    print(f"  {icon} NER   : "
          + (f"found {found}" if ner_ok else "not available (NER model not trained yet)"))
    if result.get("ner_error"):
        print(f"       ℹ  {result['ner_error']}")

    # Issues
    issues = result["issues"]
    high   = sum(1 for i in issues if i["severity"] == "HIGH")
    med    = sum(1 for i in issues if i["severity"] == "MEDIUM")
    icon = f"{_G}✅{_RESET}"
    print(f"  {icon} Errors: {len(issues)} issue(s) flagged "
          f"({_R}{high} HIGH{_RESET}, {_Y}{med} MEDIUM{_RESET})")

    # Timing
    print(f"\n  ⏱  Total time: {result['total_time_s']}s")

    # ── Extracted text ────────────────────────────────────────────────────────
    print(f"\n{_BOLD}  Extracted Text{_RESET}")
    print(f"  {'─'*54}")
    text = result["extracted_text"] or "(empty)"
    for line in text.split("\n"):
        print(f"  {line}")

    # ── Entities ──────────────────────────────────────────────────────────────
    if ner_ok:
        print(f"\n{_BOLD}  Structured Entities{_RESET}")
        print(f"  {'─'*54}")
        for lbl, vals in result["entities"].items():
            val_str = ", ".join(f'"{v}"' for v in vals) if vals else "(not found)"
            print(f"  {lbl:<12}: {val_str}")

    # ── Issues ────────────────────────────────────────────────────────────────
    if issues:
        print(f"\n{_BOLD}  Flagged Issues{_RESET}")
        print(f"  {'─'*54}")
        for iss in issues:
            clr = SEV_COLOR.get(iss["severity"], "")
            print(f"  {clr}[{iss['severity']:6}]{_RESET}  {iss['error_type']:20}  "
                  f"{iss['field']}: \"{iss['value']}\"")
            print(f"           {iss['message']}")
            if iss.get("suggestion"):
                print(f"           → {iss['suggestion']}")

    # ── Final verdict ─────────────────────────────────────────────────────────
    print(f"\n{'='*62}")
    if ocr_ok and result["extracted_text"].strip():
        print(f"  {_G}{_BOLD}✅  End-to-end pipeline working.{_RESET}")
        if not ner_ok:
            print(f"  {_Y}⚠  NER not yet available — train it to get entity extraction.{_RESET}")
    else:
        print(f"  {_R}❌  Pipeline ran but produced no text. Check the image.{_RESET}")
    print(f"{'='*62}\n")

    # ── Optional JSON dump ────────────────────────────────────────────────────
    if args.json:
        print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
