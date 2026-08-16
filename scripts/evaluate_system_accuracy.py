"""
scripts/evaluate_system_accuracy.py

Runs run_full_pipeline() on every image in a folder and compares extracted
fields against a verified ground_truth.json. Reports per-field accuracy,
overall exact-match accuracy, and a per-image breakdown.

IMPORTANT: only use this against a ground_truth.json where every value was
verified by a person, not guessed/inferred by an AI. See the accompanying
guidance on building a genuinely verified holdout set.

Usage:
    python scripts/evaluate_system_accuracy.py \
        --images data/prescriptions_demo/holdout \
        --ground-truth data/prescriptions_demo/ground_truth.json \
        --label "HOLDOUT (report this number)"

Expected ground_truth.json format:
    {
      "rx_03.jpeg": {
        "medicine": "Flunarizine",
        "dosage": "10 mg",
        "frequency": "7 PM daily",
        "duration": "1 month"
      },
      ...
    }
"""

import argparse
import json
import re
import sys
from pathlib import Path

# ── Make sure the project root is importable ────────────────────────────────
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ── ADJUST THIS IMPORT if your project structure differs ────────────────────
try:
    from src.pipeline.full_pipeline import run_full_pipeline
except ImportError as e:
    print(f"Could not import run_full_pipeline: {e}")
    print("Edit the import at the top of this script to match your actual "
          "pipeline entry point.")
    sys.exit(1)

FIELDS = ["medicine", "dosage", "frequency", "duration"]
IMG_EXTS = (".png", ".jpg", ".jpeg")


def normalize(value: str) -> str:
    """Lowercase, strip whitespace, remove punctuation/extra spaces so minor
    formatting differences ('500 mg' vs '500mg') don't count as mismatches."""
    if not value:
        return ""
    value = str(value).lower().strip()
    value = re.sub(r"[^\w\s]", "", value)   # strip punctuation
    value = re.sub(r"\s+", " ", value)       # collapse whitespace
    value = value.replace(" mg", "mg").replace(" ml", "ml")  # unit spacing
    return value


def fields_match(predicted: str, expected: str) -> bool:
    norm_p = normalize(predicted)
    norm_e = normalize(expected)
    if norm_p == norm_e:
        return True
    # If multiple tokens, check if expected is contained or matches substring
    if norm_e and norm_e in norm_p:
        return True
    if norm_p and norm_p in norm_e:
        return True
    return False


def evaluate(images_dir: Path, ground_truth_path: Path, label: str) -> None:
    if not images_dir.exists():
        print(f"Images directory not found: {images_dir}")
        sys.exit(1)
    if not ground_truth_path.exists():
        print(f"Ground truth file not found: {ground_truth_path}")
        sys.exit(1)

    with open(ground_truth_path, "r", encoding="utf-8") as f:
        ground_truth = json.load(f)

    image_files = sorted(
        p for p in images_dir.iterdir() if p.suffix.lower() in IMG_EXTS
    )

    if not image_files:
        print(f"No images found in {images_dir}")
        sys.exit(1)

    field_correct = {f: 0 for f in FIELDS}
    field_total = {f: 0 for f in FIELDS}
    exact_match_count = 0
    evaluated_count = 0
    rows = []

    for img_path in image_files:
        fname = img_path.name
        if fname not in ground_truth:
            print(f"⚠ Skipping {fname} — no entry in ground_truth.json")
            continue

        expected = ground_truth[fname]

        # ── Pipeline Call ───────────────────────────────────────────────────
        try:
            result = run_full_pipeline(str(img_path))
        except Exception as e:
            print(f"❌ Pipeline failed on {fname}: {e}")
            continue

        # ── Extract Entities (handles both dict of lists and dict of strings) ──
        entities = result.get("entities", {})

        def _extract_field(field_name: str) -> str:
            # Check lowercase and uppercase keys
            val = entities.get(field_name) or entities.get(field_name.upper()) or entities.get(field_name.lower()) or ""
            if isinstance(val, (list, tuple, set)):
                return " ".join(str(x) for x in val if x)
            return str(val)

        predicted = {
            "medicine": _extract_field("medicine"),
            "dosage": _extract_field("dosage"),
            "frequency": _extract_field("frequency"),
            "duration": _extract_field("duration"),
        }

        row_matches = {}
        for field in FIELDS:
            exp_val = expected.get(field, "")
            pred_val = predicted.get(field, "")
            if not exp_val:
                continue  # skip fields not specified in ground truth
            field_total[field] += 1
            is_match = fields_match(pred_val, exp_val)
            row_matches[field] = is_match
            if is_match:
                field_correct[field] += 1

        all_correct = all(row_matches.values()) if row_matches else False
        if all_correct:
            exact_match_count += 1
        evaluated_count += 1

        rows.append({
            "file": fname,
            "expected": expected,
            "predicted": predicted,
            "matches": row_matches,
            "all_correct": all_correct,
        })

    # ── Report ────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"EVALUATION: {label}")
    print(f"Images dir: {images_dir}")
    print("=" * 70)

    print(f"\nEvaluated {evaluated_count} images\n")

    print("Per-field accuracy:")
    for field in FIELDS:
        total = field_total[field]
        correct = field_correct[field]
        pct = (correct / total * 100) if total else 0.0
        print(f"  {field:<12} {correct}/{total}  ({pct:.1f}%)")

    overall_pct = (exact_match_count / evaluated_count * 100) if evaluated_count else 0.0
    print(f"\nOverall exact-match accuracy (all fields correct): "
          f"{exact_match_count}/{evaluated_count}  ({overall_pct:.1f}%)")

    print("\nPer-image breakdown:")
    print("-" * 70)
    for row in rows:
        status = "✓ ALL CORRECT" if row["all_correct"] else "✗ mismatch(es)"
        print(f"\n{row['file']}  —  {status}")
        for field in FIELDS:
            if field not in row["matches"]:
                continue
            mark = "✓" if row["matches"][field] else "✗"
            print(f"    {mark} {field:<10} expected: {str(row['expected'].get(field, '')):<25} "
                  f"predicted: {row['predicted'].get(field, '')}")
    print("-" * 70)
    print(f"\n>>> Report this number for '{label}': {overall_pct:.1f}% overall, "
          f"per-field breakdown above <<<\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=Path, required=True,
                         help="Folder of prescription images to evaluate")
    parser.add_argument("--ground-truth", type=Path, required=True,
                         help="Path to ground_truth.json")
    parser.add_argument("--label", type=str, default="EVALUATION",
                         help="Label for this run, e.g. 'TUNING' or 'HOLDOUT'")
    args = parser.parse_args()

    evaluate(args.images, args.ground_truth, args.label)
