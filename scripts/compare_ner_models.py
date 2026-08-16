"""
scripts/compare_ner_models.py
==============================
Side-by-side comparison of SpacyNERModel vs HFClinicalNERModel on your
prescription data.

Reads a few samples from data/ner/val.jsonl (or train.jsonl as fallback),
runs both models on the same texts, and prints a formatted comparison table.

Usage
-----
    python scripts/compare_ner_models.py
    python scripts/compare_ner_models.py --n 20
    python scripts/compare_ner_models.py --input data/ner/val.jsonl --n 10
    python scripts/compare_ner_models.py --text "Paracetamol 1g every 8h for 5 days"

Output
------
For each sample:
    TEXT      : the original prescription line
    SPACY     : entities extracted by the custom spaCy model
    HF NER    : entities extracted by Posos/ClinicalNER
    GROUND    : ground-truth labels (from the JSONL file, if available)

Purpose
-------
Eyeball which model performs better on YOUR actual prescription data
before changing active_ner_model in config.yaml.
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
from pathlib import Path
from typing import Dict, List, Optional

# ── Project root on path ──────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.models.ner_base import ENTITY_KEYS

# ── ANSI colors (graceful fallback on Windows without ANSI support) ───────────
try:
    import colorama
    colorama.init()
    _BLUE  = "\033[94m"
    _GREEN = "\033[92m"
    _AMBER = "\033[93m"
    _RESET = "\033[0m"
    _BOLD  = "\033[1m"
except ImportError:
    _BLUE = _GREEN = _AMBER = _RESET = _BOLD = ""


# ─── Data Loading ─────────────────────────────────────────────────────────────

def load_samples(path: str, n: int) -> List[dict]:
    """Load up to n samples from a JSONL file."""
    p = Path(path)
    if not p.exists():
        return []
    records = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
            if len(records) >= n:
                break
    return records


# ─── Model Loader ─────────────────────────────────────────────────────────────

def _load_spacy(cfg: dict):
    """Load SpacyNERModel, return None if not available."""
    try:
        from src.models.model_registry import get_ner_model
        spacy_cfg = dict(cfg)
        spacy_cfg["active_ner_model"] = "spacy"
        model = get_ner_model(spacy_cfg)
        # Trigger lazy load — catch model-not-found gracefully
        model.extract_entities("test")
        return model
    except FileNotFoundError as e:
        print(f"  [spaCy]  Model not found — {e}")
        return None
    except Exception as e:
        print(f"  [spaCy]  Load failed — {e}")
        return None


def _load_hf(cfg: dict):
    """Load HFClinicalNERModel, return None if unavailable."""
    try:
        from src.models.model_registry import get_ner_model
        hf_cfg = dict(cfg)
        hf_cfg["active_ner_model"] = "hf_clinical"
        model = get_ner_model(hf_cfg)
        # Trigger lazy load
        model.extract_entities("test")
        return model
    except Exception as e:
        print(f"  [HF NER] Load failed — {e}")
        return None


# ─── Formatting Helpers ───────────────────────────────────────────────────────

def _format_entities(entities: Dict[str, List[str]]) -> str:
    """Format entity dict as compact single line."""
    parts = []
    for key in ENTITY_KEYS:
        vals = entities.get(key, [])
        if vals:
            parts.append(f"{key[:3]}={', '.join(vals)}")
    return " | ".join(parts) if parts else "(none)"


def _format_ground_truth(record: dict) -> str:
    """Format ground truth entity spans from a JSONL record."""
    text = record.get("text", "")
    spans = record.get("entities", [])
    if not spans:
        return "(no annotations)"
    grouped: Dict[str, List[str]] = {k: [] for k in ENTITY_KEYS}
    for start, end, label in spans:
        if label in grouped:
            val = text[start:end].strip()
            if val not in grouped[label]:
                grouped[label].append(val)
    return _format_entities(grouped)


def _wrap(text: str, width: int = 55) -> str:
    return textwrap.fill(text, width=width)


# ─── Main Comparison Logic ────────────────────────────────────────────────────

def run_comparison(
    samples: List[dict],
    spacy_model,
    hf_model,
    extra_texts: Optional[List[str]] = None,
) -> None:
    """Run both models on all samples and print comparison table."""

    all_items = []
    for rec in samples:
        all_items.append({"text": rec.get("text", ""), "ground": rec})
    for t in (extra_texts or []):
        all_items.append({"text": t, "ground": None})

    sep = "-" * 80

    print(f"\n{_BOLD}{'='*80}{_RESET}")
    print(f"{_BOLD}  Smart Prescription — NER Model Comparison{_RESET}")
    print(f"{'='*80}")
    print(f"  spaCy  : {'loaded' if spacy_model else _AMBER + 'NOT AVAILABLE' + _RESET}")
    print(f"  HF NER : {'loaded' if hf_model else _AMBER + 'NOT AVAILABLE' + _RESET}")
    print(f"  Samples: {len(all_items)}")
    print(f"{'='*80}\n")

    for idx, item in enumerate(all_items, 1):
        text = item["text"].strip()
        ground = item["ground"]

        print(f"{_BOLD}[{idx}/{len(all_items)}]{_RESET}  {_wrap(text, 72)}")
        print(sep)

        # spaCy result
        if spacy_model:
            try:
                spacy_ents = spacy_model.extract_entities(text)
                print(f"  {_GREEN}spaCy   {_RESET}: {_format_entities(spacy_ents)}")
            except Exception as e:
                print(f"  {_GREEN}spaCy   {_RESET}: ERROR — {e}")
        else:
            print(f"  spaCy   : (not loaded)")

        # HF result
        if hf_model:
            try:
                hf_ents = hf_model.extract_entities(text)
                print(f"  {_BLUE}HF NER  {_RESET}: {_format_entities(hf_ents)}")
            except Exception as e:
                print(f"  {_BLUE}HF NER  {_RESET}: ERROR — {e}")
        else:
            print(f"  HF NER  : (not loaded)")

        # Ground truth (if from annotated file)
        if ground:
            print(f"  {_AMBER}GROUND  {_RESET}: {_format_ground_truth(ground)}")

        print()

    print(f"{'='*80}")
    print(f"  Done. Review results above, then update config.yaml:")
    print(f"    active_ner_model: \"spacy\"       # your custom-trained model")
    print(f"    active_ner_model: \"hf_clinical\"  # Posos/ClinicalNER")
    print(f"{'='*80}\n")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Side-by-side comparison of spaCy vs HF Clinical NER models.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--input", "-i",
        default=None,
        help=(
            "Path to a JSONL annotation file to sample from.\n"
            "Defaults to data/ner/val.jsonl, then data/ner/train.jsonl."
        ),
    )
    parser.add_argument(
        "--n", "-n",
        type=int,
        default=10,
        help="Number of samples to compare (default: 10).",
    )
    parser.add_argument(
        "--text", "-t",
        nargs="+",
        default=None,
        help="One or more prescription text strings to compare directly.",
    )
    parser.add_argument(
        "--skip-hf",
        action="store_true",
        help="Skip loading the HF model (faster; only shows spaCy output).",
    )
    parser.add_argument(
        "--skip-spacy",
        action="store_true",
        help="Skip loading the spaCy model (useful if not trained yet).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    # ── Load config ───────────────────────────────────────────────────────────
    from src.utils.config_loader import load_config
    cfg = load_config()

    # ── Load samples ──────────────────────────────────────────────────────────
    samples: List[dict] = []
    if args.text:
        # Direct text mode — no JSONL needed
        pass
    else:
        candidates = []
        if args.input:
            candidates = [args.input]
        else:
            candidates = [
                "data/ner/val.jsonl",
                "data/ner/train.jsonl",
            ]

        for path in candidates:
            samples = load_samples(path, args.n)
            if samples:
                print(f"  Loaded {len(samples)} samples from {path}")
                break

        if not samples and not args.text:
            print(
                "\n  No JSONL samples found. Run the NER labeler first:\n"
                "    python -m src.pipeline.ner_labeler\n\n"
                "  Or compare on raw text directly:\n"
                "    python scripts/compare_ner_models.py "
                '--text "Amoxicillin 500mg twice daily for 7 days"\n'
            )
            sys.exit(0)

    # ── Load models ───────────────────────────────────────────────────────────
    print("\n  Loading models...")
    spacy_model = None if args.skip_spacy else _load_spacy(cfg)
    hf_model    = None if args.skip_hf    else _load_hf(cfg)

    if spacy_model is None and hf_model is None:
        print("\n  Both models failed to load. Nothing to compare.")
        sys.exit(1)

    # ── Run comparison ────────────────────────────────────────────────────────
    run_comparison(
        samples=samples,
        spacy_model=spacy_model,
        hf_model=hf_model,
        extra_texts=args.text,
    )
