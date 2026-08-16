"""
src/models/ner_infer.py
========================
Inference wrapper for the trained custom NER model.

Loads the spaCy model from models/ner_model/ and extracts
MEDICINE, DOSAGE, FREQUENCY, and DURATION entities from any
prescription text string.

Implements the NERModel interface (src/models/ner_base.py) so it is
interchangeable with any other NER backend via model_registry.py.

Usage
-----
    from src.models.ner_infer import SpacyNERModel

    ner = SpacyNERModel()
    entities = ner.extract_entities("Amoxicillin 500mg twice daily for 7 days")
    # {"MEDICINE": ["Amoxicillin"], "DOSAGE": ["500mg"], ...}

    # Full result dict (with spans, missing, etc.):
    result = ner.extract("Amoxicillin 500mg twice daily for 7 days")

Backward compatibility
----------------------
    NERInferencer is kept as an alias for SpacyNERModel.

CLI
---
    python -m src.models.ner_infer --text "Paracetamol 1g every 8 hours for 5 days"
    python -m src.models.ner_infer --text "..." --model-dir models/ner_model
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

# ── Allow running from project root ──────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


from src.models.ner_base import NERModel, ENTITY_KEYS, empty_entities


# ─── Entity Type Definitions ──────────────────────────────────────────────────

ENTITY_LABELS = ENTITY_KEYS  # backwards-compat alias

# Severity level for each missing field in error detection
FIELD_SEVERITY = {
    "MEDICINE":  "HIGH",
    "DOSAGE":    "HIGH",
    "FREQUENCY": "MEDIUM",
    "DURATION":  "LOW",
}


# ─── SpaCy NER Model ──────────────────────────────────────────────────────────

class SpacyNERModel(NERModel):
    """Loads the trained custom NER model and extracts prescription entities.

    Parameters
    ----------
    model_dir : Path to the trained spaCy model (models/ner_model/).
                Must contain meta.json, config.cfg, etc.
    """

    def __init__(self, model_dir: str = "models/ner_model"):
        self.model_dir = Path(model_dir)
        self._nlp = None
        self._loaded = False

    # ── Lazy Loading ──────────────────────────────────────────────────────────

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return

        try:
            import spacy
        except ImportError:
            raise ImportError(
                "spaCy is not installed. Run: pip install spacy"
            )

        if not self.model_dir.exists():
            raise FileNotFoundError(
                f"NER model not found at: {self.model_dir}\n\n"
                "To fix this:\n"
                "  1. Label prescription texts:\n"
                "       python -m src.pipeline.ner_labeler\n"
                "  2. Train the NER model:\n"
                "       python -m src.models.train_ner\n"
            )

        print(f"  [NER] Loading model from: {self.model_dir} …")
        self._nlp = spacy.load(str(self.model_dir))
        self._loaded = True
        print(f"  [NER] Model loaded. Pipeline: {self._nlp.pipe_names}")

    # ── NERModel interface implementation ────────────────────────────────────

    @property
    def model_name(self) -> str:
        return "spacy"

    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract entities and return the canonical NERModel dict.

        Returns
        -------
        dict with keys MEDICINE, DOSAGE, FREQUENCY, DURATION.
        All keys always present; values are deduplicated text lists.
        """
        result = self.extract(text)
        return result["structured"]

    # ── Core Extraction ───────────────────────────────────────────────────────

    def extract(self, text: str) -> Dict:
        """Extract prescription entities from a text string.

        Parameters
        ----------
        text : Prescription text (one or more lines as a single string).

        Returns
        -------
        dict with keys:
            "text"       : Original input text
            "entities"   : List of entity dicts with text/label/start/end/line_index
            "structured" : {label: [text_values]} grouped by entity type
            "missing"    : List of entity labels not found in the text
        """
        self._ensure_loaded()

        doc = self._nlp(text)

        entities = []
        for ent in doc.ents:
            # Compute which line (0-indexed) the entity appears on
            line_idx = text[:ent.start_char].count("\n")
            entities.append({
                "text":       ent.text.strip(),
                "label":      ent.label_,
                "start":      ent.start_char,
                "end":        ent.end_char,
                "line_index": line_idx,
            })

        # Group by entity type
        structured: Dict[str, List[str]] = {label: [] for label in ENTITY_LABELS}
        for ent in entities:
            label = ent["label"]
            if label in structured:
                val = ent["text"]
                if val not in structured[label]:   # deduplicate
                    structured[label].append(val)

        # Find missing fields
        missing = [label for label in ENTITY_LABELS if not structured[label]]

        return {
            "text":       text,
            "entities":   entities,
            "structured": structured,
            "missing":    missing,
        }

    def extract_lines(self, lines: List[str]) -> Dict:
        """Extract entities from a list of prescription lines.

        Processes each line individually and merges results, with
        per-entity line_index tracking for the dashboard display.

        Parameters
        ----------
        lines : List of prescription line strings.

        Returns
        -------
        Merged result dict (same structure as extract()).
        """
        all_entities = []
        combined_structured: Dict[str, List[str]] = {l: [] for l in ENTITY_LABELS}

        for line_idx, line in enumerate(lines):
            if not line.strip():
                continue
            doc = self._nlp(line)
            for ent in doc.ents:
                entry = {
                    "text":       ent.text.strip(),
                    "label":      ent.label_,
                    "start":      ent.start_char,
                    "end":        ent.end_char,
                    "line_index": line_idx,
                    "source_line": line.strip(),
                }
                all_entities.append(entry)
                lbl = ent.label_
                if lbl in combined_structured:
                    val = ent.text.strip()
                    if val not in combined_structured[lbl]:
                        combined_structured[lbl].append(val)

        missing = [l for l in ENTITY_LABELS if not combined_structured[l]]

        return {
            "text":       "\n".join(lines),
            "entities":   all_entities,
            "structured": combined_structured,
            "missing":    missing,
        }

    def extract_from_ocr_result(self, ocr_result) -> Dict:
        """Convenience method: extract entities from an OCRResult object.

        Parameters
        ----------
        ocr_result : OCRResult from src.pipeline.ocr_pipeline

        Returns
        -------
        NER result dict with full structured entities.
        """
        lines = [lr.text for lr in ocr_result.lines if lr.text.strip()]
        if not lines:
            return {
                "text": "",
                "entities": [],
                "structured": {l: [] for l in ENTITY_LABELS},
                "missing": list(ENTITY_LABELS),
            }
        return self.extract_lines(lines)

    # ── Pretty Printing ───────────────────────────────────────────────────────

    @staticmethod
    def format_result(result: Dict, use_color: bool = True) -> str:
        """Format an extraction result as a readable string."""
        COLORS = {
            "MEDICINE":  "\033[92m",
            "DOSAGE":    "\033[94m",
            "FREQUENCY": "\033[93m",
            "DURATION":  "\033[95m",
        }
        RESET = "\033[0m"

        lines_out = []
        lines_out.append("─" * 50)
        lines_out.append("  NER Extraction Results")
        lines_out.append("─" * 50)

        struct = result["structured"]
        for label in ENTITY_LABELS:
            values = struct.get(label, [])
            color = COLORS.get(label, "") if use_color else ""
            reset = RESET if use_color else ""
            if values:
                lines_out.append(f"  {color}{label:<12}{reset}: {', '.join(values)}")
            else:
                lines_out.append(f"  {label:<12}: (not found)")

        if result.get("missing"):
            lines_out.append("")
            lines_out.append(f"  ⚠ Missing fields: {', '.join(result['missing'])}")

        lines_out.append("─" * 50)
        return "\n".join(lines_out)


# ── Backward-compatibility alias ─────────────────────────────────────────────
# Code that already imports NERInferencer continues to work unchanged.
NERInferencer = SpacyNERModel


# ─── CLI Entry Point ──────────────────────────────────────────────────────────

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Extract prescription entities from text using the trained NER model."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--text", "-t",
                       help="Prescription text string to extract entities from.")
    group.add_argument("--file", "-f",
                       help="Text file to process (one prescription per line or full text).")
    parser.add_argument("--model-dir", "-m", default="models/ner_model",
                        help="Path to the trained spaCy NER model dir.")
    parser.add_argument("--json-out",  "-o", default=None,
                        help="Save extraction result to this JSON file.")
    parser.add_argument("--no-color", action="store_true",
                        help="Disable ANSI color output.")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    ner = NERInferencer(model_dir=args.model_dir)

    # Get text to process
    if args.text:
        text = args.text
    else:
        fpath = Path(args.file)
        if not fpath.exists():
            print(f"ERROR: File not found: {fpath}", file=sys.stderr)
            sys.exit(1)
        with open(fpath, "r", encoding="utf-8") as f:
            text = f.read()

    print(f"\n{'='*55}")
    print(f"  NER Inference")
    print(f"{'='*55}")
    print(f"  Text: \"{text[:80]}{'...' if len(text) > 80 else ''}\"")
    print(f"{'='*55}\n")

    result = ner.extract(text)
    print(NERInferencer.format_result(result, use_color=not args.no_color))

    print("\n  Raw entity spans:")
    for ent in result["entities"]:
        print(f"    [{ent['start']:>3},{ent['end']:>3}]  "
              f"{ent['label']:<12}  \"{ent['text']}\"")

    if args.json_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.json_out)), exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\n  Result saved → {args.json_out}")
