"""
src/pipeline/medicine_matcher.py
=================================
Dedicated medicine-name fuzzy-matching module.

Extracted from error_detection.py so it can be:
  - Unit-tested independently
  - Reused by any module that needs medicine name resolution
  - Extended (e.g. phonetic matching, brand→generic mapping) in one place

Design inspired by RxVision's text_extraction.py (Pjanhavi24/RxVision-ocr-prescription-reader),
adapted to our existing pipeline structure with the following changes:
  - No Tesseract / Google Vision / AWS Textract dependencies
  - Threshold raised to 80 (better precision for clinical context)
  - Normalisation expanded to handle common OCR confusions beyond just "mg" stripping
  - Returns a structured dict rather than a bare string for downstream use

Usage
-----
    from src.pipeline.medicine_matcher import match_medicine, normalize_medicine_text

    result = match_medicine(
        raw_text="amoxcillin",
        known_medicines=["Amoxicillin", "Amoxil", "Paracetamol"],
    )
    # {"matched_name": "Amoxicillin", "confidence": 91.3,
    #  "raw_ocr_text": "amoxcillin", "is_confident_match": True}

CLI self-test
-------------
    python -m src.pipeline.medicine_matcher
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import List, Optional

# ── Project root on path ──────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    from rapidfuzz import fuzz, process as fuzz_process
    _RAPIDFUZZ_AVAILABLE = True
except ImportError:
    _RAPIDFUZZ_AVAILABLE = False


# ── OCR confusion table ───────────────────────────────────────────────────────
# Common single-character substitutions produced by TrOCR / handwriting OCR.
# Applied BEFORE fuzzy matching so that "0" vs "o" confusion in medicine names
# does not tank the similarity score.
#
# Key insight from debugging our pipeline:
#   TrOCR sometimes outputs "Am0xicillin" (digit zero) for "Amoxicillin".
#   Without pre-normalisation, token_sort_ratio("Am0xicillin", "Amoxicillin") ≈ 73,
#   which falls below our 80 threshold and flags a false UNRECOGNISED error.
#   With normalisation the match scores ≥ 95.
_OCR_CONFUSION_PAIRS: list[tuple[str, str]] = [
    # Zero ↔ letter O  (very common in handwriting OCR)
    # Only applied when 0 appears between two letters (contextual).
    # "Am0xicillin" → "Amoxicillin", but "500" is left untouched.
    (r"(?<=[A-Za-z])0(?=[A-Za-z])", "o"),
    # NOTE: 1/l/I confusion is intentionally NOT substituted here.
    # '1' can mean 'l' (Fluconazole) or 'i' (Metformin) — guessing wrong
    # makes the fuzzy score WORSE.  WRatio handles this without substitution.
]

# Strip dosage suffixes that appear AFTER the medicine name.
# Anchored on whitespace + digit OR whitespace + known unit word, so that
# "Amoxicillin 500mg" → "Amoxicillin" but "mg" inside a name is untouched.
# The critical fix vs the old pattern: \b500\b doesn't match in "500mg"
# because 500 and mg share a word-character adjacency — no word boundary exists
# between them. Anchoring on \s+ before the digit solves this.
_SUFFIX_STRIP_PATTERN = re.compile(
    r"\s+\d[\w./]*.*$"  # " 500mg", " 10", " 2.5ml" etc.
    r"|\s+(?:mg|mcg|µg|ml|g|iu|tab(?:let)?s?|cap(?:sule)?s?"
    r"|syrup|suspension|ointment|cream|injection|inj|drops?|puffs?)\b.*$",
    re.I,
)


# ── Normalisation ─────────────────────────────────────────────────────────────

def normalize_medicine_text(raw: str) -> str:
    """Normalise a raw OCR medicine token for fuzzy matching.

    Steps
    -----
    1. Strip leading/trailing whitespace and punctuation
    2. Strip dosage suffixes ("500mg", "10ml", "tablet", etc.)
    3. Apply OCR confusion substitutions (0→o between letters, etc.)
    4. Lowercase and collapse internal whitespace

    The normalisation is intentionally conservative — only clear, context-
    sensitive substitutions are applied so we don't transform real medicine
    names into wrong ones.

    Parameters
    ----------
    raw : Raw string from OCR output or NER entity extraction.

    Returns
    -------
    Normalised string ready for fuzzy matching.
    """
    text = raw.strip().strip(".,;:!?\"'()")

    # Strip trailing dosage suffixes so "Amoxicillin 500mg" → "Amoxicillin"
    text = _SUFFIX_STRIP_PATTERN.sub("", text).strip()

    # Apply OCR confusion substitutions
    for pattern, replacement in _OCR_CONFUSION_PAIRS:
        text = re.sub(pattern, replacement, text)

    # Lowercase and normalise whitespace
    text = re.sub(r"\s+", " ", text).lower().strip()
    return text


# ── Core Matcher ──────────────────────────────────────────────────────────────

def match_medicine(
    raw_text: str,
    known_medicines: List[str],
    threshold: int = 80,
) -> dict:
    """Fuzzy-match a raw OCR medicine string against a known-medicines list.

    Normalisation is applied BEFORE fuzzy matching (order matters — see
    `normalize_medicine_text` for why).

    Parameters
    ----------
    raw_text        : Raw medicine string from OCR / NER output.
    known_medicines : List of canonical medicine name strings (from CSV).
    threshold       : Minimum RapidFuzz similarity score (0–100) to accept
                      a match as confident. Default 80 gives a good balance
                      between catching typos and avoiding false matches.
                      Lower for looser matching, raise for stricter.

    Returns
    -------
    dict with keys:
        raw_ocr_text      (str)  : Input as received (unmodified).
        normalised_text   (str)  : After apply normalize_medicine_text().
        matched_name      (str | None) : Canonical name if a match was found.
        confidence        (float): RapidFuzz similarity score 0.0–100.0.
                                   0.0 if no match found or rapidfuzz unavailable.
        is_exact_match    (bool) : True if normalised input == canonical (case-insensitive).
        is_confident_match(bool) : True if confidence >= threshold.
        match_type        (str)  : "exact" | "fuzzy" | "none"
    """
    result: dict = {
        "raw_ocr_text":       raw_text,
        "normalised_text":    "",
        "matched_name":       None,
        "confidence":         0.0,
        "is_exact_match":     False,
        "is_confident_match": False,
        "match_type":         "none",
    }

    if not raw_text or not raw_text.strip():
        return result

    if not _RAPIDFUZZ_AVAILABLE:
        result["matched_name"] = None
        return result

    normalised = normalize_medicine_text(raw_text)
    result["normalised_text"] = normalised

    if not normalised:
        return result

    # ── Exact match (case-insensitive) ────────────────────────────────────────
    known_lower_map = {n.lower(): n for n in known_medicines}
    if normalised in known_lower_map:
        canonical = known_lower_map[normalised]
        result.update({
            "matched_name":       canonical,
            "confidence":         100.0,
            "is_exact_match":     True,
            "is_confident_match": True,
            "match_type":         "exact",
        })
        return result

    # ── Fuzzy match ───────────────────────────────────────────────────────────
    # Bidirectional partial_ratio: max(partial_ratio(q→c), partial_ratio(c→q))
    #
    # Why bidirectional?
    # partial_ratio finds windows of len(shorter) inside len(longer).
    # When the query is longer (e.g. "ibuprophen" > "ibuprofen"), the useful
    # direction is partial_ratio(candidate, query) — i.e. look for the
    # 9-char candidate as a window inside the 10-char query.
    # Taking the max covers both insertion and deletion errors automatically.
    #
    # Why not process.extractOne with a custom scorer?
    # rapidfuzz's extractOne doesn't forward **kwargs to lambda scorers
    # correctly in all installed versions, causing scores to differ from the
    # direct fuzz.partial_ratio() call. Manual loop avoids this.
    best_name: str | None = None
    best_score: float = 0.0

    for candidate in known_medicines:
        cand_lower = candidate.lower()
        score = max(
            fuzz.partial_ratio(normalised, cand_lower),
            fuzz.partial_ratio(cand_lower, normalised),
        )
        if score > best_score:
            best_score = score
            best_name = candidate

    is_confident = best_score >= threshold

    result.update({
        "matched_name":       best_name if is_confident else None,
        "confidence":         float(best_score),
        "is_exact_match":     False,
        "is_confident_match": is_confident,
        "match_type":         "fuzzy" if is_confident else "none",
    })
    return result


# ── Convenience: load medicines from CSV ──────────────────────────────────────

def load_known_medicines(csv_path: str) -> List[str]:
    """Load the canonical medicine name list from known_medicines.csv.

    Returns flat list of names + aliases. Returns empty list if file not found.
    """
    import pandas as pd
    p = Path(csv_path)
    if not p.exists():
        print(f"  [medicine_matcher] WARNING: {csv_path} not found.", file=sys.stderr)
        return []
    df = pd.read_csv(p)
    names = list(df["name"].dropna().str.strip())
    for alias_str in df.get("aliases", pd.Series(dtype=str)).dropna():
        names.extend([a.strip() for a in str(alias_str).split(",") if a.strip()])
    return [n for n in names if n]


# ── Self-test / CLI demo ──────────────────────────────────────────────────────

if __name__ == "__main__":
    # Quick visual confirmation of normalisation + fuzzy matching on
    # realistic OCR typo inputs — Part C requirement.

    DEMO_MEDICINES = [
        "Amoxicillin", "Amoxil", "Paracetamol", "Ibuprofen", "Metformin",
        "Atorvastatin", "Lisinopril", "Omeprazole", "Azithromycin",
        "Cetirizine", "Doxycycline", "Fluoxetine", "Metronidazole",
        "Ciprofloxacin", "Salbutamol", "Prednisolone", "Diazepam",
    ]

    TEST_CASES = [
        # (input,              expected_match,   note)
        ("amoxcillin",         "Amoxicillin",    "classic OCR transposition"),
        ("paracetmol",         "Paracetamol",    "dropped letter"),
        ("Ibuprophen",         "Ibuprofen",      "extra 'h'"),
        ("metform1n",          "Metformin",      "digit-1 OCR confusion"),
        ("Am0xicillin 500mg",  "Amoxicillin",    "zero-vs-O + dosage suffix"),
        ("azithromycin",       "Azithromycin",   "exact (lowercased)"),
        ("ciprofioxacin",      "Ciprofloxacin",  "fi-ligature OCR error"),
        ("xyzgarbage",         None,             "no match expected"),
    ]

    print(f"\n{'='*70}")
    print("  Medicine Matcher — Self-Test")
    print(f"  Using {len(DEMO_MEDICINES)} reference medicines | threshold=80")
    print(f"{'='*70}")
    print(f"  {'INPUT':<25} {'NORM':<22} {'MATCH':<18} {'CONF':>6}  STATUS")
    print(f"  {'-'*67}")

    all_pass = True
    for raw, expected, note in TEST_CASES:
        res = match_medicine(raw, DEMO_MEDICINES, threshold=80)
        norm  = res["normalised_text"]
        match = res["matched_name"] or "(none)"
        conf  = res["confidence"]
        ok    = res["matched_name"] == expected
        icon  = "PASS" if ok else "FAIL"
        all_pass = all_pass and ok
        print(f"  {raw:<25} {norm:<22} {match:<18} {conf:>6.1f}  {icon}  {note}")

    print(f"{'='*70}")
    print(f"  {'ALL PASSED' if all_pass else 'SOME FAILED -- check output above'}")
    print(f"{'='*70}\n")
