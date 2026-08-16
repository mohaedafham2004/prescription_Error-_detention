"""
src/pipeline/error_detection.py
=================================
Rule-based + lightweight ML error detection for extracted prescription entities.

Given structured entities from the NER model, this module checks for:

1. MEDICINE NAME ERRORS
   - Fuzzy match against data/error_rules/known_medicines.csv
   - Flags likely misspellings (e.g. "Amoxicilin" → "Amoxicillin")
   - Flags completely unrecognised medicine names

2. DOSAGE RANGE ERRORS
   - Parses dosage strings (e.g. "500mg", "1g", "2 tablets")
   - Checks against data/error_rules/dosage_ranges.csv for matched medicine
   - Flags values outside the typical min/max range

3. FREQUENCY FORMAT ERRORS
   - Checks against a list of known frequency patterns (OD, BD, TDS, etc.)
   - Flags unparseable or highly unusual frequency strings

4. DRUG INTERACTION WARNINGS
   - Checks all MEDICINE pairs against data/error_rules/interactions.csv
   - Flags known interactions with severity level and clinical recommendations

5. THERAPEUTIC DUPLICATION WARNINGS
   - Checks all MEDICINEs against data/error_rules/therapeutic_classes.csv
   - Flags duplicate prescribing within the same pharmacological class (e.g. 2 NSAIDs or 2 PPIs)

6. MISSING FIELD WARNINGS
   - Flags if MEDICINE, DOSAGE, FREQUENCY, or DURATION are absent

All flags include:
  - error_type : category of the issue
  - severity   : HIGH / MEDIUM / LOW / INFO
  - message    : human-readable description
  - field      : which field triggered the flag
  - value      : the extracted value that was checked
  - suggestion : correction hint where available

⚠ DISCLAIMER:
All flags are "for professional review only" and do not constitute
clinical advice or diagnosis. Always defer to a licensed pharmacist
or physician.

Usage
-----
    from src.pipeline.error_detection import ErrorDetector

    detector = ErrorDetector()
    issues = detector.check(structured_entities)
    # structured_entities = {"MEDICINE": ["Amoxicilin"], "DOSAGE": ["500mg"], ...}

    for issue in issues:
        print(issue)

CLI
---
    python -m src.pipeline.error_detection \
        --medicine "Amoxicilin" --dosage "5000mg" \
        --frequency "twice daily" --duration "7 days"

    python -m src.pipeline.error_detection --json-file ner_result.json
"""

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set

import pandas as pd

# ── Allow running from project root ──────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Safe UTF-8 output on Windows terminals
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

try:
    from rapidfuzz import fuzz, process as fuzz_process
    _RAPIDFUZZ_AVAILABLE = True
except ImportError:
    _RAPIDFUZZ_AVAILABLE = False
    print("  WARNING: rapidfuzz not installed. Fuzzy matching disabled. "
          "Run: pip install rapidfuzz", file=sys.stderr)

# Medicine matching is now handled by the dedicated medicine_matcher module.
# This keeps the fuzzy logic testable independently and normalisation is applied
# BEFORE matching (order matters — see medicine_matcher.normalize_medicine_text).
try:
    from src.pipeline.medicine_matcher import match_medicine as _match_medicine, normalize_medicine_text as _normalize_medicine_text
    _MATCHER_AVAILABLE = True
except ImportError:
    _MATCHER_AVAILABLE = False
    print("  WARNING: medicine_matcher not available. Medicine name matching disabled.",
          file=sys.stderr)


# ─── Issue Dataclass ──────────────────────────────────────────────────────────

@dataclass
class PrescriptionIssue:
    """A single detected prescription error or warning."""
    error_type: str       # MISSPELLING | OUT_OF_RANGE | BAD_FREQUENCY |
                          # INTERACTION | DUPLICATE_THERAPY | MISSING_FIELD | UNRECOGNISED
    severity:   str       # HIGH | MEDIUM | LOW | INFO
    field:      str       # MEDICINE | DOSAGE | FREQUENCY | DURATION
    value:      str       # The extracted value that triggered the flag
    message:    str       # Human-readable description
    suggestion: Optional[str] = None   # Correction hint

    def __str__(self) -> str:
        sev_icons = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🔵", "INFO": "⚪"}
        icon = sev_icons.get(self.severity, "⚪")
        lines = [
            f"{icon} [{self.severity}] {self.error_type}",
            f"   Field     : {self.field}",
            f"   Value     : \"{self.value}\"",
            f"   Message   : {self.message}",
        ]
        if self.suggestion:
            lines.append(f"   Suggestion: {self.suggestion}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return asdict(self)


# ─── Known Frequency Patterns ─────────────────────────────────────────────────

# Compiled regex patterns for standard medical frequency strings
FREQUENCY_PATTERNS = [
    # Abbreviations
    re.compile(r'\b(od|once\s*daily|once\s*a\s*day|qd)\b',         re.I),
    re.compile(r'\b(bd|bid|twice\s*daily|twice\s*a\s*day)\b',       re.I),
    re.compile(r'\b(tds|tid|three\s*times\s*(a\s*)?daily)\b',       re.I),
    re.compile(r'\b(qds|qid|four\s*times\s*(a\s*)?daily)\b',        re.I),
    re.compile(r'\b(prn|as\s*needed|when\s*required|sos)\b',        re.I),
    re.compile(r'\b(hs|at\s*bedtime|at\s*night|nocte)\b',           re.I),
    re.compile(r'\b(stat|immediately|now)\b',                        re.I),
    re.compile(r'\b(ac|before\s*meals?|before\s*food)\b',           re.I),
    re.compile(r'\b(pc|after\s*meals?|after\s*food)\b',             re.I),
    re.compile(r'\b(with\s*meals?|with\s*food)\b',                  re.I),
    # "Every N hours"
    re.compile(r'\bevery\s*\d+\s*(hr?s?|hours?)\b',                 re.I),
    # "N times daily/a day"
    re.compile(r'\b\d+\s*times?\s*(a\s*)?(daily|day|week|month)\b', re.I),
    # "N-hourly"
    re.compile(r'\b\d+\s*-?\s*hourly\b',                            re.I),
    # Weekly / monthly
    re.compile(r'\b(weekly|monthly|fortnightly|alternate\s*days?)\b',re.I),
    # "morning and night", "morning and evening"
    re.compile(r'\b(morning\s*and\s*(night|evening|afternoon))\b',  re.I),
]

DOSE_PARSE_PATTERN = re.compile(
    r'(?P<value>[\d,]+(?:\.\d+)?)\s*'
    r'(?P<unit>mg|g|mcg|µg|ml|l|iu|units?|tabs?|tablets?|caps?|capsules?|tsp|tbsp|drops?|puffs?)',
    re.I,
)

UNIT_TO_MG = {
    "g":   1000.0,
    "mcg": 0.001,
    "µg":  0.001,
    "mg":  1.0,
}


# ─── ErrorDetector ────────────────────────────────────────────────────────────

class ErrorDetector:
    """Checks structured prescription entities against reference rules.

    Parameters
    ----------
    medicines_csv   : Path to data/error_rules/known_medicines.csv
    dosage_csv      : Path to data/error_rules/dosage_ranges.csv
    interactions_csv: Path to data/error_rules/interactions.csv
    classes_csv     : Path to data/error_rules/therapeutic_classes.csv
    fuzzy_threshold : Minimum rapidfuzz score to accept a spelling match (0–100)
    """

    def __init__(self,
                 medicines_csv:    str = "data/error_rules/known_medicines.csv",
                 dosage_csv:       str = "data/error_rules/dosage_ranges.csv",
                 interactions_csv: str = "data/error_rules/interactions.csv",
                 classes_csv:      str = "data/error_rules/therapeutic_classes.csv",
                 fuzzy_threshold:  int = 80):

        self.fuzzy_threshold = fuzzy_threshold
        self._medicines_df    = None
        self._dosage_df       = None
        self._interactions_df = None
        self._classes_df      = None
        self._classes_dict: Dict[str, dict] = {}
        self._known_names: List[str] = []   # flat list for fuzzy search

        self._load_reference_data(medicines_csv, dosage_csv, interactions_csv, classes_csv)

    # ── Reference Data Loading ────────────────────────────────────────────────

    def _load_reference_data(self, medicines_csv: str,
                              dosage_csv: str,
                              interactions_csv: str,
                              classes_csv: str) -> None:
        """Load all CSV reference files. Missing files emit warnings, not errors."""

        # Known medicines
        if os.path.exists(medicines_csv):
            self._medicines_df = pd.read_csv(medicines_csv)
            names = list(self._medicines_df["name"].dropna().str.strip())
            # Also include aliases
            for alias_str in self._medicines_df["aliases"].dropna():
                names.extend([a.strip() for a in str(alias_str).split(",") if a.strip()])
            self._known_names = [n for n in names if n]
        else:
            print(f"  WARNING: medicines CSV not found: {medicines_csv}. "
                  "Medicine checking disabled.", file=sys.stderr)

        # Dosage ranges
        if os.path.exists(dosage_csv):
            self._dosage_df = pd.read_csv(dosage_csv)
            # Normalise medicine names for lookup
            self._dosage_df["medicine_lower"] = (
                self._dosage_df["medicine"].str.strip().str.lower()
            )
        else:
            print(f"  WARNING: dosage CSV not found: {dosage_csv}. "
                  "Dosage range checking disabled.", file=sys.stderr)

        # Drug interactions
        if os.path.exists(interactions_csv):
            self._interactions_df = pd.read_csv(interactions_csv)
            self._interactions_df["drug_a_lower"] = (
                self._interactions_df["drug_a"].str.strip().str.lower()
            )
            self._interactions_df["drug_b_lower"] = (
                self._interactions_df["drug_b"].str.strip().str.lower()
            )
        else:
            print(f"  WARNING: interactions CSV not found: {interactions_csv}. "
                  "Interaction checking disabled.", file=sys.stderr)

        # Therapeutic classes
        if os.path.exists(classes_csv):
            self._classes_df = pd.read_csv(classes_csv)
            self._classes_dict = {}
            for _, row in self._classes_df.iterrows():
                c_name = str(row["class_name"]).strip()
                d_name = str(row["drug_name"]).strip().lower()
                sev = str(row.get("severity", "MEDIUM")).strip().upper()
                note = str(row.get("risk_note", "")).strip()
                if c_name not in self._classes_dict:
                    self._classes_dict[c_name] = {
                        "drugs": set(),
                        "severity": sev if sev in {"HIGH", "MEDIUM", "LOW"} else "MEDIUM",
                        "risk_note": note,
                    }
                self._classes_dict[c_name]["drugs"].add(d_name)
        else:
            # Clinical fallback defaults
            self._classes_dict = {
                "NSAIDs (Non-Steroidal Anti-Inflammatory)": {
                    "drugs": {"ibuprofen", "diclofenac", "naproxen", "aspirin", "mefenamic acid", "etoricoxib", "celecoxib", "ketoprofen"},
                    "severity": "HIGH",
                    "risk_note": "Concurrent use of multiple NSAIDs markedly increases gastrointestinal ulceration and bleeding risk without added analgesic efficacy.",
                },
                "Proton Pump Inhibitors (PPIs)": {
                    "drugs": {"omeprazole", "pantoprazole", "esomeprazole", "rabeprazole", "lansoprazole"},
                    "severity": "MEDIUM",
                    "risk_note": "Multiple concurrent PPIs provide no additional acid suppression benefit and increase risk of adverse events.",
                },
                "ACE Inhibitors / ARBs": {
                    "drugs": {"lisinopril", "enalapril", "ramipril", "losartan", "valsartan", "telmisartan"},
                    "severity": "HIGH",
                    "risk_note": "Dual renin-angiotensin-aldosterone blockade markedly elevates risks of severe hypotension hyperkalemia and acute renal impairment.",
                },
                "Statins (Lipid Lowering)": {
                    "drugs": {"atorvastatin", "rosuvastatin", "simvastatin", "pravastatin"},
                    "severity": "HIGH",
                    "risk_note": "Multiple concurrent statins increase the risk of severe myopathy and rhabdomyolysis without additional cardiovascular benefit.",
                },
                "Antihistamines (H1 Blockers)": {
                    "drugs": {"cetirizine", "levocetirizine", "loratadine", "fexofenadine", "chlorpheniramine"},
                    "severity": "MEDIUM",
                    "risk_note": "Co-prescribing multiple antihistamines increases anticholinergic load sedation and psychomotor impairment.",
                },
                "Fluoroquinolone Antibiotics": {
                    "drugs": {"ciprofloxacin", "levofloxacin", "ofloxacin", "moxifloxacin"},
                    "severity": "HIGH",
                    "risk_note": "Redundant fluoroquinolone therapy amplifies risks of tendinopathy QT prolongation and CNS adverse effects.",
                },
                "Macrolide Antibiotics": {
                    "drugs": {"azithromycin", "clarithromycin", "erythromycin"},
                    "severity": "HIGH",
                    "risk_note": "Concurrent macrolides represent duplicate antimicrobial coverage and increase risks of QT prolongation and hepatotoxicity.",
                },
                "Beta Blockers": {
                    "drugs": {"metoprolol", "bisoprolol", "atenolol", "carvedilol", "propranolol"},
                    "severity": "HIGH",
                    "risk_note": "Combining multiple beta blockers carries high risk of profound bradycardia heart block and hypotension.",
                },
                "Opioids / Central Analgesics": {
                    "drugs": {"tramadol", "codeine", "morphine"},
                    "severity": "HIGH",
                    "risk_note": "Concurrent centrally acting opioid analgesics carry substantial risk of profound CNS depression respiratory depression and overdose.",
                },
            }

    # ── Medicine Name Checking ────────────────────────────────────────────────

    def _check_medicine_name(self, medicine: str) -> List[PrescriptionIssue]:
        """Fuzzy-match a medicine name against the known list via medicine_matcher.

        Delegates to src.pipeline.medicine_matcher.match_medicine() so that:
          - Normalisation (OCR confusion fixes, suffix stripping) is applied
            BEFORE fuzzy matching, not after.
          - The matching logic is testable independently of error detection.
          - Confidence scores are surfaced in the issue message.
        """
        issues = []

        if not self._known_names or not _MATCHER_AVAILABLE:
            return issues

        med_clean = medicine.strip()
        if not med_clean:
            return issues

        result = _match_medicine(
            raw_text=med_clean,
            known_medicines=self._known_names,
            threshold=self.fuzzy_threshold,
        )

        if result["match_type"] == "exact":
            return []   # Perfect match — no issue

        if result["match_type"] == "fuzzy":
            # Close but not exact — likely a misspelling or OCR typo
            best_match = result["matched_name"]
            score = result["confidence"]
            if best_match and best_match.lower() != med_clean.lower():
                issues.append(PrescriptionIssue(
                    error_type="MISSPELLING",
                    severity="HIGH",
                    field="MEDICINE",
                    value=med_clean,
                    message=(
                        f'"{med_clean}" may be a misspelling or OCR error. '
                        f'Closest match: "{best_match}" '
                        f'(similarity {score:.0f}/100).'
                    ),
                    suggestion=f'Did you mean "{best_match}"?',
                ))
        else:
            # No close match — unrecognised medicine name
            score = result["confidence"]
            issues.append(PrescriptionIssue(
                error_type="UNRECOGNISED",
                severity="MEDIUM",
                field="MEDICINE",
                value=med_clean,
                message=(
                    f'"{med_clean}" was not found in the known medicines list '
                    f'and no close spelling match was identified '
                    f'(best score: {score:.0f}/100).'
                ),
                suggestion=(
                    "Verify this is a real medicine name. "
                    "It may be a trade name not in the reference list, "
                    "or a significant spelling error."
                ),
            ))

        return issues

    # ── Dosage Parsing ────────────────────────────────────────────────────────

    @staticmethod
    def _parse_dosage_mg(dosage_str: str) -> Optional[float]:
        """Attempt to parse a dosage string into milligrams.

        Returns the numeric value in mg, or None if parsing fails.
        Examples:
            "500mg"  → 500.0
            "1g"     → 1000.0
            "250mcg" → 0.25
            "2 tabs" → None  (count-based, not directly comparable)
        """
        m = DOSE_PARSE_PATTERN.search(dosage_str)
        if not m:
            return None
        value = float(m.group("value").replace(",", ""))
        unit  = m.group("unit").lower().rstrip("s")   # normalise plural
        if unit in UNIT_TO_MG:
            return value * UNIT_TO_MG[unit]
        return None   # tablet/cap counts — skip numeric range check

    # ── Dosage Range Checking ─────────────────────────────────────────────────

    def _check_dosage_range(self, medicine: str,
                             dosage: str) -> List[PrescriptionIssue]:
        """Check a dosage against the expected range for a matched medicine."""
        issues = []

        if self._dosage_df is None or not _RAPIDFUZZ_AVAILABLE:
            return issues

        med_clean = medicine.strip().lower()
        if not med_clean or not dosage.strip():
            return issues

        # Find the best matching row in dosage_ranges.csv
        match = fuzz_process.extractOne(
            med_clean,
            list(self._dosage_df["medicine_lower"]),
            scorer=fuzz.token_sort_ratio,
            score_cutoff=75,
        )
        if match is None:
            return []   # Medicine not in dosage reference — can't check

        _, score, match_idx = match
        row = self._dosage_df.iloc[match_idx]

        dose_mg = self._parse_dosage_mg(dosage)
        if dose_mg is None:
            return []   # Can't parse numerically (e.g. "2 tablets") — skip

        try:
            min_dose = float(row["min_dose_mg"])
            max_dose = float(row["max_dose_mg"])
        except (ValueError, KeyError):
            return []

        matched_med = row["medicine"]
        unit = str(row.get("unit", "mg"))

        if dose_mg < min_dose:
            issues.append(PrescriptionIssue(
                error_type="OUT_OF_RANGE",
                severity="MEDIUM",
                field="DOSAGE",
                value=dosage,
                message=(
                    f'Dosage {dosage} for {matched_med} appears LOW. '
                    f'Typical range: {min_dose}–{max_dose} {unit}.'
                ),
                suggestion=(
                    f"Verify the dose is intentional. "
                    f"Typical minimum for {matched_med} is {min_dose} {unit}."
                ),
            ))
        elif dose_mg > max_dose:
            issues.append(PrescriptionIssue(
                error_type="OUT_OF_RANGE",
                severity="HIGH",
                field="DOSAGE",
                value=dosage,
                message=(
                    f'Dosage {dosage} for {matched_med} appears HIGH. '
                    f'Typical range: {min_dose}–{max_dose} {unit}.'
                ),
                suggestion=(
                    f"This exceeds the typical maximum of {max_dose} {unit} "
                    f"for {matched_med}. Verify with prescriber."
                ),
            ))

        return issues

    # ── Frequency Checking ────────────────────────────────────────────────────

    @staticmethod
    def _check_frequency(frequency: str) -> List[PrescriptionIssue]:
        """Check that a frequency string is in a recognised format."""
        issues = []
        freq_clean = frequency.strip()
        if not freq_clean:
            return issues

        # Check against all known patterns
        for pattern in FREQUENCY_PATTERNS:
            if pattern.search(freq_clean):
                return []   # Recognised — no issue

        # Not matched by any pattern
        issues.append(PrescriptionIssue(
            error_type="BAD_FREQUENCY",
            severity="MEDIUM",
            field="FREQUENCY",
            value=freq_clean,
            message=(
                f'Frequency "{freq_clean}" is not in a recognised format. '
                f'It may be unclear or non-standard.'
            ),
            suggestion=(
                "Use a standard frequency expression such as: "
                "OD, BD, TDS, QDS, every 8 hours, twice daily, PRN, etc."
            ),
        ))
        return issues

    # ── Drug Interaction Checking ─────────────────────────────────────────────

    def _check_interactions(self,
                             medicines: List[str]) -> List[PrescriptionIssue]:
        """Check all medicine pairs against the interactions table."""
        issues = []

        if self._interactions_df is None or len(medicines) < 2:
            return issues

        # Normalise for lookup
        meds_lower = [m.strip().lower() for m in medicines]

        # Check every unique pair
        for i in range(len(meds_lower)):
            for j in range(i + 1, len(meds_lower)):
                med_a = meds_lower[i]
                med_b = meds_lower[j]

                # Look for (a, b) or (b, a) in the table
                mask = (
                    (
                        self._interactions_df["drug_a_lower"].str.contains(
                            re.escape(med_a), case=False, na=False
                        ) &
                        self._interactions_df["drug_b_lower"].str.contains(
                            re.escape(med_b), case=False, na=False
                        )
                    ) | (
                        self._interactions_df["drug_a_lower"].str.contains(
                            re.escape(med_b), case=False, na=False
                        ) &
                        self._interactions_df["drug_b_lower"].str.contains(
                            re.escape(med_a), case=False, na=False
                        )
                    )
                )

                matched_rows = self._interactions_df[mask]
                for _, row in matched_rows.iterrows():
                    sev_raw = str(row.get("severity", "LOW")).upper().strip()
                    if sev_raw in {"CRITICAL", "HIGH"}:
                        severity = "HIGH"
                    elif sev_raw in {"WARNING", "MODERATE", "MEDIUM"}:
                        severity = "MEDIUM"
                    else:
                        severity = "LOW"

                    note = str(row.get("note", "")).strip()
                    rec = str(row.get("recommendation", "")).strip()
                    suggestion = rec if rec else "Review with prescriber or pharmacist before dispensing."

                    issues.append(PrescriptionIssue(
                        error_type="INTERACTION",
                        severity=severity,
                        field="MEDICINE",
                        value=f"{medicines[i]} + {medicines[j]}",
                        message=(
                            f'Potential drug interaction: '
                            f'{row["drug_a"]} + {row["drug_b"]}. {note}'
                        ),
                        suggestion=suggestion,
                    ))

        return issues

    # ── Therapeutic Duplication Checking ──────────────────────────────────────

    def _check_therapeutic_duplication(self,
                                        medicines: List[str]) -> List[PrescriptionIssue]:
        """Check if multiple medicines belonging to the same therapeutic class are prescribed.

        Detects duplicate prescribing within the same pharmacological group (e.g.
        concurrently prescribing 2 NSAIDs or 2 PPIs).
        """
        issues = []
        if not self._classes_dict or len(medicines) < 2:
            return issues

        # Clean and normalize medicine inputs
        cleaned_meds = []
        for raw_med in medicines:
            m = str(raw_med).strip()
            if not m:
                continue
            norm = _normalize_medicine_text(m) if _MATCHER_AVAILABLE else m.lower()
            cleaned_meds.append((raw_med, norm))

        # Check each therapeutic class
        for class_name, class_info in self._classes_dict.items():
            drug_set: Set[str] = class_info["drugs"]
            severity: str = class_info.get("severity", "MEDIUM")
            risk_note: str = class_info.get("risk_note", "")

            # Identify which detected medicines belong to this class
            matched_items: List[Tuple[str, str]] = []  # (raw_display, matched_drug_key)
            seen_drug_keys = set()

            for raw_med, norm_med in cleaned_meds:
                for target_drug in drug_set:
                    # Match if target drug name is a substring or exact token
                    if target_drug in norm_med:
                        if target_drug not in seen_drug_keys:
                            seen_drug_keys.add(target_drug)
                            matched_items.append((raw_med, target_drug))
                        break

            # If 2 or more distinct drugs in this class are detected
            if len(matched_items) > 1:
                display_names = [item[0] for item in matched_items]
                joined_names = ", ".join(display_names)
                issues.append(PrescriptionIssue(
                    error_type="DUPLICATE_THERAPY",
                    severity=severity,
                    field="MEDICINE",
                    value=" + ".join(display_names),
                    message=(
                        f"Duplicate therapy detected ({class_name}): {joined_names}. "
                        f"{risk_note}"
                    ),
                    suggestion=(
                        f"Verify whether concurrent use of {joined_names} is clinically "
                        f"indicated or represents unintended duplicate prescribing."
                    ),
                ))

        return issues

    # ── Missing Field Checking ────────────────────────────────────────────────

    @staticmethod
    def _check_missing_fields(structured: Dict[str, List[str]]) -> List[PrescriptionIssue]:
        """Flag any required NER fields that are absent."""
        severity_map = {
            "MEDICINE":  "HIGH",
            "DOSAGE":    "HIGH",
            "FREQUENCY": "MEDIUM",
            "DURATION":  "LOW",
        }
        issues = []
        for field_name, severity in severity_map.items():
            if not structured.get(field_name):
                issues.append(PrescriptionIssue(
                    error_type="MISSING_FIELD",
                    severity=severity,
                    field=field_name,
                    value="(not found)",
                    message=f'{field_name} could not be extracted from this prescription.',
                    suggestion=(
                        f"Check whether the {field_name.lower()} is written clearly "
                        f"in the prescription or was missed by the OCR/NER step."
                    ),
                ))
        return issues

    # ── Main Entry Point ──────────────────────────────────────────────────────

    def check(self, structured: Dict[str, List[str]]) -> List[PrescriptionIssue]:
        """Run all checks on a structured NER result.

        Parameters
        ----------
        structured : Dict mapping entity labels to lists of extracted values.
            Expected keys: MEDICINE, DOSAGE, FREQUENCY, DURATION
            e.g.: {"MEDICINE": ["Amoxicilin"], "DOSAGE": ["500mg"],
                   "FREQUENCY": ["twice daily"], "DURATION": ["7 days"]}

        Returns
        -------
        issues : List of PrescriptionIssue, sorted by severity (HIGH first).
        """
        all_issues: List[PrescriptionIssue] = []

        medicines  = structured.get("MEDICINE",  [])
        dosages    = structured.get("DOSAGE",    [])
        frequencies= structured.get("FREQUENCY", [])

        # 1. Missing field checks (always run first)
        all_issues.extend(self._check_missing_fields(structured))

        # 2. Medicine name checks
        for med in medicines:
            all_issues.extend(self._check_medicine_name(med))

        # 3. Dosage range checks (paired with matching medicine)
        if len(medicines) == len(dosages):
            for med, dosage in zip(medicines, dosages):
                all_issues.extend(self._check_dosage_range(med, dosage))
        else:
            for dosage in dosages:
                for med in medicines:
                    all_issues.extend(self._check_dosage_range(med, dosage))

        # 4. Frequency format checks
        for freq in frequencies:
            all_issues.extend(self._check_frequency(freq))

        # 5. Drug interaction checks
        all_issues.extend(self._check_interactions(medicines))

        # 6. Therapeutic duplication checks
        all_issues.extend(self._check_therapeutic_duplication(medicines))

        # Sort: HIGH > MEDIUM > LOW > INFO
        severity_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2, "INFO": 3}
        all_issues.sort(key=lambda x: severity_order.get(x.severity, 9))

        return all_issues

    def check_from_ner_result(self, ner_result: Dict) -> List[PrescriptionIssue]:
        """Convenience: run checks directly on a NER inference result dict."""
        return self.check(ner_result.get("structured", {}))


# ─── Report Formatting ────────────────────────────────────────────────────────

def format_issues_report(issues: List[PrescriptionIssue],
                          structured: Optional[Dict] = None,
                          use_color: bool = True) -> str:
    """Format a full prescription error report as a string.

    Parameters
    ----------
    issues     : List of PrescriptionIssue from ErrorDetector.check()
    structured : Optional structured NER dict for display context
    use_color  : Whether to use ANSI color codes

    Returns
    -------
    report_str : Multi-line formatted report string
    """
    BOLD    = "\033[1m"  if use_color else ""
    RESET   = "\033[0m"  if use_color else ""
    RED     = "\033[91m" if use_color else ""
    YELLOW  = "\033[93m" if use_color else ""
    BLUE    = "\033[94m" if use_color else ""
    GRAY    = "\033[90m" if use_color else ""

    SEV_COLOR = {"HIGH": RED, "MEDIUM": YELLOW, "LOW": BLUE, "INFO": GRAY}

    lines = []
    lines.append("=" * 60)
    lines.append(f"{BOLD}  Prescription Error Detection Report{RESET}")
    lines.append(f"  ⚠  FOR PROFESSIONAL REVIEW ONLY — NOT A CLINICAL DIAGNOSIS")
    lines.append("=" * 60)

    if structured:
        lines.append(f"\n  {BOLD}Extracted Fields:{RESET}")
        for label, values in structured.items():
            val_str = ", ".join(f'"{v}"' for v in values) if values else "(not found)"
            lines.append(f"    {label:<12}: {val_str}")

    lines.append(f"\n  {BOLD}Issues Found: {len(issues)}{RESET}")

    if not issues:
        lines.append(f"  ✅  No issues detected. Prescription appears well-formed.")
        lines.append("=" * 60)
        return "\n".join(lines)

    # Count by severity
    counts = {"HIGH": 0, "MEDIUM": 0, "LOW": 0, "INFO": 0}
    for issue in issues:
        counts[issue.severity] = counts.get(issue.severity, 0) + 1

    summary_parts = []
    for sev, count in counts.items():
        if count > 0:
            color = SEV_COLOR.get(sev, "")
            summary_parts.append(f"{color}{count} {sev}{RESET}")
    lines.append(f"  Summary: {' | '.join(summary_parts)}")
    lines.append("")

    for i, issue in enumerate(issues, 1):
        color = SEV_COLOR.get(issue.severity, "")
        lines.append(f"  ─── Issue {i} {'─'*40}")
        lines.append(f"  {color}[{issue.severity}] {issue.error_type}{RESET}")
        lines.append(f"    Field     : {issue.field}")
        lines.append(f"    Value     : \"{issue.value}\"")
        lines.append(f"    Message   : {issue.message}")
        if issue.suggestion:
            lines.append(f"    Suggestion: {issue.suggestion}")

    lines.append("=" * 60)
    lines.append(f"  ⚠  All flags require professional clinical review.")
    lines.append("=" * 60)
    return "\n".join(lines)


def issues_to_dict_list(issues: List[PrescriptionIssue]) -> List[dict]:
    """Convert issues to a list of plain dicts (for JSON serialisation)."""
    return [i.to_dict() for i in issues]


# ─── CLI Entry Point ──────────────────────────────────────────────────────────

def _parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Prescription error detection.\n"
            "Provide entities directly or pass a NER result JSON file."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--medicine",  nargs="+", default=[], metavar="MEDICINE",
                        help="One or more medicine names (space-separated).")
    parser.add_argument("--dosage",    nargs="+", default=[], metavar="DOSAGE",
                        help="One or more dosage strings.")
    parser.add_argument("--frequency", nargs="+", default=[], metavar="FREQUENCY",
                        help="One or more frequency strings.")
    parser.add_argument("--duration",  nargs="+", default=[], metavar="DURATION",
                        help="One or more duration strings.")
    parser.add_argument("--json-file", default=None,
                        help="Path to a NER result JSON file (from ner_infer.py --json-out).")
    parser.add_argument("--classes-csv", default="data/error_rules/therapeutic_classes.csv",
                        help="Path to therapeutic_classes.csv (default: data/error_rules/therapeutic_classes.csv).")
    parser.add_argument("--save-json", default=None,
                        help="Save the issues report to this JSON file.")
    parser.add_argument("--fuzzy-threshold", type=int, default=80,
                        help="Fuzzy match threshold 0–100 (default: 80).")
    parser.add_argument("--no-color",  action="store_true",
                        help="Disable ANSI color output.")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    # Build structured dict
    if args.json_file:
        with open(args.json_file, "r", encoding="utf-8") as f:
            ner_result = json.load(f)
        structured = ner_result.get("structured", {})
        print(f"  Loaded NER result from: {args.json_file}")
    else:
        structured = {
            "MEDICINE":  args.medicine,
            "DOSAGE":    args.dosage,
            "FREQUENCY": args.frequency,
            "DURATION":  args.duration,
        }

    if not any(v for v in structured.values()):
        print(
            "ERROR: No entities provided.\n"
            "Use --medicine, --dosage, --frequency, --duration,\n"
            "or --json-file to provide a NER result.",
            file=sys.stderr
        )
        sys.exit(1)

    detector = ErrorDetector(
        classes_csv=args.classes_csv,
        fuzzy_threshold=args.fuzzy_threshold,
    )
    issues = detector.check(structured)

    print(format_issues_report(issues, structured, use_color=not args.no_color))

    if args.save_json:
        report = {
            "structured": structured,
            "issues": issues_to_dict_list(issues),
            "summary": {
                "total": len(issues),
                "HIGH":   sum(1 for i in issues if i.severity == "HIGH"),
                "MEDIUM": sum(1 for i in issues if i.severity == "MEDIUM"),
                "LOW":    sum(1 for i in issues if i.severity == "LOW"),
            }
        }
        os.makedirs(os.path.dirname(os.path.abspath(args.save_json)), exist_ok=True)
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\n  Issues saved → {args.save_json}")
