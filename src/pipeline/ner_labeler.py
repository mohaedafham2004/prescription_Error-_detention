"""
src/pipeline/ner_labeler.py
============================
Interactive CLI tool for labeling prescription text with NER spans.

Builds data/ner/train.jsonl and data/ner/val.jsonl in spaCy-compatible format:
    {"text": "Amoxicillin 500mg twice daily for 7 days",
     "entities": [[0, 11, "MEDICINE"], [12, 17, "DOSAGE"],
                  [18, 29, "FREQUENCY"], [34, 41, "DURATION"]]}

Entity types
------------
  MEDICINE   — drug / medicine name  (e.g. "Amoxicillin", "Paracetamol 500mg Tab")
  DOSAGE     — amount + unit         (e.g. "500mg", "2 tablets", "1 tsp")
  FREQUENCY  — how often             (e.g. "twice daily", "every 8 hours", "TID")
  DURATION   — how long              (e.g. "7 days", "2 weeks", "for a month")

Workflow
--------
1. Paste / type the prescription text when prompted.
2. The tool displays the text with character positions shown.
3. For each entity, type:  <start> <end> <LABEL>
   Example:  0 11 MEDICINE
4. Type 'done' when all entities for this text are labeled.
5. Type 'skip' to discard this sample.
6. Type 'q' to quit and save progress.

The tool automatically sends 85% of samples to train.jsonl and 15% to val.jsonl
(or you can force a split with --split).

Usage
-----
    python -m src.pipeline.ner_labeler

    # Resume a previous session (appends to existing files):
    python -m src.pipeline.ner_labeler --resume

    # Specify output dirs explicitly:
    python -m src.pipeline.ner_labeler \
        --train data/ner/train.jsonl \
        --val   data/ner/val.jsonl

    # Pre-load texts from a file (one text per line) instead of manual typing:
    python -m src.pipeline.ner_labeler --input-file extracted_texts.txt

Tips for faster labeling
------------------------
- Use Tab-completion in the terminal for the label names.
- Common frequency abbreviations to watch: OD, BD, TDS, QDS, PRN, SOS, HS.
- Dosage always includes the unit: "500mg", "1 tab", "2 caps", "5ml".
- DURATION answers "for how long": "for 5 days", "x 1 week", "10/7".
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import List, Optional, Tuple

# ── Allow running from project root ──────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ─── Constants ────────────────────────────────────────────────────────────────

ENTITY_LABELS = ["MEDICINE", "DOSAGE", "FREQUENCY", "DURATION"]
VAL_FRACTION  = 0.15   # fraction of labeled samples routed to val.jsonl

LABEL_COLORS = {
    "MEDICINE":  "\033[92m",   # green
    "DOSAGE":    "\033[94m",   # blue
    "FREQUENCY": "\033[93m",   # yellow
    "DURATION":  "\033[95m",   # magenta
}
RESET = "\033[0m"
BOLD  = "\033[1m"


# ─── Display Helpers ──────────────────────────────────────────────────────────

def _color(text: str, label: str) -> str:
    return LABEL_COLORS.get(label, "") + text + RESET


def display_text_with_positions(text: str, entities: List[Tuple]) -> None:
    """Print the text with character position ruler and color-highlighted spans."""
    print()

    # Ruler: every 10 chars
    ruler_top = ""
    ruler_bot = ""
    for i in range(len(text)):
        if i % 10 == 0:
            marker = str(i // 10 % 10)
            ruler_top += marker
            ruler_bot += str(i % 10)
        else:
            ruler_top += " "
            ruler_bot += str(i % 10)

    print(f"  {ruler_top}")
    print(f"  {ruler_bot}")

    # Highlighted text
    if entities:
        # Build colored version of the text
        colored = list(text)
        # Apply color markers (simple: highlight each entity span)
        highlighted = ""
        pos = 0
        sorted_ents = sorted(entities, key=lambda e: e[0])
        for start, end, label in sorted_ents:
            highlighted += text[pos:start]
            highlighted += _color(text[start:end], label)
            pos = end
        highlighted += text[pos:]
        print(f"  {highlighted}")
    else:
        print(f"  {text}")

    print()


def display_entities(entities: List[Tuple], text: str) -> None:
    """Print current labeled entities in a table."""
    if not entities:
        print("  (no entities labeled yet)")
        return
    print()
    print(f"  {'#':<4} {'Start':<6} {'End':<6} {'Label':<12} {'Span'}")
    print(f"  {'-'*55}")
    for i, (start, end, label) in enumerate(entities):
        span = text[start:end]
        color = LABEL_COLORS.get(label, "")
        print(f"  {i:<4} {start:<6} {end:<6} {color}{label:<12}{RESET} \"{span}\"")
    print()


def display_legend() -> None:
    """Print the entity label legend."""
    print(f"\n  {'-'*55}")
    print(f"  Entity Labels:")
    for label in ENTITY_LABELS:
        color = LABEL_COLORS.get(label, "")
        print(f"    {color}{label}{RESET}")
    print(f"  {'-'*55}\n")


# ─── Validation ───────────────────────────────────────────────────────────────

def validate_span(text: str, start: int, end: int, label: str,
                  existing: List[Tuple]) -> Optional[str]:
    """Return an error message if the span is invalid, else None."""
    if start < 0 or end > len(text) or start >= end:
        return f"Invalid range [{start}, {end}] for text of length {len(text)}"
    if label not in ENTITY_LABELS:
        return f"Unknown label '{label}'. Must be one of: {ENTITY_LABELS}"
    # Check for overlaps
    for es, ee, el in existing:
        if not (end <= es or start >= ee):
            overlap_span = text[es:ee]
            return (f"Overlaps with existing entity [{es},{ee}] "
                    f"'{overlap_span}' ({el})")
    return None


# ─── I/O Helpers ─────────────────────────────────────────────────────────────

def load_existing(filepath: str) -> List[dict]:
    """Load existing JSONL records (for --resume mode)."""
    records = []
    p = Path(filepath)
    if p.exists():
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    return records


def append_record(filepath: str, record: dict) -> None:
    """Append one JSONL record to a file."""
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    with open(filepath, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ─── Single-Sample Labeling Session ──────────────────────────────────────────

def label_one_sample(text: str, sample_index: int,
                     total: Optional[int] = None) -> Optional[List[Tuple]]:
    """Interactively label entities in one text sample.

    Returns
    -------
    entities : List of (start, end, label) tuples, or None if skipped.
    """
    total_str = f"/{total}" if total else ""
    print(f"\n  {'='*55}")
    print(f"  {BOLD}Sample {sample_index}{total_str}{RESET}")
    print(f"  {'='*55}")

    entities: List[Tuple] = []
    display_text_with_positions(text, entities)

    print("  Commands:")
    print("    <start> <end> <LABEL>   — add entity (e.g. '0 11 MEDICINE')")
    print("    del <#>                 — remove entity by index")
    print("    show                    — redisplay text with highlights")
    print("    done                    — save this sample")
    print("    skip                    — discard this sample")
    print("    q                       — quit and save all progress so far")

    display_legend()

    while True:
        try:
            raw = input("  > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Interrupted — saving progress.")
            return None

        if not raw:
            continue

        parts = raw.split()

        # ── quit ─────────────────────────────────────────────────────────────
        if parts[0].lower() == "q":
            return None

        # ── skip ─────────────────────────────────────────────────────────────
        if parts[0].lower() == "skip":
            print("  → Sample skipped.")
            return []    # empty list = skipped (distinct from None = quit)

        # ── done ─────────────────────────────────────────────────────────────
        if parts[0].lower() == "done":
            if not entities:
                confirm = input("  No entities labeled. Save as empty? [y/N] ").strip().lower()
                if confirm != "y":
                    continue
            display_entities(entities, text)
            return entities

        # ── show ─────────────────────────────────────────────────────────────
        if parts[0].lower() == "show":
            display_text_with_positions(text, entities)
            display_entities(entities, text)
            continue

        # ── del <#> ──────────────────────────────────────────────────────────
        if parts[0].lower() == "del" and len(parts) == 2:
            try:
                idx = int(parts[1])
                removed = entities.pop(idx)
                print(f"  → Removed entity {idx}: {removed}")
                display_text_with_positions(text, entities)
            except (ValueError, IndexError):
                print(f"  ✗ Invalid index '{parts[1]}'")
            continue

        # ── <start> <end> <LABEL> ─────────────────────────────────────────────
        if len(parts) == 3:
            try:
                start = int(parts[0])
                end   = int(parts[1])
                label = parts[2].upper()
            except ValueError:
                print(f"  ✗ Could not parse '{raw}'. Format: <start> <end> <LABEL>")
                continue

            err = validate_span(text, start, end, label, entities)
            if err:
                print(f"  ✗ {err}")
                continue

            span_preview = text[start:end]
            entities.append((start, end, label))
            color = LABEL_COLORS.get(label, "")
            print(f"  ✓ Added: {color}{label}{RESET} [{start},{end}] \"{span_preview}\"")
            display_text_with_positions(text, entities)
            display_entities(entities, text)
            continue

        print(f"  ✗ Unrecognised command: '{raw}'")
        print("    Use '<start> <end> <LABEL>', 'done', 'skip', 'del <#>', 'show', or 'q'")


# ─── Main Labeling Session ────────────────────────────────────────────────────

def run_labeling_session(train_path: str,
                         val_path: str,
                         input_texts: Optional[List[str]] = None,
                         resume: bool = False) -> None:
    """Run the full interactive labeling session.

    Parameters
    ----------
    train_path   : Path to output train.jsonl
    val_path     : Path to output val.jsonl
    input_texts  : Pre-loaded texts. If None, prompts the user to type each one.
    resume       : If True, counts existing records for progress display.
    """
    os.makedirs(os.path.dirname(os.path.abspath(train_path)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(val_path)),   exist_ok=True)

    # Count existing samples for progress tracking
    existing_train = load_existing(train_path) if resume else []
    existing_val   = load_existing(val_path)   if resume else []
    n_existing = len(existing_train) + len(existing_val)

    print(f"\n{'='*60}")
    print(f"  Smart Prescription — NER Labeling Tool")
    print(f"{'='*60}")
    print(f"  Train output : {train_path}")
    print(f"  Val output   : {val_path}")
    if resume:
        print(f"  Resuming     : {len(existing_train)} train + {len(existing_val)} val samples already saved")
    print(f"  Val fraction : {VAL_FRACTION*100:.0f}%  (auto-routed)")
    print(f"  Target       : 200–300 total labeled samples for good NER training")
    print(f"{'='*60}")

    labeled_this_session = 0
    skipped_this_session = 0
    sample_idx = n_existing + 1

    # ── Main loop ─────────────────────────────────────────────────────────────
    text_source = iter(input_texts) if input_texts else None

    while True:
        # ── Get text ──────────────────────────────────────────────────────────
        if text_source is not None:
            try:
                text = next(text_source).strip()
                if not text:
                    continue
                print(f"\n  {'-'*55}")
                print(f"  Text loaded from file: (showing below)")
            except StopIteration:
                print("\n  All texts from the input file have been processed.")
                break
        else:
            print(f"\n  {'-'*55}")
            print(f"  Paste or type the prescription text to label.")
            print(f"  (Press ENTER twice when done, or type 'q' to quit)")
            print(f"  {'-'*55}")
            lines = []
            try:
                while True:
                    line = input()
                    if line.lower() == "q":
                        print("  Quit signal. Saving progress.")
                        _print_session_summary(labeled_this_session, skipped_this_session,
                                               sample_idx - 1, train_path, val_path)
                        return
                    if line == "" and lines and lines[-1] == "":
                        lines.pop()  # remove trailing blank
                        break
                    lines.append(line)
            except (EOFError, KeyboardInterrupt):
                print("\n  Saving progress.")
                _print_session_summary(labeled_this_session, skipped_this_session,
                                       sample_idx - 1, train_path, val_path)
                return

            text = "\n".join(lines).strip()
            if not text:
                continue

        # ── Label this text ───────────────────────────────────────────────────
        total = len(input_texts) + n_existing if input_texts else None
        result = label_one_sample(text, sample_idx, total=total)

        if result is None:
            # User quit
            _print_session_summary(labeled_this_session, skipped_this_session,
                                   sample_idx - 1, train_path, val_path)
            return

        if result == []:
            # User skipped
            skipped_this_session += 1
            continue

        # ── Route to train or val ─────────────────────────────────────────────
        record = {
            "text": text,
            "entities": [[s, e, l] for s, e, l in result],
        }
        total_so_far = sample_idx - 1
        go_to_val = (random.random() < VAL_FRACTION) and total_so_far > 0
        dest = val_path if go_to_val else train_path

        append_record(dest, record)
        split_label = "val" if go_to_val else "train"
        print(f"\n  ✓ Sample saved to {split_label}.jsonl  "
              f"(entities: {len(result)})")

        labeled_this_session += 1
        sample_idx += 1

    _print_session_summary(labeled_this_session, skipped_this_session,
                           sample_idx - 1, train_path, val_path)


def _print_session_summary(labeled: int, skipped: int,
                            total: int, train_path: str, val_path: str) -> None:
    train_count = len(load_existing(train_path))
    val_count   = len(load_existing(val_path))
    print(f"\n{'='*55}")
    print(f"  Session Summary")
    print(f"{'='*55}")
    print(f"  Labeled this session : {labeled}")
    print(f"  Skipped this session : {skipped}")
    print(f"  Total in train.jsonl : {train_count}")
    print(f"  Total in val.jsonl   : {val_count}")
    print(f"  Grand total          : {train_count + val_count}")
    print(f"{'='*55}")
    if train_count + val_count < 200:
        needed = 200 - (train_count + val_count)
        print(f"  ℹ  Need ~{needed} more samples for good NER training.")
    else:
        print(f"  ✓  Sufficient samples for training. Ready for step 6 NER training.")
    print()


# ─── CLI Entry Point ──────────────────────────────────────────────────────────

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Interactive NER labeling tool for prescription text.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--train",  default="data/ner/train.jsonl",
                        help="Output path for training samples (default: data/ner/train.jsonl)")
    parser.add_argument("--val",    default="data/ner/val.jsonl",
                        help="Output path for validation samples (default: data/ner/val.jsonl)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume a previous session (appends to existing files, "
                             "counts existing samples for progress display)")
    parser.add_argument("--input-file", "-f", default=None,
                        help="Text file with one prescription text per line. "
                             "If not provided, texts are entered interactively.")
    return parser.parse_args()


if __name__ == "__main__":
    random.seed()   # fresh randomness for val split each session
    args = _parse_args()

    input_texts = None
    if args.input_file:
        input_path = Path(args.input_file)
        if not input_path.exists():
            print(f"ERROR: Input file not found: {input_path}", file=sys.stderr)
            sys.exit(1)
        with open(input_path, "r", encoding="utf-8") as f:
            input_texts = [l.strip() for l in f if l.strip()]
        print(f"  Loaded {len(input_texts)} texts from {args.input_file}")

    run_labeling_session(
        train_path=args.train,
        val_path=args.val,
        input_texts=input_texts,
        resume=args.resume,
    )
