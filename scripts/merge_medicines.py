"""
scripts/merge_medicines.py
==========================
Merges and deduplicates known medicines databases (e.g. name, aliases).

Usage:
    python scripts/merge_medicines.py \
        --existing data/error_rules/known_medicines.csv \
        --new known_medicines_expanded.csv \
        --output data/error_rules/known_medicines.csv
"""

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, Set

# Ensure project root is importable
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def load_csv(path: Path) -> Dict[str, Set[str]]:
    """Load medicine names and their set of aliases from a CSV."""
    medicines: Dict[str, Set[str]] = {}
    if not path.exists():
        return medicines

    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = (row.get("name") or row.get("Medicine") or row.get("medicine") or "").strip()
            if not name:
                continue

            aliases_raw = row.get("aliases") or row.get("Aliases") or ""
            aliases = {a.strip() for a in aliases_raw.split(",") if a.strip()}

            # Key by lowercase for case-insensitive merge, but keep clean title name
            key = name.lower()
            if key not in medicines:
                medicines[key] = (name, set(aliases))
            else:
                existing_name, existing_aliases = medicines[key]
                existing_aliases.update(aliases)
                medicines[key] = (existing_name, existing_aliases)

    return medicines


def merge_medicines(existing_path: Path, new_path: Path, output_path: Path) -> None:
    print(f"Loading existing medicines from: {existing_path}")
    existing_meds = load_csv(existing_path)
    print(f"  Found {len(existing_meds)} existing medicine entries.")

    print(f"Loading new medicines from: {new_path}")
    new_meds = load_csv(new_path)
    print(f"  Found {len(new_meds)} new medicine entries.")

    # Merge
    merged: Dict[str, tuple[str, Set[str]]] = dict(existing_meds)
    added_count = 0
    updated_count = 0

    for key, (name, aliases) in new_meds.items():
        if key in merged:
            orig_name, orig_aliases = merged[key]
            before_len = len(orig_aliases)
            orig_aliases.update(aliases)
            if len(orig_aliases) > before_len:
                updated_count += 1
            merged[key] = (orig_name, orig_aliases)
        else:
            merged[key] = (name, set(aliases))
            added_count += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "aliases"])
        # Sort alphabetically by display name
        for key, (name, aliases) in sorted(merged.items(), key=lambda item: item[1][0].lower()):
            alias_str = ",".join(sorted(aliases))
            writer.writerow([name, alias_str])

    print("\n" + "=" * 60)
    print("MERGE COMPLETE:")
    print(f"  Total Medicines: {len(merged)}")
    print(f"  New Entities Added: {added_count}")
    print(f"  Existing Entities Updated with New Aliases: {updated_count}")
    print(f"  Saved Output to: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge and deduplicate medicine databases.")
    parser.add_argument("--existing", type=Path, required=True,
                        help="Path to existing known_medicines.csv")
    parser.add_argument("--new", type=Path, required=True,
                        help="Path to new/expanded medicine CSV")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output path (defaults to overwrite --existing)")
    args = parser.parse_args()

    out = args.output if args.output is not None else args.existing
    merge_medicines(args.existing, args.new, out)
