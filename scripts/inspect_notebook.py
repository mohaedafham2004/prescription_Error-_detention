import json
from pathlib import Path

nb_path = Path("notebooks/cnn_character_training_colab.ipynb")
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

print(f"Total cells in {nb_path}: {len(nb['cells'])}")
for i, cell in enumerate(nb['cells']):
    ctype = cell.get('cell_type')
    src = "".join(cell.get('source', []))
    lines = src.splitlines()
    first_line = lines[0] if lines else "[EMPTY]"
    print(f"\n=== CELL {i} ({ctype}) | Lines: {len(lines)} ===")
    for l in lines[:10]:
        print(f"  {l}")
    if len(lines) > 10:
        print(f"  ... (+{len(lines)-10} lines)")
