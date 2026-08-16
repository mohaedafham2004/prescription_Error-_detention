import json
from pathlib import Path

nb_path = Path("notebooks/cnn_character_training_colab.ipynb")
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# 1. Cell 2: use %pip install instead of !pip install
cell2_src = nb["cells"][2]["source"]
nb["cells"][2]["source"] = [s.replace("!pip install", "%pip install") for s in cell2_src]

# 2. Cell 6: guard google.colab import
cell6_src = nb["cells"][6]["source"]
new_cell6_src = []
for line in cell6_src:
    if "from google.colab import drive" in line:
        new_cell6_src.append("try:\n")
        new_cell6_src.append("    from google.colab import drive\n")
        new_cell6_src.append("    drive.mount('/content/drive')\n")
        new_cell6_src.append("except (ImportError, ModuleNotFoundError):\n")
        new_cell6_src.append("    print('Running outside Google Colab — skipping drive mount.')\n")
    elif "drive.mount('/content/drive')" in line:
        continue
    else:
        new_cell6_src.append(line)
nb["cells"][6]["source"] = new_cell6_src

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2)

print("Safely updated notebooks/cnn_character_training_colab.ipynb")
