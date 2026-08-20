"""
src/models/train_ner.py
========================
Trains a custom spaCy NER model from scratch on labeled prescription data.

What this script does
---------------------
1. Loads data/ner/train.jsonl and data/ner/val.jsonl
2. Converts JSONL records to spaCy DocBin format
3. Writes train.spacy and dev.spacy to a temp training dir
4. Trains a blank English NER model using spaCy's `spacy train` CLI
   (wrapped in Python so you don't need to write a config file manually)
5. Evaluates on val set → per-entity Precision, Recall, F1
6. Saves the best model to models/ner_model/
7. Saves an evaluation JSON for the Streamlit dashboard

Entity types trained
--------------------
    MEDICINE   DOSAGE   FREQUENCY   DURATION

Requirements
------------
    pip install spacy
    python -m spacy download en_core_web_sm   ← used as pipeline backbone

Usage
-----
    python -m src.models.train_ner

    # Override defaults:
    python -m src.models.train_ner \
        --train  data/ner/train.jsonl \
        --val    data/ner/val.jsonl \
        --model-dir models/ner_model \
        --epochs 30 \
        --dropout 0.3

    # Evaluate only (skip training):
    python -m src.models.train_ner --eval-only
"""

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ── Allow running from project root ──────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# ─── Data Loading ────────────────────────────────────────────────────────────

def load_ner_jsonl(filepath: str) -> List[Dict]:
    """Load NER labeled samples from JSONL.

    Expected format per line:
        {"text": "...", "entities": [[start, end, label], ...]}

    Returns list of dicts.
    """
    records = []
    p = Path(filepath)
    if not p.exists():
        return []
    with open(filepath, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                text = obj.get("text", "")
                entities = obj.get("entities", [])
                if text and isinstance(entities, list):
                    records.append({"text": text, "entities": entities})
            except json.JSONDecodeError as e:
                print(f"  WARNING: Could not parse line {i} in {filepath}: {e}")
    return records


# ─── spaCy DocBin Conversion ─────────────────────────────────────────────────

def records_to_docbin(records: List[Dict], nlp) -> "spacy.tokens.DocBin":
    """Convert JSONL records to a spaCy DocBin (the training format).

    Filters out records where entity spans don't align to token boundaries
    (spaCy requires span alignment).
    """
    import spacy
    from spacy.tokens import DocBin

    db = DocBin()
    skipped = 0

    for rec in records:
        text = rec["text"]
        entities = rec["entities"]

        doc = nlp.make_doc(text)
        ents = []
        valid = True

        for item in entities:
            start, end, label = int(item[0]), int(item[1]), str(item[2])
            span = doc.char_span(start, end, label=label, alignment_mode="contract")
            if span is None:
                print(f"  WARNING: Could not align span [{start},{end}] "
                      f"'{text[start:end]}' ({label}) — sample skipped")
                valid = False
                break
            ents.append(span)

        if valid:
            try:
                doc.ents = ents
                db.add(doc)
            except Exception as e:
                print(f"  WARNING: Error setting entities for '{text[:40]}…': {e} — skipped")
                skipped += 1

    if skipped:
        print(f"  Skipped {skipped} sample(s) due to entity errors.")
    return db


# ─── spaCy Config Generation ─────────────────────────────────────────────────

def generate_spacy_config(output_path: str,
                           train_path: str,
                           dev_path: str,
                           n_iter: int = 30,
                           dropout: float = 0.3,
                           batch_size: int = 8) -> None:
    """Write a minimal spaCy config.cfg for NER training."""
    config_content = f"""
[paths]
train = "{train_path}"
dev = "{dev_path}"

[system]
gpu_allocator = null
seed = 42

[nlp]
lang = "en"
pipeline = ["ner"]
batch_size = {batch_size}

[components]

[components.ner]
factory = "ner"

[components.ner.model]
@architectures = "spacy.TransitionBasedParser.v2"
state_type = "ner"
extra_state_tokens = false
hidden_width = 128
maxout_pieces = 3
use_upper = true
nO = null

[components.ner.model.tok2vec]
@architectures = "spacy.HashEmbedCNN.v2"
pretrained_vectors = null
width = 128
depth = 4
embed_size = 2000
window_size = 1
maxout_pieces = 3
subword_features = true

[training]
dev_corpus = "corpora.dev"
train_corpus = "corpora.train"
seed = 42
gpu_id = -1
dropout = {dropout}
accumulate_gradient = 1
patience = 1600
max_steps = 0
eval_frequency = 200
frozen_components = []
annotating_components = []
max_epochs = {n_iter}
before_to_disk = null
before_update = null

[training.batcher]
@batchers = "spacy.batch_by_words.v1"
discard_oversize = false
tolerance = 0.2
get_length = null

[training.batcher.size]
@schedules = "compounding.v1"
start = 100
stop = 1000
compound = 1.001
t = 0.0

[training.logger]
@loggers = "spacy.ConsoleLogger.v1"
progress_bar = true

[training.optimizer]
@optimizers = "Adam.v1"
beta1 = 0.9
beta2 = 0.999
L2_is_weight_decay = true
L2 = 0.01
grad_clip = 1.0
use_averages = false
eps = 0.00000001
learn_rate = 0.001

[training.optimizer.learn_rate]
@schedules = "warmup_linear.v1"
warmup_steps = 250
total_steps = 20000
initial_rate = 0.00005

[training.score_weights]
ents_f = 1.0
ents_p = 0.0
ents_r = 0.0
ents_per_type = null

[pretraining]

[initialize]
vectors = null
init_tok2vec = null

[corpora]

[corpora.train]
@readers = "spacy.Corpus.v1"
path = ${{paths.train}}
max_length = 0

[corpora.dev]
@readers = "spacy.Corpus.v1"
path = ${{paths.dev}}
max_length = 0
""".strip()

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(config_content)


# ─── Evaluation ──────────────────────────────────────────────────────────────

def evaluate_ner_model(model_path: str,
                        val_records: List[Dict],
                        output_dir: str) -> Dict:
    """Evaluate the trained NER model on the validation set.

    Returns a dict with per-entity and overall precision/recall/F1.
    """
    import spacy

    print(f"\n  Loading model from: {model_path}")
    nlp = spacy.load(model_path)

    tp: Dict[str, int] = {}
    fp: Dict[str, int] = {}
    fn: Dict[str, int] = {}

    for rec in val_records:
        text = rec["text"]
        gold_ents = {(int(s), int(e), l) for s, e, l in rec["entities"]}
        pred_doc  = nlp(text)
        pred_ents = {(ent.start_char, ent.end_char, ent.label_) for ent in pred_doc.ents}

        for ent in pred_ents:
            label = ent[2]
            tp.setdefault(label, 0)
            fp.setdefault(label, 0)
            fn.setdefault(label, 0)
            if ent in gold_ents:
                tp[label] += 1
            else:
                fp[label] += 1

        for ent in gold_ents:
            label = ent[2]
            tp.setdefault(label, 0)
            fp.setdefault(label, 0)
            fn.setdefault(label, 0)
            if ent not in pred_ents:
                fn[label] += 1

    # Per-entity metrics
    results = {}
    all_labels = sorted(set(list(tp.keys()) + list(fn.keys())))

    print(f"\n  {'Label':<14} {'P':>8} {'R':>8} {'F1':>8} {'TP':>6} {'FP':>6} {'FN':>6}")
    print(f"  {'─'*57}")

    total_tp = total_fp = total_fn = 0
    for label in all_labels:
        t = tp.get(label, 0)
        f = fp.get(label, 0)
        n = fn.get(label, 0)
        total_tp += t; total_fp += f; total_fn += n

        p  = t / (t + f) if (t + f) > 0 else 0.0
        r  = t / (t + n) if (t + n) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        results[label] = {"precision": round(p, 4), "recall": round(r, 4),
                           "f1": round(f1, 4), "tp": t, "fp": f, "fn": n}
        print(f"  {label:<14} {p:>8.2%} {r:>8.2%} {f1:>8.2%} {t:>6} {f:>6} {n:>6}")

    # Overall micro-average
    p_all  = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    r_all  = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1_all = 2 * p_all * r_all / (p_all + r_all) if (p_all + r_all) > 0 else 0.0

    print(f"  {'─'*57}")
    print(f"  {'OVERALL (micro)':<14} {p_all:>8.2%} {r_all:>8.2%} {f1_all:>8.2%} "
          f"{total_tp:>6} {total_fp:>6} {total_fn:>6}")

    summary = {
        "model_path": str(model_path),
        "n_val_samples": len(val_records),
        "overall": {
            "precision": round(p_all, 4),
            "recall":    round(r_all, 4),
            "f1":        round(f1_all, 4),
        },
        "per_entity": results,
    }

    os.makedirs(output_dir, exist_ok=True)
    eval_path = os.path.join(output_dir, "ner_eval_summary.json")
    with open(eval_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Eval summary saved → {eval_path}")
    return summary


# ─── Main Training Function ───────────────────────────────────────────────────

def train(train_path:  str = "data/ner/train.jsonl",
          val_path:    str = "data/ner/val.jsonl",
          model_dir:   str = "models/ner_model",
          eval_dir:    str = "evaluation/ner_eval",
          n_iter:      int = 30,
          dropout:    float = 0.3,
          batch_size:  int = 8,
          eval_only:  bool = False) -> None:
    """Full NER training pipeline."""

    try:
        import spacy
    except ImportError:
        print("ERROR: spaCy not installed. Run: pip install spacy", file=sys.stderr)
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  Custom NER Training  (spaCy)")
    print(f"{'='*60}")
    print(f"  Train data   : {train_path}")
    print(f"  Val data     : {val_path}")
    print(f"  Model dir    : {model_dir}")
    print(f"  Epochs       : {n_iter}")
    print(f"  Dropout      : {dropout}")
    print(f"{'='*60}\n")

    # ── Load data ─────────────────────────────────────────────────────────────
    print("  Loading training data …")
    train_records = load_ner_jsonl(train_path)
    val_records   = load_ner_jsonl(val_path)

    print(f"  Train samples : {len(train_records)}")
    print(f"  Val samples   : {len(val_records)}")

    if not train_records:
        print(
            "\n  ERROR: No training samples found.\n"
            "  Run the NER labeling tool first:\n"
            "    python -m src.pipeline.ner_labeler\n"
            "  Target: 200–300 labeled samples.",
            file=sys.stderr
        )
        sys.exit(1)

    if len(train_records) < 20:
        print(f"\n  WARNING: Only {len(train_records)} training samples. "
              "NER quality will be low. Aim for 200+ samples.\n")

    if eval_only:
        if not os.path.exists(model_dir):
            print(f"ERROR: Model not found at {model_dir}. Train first.", file=sys.stderr)
            sys.exit(1)
        evaluate_ner_model(model_dir, val_records, eval_dir)
        return

    # ── Build a blank English NLP pipeline ────────────────────────────────────
    print("\n  Building blank English NLP pipeline …")
    nlp = spacy.blank("en")

    # ── Convert records to DocBin ─────────────────────────────────────────────
    tmp_dir = Path(tempfile.mkdtemp(prefix="ner_training_"))
    try:
        print("  Converting train set to DocBin …")
        train_db = records_to_docbin(train_records, nlp)
        train_spacy = str(tmp_dir / "train.spacy")
        train_db.to_disk(train_spacy)
        print(f"  Wrote {len(train_db)} train docs → {train_spacy}")

        print("  Converting val set to DocBin …")
        val_db = records_to_docbin(val_records, nlp)
        dev_spacy = str(tmp_dir / "dev.spacy")
        val_db.to_disk(dev_spacy)
        print(f"  Wrote {len(val_db)} val docs → {dev_spacy}")

        # ── Write spaCy config ────────────────────────────────────────────────
        config_path = str(tmp_dir / "config.cfg")
        output_path = str(tmp_dir / "output")
        os.makedirs(output_path, exist_ok=True)

        # Use forward slashes in the config (spaCy requires this on Windows too)
        generate_spacy_config(
            output_path=config_path,
            train_path=train_spacy.replace("\\", "/"),
            dev_path=dev_spacy.replace("\\", "/"),
            n_iter=n_iter,
            dropout=dropout,
            batch_size=batch_size,
        )
        print(f"  Config written → {config_path}")

        # ── Run spaCy train ───────────────────────────────────────────────────
        print(f"\n  Starting spaCy NER training …")
        print(f"  (This runs on CPU — expect ~1–5 min for 200 samples / 30 epochs)\n")

        cmd = [
            sys.executable, "-m", "spacy", "train",
            config_path,
            "--output", output_path,
            "--paths.train", train_spacy.replace("\\", "/"),
            "--paths.dev",   dev_spacy.replace("\\", "/"),
        ]

        result = subprocess.run(cmd, capture_output=False, text=True)
        if result.returncode != 0:
            print(f"\n  ERROR: spaCy training failed (exit code {result.returncode}).",
                  file=sys.stderr)
            sys.exit(1)

        # ── Copy best model to models/ner_model/ ─────────────────────────────
        best_src = Path(output_path) / "model-best"
        last_src = Path(output_path) / "model-last"
        model_src = best_src if best_src.exists() else last_src

        if not model_src.exists():
            print(f"  ERROR: No trained model found at {output_path}/model-best", file=sys.stderr)
            sys.exit(1)

        os.makedirs(model_dir, exist_ok=True)
        if os.path.exists(model_dir) and os.listdir(model_dir):
            print(f"  Clearing existing model at {model_dir} …")
            shutil.rmtree(model_dir)

        shutil.copytree(str(model_src), model_dir)
        print(f"\n  ✓ Best model saved → {model_dir}")

    finally:
        shutil.rmtree(str(tmp_dir), ignore_errors=True)

    # ── Evaluate ──────────────────────────────────────────────────────────────
    print(f"\n  Running evaluation on val set …")
    summary = evaluate_ner_model(model_dir, val_records, eval_dir)

    print(f"\n{'='*60}")
    print(f"  Training complete!")
    print(f"  Overall F1 : {summary['overall']['f1']*100:.2f}%")
    print(f"  Model saved: {model_dir}")
    print(f"  Eval saved : {eval_dir}")
    print(f"{'='*60}\n")

    print("  Next: use the model with:")
    print("    python -m src.models.ner_infer --text \"Amoxicillin 500mg twice daily\"")


# ─── CLI Entry Point ──────────────────────────────────────────────────────────

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Train a custom spaCy NER model on labeled prescription data."
    )
    parser.add_argument("--train",     default="data/ner/train.jsonl")
    parser.add_argument("--val",       default="data/ner/val.jsonl")
    parser.add_argument("--model-dir", default="models/ner_model")
    parser.add_argument("--eval-dir",  default="evaluation/ner_eval")
    parser.add_argument("--epochs",    type=int,   default=30)
    parser.add_argument("--dropout",   type=float, default=0.3)
    parser.add_argument("--batch-size",type=int,   default=8)
    parser.add_argument("--eval-only", action="store_true",
                        help="Skip training; only evaluate an existing model.")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    train(
        train_path=args.train,
        val_path=args.val,
        model_dir=args.model_dir,
        eval_dir=args.eval_dir,
        n_iter=args.epochs,
        dropout=args.dropout,
        batch_size=args.batch_size,
        eval_only=args.eval_only,
    )
