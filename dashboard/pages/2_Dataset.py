"""
dashboard/pages/2_Dataset.py
=============================
Dataset Overview & Corpus Visualizer
"""

import json
import sys
from pathlib import Path
import streamlit as st
import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.utils.ui_theme import get_theme_css, render_sidebar

st.set_page_config(
    page_title="Dataset · Smart Prescription",
    page_icon="📁",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(get_theme_css(), unsafe_allow_html=True)
render_sidebar()

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='margin-bottom: 20px;'>
    <h1 style='margin:0; font-size:1.8rem; color:#f8fafc !important; font-weight:800;'>
        Prescription Dataset
    </h1>
    <p style='margin:4px 0 0; color:#94a3b8; font-size:0.92rem;'>
        Corpus breakdown across prescription scans, line slices, character crops, and NER annotations.
    </p>
</div>
<hr style='border-color:rgba(255,255,255,0.07); margin:10px 0 20px;'>
""", unsafe_allow_html=True)

# ── Count Data on Disk ────────────────────────────────────────────────────────
raw_dir = _ROOT / "data" / "raw" / "prescriptions"
char_dir = _ROOT / "data" / "characters"
word_dir = _ROOT / "data" / "words_lines" / "images"
labels_file = _ROOT / "data" / "words_lines" / "labels.jsonl"
ner_train_file = _ROOT / "data" / "ner" / "train.jsonl"

n_raw = len([f for f in raw_dir.glob("*.*") if f.is_file() and f.name != ".gitkeep"]) if raw_dir.exists() else 0
n_char = len([f for f in char_dir.rglob("*.*") if f.is_file() and f.name != ".gitkeep"]) if char_dir.exists() else 0

n_words = 0
if labels_file.exists():
    try:
        with open(labels_file, "r", encoding="utf-8") as f:
            n_words = sum(1 for line in f if line.strip())
    except Exception:
        pass
elif word_dir.exists():
    n_words = len([f for f in word_dir.glob("*.*") if f.is_file()])

n_ner = 0
if ner_train_file.exists():
    try:
        with open(ner_train_file, "r", encoding="utf-8") as f:
            n_ner = sum(1 for line in f if line.strip())
    except Exception:
        pass

# ── Stat Cards ────────────────────────────────────────────────────────────────
d1, d2, d3, d4 = st.columns(4)

with d1:
    st.markdown(f"""
    <div class='ui-card' style='text-align:center;'>
        <div style='font-size:0.75rem; font-weight:700; color:#94a3b8; text-transform:uppercase;'>Raw Prescriptions</div>
        <div style='font-size:2rem; font-weight:800; color:#38bdf8; margin:6px 0 2px;'>{n_raw}</div>
        <div style='font-size:0.78rem; color:#64748b;'>Scans in <code>data/raw/</code></div>
    </div>
    """, unsafe_allow_html=True)

with d2:
    st.markdown(f"""
    <div class='ui-card' style='text-align:center;'>
        <div style='font-size:0.75rem; font-weight:700; color:#94a3b8; text-transform:uppercase;'>Line & Word Crops</div>
        <div style='font-size:2rem; font-weight:800; color:#34d399; margin:6px 0 2px;'>{n_words}</div>
        <div style='font-size:0.78rem; color:#64748b;'>TrOCR fine-tuning slices</div>
    </div>
    """, unsafe_allow_html=True)

with d3:
    st.markdown(f"""
    <div class='ui-card' style='text-align:center;'>
        <div style='font-size:0.75rem; font-weight:700; color:#94a3b8; text-transform:uppercase;'>Character Crops</div>
        <div style='font-size:2rem; font-weight:800; color:#fbbf24; margin:6px 0 2px;'>{n_char}</div>
        <div style='font-size:0.78rem; color:#64748b;'>CNN character samples</div>
    </div>
    """, unsafe_allow_html=True)

with d4:
    st.markdown(f"""
    <div class='ui-card' style='text-align:center;'>
        <div style='font-size:0.75rem; font-weight:700; color:#94a3b8; text-transform:uppercase;'>NER Annotations</div>
        <div style='font-size:2rem; font-weight:800; color:#c084fc; margin:6px 0 2px;'>{n_ner}</div>
        <div style='font-size:0.78rem; color:#64748b;'>Labeled training samples</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)

# ── Dataset Structure Breakdown ───────────────────────────────────────────────
st.markdown("""
<div class='ui-card'>
    <div class='ui-card-header'>
        <span>Dataset Folder Conventions</span>
        <span style='color:#64748b; font-size:0.75rem;'>Partitioning Guide</span>
    </div>
    <table class='table-container'>
        <thead>
            <tr>
                <th>Directory Path</th>
                <th>Content Type</th>
                <th>Target Pipeline Stage</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td><code>data/raw/prescriptions/</code></td>
                <td>Raw unprocessed handwritten prescription images</td>
                <td>Input Inference</td>
            </tr>
            <tr>
                <td><code>data/words_lines/</code></td>
                <td>Segmented line slices + <code>labels.jsonl</code></td>
                <td>TrOCR Fine-Tuning (Colab GPU)</td>
            </tr>
            <tr>
                <td><code>data/characters/</code></td>
                <td>Individual character crops (A-Z, a-z, 0-9)</td>
                <td>Custom CharCNN Training</td>
            </tr>
            <tr>
                <td><code>data/ner/</code></td>
                <td>Annotated entity spans in JSONL format</td>
                <td>spaCy Clinical NER Training</td>
            </tr>
            <tr>
                <td><code>data/error_rules/</code></td>
                <td>Pharmacology reference tables (medicines, dosages, interactions)</td>
                <td>Rule-based Error Screening</td>
            </tr>
        </tbody>
    </table>
</div>
""", unsafe_allow_html=True)
