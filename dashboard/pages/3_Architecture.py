"""
dashboard/pages/3_Architecture.py
==================================
System Architecture & Data Flow Visualization
"""

import sys
from pathlib import Path
import streamlit as st

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.utils.ui_theme import get_theme_css, render_sidebar

st.set_page_config(
    page_title="Architecture · Smart Prescription",
    page_icon="🔗",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(get_theme_css(), unsafe_allow_html=True)
render_sidebar()

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='margin-bottom: 20px;'>
    <h1 style='margin:0; font-size:1.8rem; color:#f8fafc !important; font-weight:800;'>
        System Architecture
    </h1>
    <p style='margin:4px 0 0; color:#94a3b8; font-size:0.92rem;'>
        End-to-end multi-stage pipeline: from raw handwritten prescription scan to verified clinical output.
    </p>
</div>
<hr style='border-color:rgba(255,255,255,0.07); margin:10px 0 24px;'>
""", unsafe_allow_html=True)

# ── Visual Pipeline Diagram ───────────────────────────────────────────────────
pipeline_stages = [
    ("1", "Prescription Image", "Input scan (JPG / PNG / BMP) of handwritten doctor's prescription", "#38bdf8"),
    ("2", "Image Processing", "Deskewing (Hough transforms), adaptive Gaussian thresholding, and horizontal projection profile line segmentation", "#0284c7"),
    ("3", "OCR Engine", "Vision Transformer (TrOCR) encoder-decoder transcribes segmented handwriting lines into digital text", "#0f766e"),
    ("4", "Text Extraction", "Consolidation of per-line token sequences and confidence proxy calculation", "#34d399"),
    ("5", "NLP Entity Structuring", "Custom spaCy Named Entity Recognition model extracts MEDICINE, DOSAGE, FREQUENCY, and DURATION entities", "#c084fc"),
    ("6", "Error Detection Engine", "Pharmacological rule screening: RapidFuzz medicine spelling, dosage threshold validation, frequency regex syntax, and interaction pairs", "#fbbf24"),
    ("7", "Final Analysis & Review", "Structured clinical report with confidence metrics, flagged warnings, and exportable JSON payload", "#10b981"),
]

st.markdown("<div class='ui-card'>", unsafe_allow_html=True)
st.markdown("<div class='ui-card-header'><span>Pipeline Data Flow</span></div>", unsafe_allow_html=True)

for num, title, desc, color in pipeline_stages:
    st.markdown(f"""
    <div style='display:flex; align-items:flex-start; gap:16px; margin-bottom:14px;'>
        <div style='background:{color}; color:#090d16; font-weight:800; font-size:0.88rem; width:30px; height:30px; border-radius:50%; display:flex; align-items:center; justify-content:center; flex-shrink:0; margin-top:2px;'>
            {num}
        </div>
        <div style='flex:1; background:#070b12; border:1px solid rgba(255,255,255,0.08); border-radius:8px; padding:12px 16px;'>
            <div style='font-weight:700; color:#f8fafc; font-size:0.92rem;'>{title}</div>
            <div style='color:#94a3b8; font-size:0.84rem; margin-top:2px;'>{desc}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    if num != "7":
        st.markdown("<div style='margin-left:14px; height:12px; border-left:2px dashed rgba(255,255,255,0.15); margin-bottom:6px;'></div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

# ── Architecture Design Principles ────────────────────────────────────────────
st.markdown("""
<div class='ui-card'>
    <div class='ui-card-header'><span>Key Architectural Principles</span></div>
    <div style='display:grid; grid-template-columns: 1fr 1fr; gap:16px;'>
        <div style='background:#070b12; border:1px solid rgba(255,255,255,0.08); border-radius:8px; padding:14px;'>
            <div style='font-weight:700; color:#38bdf8; font-size:0.88rem; margin-bottom:4px;'>🧩 Abstract OCR Interface</div>
            <div style='color:#94a3b8; font-size:0.82rem; line-height:1.6;'>
                All OCR engines adhere to the <code>OCRModel</code> abstract base class (<code>src/models/ocr_base.py</code>). Swapping between TrOCR and custom CharCNN is a one-line config change without modifying the pipeline.
            </div>
        </div>
        <div style='background:#070b12; border:1px solid rgba(255,255,255,0.08); border-radius:8px; padding:14px;'>
            <div style='font-weight:700; color:#34d399; font-size:0.88rem; margin-bottom:4px;'>⚡ CPU Optimized & Cloud Deployable</div>
            <div style='color:#94a3b8; font-size:0.82rem; line-height:1.6;'>
                Fine-tuning runs on GPU (Google Colab), while inference is optimized for local CPU and Streamlit Community Cloud with minimal memory overhead and zero GPU dependencies.
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)
