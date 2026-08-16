"""
dashboard/pages/4_About.py
===========================
Project Information & Technical Specifications
"""

import sys
from pathlib import Path
import streamlit as st

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.utils.ui_theme import get_theme_css, render_sidebar

st.set_page_config(
    page_title="About · Smart Prescription",
    page_icon="ℹ️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(get_theme_css(), unsafe_allow_html=True)
render_sidebar()

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='margin-bottom: 20px;'>
    <h1 style='margin:0; font-size:1.8rem; color:#f8fafc !important; font-weight:800;'>
        About the Project
    </h1>
    <p style='margin:4px 0 0; color:#94a3b8; font-size:0.92rem;'>
        Smart Prescription Error Detection — an AI-assisted handwriting recognition and clinical safety platform.
    </p>
</div>
<hr style='border-color:rgba(255,255,255,0.07); margin:10px 0 24px;'>
""", unsafe_allow_html=True)

# ── Project Overview ──────────────────────────────────────────────────────────
st.markdown("""
<div class='ui-card'>
    <div class='ui-card-header'><span>Project Objective</span></div>
    <p style='color:#cbd5e1; font-size:0.92rem; line-height:1.7; margin:0 0 12px;'>
        Handwritten medical prescriptions remain a leading cause of medication dispensing errors due to illegible physician handwriting, non-standard abbreviations, and complex dosage schedules. 
        <strong>Smart Prescription Error Detection</strong> provides an end-to-end intelligent document analysis pipeline combining deep computer vision (Vision Transformers) with clinical NLP (Named Entity Recognition and fuzzy rule validation) to extract prescription fields and flag potential errors before medications are dispensed.
    </p>
</div>
""", unsafe_allow_html=True)

# ── Technologies & Badges ─────────────────────────────────────────────────────
st.markdown("""
<div class='ui-card'>
    <div class='ui-card-header'><span>Technologies & Frameworks</span></div>
    <div style='display:flex; flex-wrap:wrap; gap:8px;'>
        <span style='background:#1e293b; border:1px solid rgba(255,255,255,0.1); color:#f8fafc; padding:4px 14px; border-radius:6px; font-weight:600; font-size:0.82rem;'>Python</span>
        <span style='background:#1e293b; border:1px solid rgba(255,255,255,0.1); color:#f8fafc; padding:4px 14px; border-radius:6px; font-weight:600; font-size:0.82rem;'>OpenCV</span>
        <span style='background:#1e293b; border:1px solid rgba(255,255,255,0.1); color:#f8fafc; padding:4px 14px; border-radius:6px; font-weight:600; font-size:0.82rem;'>PyTorch CNN</span>
        <span style='background:#1e293b; border:1px solid rgba(255,255,255,0.1); color:#f8fafc; padding:4px 14px; border-radius:6px; font-weight:600; font-size:0.82rem;'>Hugging Face</span>
        <span style='background:#1e293b; border:1px solid rgba(255,255,255,0.1); color:#f8fafc; padding:4px 14px; border-radius:6px; font-weight:600; font-size:0.82rem;'>TrOCR</span>
        <span style='background:#1e293b; border:1px solid rgba(255,255,255,0.1); color:#f8fafc; padding:4px 14px; border-radius:6px; font-weight:600; font-size:0.82rem;'>spaCy NLP</span>
        <span style='background:#1e293b; border:1px solid rgba(255,255,255,0.1); color:#f8fafc; padding:4px 14px; border-radius:6px; font-weight:600; font-size:0.82rem;'>Clinical NER</span>
        <span style='background:#1e293b; border:1px solid rgba(255,255,255,0.1); color:#f8fafc; padding:4px 14px; border-radius:6px; font-weight:600; font-size:0.82rem;'>RapidFuzz</span>
        <span style='background:#1e293b; border:1px solid rgba(255,255,255,0.1); color:#f8fafc; padding:4px 14px; border-radius:6px; font-weight:600; font-size:0.82rem;'>Streamlit</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Models Used ───────────────────────────────────────────────────────────────
st.markdown("""
<div class='ui-card'>
    <div class='ui-card-header'><span>Core Models & Algorithms</span></div>
    <div style='display:grid; grid-template-columns: 1fr 1fr; gap:14px;'>
        <div style='background:#070b12; border:1px solid rgba(255,255,255,0.08); border-radius:8px; padding:12px;'>
            <div style='font-weight:700; color:#38bdf8; font-size:0.88rem;'>Vision Transformer (TrOCR)</div>
            <div style='color:#94a3b8; font-size:0.82rem; margin-top:2px;'>Encoder: Vision Transformer (ViT) · Decoder: RoBERTa Causal LM</div>
        </div>
        <div style='background:#070b12; border:1px solid rgba(255,255,255,0.08); border-radius:8px; padding:12px;'>
            <div style='font-weight:700; color:#34d399; font-size:0.88rem;'>Custom CharCNN</div>
            <div style='color:#94a3b8; font-size:0.82rem; margin-top:2px;'>3-Block Convolutional Network with BatchNorm & Dropout</div>
        </div>
        <div style='background:#070b12; border:1px solid rgba(255,255,255,0.08); border-radius:8px; padding:12px;'>
            <div style='font-weight:700; color:#c084fc; font-size:0.88rem;'>Transition-based NER</div>
            <div style='color:#94a3b8; font-size:0.82rem; margin-top:2px;'>HashEmbedCNN feature extractor with transition parser</div>
        </div>
        <div style='background:#070b12; border:1px solid rgba(255,255,255,0.08); border-radius:8px; padding:12px;'>
            <div style='font-weight:700; color:#fbbf24; font-size:0.88rem;'>Pharmacology Rule Matcher</div>
            <div style='color:#94a3b8; font-size:0.82rem; margin-top:2px;'>Token-sort fuzzy ratio, range parsing & interaction lookup</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Project Team & Academic Credentials ───────────────────────────────────────
st.markdown("""
<div class='ui-card'>
    <div class='ui-card-header'><span>Project Team & Affiliation</span></div>
    <div style='display:flex; justify-content:space-between; align-items:center;'>
        <div>
            <div style='font-weight:700; color:#f8fafc; font-size:0.95rem;'>Natural Language Processing (NLP) · 5th Semester</div>
            <div style='color:#94a3b8; font-size:0.85rem;'>SLTC Research University</div>
        </div>
        <div class='sev-badge-none'>Academic Final Year Project</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Small Academic Disclaimer ─────────────────────────────────────────────────
st.markdown("""
<div class='academic-disclaimer'>
    <strong>Academic Disclaimer:</strong> This system is an academic AI prototype intended to assist prescription review and does not replace professional medical judgment.
</div>
""", unsafe_allow_html=True)
