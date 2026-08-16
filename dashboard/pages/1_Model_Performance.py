"""
dashboard/pages/1_Model_Performance.py
========================================
Model Performance Evaluation & Benchmarks
"""

import json
import sys
from pathlib import Path
import streamlit as st
import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.utils.config_loader import load_config
from src.utils.ui_theme import get_theme_css, render_sidebar

st.set_page_config(
    page_title="Model Performance · Smart Prescription",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(get_theme_css(), unsafe_allow_html=True)
render_sidebar()

cfg = load_config()

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='margin-bottom: 20px;'>
    <h1 style='margin:0; font-size:1.8rem; color:#f8fafc !important; font-weight:800;'>
        Model Performance
    </h1>
    <p style='margin:4px 0 0; color:#94a3b8; font-size:0.92rem;'>
        Evaluation metrics and benchmark comparisons across OCR, character classification, and clinical entity extraction models.
    </p>
</div>
<hr style='border-color:rgba(255,255,255,0.07); margin:10px 0 20px;'>
""", unsafe_allow_html=True)

# ── Overview Comparison Cards ─────────────────────────────────────────────────
c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("""
    <div class='ui-card'>
        <div class='ui-card-header'>
            <span>Hugging Face TrOCR</span>
            <span class='sev-badge-none'>Line OCR</span>
        </div>
        <div style='font-size:1.8rem; font-weight:700; color:#38bdf8; margin:6px 0 2px;'>
            TrOCR Base / Small
        </div>
        <div style='color:#94a3b8; font-size:0.84rem;'>
            Task: End-to-end handwriting line transcription using Vision Transformer & RoBERTa.
        </div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown("""
    <div class='ui-card'>
        <div class='ui-card-header'>
            <span>Custom CNN</span>
            <span class='sev-badge-possible'>Fallback</span>
        </div>
        <div style='font-size:1.8rem; font-weight:700; color:#cbd5e1; margin:6px 0 2px;'>
            CharCNN (3-Block)
        </div>
        <div style='color:#94a3b8; font-size:0.84rem;'>
            Task: Single-character (A-Z, a-z, 0-9) classification on segmented letter crops.
        </div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown("""
    <div class='ui-card'>
        <div class='ui-card-header'>
            <span>Custom spaCy NER</span>
            <span class='sev-badge-none'>NLP Extraction</span>
        </div>
        <div style='font-size:1.8rem; font-weight:700; color:#34d399; margin:6px 0 2px;'>
            Transition NER
        </div>
        <div style='color:#94a3b8; font-size:0.84rem;'>
            Task: Structuring extracted lines into Medicine, Dosage, Frequency & Duration spans.
        </div>
    </div>
    """, unsafe_allow_html=True)

# ── Detailed Model Breakdown ──────────────────────────────────────────────────
st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)

tab_trocr, tab_cnn, tab_ner = st.tabs([
    "✍️ TrOCR (Line OCR)",
    "🔤 Custom CNN (Single Character)",
    "🏷️ Clinical NER (Entity Extraction)",
])

# ─── TAB 1: TrOCR ─────────────────────────────────────────────────────────────
with tab_trocr:
    trocr_eval_path = _ROOT / "models" / "trocr_finetuned" / "eval_results.json"
    
    cer = None
    wer = None
    loss = None
    
    if trocr_eval_path.exists():
        try:
            ev = json.loads(trocr_eval_path.read_text())
            cer = ev.get("eval_cer", ev.get("cer"))
            wer = ev.get("eval_wer", ev.get("wer"))
            loss = ev.get("eval_loss")
        except Exception:
            pass

    st.markdown("<div class='ui-card'>", unsafe_allow_html=True)
    st.markdown("<div class='ui-card-header'><span>TrOCR Performance Metrics</span></div>", unsafe_allow_html=True)

    if cer is not None:
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Character Error Rate (CER)", f"{cer*100:.2f}%", delta="Lower is better", delta_color="inverse")
        with m2:
            st.metric("Word Error Rate (WER)", f"{wer*100:.2f}%" if wer else "N/A", delta="Lower is better", delta_color="inverse")
        with m3:
            st.metric("Evaluation Loss", f"{loss:.4f}" if loss else "N/A")
    else:
        st.markdown("""
        <div style='background:#f8fafc; border:1px solid #e2e8f0; border-radius:8px; padding:16px;'>
            <p style='color:#334155; margin:0 0 6px; font-weight:600;'>Pretrained Model Active</p>
            <p style='color:#64748b; font-size:0.86rem; margin:0;'>
                Currently running <code>microsoft/trocr-small-handwritten</code> from Hugging Face Hub.
                When fine-tuning on Google Colab (<code>notebooks/03_trocr_finetuning_colab.ipynb</code>), 
                the generated <code>eval_results.json</code> will populate CER and WER benchmarks here.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ─── TAB 2: Custom CNN ────────────────────────────────────────────────────────
with tab_cnn:
    cnn_eval_path = _ROOT / "evaluation" / "cnn_eval" / "eval_summary.json"
    
    st.markdown("<div class='ui-card'>", unsafe_allow_html=True)
    st.markdown("<div class='ui-card-header'><span>CharCNN Benchmark</span></div>", unsafe_allow_html=True)

    if cnn_eval_path.exists():
        try:
            cnn_data = json.loads(cnn_eval_path.read_text())
            c_acc = cnn_data.get("accuracy", 0.0)
            c_prec = cnn_data.get("macro_precision", 0.0)
            c_rec = cnn_data.get("macro_recall", 0.0)
            c_f1 = cnn_data.get("macro_f1", 0.0)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Accuracy", f"{c_acc*100:.1f}%")
            c2.metric("Precision", f"{c_prec*100:.1f}%")
            c3.metric("Recall", f"{c_rec*100:.1f}%")
            c4.metric("F1 Score", f"{c_f1*100:.1f}%")
        except Exception:
            cnn_eval_path = None

    if not cnn_eval_path or not cnn_eval_path.exists():
        st.markdown("""
        <div style='background:#f8fafc; border:1px solid #e2e8f0; border-radius:8px; padding:16px;'>
            <p style='color:#334155; margin:0 0 6px; font-weight:600;'>Custom Character Model (Plug-in Roadmap)</p>
            <p style='color:#64748b; font-size:0.86rem; margin:0;'>
                The Custom CNN character model can be trained locally on character crops (<code>python -m src.models.train_cnn</code>). 
                Once trained, accuracy, precision, recall, and confusion matrices will display automatically.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ─── TAB 3: Clinical NER ──────────────────────────────────────────────────────
with tab_ner:
    # Read which model is active from config
    active_ner = cfg.get("active_ner_model", "spacy")
    hf_ner_name = cfg.get("hf_ner_model_name", "Posos/ClinicalNER")

    # ── Active model status banner ────────────────────────────────────────────
    if active_ner == "hf_clinical":
        badge_html = f"<span class='sev-badge-none'>● Active: HF Clinical NER ({hf_ner_name})</span>"
        active_label = f"Hugging Face — {hf_ner_name}"
        active_desc = "Pre-trained BERT clinical NER model, no custom training required."
    else:
        badge_html = "<span class='sev-badge-none'>● Active: Custom spaCy NER</span>"
        active_label = "Custom spaCy (trained on your data)"
        active_desc = "Transition-based NER model trained on YOUR prescription corpus."

    st.markdown(f"""
    <div class='ui-card'>
        <div class='ui-card-header'>
            <span>Active NER Model</span>
            {badge_html}
        </div>
        <div style='display:grid; grid-template-columns:1fr 1fr; gap:14px; margin-top:8px;'>
            <div style='background:#070b12; border:1px solid rgba(255,255,255,0.08); border-radius:8px; padding:14px;'>
                <div style='font-weight:700; color:#34d399; font-size:0.85rem; margin-bottom:4px;'>
                    {"✅ " if active_ner == "spacy" else "○ "} Custom spaCy NER
                </div>
                <div style='color:#94a3b8; font-size:0.82rem; line-height:1.5;'>
                    Trained on YOUR prescription corpus (<code>data/ner/</code>).<br>
                    Path: <code>models/ner_model/</code><br>
                    Train: <code>python -m src.models.train_ner</code>
                </div>
            </div>
            <div style='background:#070b12; border:1px solid rgba(255,255,255,0.08); border-radius:8px; padding:14px;'>
                <div style='font-weight:700; color:#38bdf8; font-size:0.85rem; margin-bottom:4px;'>
                    {"✅ " if active_ner == "hf_clinical" else "○ "} HF Clinical NER
                </div>
                <div style='color:#94a3b8; font-size:0.82rem; line-height:1.5;'>
                    <code>{hf_ner_name}</code> from Hugging Face Hub.<br>
                    No training required — downloads on first use.<br>
                    Switch: <code>active_ner_model: "hf_clinical"</code> in config.yaml
                </div>
            </div>
        </div>
        <div style='margin-top:12px; padding:10px; background:rgba(56,189,248,0.07); border:1px solid rgba(56,189,248,0.2); border-radius:8px; font-size:0.82rem; color:#94a3b8;'>
            ℹ To switch models: edit <code>active_ner_model</code> in <code>config.yaml</code> — no code changes needed.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Custom spaCy evaluation metrics ──────────────────────────────────────
    st.markdown("<div class='ui-card'>", unsafe_allow_html=True)
    st.markdown("<div class='ui-card-header'><span>Custom spaCy NER — Evaluation</span></div>", unsafe_allow_html=True)

    ner_eval_path = _ROOT / "evaluation" / "ner_eval" / "ner_eval_summary.json"

    if ner_eval_path.exists():
        try:
            ner_ev = json.loads(ner_eval_path.read_text())
            overall = ner_ev.get("overall", {})

            n1, n2, n3 = st.columns(3)
            n1.metric("Overall Precision", f"{overall.get('precision', 0)*100:.1f}%")
            n2.metric("Overall Recall",    f"{overall.get('recall',    0)*100:.1f}%")
            n3.metric("Overall F1-Score",  f"{overall.get('f1',        0)*100:.1f}%")
        except Exception:
            ner_eval_path = None

    if not ner_eval_path or not ner_eval_path.exists():
        st.markdown("""
        <div style='background:#070b12; border:1px solid rgba(255,255,255,0.08); border-radius:8px; padding:14px;'>
            <p style='color:#f8fafc; margin:0 0 6px; font-weight:600;'>Evaluation Metrics</p>
            <p style='color:#94a3b8; font-size:0.86rem; margin:0;'>
                Evaluates exact token boundary extraction across <code>MEDICINE</code>, <code>DOSAGE</code>,
                <code>FREQUENCY</code>, and <code>DURATION</code> entity types.
                Run <code>python -m src.models.train_ner</code> to generate evaluation summaries.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # ── Model comparison placeholder ──────────────────────────────────────────
    st.markdown("""
    <div class='ui-card'>
        <div class='ui-card-header'>
            <span>Side-by-Side Comparison Results</span>
            <span class='sev-badge-possible'>Pending Review</span>
        </div>
        <div style='background:#070b12; border:1px solid rgba(255,255,255,0.08); border-radius:8px; padding:14px;'>
            <p style='color:#f8fafc; margin:0 0 8px; font-weight:600;'>Run the comparison script to populate this section</p>
            <p style='color:#94a3b8; font-size:0.84rem; margin:0 0 10px;'>
                The comparison script runs both NER models on your annotated prescription samples
                and prints a side-by-side table so you can eyeball which performs better on your data.
            </p>
            <code style='color:#38bdf8; font-size:0.82rem;'>
                python scripts/compare_ner_models.py --n 15
            </code>
            <p style='color:#64748b; font-size:0.79rem; margin:8px 0 0;'>
                Options: <code>--skip-spacy</code> · <code>--skip-hf</code> · 
                <code>--text "your prescription text"</code> · <code>--input data/ner/val.jsonl</code>
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

