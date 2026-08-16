"""
src/utils/ui_theme.py
======================
Shared modern dark-mode AI medical SaaS theme, typography, and styling for Streamlit.
"""

def get_theme_css() -> str:
    """Return sleek, modern dark-mode CSS tailored for clinical document intelligence."""
    return """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

/* Global Reset & Dark Canvas */
html, body, [class*="css"], .stApp {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
    background-color: #090d16 !important;
    color: #f1f5f9 !important;
}

/* Header blur */
header[data-testid="stHeader"] {
    background-color: rgba(9, 13, 22, 0.85) !important;
    backdrop-filter: blur(10px);
}
#MainMenu, footer { visibility: hidden; }

/* Dark Sidebar */
section[data-testid="stSidebar"] {
    background-color: #070a10 !important;
    border-right: 1px solid rgba(255, 255, 255, 0.07) !important;
    box-shadow: 2px 0 10px rgba(0, 0, 0, 0.4) !important;
}
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] a {
    color: #94a3b8 !important;
    font-size: 0.9rem !important;
    font-weight: 500 !important;
}
section[data-testid="stSidebar"] a:hover {
    color: #38bdf8 !important;
}

/* Modern Dark Cards */
.ui-card {
    background: #0f172a;
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 12px;
    padding: 20px;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
    margin-bottom: 16px;
    transition: all 0.2s ease-in-out;
}
.ui-card:hover {
    border-color: rgba(56, 189, 248, 0.35);
    box-shadow: 0 6px 20px rgba(0, 0, 0, 0.35);
}

.ui-card-header {
    font-size: 0.82rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #94a3b8;
    margin-bottom: 12px;
    display: flex;
    align-items: center;
    justify-content: space-between;
}

/* System Status Pill */
.status-pill-ready {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background-color: rgba(16, 185, 129, 0.12);
    border: 1px solid rgba(16, 185, 129, 0.3);
    color: #34d399;
    padding: 4px 12px;
    border-radius: 9999px;
    font-size: 0.76rem;
    font-weight: 600;
}
.status-dot-green {
    width: 6px;
    height: 6px;
    background-color: #10b981;
    border-radius: 50%;
    box-shadow: 0 0 8px #10b981;
}

/* Visual Workflow Tracker */
.workflow-track {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 10px 16px;
    background: #0f172a;
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 10px;
    margin-bottom: 20px;
    font-size: 0.82rem;
    color: #64748b;
}
.workflow-step-done {
    color: #38bdf8;
    font-weight: 600;
}
.workflow-step-active {
    color: #f8fafc;
    font-weight: 700;
    background: rgba(56, 189, 248, 0.15);
    border: 1px solid rgba(56, 189, 248, 0.3);
    padding: 3px 10px;
    border-radius: 6px;
}

/* Primary Action Button */
.stButton > button {
    background: linear-gradient(135deg, #0284c7 0%, #2563eb 100%) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 10px 22px !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    letter-spacing: -0.01em !important;
    box-shadow: 0 4px 14px rgba(2, 132, 199, 0.3) !important;
    transition: all 0.15s ease-in-out !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #0369a1 0%, #1d4ed8 100%) !important;
    box-shadow: 0 6px 20px rgba(2, 132, 199, 0.5) !important;
    transform: translateY(-1px);
}
.stButton > button:active {
    transform: translateY(0);
}

/* Secondary Download Button */
.stDownloadButton > button {
    background: #1e293b !important;
    color: #f8fafc !important;
    border: 1px solid rgba(255, 255, 255, 0.12) !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
}
.stDownloadButton > button:hover {
    background: #334155 !important;
    border-color: rgba(255, 255, 255, 0.25) !important;
}

/* Monospace OCR Box */
.ocr-document-box {
    background: #060911;
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 10px;
    padding: 18px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.9rem;
    line-height: 1.7;
    color: #e2e8f0;
    white-space: pre-wrap;
    box-shadow: inset 0 2px 4px rgba(0,0,0,0.4);
}

/* Data Table Styling (Dark) */
.table-container {
    width: 100%;
    border-collapse: separate;
    border-spacing: 0;
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 10px;
    overflow: hidden;
    margin-top: 10px;
    background: #0f172a;
}
.table-container th {
    background: #1e293b;
    color: #94a3b8;
    font-size: 0.78rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    padding: 10px 16px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.08);
    text-align: left;
}
.table-container td {
    padding: 12px 16px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.05);
    color: #f1f5f9;
    font-size: 0.88rem;
}
.table-container tr:last-child td {
    border-bottom: none;
}
.table-container tr:hover td {
    background-color: rgba(255, 255, 255, 0.025);
}

/* Dark Severity Badges */
.sev-badge-none {
    background-color: rgba(16, 185, 129, 0.15);
    border: 1px solid rgba(16, 185, 129, 0.35);
    color: #34d399;
    padding: 2px 10px;
    border-radius: 6px;
    font-size: 0.78rem;
    font-weight: 600;
}
.sev-badge-possible {
    background-color: rgba(245, 158, 11, 0.15);
    border: 1px solid rgba(245, 158, 11, 0.35);
    color: #fbbf24;
    padding: 2px 10px;
    border-radius: 6px;
    font-size: 0.78rem;
    font-weight: 600;
}
.sev-badge-required {
    background-color: rgba(239, 68, 68, 0.15);
    border: 1px solid rgba(239, 68, 68, 0.35);
    color: #f87171;
    padding: 2px 10px;
    border-radius: 6px;
    font-size: 0.78rem;
    font-weight: 600;
}

/* Dark File Uploader */
[data-testid="stFileUploader"] {
    background: #0f172a;
    border: 2px dashed rgba(255, 255, 255, 0.15);
    border-radius: 12px;
    padding: 20px 14px;
    transition: border-color 0.2s ease;
}
[data-testid="stFileUploader"]:hover {
    border-color: #38bdf8;
}

/* Metric Bars */
.confidence-meter {
    background-color: #1e293b;
    border-radius: 9999px;
    height: 7px;
    overflow: hidden;
    margin-top: 4px;
}
.confidence-fill {
    height: 100%;
    border-radius: 9999px;
    background: linear-gradient(90deg, #0284c7, #38bdf8);
}

/* Dark Tabs */
.stTabs [data-baseweb="tab-list"] {
    background-color: #0f172a;
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 8px;
    padding: 4px;
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    background-color: transparent;
    color: #94a3b8;
    border-radius: 6px;
    font-size: 0.85rem;
    font-weight: 600;
    padding: 6px 16px;
    border: none !important;
}
.stTabs [aria-selected="true"] {
    background-color: #1e293b !important;
    color: #f8fafc !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.3);
}

/* Text Area & Code in Dark Mode */
.stTextArea textarea {
    background-color: #060911 !important;
    color: #f1f5f9 !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    font-family: 'JetBrains Mono', monospace !important;
}

/* Streamlit Metrics */
[data-testid="stMetric"] {
    background: #0f172a !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 10px !important;
    padding: 12px 16px !important;
}
[data-testid="stMetricValue"] {
    color: #f8fafc !important;
    font-weight: 700 !important;
}
[data-testid="stMetricLabel"] {
    color: #94a3b8 !important;
    font-size: 0.8rem !important;
}

/* Headings color */
h1, h2, h3, h4 {
    color: #f8fafc !important;
    font-weight: 700 !important;
    letter-spacing: -0.01em;
}
p, li { color: #94a3b8; }
code {
    font-family: 'JetBrains Mono', monospace;
    background: #070b12;
    border: 1px solid rgba(255, 255, 255, 0.08);
    color: #38bdf8;
    padding: 2px 6px;
    border-radius: 6px;
}

/* Disclaimer text */
.academic-disclaimer {
    background: #0f172a;
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 8px;
    padding: 12px 16px;
    color: #94a3b8;
    font-size: 0.8rem;
    line-height: 1.5;
}
</style>
"""

def render_sidebar():
    """Render sleek dark-mode sidebar navigation."""
    import streamlit as st
    with st.sidebar:
        st.markdown("""
        <div style='padding: 8px 0 12px;'>
            <div style='display:flex; align-items:center; gap:10px;'>
                <div style='background:linear-gradient(135deg, #0284c7, #2563eb); color:#ffffff; font-weight:800; font-size:0.9rem; width:34px; height:34px; border-radius:8px; display:flex; align-items:center; justify-content:center; box-shadow:0 0 12px rgba(2, 132, 199, 0.4);'>
                    SP
                </div>
                <div>
                    <h3 style='margin:0; font-size:1.05rem; color:#f8fafc !important;'>Smart Prescription</h3>
                    <p style='margin:0; font-size:0.72rem; color:#64748b;'>Document Intelligence</p>
                </div>
            </div>
        </div>
        <div class='status-pill-ready' style='margin-bottom:18px;'>
            <div class='status-dot-green'></div> AI System Ready
        </div>
        <hr style='border-color:rgba(255, 255, 255, 0.07); margin:8px 0 16px;'>
        """, unsafe_allow_html=True)

        st.markdown("<p style='font-size:0.75rem; font-weight:700; text-transform:uppercase; color:#64748b; margin-bottom:8px; letter-spacing:0.05em;'>Platform</p>", unsafe_allow_html=True)
        st.page_link("app.py", label="Prescription Analysis", icon="🏠")
        st.page_link("pages/1_Model_Performance.py", label="Model Performance", icon="📊")
        st.page_link("pages/2_Dataset.py", label="Dataset", icon="📁")
        st.page_link("pages/3_Architecture.py", label="Architecture", icon="🔗")
        st.page_link("pages/4_About.py", label="About", icon="ℹ️")

        st.markdown("""
        <hr style='border-color:rgba(255, 255, 255, 0.07); margin:24px 0 16px;'>
        <div style='background:#0f172a; border:1px solid rgba(255, 255, 255, 0.07); border-radius:8px; padding:10px; font-size:0.75rem; color:#64748b;'>
            <div style='font-weight:600; color:#cbd5e1; margin-bottom:2px;'>Academic Prototype</div>
            <div>SLTC Research University</div>
        </div>
        """, unsafe_allow_html=True)
