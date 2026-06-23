# app.py - Predicción Mundial 2026 - Diseño Predictor Pro
import streamlit as st
import pandas as pd
from modules import (
    config,
    data_loader,
    models,
    corrections,
    validation,
    team_roster,
    predictors,
    ui
)
import warnings
warnings.filterwarnings('ignore')

# ⚠️ set_page_config DEBE SER LA PRIMERA INSTRUCCIÓN DE STREAMLIT
st.set_page_config(
    page_title="Predicción Mundial 2026",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CSS PERSONALIZADO - DISEÑO PREDICTOR PRO
# ============================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800;900&family=Lexend:wght@400;600;700&family=JetBrains+Mono:wght@500&display=swap');

    /* ============================================================
       🎨 VARIABLES DE COLOR - PALETA PREDICTOR PRO
       ============================================================ */
    :root {
        /* Colores de fondo */
        --bg-background: #0b1326;
        --bg-surface: #0b1326;
        --bg-surface-container-low: #131b2e;
        --bg-surface-container: #171f33;
        --bg-surface-container-high: #222a3d;
        --bg-surface-container-lowest: #060e20;
        --bg-surface-variant: #2d3449;
        
        /* Colores de borde */
        --border-outline: #849495;
        --border-outline-variant: #3b494b;
        
        /* Colores de texto */
        --text-on-surface: #dae2fd;
        --text-on-surface-variant: #b9cacb;
        --text-primary: #dbfcff;
        --text-secondary: #d1bcff;
        --text-on-secondary: #3c0090;
        --text-on-primary: #00363a;
        --text-on-tertiary: #67001f;
        
        /* Colores de acento */
        --accent-primary-fixed-dim: #00dbe9;
        --accent-primary-fixed: #7df4ff;
        --accent-secondary: #d1bcff;
        --accent-secondary-container: #7000ff;
        --accent-tertiary: #fff3f3;
        --accent-error: #ffb4ab;
        --accent-error-container: #93000a;
        
        /* Sombras y efectos */
        --shadow-glow: 0 0 30px rgba(0, 219, 233, 0.05);
        --glass-bg: rgba(23, 31, 51, 0.7);
    }

    /* ============================================================
       🎨 FONDO DE LA APP
       ============================================================ */
    .stApp {
        background: var(--bg-background);
        font-family: 'Inter', sans-serif;
        color: var(--text-on-surface);
    }

    /* ============================================================
       🎨 SIDEBAR - ESTILO DARK MODERN
       ============================================================ */
    [data-testid="stSidebar"] {
        background: var(--bg-surface-container-low) !important;
        border-right: 1px solid var(--border-outline-variant) !important;
        padding-top: 0 !important;
    }

    [data-testid="stSidebar"] .stMarkdown,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stCaption {
        color: var(--text-on-surface) !important;
        font-family: 'Inter', sans-serif !important;
    }

    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3 {
        color: var(--accent-primary-fixed-dim) !important;
        font-family: 'Lexend', sans-serif !important;
        font-weight: 700 !important;
    }

    /* Sidebar - Selectores personalizados */
    [data-testid="stSidebar"] .stSelectbox > div > div {
        background: var(--bg-surface-container-lowest) !important;
        border: 1px solid var(--border-outline-variant) !important;
        border-radius: 8px !important;
        color: var(--text-on-surface) !important;
        padding: 8px 12px !important;
        transition: all 0.3s ease !important;
    }
    [data-testid="stSidebar"] .stSelectbox > div > div:hover {
        border-color: var(--accent-primary-fixed-dim) !important;
        box-shadow: 0 0 20px rgba(0, 219, 233, 0.1) !important;
    }
    [data-testid="stSidebar"] .stSelectbox > div > div:focus {
        ring: 2px solid var(--accent-primary-fixed-dim) !important;
    }

    /* Sidebar - Checkboxes */
    [data-testid="stSidebar"] .stCheckbox > div > div > div {
        background: var(--accent-secondary-container) !important;
        border-radius: 4px !important;
    }
    [data-testid="stSidebar"] .stCheckbox label {
        color: var(--text-on-surface) !important;
    }

    /* Sidebar - Sliders */
    [data-testid="stSidebar"] .stSlider > div > div > div {
        background: var(--accent-primary-fixed-dim) !important;
    }

    /* Sidebar - Date Input */
    [data-testid="stSidebar"] .stDateInput > div > div {
        background: var(--bg-surface-container-lowest) !important;
        border: 1px solid var(--border-outline-variant) !important;
        border-radius: 8px !important;
        color: var(--text-on-surface) !important;
    }

    /* Sidebar - Buttons */
    [data-testid="stSidebar"] .stButton > button {
        background: var(--accent-secondary) !important;
        color: var(--text-on-secondary) !important;
        font-weight: 700 !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 16px 24px !important;
        font-family: 'Inter', sans-serif !important;
        letter-spacing: 0.5px !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 20px rgba(209, 188, 255, 0.3) !important;
        text-transform: uppercase;
        font-size: 0.85rem !important;
    }
    [data-testid="stSidebar"] .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(209, 188, 255, 0.4) !important;
        opacity: 0.9;
    }
    [data-testid="stSidebar"] .stButton > button:active {
        transform: scale(0.95);
    }

    /* Sidebar - Expander */
    [data-testid="stSidebar"] .streamlit-expanderHeader {
        background: var(--bg-surface-container) !important;
        border: 1px solid var(--border-outline-variant) !important;
        border-radius: 8px !important;
        color: var(--text-primary) !important;
    }

    /* Sidebar - Captions */
    [data-testid="stSidebar"] .stCaption {
        color: var(--text-on-surface-variant) !important;
        font-size: 0.75rem !important;
    }

    /* ============================================================
       🎨 HEADERS
       ============================================================ */
    h1, h2, h3, h4 {
        font-family: 'Lexend', sans-serif !important;
        color: var(--text-primary) !important;
    }
    h1 {
        font-weight: 800 !important;
        letter-spacing: -0.5px;
        font-size: 2.5rem !important;
    }

    /* ============================================================
       🎨 TITLE BAR
       ============================================================ */
    .title-bar {
        background: linear-gradient(135deg, #0b1326, #171f33);
        border: 1px solid var(--border-outline-variant);
        border-radius: 16px;
        padding: 20px 28px;
        margin-bottom: 24px;
        position: relative;
        overflow: hidden;
        transition: all 0.3s ease;
    }
    .title-bar:hover {
        border-color: rgba(0, 219, 233, 0.3);
        box-shadow: 0 0 40px rgba(0, 219, 233, 0.05);
    }
    .title-bar::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, #00dbe9, #d1bcff, #00dbe9);
        background-size: 200% 100%;
        animation: shimmer 3s linear infinite;
    }
    @keyframes shimmer {
        0% { background-position: 200% 0; }
        100% { background-position: -200% 0; }
    }
    .title-bar h1 {
        font-family: 'Lexend', sans-serif !important;
        font-size: 2rem !important;
        letter-spacing: -0.02em !important;
        margin: 0 !important;
        color: var(--text-primary);
        text-shadow: 0 0 30px rgba(0, 219, 233, 0.2);
    }
    .title-bar p {
        color: var(--text-on-surface-variant);
        margin: 4px 0 0 0;
        font-size: 0.95rem;
        letter-spacing: 0.3px;
    }
    .title-bar .badge-row {
        display: flex;
        gap: 8px;
        margin-top: 12px;
        flex-wrap: wrap;
    }

    /* ============================================================
       🎨 BADGES
       ============================================================ */
    .badge {
        background: rgba(0, 219, 233, 0.08);
        border: 1px solid rgba(0, 219, 233, 0.15);
        color: var(--accent-primary-fixed-dim);
        padding: 4px 14px;
        border-radius: 20px;
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        transition: all 0.3s ease;
        font-family: 'JetBrains Mono', monospace;
    }
    .badge:hover {
        background: rgba(0, 219, 233, 0.15);
        transform: translateY(-1px);
    }
    .badge.gold {
        background: rgba(209, 188, 255, 0.12);
        border-color: rgba(209, 188, 255, 0.2);
        color: var(--accent-secondary);
    }
    .badge.gold:hover {
        background: rgba(209, 188, 255, 0.2);
    }

    /* ============================================================
       🎨 STATUS BAR
       ============================================================ */
    .status-bar {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 12px;
        margin-bottom: 16px;
    }
    .status-item {
        background: var(--glass-bg);
        backdrop-filter: blur(12px);
        border: 1px solid var(--border-outline-variant);
        border-radius: 12px;
        padding: 12px 16px;
        text-align: center;
        transition: all 0.3s ease;
    }
    .status-item:hover {
        border-color: rgba(0, 219, 233, 0.3);
        box-shadow: var(--shadow-glow);
    }
    .status-item .label {
        color: var(--text-on-surface-variant);
        font-size: 0.65rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        font-family: 'JetBrains Mono', monospace;
        font-weight: 500;
    }
    .status-item .value {
        color: var(--text-primary);
        font-size: 1.2rem;
        font-weight: 700;
        margin-top: 4px;
        font-family: 'Lexend', sans-serif;
    }
    .status-item .value.cyan { color: var(--accent-primary-fixed-dim); }
    .status-item .value.gold { color: var(--accent-secondary); }
    .status-item .value.green { color: #7df4ff; }
    .status-item .value.red { color: var(--accent-error); }

    /* ============================================================
       🎨 METRIC CARDS
       ============================================================ */
    [data-testid="stMetric"] {
        background: var(--glass-bg) !important;
        backdrop-filter: blur(12px) !important;
        border: 1px solid var(--border-outline-variant) !important;
        border-radius: 12px !important;
        padding: 16px 20px !important;
        transition: all 0.3s ease !important;
    }
    [data-testid="stMetric"]:hover {
        border-color: rgba(0, 219, 233, 0.3) !important;
        box-shadow: var(--shadow-glow) !important;
        transform: translateY(-2px) !important;
    }
    [data-testid="stMetric"] label {
        color: var(--text-on-surface-variant) !important;
        font-size: 0.7rem !important;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        font-family: 'JetBrains Mono', monospace;
        font-weight: 500 !important;
    }
    [data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: var(--text-primary) !important;
        font-size: 1.8rem !important;
        font-weight: 700 !important;
        font-family: 'Lexend', sans-serif !important;
    }
    [data-testid="stMetric"] [data-testid="stMetricDelta"] {
        color: var(--accent-primary-fixed-dim) !important;
        font-size: 0.85rem !important;
        font-family: 'Inter', sans-serif !important;
    }

    /* ============================================================
       🎨 DATA FRAMES
       ============================================================ */
    .stDataFrame {
        border: 1px solid var(--border-outline-variant) !important;
        border-radius: 12px !important;
        overflow: hidden !important;
        background: var(--bg-surface-container) !important;
    }
    .stDataFrame th {
        background: var(--bg-surface-container-low) !important;
        color: var(--text-primary) !important;
        font-weight: 600 !important;
        font-family: 'Inter', sans-serif !important;
        border-bottom: 1px solid var(--border-outline-variant) !important;
    }
    .stDataFrame td {
        color: var(--text-on-surface) !important;
        border-bottom: 1px solid var(--border-outline-variant) !important;
    }

    /* ============================================================
       🎨 EXPANDERS
       ============================================================ */
    .streamlit-expanderHeader {
        background: var(--bg-surface-container) !important;
        border: 1px solid var(--border-outline-variant) !important;
        border-radius: 8px !important;
        color: var(--text-primary) !important;
        font-family: 'Inter', sans-serif !important;
        transition: all 0.3s ease !important;
    }
    .streamlit-expanderHeader:hover {
        border-color: rgba(0, 219, 233, 0.3) !important;
        background: var(--bg-surface-container-high) !important;
    }
    .streamlit-expanderContent {
        background: var(--bg-surface-container-low) !important;
        border: 1px solid var(--border-outline-variant) !important;
        border-top: none !important;
        border-radius: 0 0 8px 8px !important;
        padding: 16px !important;
    }

    /* ============================================================
       🎨 ALERTAS
       ============================================================ */
    .stAlert {
        border-radius: 12px !important;
        border: 1px solid var(--border-outline-variant) !important;
    }
    .stAlert [data-testid="stAlertContentSuccess"] {
        background: rgba(0, 219, 233, 0.08) !important;
        border: 1px solid rgba(0, 219, 233, 0.2) !important;
        border-radius: 12px !important;
        color: var(--text-primary) !important;
    }
    .stAlert [data-testid="stAlertContentInfo"] {
        background: rgba(209, 188, 255, 0.08) !important;
        border: 1px solid rgba(209, 188, 255, 0.2) !important;
        border-radius: 12px !important;
        color: var(--text-primary) !important;
    }
    .stAlert [data-testid="stAlertContentWarning"] {
        background: rgba(255, 180, 171, 0.08) !important;
        border: 1px solid rgba(255, 180, 171, 0.2) !important;
        border-radius: 12px !important;
        color: var(--text-primary) !important;
    }
    .stAlert [data-testid="stAlertContentError"] {
        background: rgba(147, 0, 10, 0.2) !important;
        border: 1px solid var(--accent-error-container) !important;
        border-radius: 12px !important;
        color: var(--text-primary) !important;
    }

    /* ============================================================
       🎨 SPINNER
       ============================================================ */
    .stSpinner > div {
        border-color: var(--accent-primary-fixed-dim) !important;
        border-top-color: transparent !important;
        border-width: 4px !important;
        width: 48px !important;
        height: 48px !important;
    }

    /* ============================================================
       🎨 DIVIDER
       ============================================================ */
    hr {
        border-color: var(--border-outline-variant) !important;
        margin: 24px 0 !important;
        opacity: 0.5;
    }

    /* ============================================================
       🎨 PROBABILITY BARS
       ============================================================ */
    .prob-bar {
        height: 6px;
        border-radius: 4px;
        background: var(--bg-surface-container-low);
        overflow: hidden;
        margin-top: 8px;
        position: relative;
    }
    .prob-bar-fill {
        height: 100%;
        border-radius: 4px;
        transition: width 0.8s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
    }
    .prob-bar-fill::after {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.15), transparent);
        animation: shimmer-bar 2s infinite;
    }
    @keyframes shimmer-bar {
        0% { transform: translateX(-100%); }
        100% { transform: translateX(100%); }
    }

    /* ============================================================
       🎨 DISCLAIMER
       ============================================================ */
    .disclaimer-box {
        background: rgba(147, 0, 10, 0.15);
        border: 1px solid rgba(255, 180, 171, 0.2);
        border-radius: 12px;
        padding: 16px 20px;
        margin: 12px 0 24px 0;
    }
    .disclaimer-box p {
        color: var(--text-on-surface-variant);
        font-size: 0.85rem;
        margin: 0;
        line-height: 1.6;
        font-family: 'Inter', sans-serif;
    }
    .disclaimer-box strong {
        color: var(--accent-error);
    }
    .disclaimer-box .block-icon {
        color: var(--accent-error);
        font-size: 0.85rem;
        margin-right: 6px;
    }

    /* ============================================================
       🎨 FOOTER
       ============================================================ */
    .footer {
        text-align: center;
        padding: 24px 16px;
        color: var(--text-on-surface-variant);
        font-size: 0.8rem;
        border-top: 1px solid var(--border-outline-variant);
        margin-top: 40px;
        background: var(--bg-surface-container);
        border-radius: 12px;
    }
    .footer a {
        color: var(--accent-primary-fixed-dim);
        text-decoration: none;
        transition: color 0.2s;
        font-weight: 600;
    }
    .footer a:hover {
        color: var(--accent-primary-fixed);
        text-decoration: underline;
    }
    .footer .separator {
        color: var(--border-outline-variant);
        margin: 0 8px;
    }

    /* ============================================================
       🎨 SCROLLBAR
       ============================================================ */
    ::-webkit-scrollbar {
        width: 6px;
        height: 6px;
    }
    ::-webkit-scrollbar-track {
        background: var(--bg-background);
    }
    ::-webkit-scrollbar-thumb {
        background: var(--border-outline-variant);
        border-radius: 8px;
        transition: background 0.3s;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: var(--accent-primary-fixed-dim);
    }

    /* ============================================================
       🎨 OCULTAR ELEMENTOS POR DEFECTO DE STREAMLIT
       ============================================================ */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* ============================================================
       🎨 BOTÓN DE PREDICCIÓN EN SIDEBAR (Estilo especial)
       ============================================================ */
    .predict-btn-container .stButton > button {
        background: linear-gradient(135deg, #d1bcff, #7000ff) !important;
        color: #ffffff !important;
        font-weight: 700 !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 16px 24px !important;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-size: 0.85rem !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 20px rgba(209, 188, 255, 0.3) !important;
    }
    .predict-btn-container .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(209, 188, 255, 0.4) !important;
    }
    .predict-btn-container .stButton > button:active {
        transform: scale(0.95);
    }

    /* ============================================================
       🎨 RESPONSIVE
       ============================================================ */
    @media (max-width: 768px) {
        .title-bar { padding: 16px 20px; }
        .title-bar h1 { font-size: 1.5rem !important; }
        .title-bar p { font-size: 0.8rem; }
        .badge { font-size: 0.55rem; padding: 2px 10px; }
        .status-bar { grid-template-columns: repeat(2, 1fr); gap: 8px; }
        [data-testid="stMetric"] { padding: 10px 14px !important; }
        [data-testid="stMetric"] [data-testid="stMetricValue"] { font-size: 1.3rem !important; }
        .footer { padding: 16px; font-size: 0.7rem; }
        .footer .separator { display: none; }
    }

    @media (max-width: 480px) {
        .status-bar { grid-template-columns: 1fr 1fr; gap: 6px; }
        .status-item { padding: 8px 12px; }
        .status-item .value { font-size: 1rem; }
        .title-bar h1 { font-size: 1.2rem !important; }
        [data-testid="stMetric"] [data-testid="stMetricValue"] { font-size: 1.1rem !important; }
    }

    /* ============================================================
       🎨 MODAL OVERLAY (Para validación)
       ============================================================ */
    .modal-overlay {
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: rgba(11, 19, 38, 0.85);
        backdrop-filter: blur(16px);
        z-index: 9999;
        display: flex;
        align-items: center;
        justify-content: center;
        padding: 20px;
        animation: fadeIn 0.3s ease;
    }
    .modal-content {
        background: linear-gradient(145deg, #0b1326, #171f33) !important;
        border: 1px solid var(--border-outline-variant) !important;
        border-radius: 20px !important;
        max-width: 900px !important;
        width: 100%;
        max-height: 90vh;
        overflow-y: auto;
        padding: 32px;
        position: relative;
        box-shadow: 0 24px 64px rgba(0,0,0,0.6);
        animation: slideUp 0.3s ease;
    }
    .modal-content h2 {
        color: var(--text-primary) !important;
        font-family: 'Lexend', sans-serif !important;
        letter-spacing: -0.02em !important;
        margin-top: 0 !important;
        font-weight: 700 !important;
    }
    .modal-content p {
        color: var(--text-on-surface-variant) !important;
    }
    .modal-content .stMetric {
        background: rgba(255,255,255,0.03) !important;
        border: 1px solid var(--border-outline-variant) !important;
    }
    .modal-content .stMetric label {
        color: var(--text-on-surface-variant) !important;
        font-family: 'JetBrains Mono', monospace !important;
    }
    .modal-content .stMetric [data-testid="stMetricValue"] {
        color: var(--text-primary) !important;
    }
    .modal-content .stDataFrame {
        border-radius: 12px !important;
        overflow: hidden !important;
    }
    .modal-content .stDataFrame th {
        background: var(--bg-surface-container-low) !important;
        color: var(--accent-primary-fixed-dim) !important;
        border-bottom: 1px solid var(--border-outline-variant) !important;
    }
    .modal-content .stDataFrame td {
        color: var(--text-on-surface) !important;
    }
    .validation-badge {
        background: rgba(0, 219, 233, 0.08);
        border: 1px solid rgba(0, 219, 233, 0.15);
        color: var(--accent-primary-fixed-dim);
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.7rem;
        font-weight: 600;
        display: inline-block;
        font-family: 'JetBrains Mono', monospace;
        letter-spacing: 0.5px;
    }
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    @keyframes slideUp {
        from { transform: translateY(30px); opacity: 0; }
        to { transform: translateY(0); opacity: 1; }
    }
    .modal-content::-webkit-scrollbar {
        width: 6px;
    }
    .modal-content::-webkit-scrollbar-track {
        background: var(--bg-surface-container-low);
        border-radius: 4px;
    }
    .modal-content::-webkit-scrollbar-thumb {
        background: var(--border-outline-variant);
        border-radius: 4px;
    }
    .modal-content::-webkit-scrollbar-thumb:hover {
        background: var(--accent-primary-fixed-dim);
    }

    /* Estilo para el botón de cerrar en el modal */
    .modal-close-btn {
        background: rgba(255,255,255,0.05) !important;
        border: 1px solid var(--border-outline-variant) !important;
        color: var(--text-on-surface-variant) !important;
        border-radius: 8px !important;
        padding: 8px 16px !important;
        transition: all 0.3s ease !important;
    }
    .modal-close-btn:hover {
        background: rgba(255,255,255,0.1) !important;
        border-color: var(--accent-primary-fixed-dim) !important;
        color: var(--text-primary) !important;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# INICIALIZACIÓN
# ============================================================================

def main():
    """Punto de entrada principal de la aplicación"""
    
    # 1. Cargar datos
    raw = data_loader.load_match_data()
    if raw is None:
        st.error("❌ No se pudieron cargar los datos.")
        st.stop()
    
    fixture_data = data_loader.get_espn_fixture()
    
    # 2. Filtrar equipos clasificados
    wc_teams = data_loader.filter_world_cup_teams(raw)
    
    # 3. UI - Header y status
    ui.render_header()
    ui.render_status_bar()
    
    # 4. Disclaimer
    ui.render_disclaimer()
    
    # 5. Sidebar
    with st.sidebar:
        st.sidebar.header("⚙️ Configuración del Partido")
        
        home_team, away_team, match_date, train_start = ui.render_match_selector(wc_teams)
        
        st.markdown("---")
        st.subheader("🏟️ Configuración del Partido")
        
        neutral_venue = st.checkbox("🏟️ Partido en sede neutral", value=False, 
                                   help="Anula la ventaja de localía")
        
        st.markdown("---")
        st.subheader("🤖 Modelos a usar")
        
        use_bayesian = st.checkbox("🔵 Bayesiano (recomendado)", value=True)
        use_xgboost = st.checkbox("🟢 XGBoost (comparativo)", value=False)
        
        st.markdown("---")
        st.subheader("🔧 Correcciones")
        
        use_dixon_coles = st.checkbox("🔧 Dixon-Coles", value=True, help="Corrige la subestimación de empates")
        use_hydration = st.checkbox("💧 Pausas de hidratación (4 tiempos)", value=True)
        
        st.markdown("---")
        st.subheader("⚡ Ajustes Dinámicos")
        
        use_dynamic, underdog_scored_first, minuto_gol = ui.render_dynamic_adjustments()
        use_momentum, minuto_gol_favorito, llegadas_previas_h, llegadas_previas_a, marcador_actual_h, marcador_actual_a = ui.render_momentum_adjustments()
        
        st.markdown("---")
        
        use_high_scoring = st.checkbox("⚽ Ajuste por alta anotación", value=True, 
                                       help="Aumenta probabilidad de partidos con +3 goles")
        
        max_goals_display = st.slider("📊 Máximo de goles a mostrar", 4, 10, 7)
        
        if fixture_data:
            with st.expander("📅 Fixture del Mundial 2026"):
                fixture_df = pd.DataFrame(fixture_data)
                fixture_df['date'] = fixture_df['date'].astype(str)
                st.dataframe(fixture_df, hide_index=True, use_container_width=True)
        
        # Botón de predicción con clase especial
        st.markdown('<div class="predict-btn-container">', unsafe_allow_html=True)
        predict_btn = st.button("🔮 Predecir", type="primary", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        st.subheader("🔬 Validación del Modelo")
        if st.button("📊 Validar modelo", use_container_width=True):
            st.session_state.show_validation = True
    
    # 9. Modal de validación
    if st.session_state.get('show_validation', False):
        validation.render_validation_modal(raw, corrections)
    
    # 10. Ejecutar predicción
    if predict_btn:
        results, errores, elo_h, elo_a = models.run_prediction(
            raw=raw,
            home_team=home_team,
            away_team=away_team,
            match_date=match_date,
            train_start=train_start,
            neutral_venue=neutral_venue,
            use_xgboost=use_xgboost,
            use_bayesian=use_bayesian,
            use_dixon_coles=use_dixon_coles,
            use_hydration=use_hydration,
            use_dynamic=use_dynamic,
            underdog_scored_first=underdog_scored_first,
            minuto_gol=minuto_gol,
            use_momentum=use_momentum,
            minuto_gol_favorito=minuto_gol_favorito,
            llegadas_previas_h=llegadas_previas_h,
            llegadas_previas_a=llegadas_previas_a,
            marcador_actual_h=marcador_actual_h,
            marcador_actual_a=marcador_actual_a,
            max_goals_display=max_goals_display,
            roster_factors=False,
            home_team_roster=None,
            away_team_roster=None,
            use_high_scoring=use_high_scoring
        )
        
        if results:
            st.session_state.results = results
            st.session_state.elo = {'home': elo_h, 'away': elo_a}
            st.success("✅ Predicción completada!")
            if errores:
                st.warning(f"⚠️ Algunos fallaron: {', '.join(errores)}")
        else:
            st.error(f"❌ No se pudo completar. Errores: {', '.join(errores)}")
    
    # 11. Mostrar resultados
    if 'results' in st.session_state:
        ui.render_results(
            st.session_state.results, 
            st.session_state.get('elo', {'home': 0, 'away': 0}),
            config.DIXON_COLES_RHO
        )
    
    # 12. Footer
    ui.render_footer()
    
    # 13. Info del sistema
    with st.expander("ℹ️ Información del modelo"):
        ui.render_system_info(len(wc_teams), raw)


if __name__ == "__main__":
    main()