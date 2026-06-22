# app.py - Predicción Mundial 2026 - Versión Final Corregida
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from scipy.stats import poisson
import warnings
import sys
import requests
import json
from datetime import datetime, timedelta
import pytz

# ⚠️ set_page_config DEBE SER LA PRIMERA INSTRUCCIÓN DE STREAMLIT
st.set_page_config(
    page_title="Predicción Mundial 2026",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

warnings.filterwarnings('ignore')

# ============================================================================
# CSS PERSONALIZADO - DISEÑO DARK PREMIUM
# ============================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Bebas+Neue&display=swap');

    :root {
        --bg-primary: #02091d;
        --bg-secondary: #111827;
        --bg-card: #1a1f2e;
        --bg-card-hover: #1e2538;
        --border-subtle: rgba(255,255,255,0.06);
        --border-active: rgba(0,212,255,0.3);
        --text-primary: #f0f5f9;
        --text-secondary: #94a3b8;
        --text-muted: #64748b;
        --accent-cyan: #0088cc;
        --accent-gold: #b8a800;
        --accent-green: #0d7a3e;
        --accent-red: #cc2233;
        --accent-purple: #a855f7;
        --shadow-glow: 0 0 30px rgba(0, 212, 255, 0.05);
    }

    .stApp {
        background: linear-gradient(to bottom, #e8edf2 0%, #c8d6e5 100%);
        font-family: 'Inter', sans-serif;
        color: #1a1a2e;
    }

    [data-testid="stSidebar"] {
        background: 
            linear-gradient(
                135deg,
                rgba(0, 212, 255, 0.03) 0%,
                rgba(0, 212, 255, 0.01) 30%,
                transparent 70%
            ),
            linear-gradient(
                180deg,
                #111827 0%,
                #1a1f2e 50%,
                #0f172a 100%
            ) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.03);
    }
    
    [data-testid="stSidebar"] .stMarkdown,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stCaption {
        color: #e2e8f0 !important;
        text-shadow: 0 4px 20px rgba(120, 231, 248, 0.5);
    }
    
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        color: var(--text-primary) !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 700 !important;
    }

    h1, h2, h3, h4 {
        font-family: 'Inter', sans-serif !important;
        color: var(--text-primary) !important;
    }
    h1 {
        font-weight: 800 !important;
        letter-spacing: -0.5px;
    }

    [data-testid="stVerticalBlock"] > [style*="flex-direction: column"] > [data-testid="stVerticalBlock"] {
        background: var(--bg-card);
        border: 1px solid var(--border-subtle);
        border-radius: 16px;
        padding: 24px;
        transition: all 0.3s ease;
    }

    [data-testid="stMetric"] {
        background: var(--bg-secondary) !important;
        border: 1px solid var(--border-subtle);
        border-radius: 12px;
        padding: 12px 16px !important;
        transition: all 0.3s ease;
    }
    [data-testid="stMetric"]:hover {
        border-color: var(--border-active);
        box-shadow: var(--shadow-glow);
        transform: translateY(-1px);
    }
    [data-testid="stMetric"] label {
        color: var(--text-secondary) !important;
        font-size: 0.7rem !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    [data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: var(--text-primary) !important;
        font-size: 1.6rem !important;
        font-weight: 700 !important;
    }
    [data-testid="stMetric"] [data-testid="stMetricDelta"] {
        color: var(--accent-cyan) !important;
        font-size: 0.8rem !important;
    }

    .stButton > button {
        background: linear-gradient(135deg, #00d4ff, #0099ff) !important;
        color: #000 !important;
        font-weight: 700 !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 12px 24px !important;
        text-transform: uppercase;
        letter-spacing: 1px;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 20px rgba(0, 212, 255, 0.3) !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(0, 212, 255, 0.4) !important;
    }

    .stSelectbox > div > div {
        background: var(--bg-card) !important;
        border: 1px solid var(--border-subtle) !important;
        border-radius: 10px !important;
        color: var(--text-primary) !important;
        transition: all 0.3s ease !important;
    }
    .stSelectbox > div > div:hover {
        border-color: var(--accent-cyan) !important;
        box-shadow: 0 0 20px rgba(0, 212, 255, 0.1) !important;
    }

    .stSlider > div > div > div {
        background: var(--accent-cyan) !important;
    }

    .stCheckbox > div > div > div {
        background: var(--accent-cyan) !important;
    }

    .stDataFrame {
        border: 1px solid var(--border-subtle) !important;
        border-radius: 12px !important;
        overflow: hidden;
    }
    .stDataFrame th {
        background: var(--bg-secondary) !important;
        color: var(--text-primary) !important;
        font-weight: 600 !important;
    }
    .stDataFrame td {
        color: var(--text-secondary) !important;
    }

    .streamlit-expanderHeader {
        background: var(--bg-card) !important;
        border: 1px solid var(--border-subtle) !important;
        border-radius: 10px !important;
        color: var(--text-primary) !important;
        transition: all 0.3s ease !important;
    }
    .streamlit-expanderHeader:hover {
        border-color: var(--border-active) !important;
    }
    .streamlit-expanderContent {
        background: var(--bg-secondary) !important;
        border: 1px solid var(--border-subtle) !important;
        border-top: none !important;
        border-radius: 0 0 10px 10px !important;
    }

    .stAlert {
        border-radius: 12px !important;
        border: 1px solid var(--border-subtle) !important;
    }
    .stAlert [data-testid="stAlertContent"] {
        background: transparent !important;
    }
    .stAlert [data-testid="stAlertContentSuccess"] {
        background: rgba(34, 197, 94, 0.6) !important;
        border: 1px solid rgba(34, 197, 94, 0.5) !important;
        border-radius: 12px !important;
    }
    .stAlert [data-testid="stAlertContentInfo"] {
        background: rgba(0, 212, 255, 0.6) !important;
        border: 1px solid rgba(0, 212, 255, 0.5) !important;
        border-radius: 12px !important;
    }

    .stSpinner > div {
        border-color: var(--accent-cyan) !important;
        border-top-color: transparent !important;
        border-width: 4px !important;
        width: 48px !important;
        height: 48px !important;
    }

    hr {
        border-color: var(--border-subtle) !important;
        margin: 24px 0 !important;
    }

    .title-bar {
        background: linear-gradient(135deg, #0f172a, #1e293b);
        border: 1px solid rgba(0, 212, 255, 0.15);
        border-radius: 16px;
        padding: 24px 32px;
        margin-bottom: 24px;
        position: relative;
        overflow: hidden;
        transition: all 0.3s ease;
    }
    .title-bar:hover {
        border-color: rgba(0, 212, 255, 0.3);
        box-shadow: 0 0 40px rgba(0, 212, 255, 0.05);
    }
    .title-bar::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, #00d4ff, #0099ff, #00d4ff);
        background-size: 200% 100%;
        animation: shimmer 3s linear infinite;
    }
    @keyframes shimmer {
        0% { background-position: 200% 0; }
        100% { background-position: -200% 0; }
    }
    .title-bar h1 {
        font-family: 'Bebas Neue', sans-serif !important;
        font-size: 2.5rem !important;
        letter-spacing: 3px !important;
        margin: 0 !important;
        color: #ffffff;
        text-shadow: 0 0 30px rgba(0, 212, 255, 0.3);
        filter: drop-shadow(0 0 20px rgba(0, 212, 255, 0.2));
    }
    .title-bar p {
        color: var(--text-secondary);
        margin: 4px 0 0 0;
        font-size: 0.95rem;
        letter-spacing: 0.5px;
    }
    .title-bar .badge-row {
        display: flex;
        gap: 8px;
        margin-top: 12px;
        flex-wrap: wrap;
    }
    .badge {
        background: rgba(0, 0, 0, 0.08);
        border: 1px solid rgba(0, 0, 0, 0.15);
        color: #0a0e1a;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        transition: all 0.3s ease;
    }
    .badge:hover {
        background: rgba(0, 0, 0, 0.15);
    }
    .badge.gold {
        background: rgba(210, 200, 15, 0.15);
        border-color: rgba(210, 200, 15, 0.3);
        color: #8a7a00;
    }
    .badge.gold:hover {
        background: rgba(255, 193, 7, 0.2);
    }

    .status-bar {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 12px;
        margin-bottom: 16px;
    }
    .status-item {
        background: rgba(255,255,255,0.6);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(0,0,0,0.08);
        border-radius: 12px;
        padding: 12px 16px;
        text-align: center;
        transition: all 0.3s ease;
    }
    .status-item:hover {
        border-color: var(--border-active);
        box-shadow: var(--shadow-glow);
    }
    .status-item .label {
        color: var(--text-muted);
        font-size: 0.65rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .status-item .value {
        color: #0a0e1a;
        font-size: 1.2rem;
        font-weight: 700;
        margin-top: 4px;
    }
    .status-item .value.cyan { color: var(--accent-cyan); }
    .status-item .value.gold { color: var(--accent-gold); }
    .status-item .value.green { color: var(--accent-green); }
    .status-item .value.red { color: var(--accent-red); }

    .prob-bar {
        height: 8px;
        border-radius: 4px;
        background: var(--bg-primary);
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
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
        animation: shimmer-bar 2s infinite;
    }
    @keyframes shimmer-bar {
        0% { transform: translateX(-100%); }
        100% { transform: translateX(100%); }
    }

    .footer {
        text-align: center;
        padding: 24px 16px;
        color: var(--text-muted);
        font-size: 0.8rem;
        border-top: 1px solid var(--border-subtle);
        margin-top: 40px;
        background: var(--bg-secondary);
        border-radius: 12px;
    }
    .footer a {
        color: var(--accent-cyan);
        text-decoration: none;
        transition: color 0.2s;
    }
    .footer a:hover {
        color: #66e5ff;
        text-decoration: underline;
    }
    .footer .separator {
        color: var(--border-subtle);
        margin: 0 8px;
    }

    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    ::-webkit-scrollbar-track {
        background: var(--bg-primary);
    }
    ::-webkit-scrollbar-thumb {
        background: var(--bg-card);
        border-radius: 4px;
        transition: background 0.3s;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: var(--accent-cyan);
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    @media (max-width: 768px) {
        .title-bar {
            padding: 16px 20px;
        }
        .title-bar h1 {
            font-size: 1.8rem !important;
        }
        .title-bar p {
            font-size: 0.8rem;
        }
        .badge {
            font-size: 0.6rem;
            padding: 3px 10px;
        }
        .status-bar {
            grid-template-columns: repeat(2, 1fr);
            gap: 8px;
        }
        [data-testid="stMetric"] {
            padding: 8px 12px !important;
        }
        [data-testid="stMetric"] [data-testid="stMetricValue"] {
            font-size: 1.2rem !important;
        }
        .footer {
            padding: 16px;
            font-size: 0.7rem;
        }
        .footer .separator {
            display: none;
        }
        .footer br {
            display: block;
        }
    }

    @media (max-width: 480px) {
        .status-bar {
            grid-template-columns: 1fr 1fr;
            gap: 6px;
        }
        .status-item {
            padding: 8px 12px;
        }
        .status-item .value {
            font-size: 1rem;
        }
        .title-bar h1 {
            font-size: 1.4rem !important;
        }
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# LISTA DE SELECCIONES CALIFICADAS
# ============================================================================
WORLD_CUP_2026_TEAMS = [
    "Spain", "Portugal", "France", "Belgium", "Netherlands", "Italy", "Germany",
    "England", "Croatia", "Switzerland", "Denmark", "Austria", "Ukraine",
    "Sweden", "Wales", "Turkey", "Scotland", "Norway", "Serbia", "Bosnia and Herzegovina",
    "Poland", "Czech Republic", "Hungary", "Slovakia",
    "Brazil", "Argentina", "Uruguay", "Ecuador", "Colombia", "Chile", "Paraguay",
    "Mexico", "USA", "Canada", "Costa Rica", "Jamaica", "Panama", "Honduras", "El Salvador",
    "Haiti", "Curacao",
    "Morocco", "Senegal", "Nigeria", "Egypt", "Ghana", "Cameroon", "Algeria",
    "Tunisia", "Mali", "Ivory Coast", "Cabo Verde", "South Africa", "Congo DR",
    "Japan", "South Korea", "Iran", "Saudi Arabia", "Australia", "Qatar",
    "UAE", "Iraq", "Jordan", "Uzbekistan",
    "New Zealand"
]
WORLD_CUP_2026_TEAMS = sorted([team for team in WORLD_CUP_2026_TEAMS])

# ============================================================================
# PARÁMETROS CONFIGURABLES
# ============================================================================
DIXON_COLES_RHO = -0.13

# ============================================================================
# FUNCIÓN DE AJUSTE HÍBRIDO
# ============================================================================
def ajustar_por_pausas_hidratacion(lam_h, lam_a, elo_h=None, elo_a=None):
    factor_general = 0.95
    media_h_general = 1.35
    media_a_general = 1.05
    
    lam_h_ajustado = lam_h * factor_general
    lam_a_ajustado = lam_a * factor_general
    
    if elo_h is not None and elo_a is not None:
        diff_elo = abs(elo_h - elo_a)
        if diff_elo > 200:
            factor_contraccion = 0.05
        elif diff_elo > 100:
            factor_contraccion = 0.08
        else:
            factor_contraccion = 0.12
    else:
        factor_contraccion = 0.12
    
    lam_h_ajustado = lam_h_ajustado * (1 - factor_contraccion) + media_h_general * factor_contraccion
    lam_a_ajustado = lam_a_ajustado * (1 - factor_contraccion) + media_a_general * factor_contraccion
    
    return lam_h_ajustado, lam_a_ajustado

# ============================================================================
# FUNCIÓN DIXON-COLES
# ============================================================================
def dixon_coles_factor(x, y, lam_h, lam_a, rho):
    mask_00 = (x == 0) & (y == 0)
    mask_01 = (x == 0) & (y == 1)
    mask_10 = (x == 1) & (y == 0)
    mask_11 = (x == 1) & (y == 1)
    tau = np.ones_like(x, dtype=float)
    tau[mask_00] = 1 - lam_h * lam_a * rho
    tau[mask_01] = 1 + lam_h * rho
    tau[mask_10] = 1 + lam_a * rho
    tau[mask_11] = 1 - rho
    tau = np.maximum(tau, 0.01)
    return tau

def aplicar_dixon_coles(score_matrix, lam_h, lam_a, rho=DIXON_COLES_RHO):
    max_g = score_matrix.shape[0] - 1
    goals = np.arange(0, max_g + 1)
    H, A = np.meshgrid(goals, goals, indexing='ij')
    tau = dixon_coles_factor(H, A, lam_h, lam_a, rho)
    score_matrix_corregida = score_matrix * tau
    suma = score_matrix_corregida.sum()
    if suma > 0:
        score_matrix_corregida = score_matrix_corregida / suma
    return score_matrix_corregida

# ============================================================================
# FUNCIÓN DE AJUSTE POR GOL TEMPRANO DEL UNDERDOG
# ============================================================================
def ajustar_por_gol_temprano(score_matrix, lam_h, lam_a, home_team, away_team,
                             underdog_scored_first, minuto_gol, favorito_elo, underdog_elo):
    if not underdog_scored_first:
        return score_matrix
    if lam_h > lam_a:
        favorito = 'home'; underdog = 'away'; diff_lam = lam_h - lam_a
    else:
        favorito = 'away'; underdog = 'home'; diff_lam = lam_a - lam_h
    if minuto_gol <= 15:
        factor_tiempo = 1.0
    elif minuto_gol <= 30:
        factor_tiempo = 0.8
    elif minuto_gol <= 45:
        factor_tiempo = 0.6
    else:
        factor_tiempo = 0.3
    factor_sorpresa = min(1.0, diff_lam * 0.5)
    factor_ajuste = factor_tiempo * factor_sorpresa
    if factor_ajuste < 0.1:
        return score_matrix
    max_g = score_matrix.shape[0] - 1
    ajuste = np.ones_like(score_matrix)
    for i in range(min(3, max_g + 1)):
        for j in range(min(3, max_g + 1)):
            if i + j <= 2:
                ajuste[i, j] = 1 - factor_ajuste * 0.3
    for i in range(1, min(5, max_g + 1)):
        for j in range(1, min(5, max_g + 1)):
            if 3 <= i + j <= 6:
                ajuste[i, j] = 1 + factor_ajuste * 0.25
    if underdog == 'away':
        for i in range(1, min(5, max_g + 1)):
            for j in range(1, min(3, max_g + 1)):
                if i >= j:
                    ajuste[i, j] = 1 + factor_ajuste * 0.2
    if underdog == 'home':
        for i in range(1, min(3, max_g + 1)):
            for j in range(1, min(5, max_g + 1)):
                if j >= i:
                    ajuste[i, j] = 1 + factor_ajuste * 0.2
    score_matrix_ajustada = score_matrix * ajuste
    suma = score_matrix_ajustada.sum()
    if suma > 0:
        score_matrix_ajustada = score_matrix_ajustada / suma
    return score_matrix_ajustada

# ============================================================================
# FUNCIÓN DE AJUSTE POR MOMENTUM (CORREGIDA)
# ============================================================================
def ajustar_por_momentum(lam_h, lam_a, home_team, away_team, 
                         minuto_gol=None, es_favorito_local=None,
                         llegadas_previas_h=None, llegadas_previas_a=None,
                         marcador_actual=None):
    # 1. Ajuste por gol tardío del favorito (minuto 80+)
    if minuto_gol is not None and minuto_gol >= 80:
        if es_favorito_local:
            lam_h *= 1.12
            lam_a *= 0.95
        else:
            lam_a *= 1.12
            lam_h *= 0.95
    
    # 2. Ajuste por relajación defensiva (si va ganando por 2+ goles)
    if marcador_actual is not None:
        diff = marcador_actual.get('home', 0) - marcador_actual.get('away', 0)
        if diff >= 2:
            lam_a *= 1.08
        elif diff <= -2:
            lam_h *= 1.08
    
    # 3. Ajuste por momentum del cuarto anterior
    if (llegadas_previas_h is not None and llegadas_previas_a is not None 
        and llegadas_previas_a > 0 and llegadas_previas_h > 0):
        ratio = llegadas_previas_h / llegadas_previas_a
        if ratio > 1.5:
            lam_h *= 1.06
        elif ratio < 0.67:
            lam_a *= 1.06
    
    return lam_h, lam_a

# ============================================================================
# CONEXIÓN A ESPN
# ============================================================================
@st.cache_data(ttl=3600)
def get_espn_fixture():
    try:
        url = "https://site.web.api.espn.com/apis/site/v2/sports/soccer/fifa.worldcup/scoreboard"
        params = {"region": "us", "lang": "en", "contentorigin": "espn"}
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        
        fixture_data = []
        # Hacer múltiples llamadas para cubrir el torneo
        for date_offset in range(0, 35):
            date_str = (datetime.strptime("2026-06-11", "%Y-%m-%d") + timedelta(days=date_offset)).strftime("%Y-%m-%d")
            params["dates"] = date_str
            response = requests.get(url, params=params, headers=headers, timeout=5)
            if response.status_code == 200:
                data = response.json()
                events = data.get('events', [])
                for event in events:
                    competitions = event.get('competitions', [])
                    for comp in competitions:
                        competitors = comp.get('competitors', [])
                        if len(competitors) >= 2:
                            home_team = competitors[0].get('team', {}).get('displayName', '')
                            away_team = competitors[1].get('team', {}).get('displayName', '')
                            date_str_event = event.get('date', '')
                            try:
                                match_date = datetime.fromisoformat(date_str_event.replace('Z', '+00:00')).date()
                            except:
                                match_date = datetime.strptime(date_str, "%Y-%m-%d").date()
                            fixture_data.append({'date': match_date, 'home_team': home_team, 'away_team': away_team})
        if fixture_data:
            return fixture_data
    except Exception as e:
        st.warning(f"⚠️ No se pudo conectar a ESPN: {str(e)}")
    
    # Fixture manual (fallback)
    return [
        {'date': datetime.strptime("2026-06-15", "%Y-%m-%d").date(), 'home_team': 'Spain', 'away_team': 'Cabo Verde'},
        {'date': datetime.strptime("2026-06-15", "%Y-%m-%d").date(), 'home_team': 'Saudi Arabia', 'away_team': 'Uruguay'},
        {'date': datetime.strptime("2026-06-21", "%Y-%m-%d").date(), 'home_team': 'Uruguay', 'away_team': 'Cabo Verde'},
        {'date': datetime.strptime("2026-06-22", "%Y-%m-%d").date(), 'home_team': 'Spain', 'away_team': 'Saudi Arabia'},
        {'date': datetime.strptime("2026-06-27", "%Y-%m-%d").date(), 'home_team': 'Cabo Verde', 'away_team': 'Saudi Arabia'},
        {'date': datetime.strptime("2026-06-27", "%Y-%m-%d").date(), 'home_team': 'Spain', 'away_team': 'Uruguay'},
    ]

@st.cache_data(ttl=3600)
def get_espn_team_stats(team_name):
    stats_cache = {
        'Spain': {'attack': 2.5, 'defense': 0.8, 'elo': 2050},
        'Uruguay': {'attack': 1.8, 'defense': 1.1, 'elo': 1850},
        'Cabo Verde': {'attack': 1.4, 'defense': 1.3, 'elo': 1680},
        'Saudi Arabia': {'attack': 1.3, 'defense': 1.4, 'elo': 1700},
        'Mexico': {'attack': 1.8, 'defense': 1.2, 'elo': 1880},
        'USA': {'attack': 1.9, 'defense': 1.1, 'elo': 1900},
        'Canada': {'attack': 1.7, 'defense': 1.3, 'elo': 1820},
        'Brazil': {'attack': 2.8, 'defense': 0.7, 'elo': 2100},
        'Argentina': {'attack': 2.6, 'defense': 0.8, 'elo': 2080},
        'France': {'attack': 2.4, 'defense': 0.9, 'elo': 2040},
        'Germany': {'attack': 2.3, 'defense': 0.9, 'elo': 2020},
        'England': {'attack': 2.2, 'defense': 0.9, 'elo': 2000},
        'Netherlands': {'attack': 2.1, 'defense': 1.0, 'elo': 1980},
        'Portugal': {'attack': 2.0, 'defense': 1.0, 'elo': 1960},
        'Italy': {'attack': 1.9, 'defense': 0.9, 'elo': 1940},
        'Belgium': {'attack': 1.9, 'defense': 1.1, 'elo': 1920},
        'Austria': {'attack': 1.6, 'defense': 1.2, 'elo': 1780},
    }
    synonyms = {'Cape Verde': 'Cabo Verde', 'USA': 'USA', 'United States': 'USA',
                'Czechia': 'Czech Republic', "Côte d'Ivoire": 'Ivory Coast'}
    search_name = synonyms.get(team_name, team_name)
    return stats_cache.get(search_name, {'attack': 1.5, 'defense': 1.3, 'elo': 1750})

# ============================================================================
# VERIFICAR DEPENDENCIAS
# ============================================================================
try:
    import pymc as pm
    import arviz as az
    PYMC_AVAILABLE = True
except ImportError:
    PYMC_AVAILABLE = False
    pm = None
    az = None

try:
    import sklearn
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# ============================================================================
# HEADER PERSONALIZADO
# ============================================================================
st.markdown("""
<div class="title-bar">
    <h1>⚽ PREDICCIÓN DE MARCADORES</h1>
    <p>Mundial FIFA 2026 — Modelos Bayesiano &amp; XGBoost</p>
    <div class="badge-row">
        <span class="badge">Dixon-Coles</span>
        <span class="badge">XGBoost</span>
        <span class="badge gold">Mundial 2026</span>
        <span class="badge">4 Tiempos</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Status bar mejorada
st.markdown('<div class="status-bar">', unsafe_allow_html=True)
status_cols = st.columns(4)
with status_cols[0]:
    st.markdown(f"""
    <div class="status-item">
        <div class="label">Python</div>
        <div class="value cyan">{sys.version.split()[0]}</div>
    </div>
    """, unsafe_allow_html=True)
with status_cols[1]:
    st.markdown(f"""
    <div class="status-item">
        <div class="label">PyMC</div>
        <div class="value {'green' if PYMC_AVAILABLE else 'red'}">{'✅ Disponible' if PYMC_AVAILABLE else '❌ No disponible'}</div>
    </div>
    """, unsafe_allow_html=True)
with status_cols[2]:
    st.markdown(f"""
    <div class="status-item">
        <div class="label">SKLearn</div>
        <div class="value {'green' if SKLEARN_AVAILABLE else 'red'}">{'✅ Disponible' if SKLEARN_AVAILABLE else '❌ No disponible'}</div>
    </div>
    """, unsafe_allow_html=True)
with status_cols[3]:
    st.markdown(f"""
    <div class="status-item">
        <div class="label">Dixon-Coles ρ</div>
        <div class="value gold">{DIXON_COLES_RHO:.3f}</div>
    </div>
    """, unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")

if not PYMC_AVAILABLE:
    st.info("ℹ️ El modelo Bayesiano no está disponible. Solo se usará XGBoost.")

# ============================================================================
# SIDEBAR - Configuración
# ============================================================================
st.sidebar.header("⚙️ Configuración del Partido")

@st.cache_data(ttl=3600)
def load_data():
    try:
        RESULTS_URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
        raw = pd.read_csv(RESULTS_URL, parse_dates=["date"])
        return raw
    except Exception as e:
        st.error(f"Error al cargar los datos: {str(e)}")
        return None

with st.spinner("🔄 Cargando datos..."):
    raw = load_data()

if raw is None:
    st.error("❌ No se pudieron cargar los datos.")
    st.stop()

fixture_data = get_espn_fixture()
all_teams_in_data = sorted(pd.concat([raw.home_team, raw.away_team]).unique())
wc_teams_in_data = [team for team in WORLD_CUP_2026_TEAMS if team in all_teams_in_data]

if len(wc_teams_in_data) < 10:
    st.warning(f"⚠️ Solo se encontraron {len(wc_teams_in_data)} equipos.")
    wc_teams_in_data = all_teams_in_data[:20]
    st.info(f"Usando {len(wc_teams_in_data)} equipos disponibles.")

st.sidebar.caption(f"🏆 {len(wc_teams_in_data)} selecciones clasificadas")

# ============================================================================
# SIDEBAR - Selectores
# ============================================================================
with st.sidebar:
    st.subheader("🏟️ Selecciona los equipos")

    home_team = st.selectbox(
        "🏠 Equipo Local",
        options=wc_teams_in_data,
        index=wc_teams_in_data.index("Mexico") if "Mexico" in wc_teams_in_data else 0
    )

    away_team = st.selectbox(
        "✈️ Equipo Visitante",
        options=wc_teams_in_data,
        index=wc_teams_in_data.index("Cabo Verde") if "Cabo Verde" in wc_teams_in_data else min(1, len(wc_teams_in_data)-1)
    )

    if home_team == away_team:
        st.warning("⚠️ Los equipos deben ser diferentes")
        if len(wc_teams_in_data) > 1:
            away_idx = 1 if wc_teams_in_data[1] != home_team else 0
            away_team = wc_teams_in_data[away_idx]

    mexico_tz = pytz.timezone('America/Mexico_City')
    match_date = st.date_input("📅 Fecha del Partido", datetime.now(mexico_tz).date())
    train_start = st.selectbox("📊 Ventana de entrenamiento", ["2018-01-01", "2016-01-01", "2014-01-01", "2010-01-01"], index=0)

    st.markdown("---")
    st.subheader("🏟️ Configuración del Partido")
    
    neutral_venue = st.checkbox("🏟️ Partido en sede neutral", value=False, 
                               help="Anula la ventaja de localía (aplica para Mundial en USA/Canadá/México)")

    st.markdown("---")
    st.subheader("🤖 Modelos a usar")

    use_xgboost = st.checkbox("✅ XGBoost", value=True)
    use_bayesian = st.checkbox("✅ Bayesiano" if PYMC_AVAILABLE else "❌ Bayesiano (no disponible)",
                               value=PYMC_AVAILABLE, disabled=not PYMC_AVAILABLE)

    st.markdown("---")
    st.subheader("🔧 Correcciones")

    use_dixon_coles = st.checkbox("🔧 Dixon-Coles", value=True, help="Corrige la subestimación de empates")
    use_hydration_adjustment = st.checkbox("💧 Pausas de hidratación (4 tiempos)", value=True)

    st.markdown("---")
    st.subheader("⚡ Ajustes Dinámicos")
    
    use_dynamic_adjustment = st.checkbox("⚡ Ajuste por gol temprano del underdog", value=False)

    if use_dynamic_adjustment:
        underdog_scored_first = st.checkbox("🏃 El underdog anotó primero", value=False)
        minuto_gol = st.slider("⏱️ Minuto del primer gol", 1, 90, 15, help="Minuto en que el underdog anotó")
        st.caption("💡 Si el underdog anota primero, el partido se vuelve más abierto.")
    
    use_momentum_adjustment = st.checkbox("⚡ Ajuste por momentum (gol tardío del favorito)", value=False)
    
    if use_momentum_adjustment:
        minuto_gol_favorito = st.slider("⏱️ Minuto del gol del favorito", 1, 90, 85, 
                                       help="Si el favorito anota en el minuto 80+, aumenta sus chances")
        llegadas_previas_h = st.number_input("Llegadas del local en el cuarto anterior", 0, 20, 5)
        llegadas_previas_a = st.number_input("Llegadas del visitante en el cuarto anterior", 0, 20, 3)
        st.markdown("**Marcador actual:**")
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            marcador_actual_h = st.number_input("Goles local", 0, 10, 0, key="marc_h")
        with col_m2:
            marcador_actual_a = st.number_input("Goles visitante", 0, 10, 0, key="marc_a")

    st.markdown("---")
    max_goals_display = st.slider("📊 Máximo de goles a mostrar", 4, 10, 7)

    st.markdown("---")
    if fixture_data:
        with st.expander("📅 Fixture del Mundial 2026"):
            fixture_df = pd.DataFrame(fixture_data)
            fixture_df['date'] = fixture_df['date'].astype(str)
            st.dataframe(fixture_df, hide_index=True, use_container_width=True)

    predict_btn = st.button("🔮 Predecir", type="primary", use_container_width=True)

# ============================================================================
# SIDEBAR - VALIDACIÓN DEL MODELO (CON MODAL)
# ============================================================================

with st.sidebar:
    st.markdown("---")
    st.subheader("🔬 Validación del Modelo")

    if st.button("📊 Validar modelo", use_container_width=True):
        st.session_state.show_validation = True
# ============================================================================
# MODAL DE VALIDACIÓN (CORREGIDO - SIN st.dialog)
# ============================================================================
if 'show_validation' not in st.session_state:
    st.session_state.show_validation = False

if st.session_state.show_validation:
    # Usar un contenedor que simula un modal
    st.markdown("""
    <style>
    .modal-overlay {
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: rgba(0,0,0,0.7);
        backdrop-filter: blur(10px);
        z-index: 9999;
        display: flex;
        align-items: center;
        justify-content: center;
        padding: 20px;
        animation: fadeIn 0.3s ease;
    }
    .modal-content {
        background: linear-gradient(135deg, #0f172a, #1e293b) !important;
        border: 1px solid rgba(0, 212, 255, 0.2) !important;
        border-radius: 20px !important;
        max-width: 900px !important;
        width: 100%;
        max-height: 90vh;
        overflow-y: auto;
        padding: 30px;
        position: relative;
        box-shadow: 0 20px 60px rgba(0,0,0,0.5);
        animation: slideUp 0.3s ease;
    }
    .modal-content h2 {
        color: #ffffff !important;
        font-family: 'Bebas Neue', sans-serif !important;
        letter-spacing: 2px !important;
        margin-top: 0 !important;
    }
    .modal-content p {
        color: #94a3b8 !important;
    }
    .modal-content .stMetric {
        background: rgba(255,255,255,0.05) !important;
        border: 1px solid rgba(255,255,255,0.08) !important;
    }
    .modal-content .stMetric label {
        color: #94a3b8 !important;
    }
    .modal-content .stMetric [data-testid="stMetricValue"] {
        color: #ffffff !important;
    }
    .modal-content .stDataFrame {
        border-radius: 12px !important;
        overflow: hidden !important;
    }
    .modal-content .stDataFrame th {
        background: #0f172a !important;
        color: #00d4ff !important;
    }
    .modal-content .stDataFrame td {
        color: #94a3b8 !important;
    }
    .modal-close {
        position: absolute;
        top: 15px;
        right: 20px;
        background: none;
        border: none;
        color: #94a3b8;
        font-size: 24px;
        cursor: pointer;
        transition: color 0.2s;
    }
    .modal-close:hover {
        color: #ffffff;
    }
    .validation-badge {
        background: rgba(0, 212, 255, 0.1);
        border: 1px solid rgba(0, 212, 255, 0.2);
        color: #00d4ff;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.7rem;
        font-weight: 600;
        display: inline-block;
    }
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    @keyframes slideUp {
        from { transform: translateY(30px); opacity: 0; }
        to { transform: translateY(0); opacity: 1; }
    }
    </style>
    <div class="modal-overlay" id="modal-overlay">
        <div class="modal-content">
    """, unsafe_allow_html=True)
    
    # Título del modal
    st.markdown("""
    <h2>🔬 Validación del Modelo</h2>
    <p style="margin-bottom: 16px;">
        <span class="validation-badge">📊 Validación fuera de muestra</span>
        <span class="validation-badge" style="margin-left: 8px;">📅 Últimos 12 meses</span>
    </p>
    """, unsafe_allow_html=True)
    
    # Botón para cerrar el modal
    if st.button("✕ Cerrar", key="close_modal_btn"):
        st.session_state.show_validation = False
        st.rerun()
    
    st.markdown("---")
    
    with st.spinner("🔄 Ejecutando validación (puede tomar 1-2 minutos)..."):
        try:
            # Usar datos reales: entrenar con todo excepto el último año
            max_date = raw["date"].max()
            TRAIN_END = (max_date - pd.Timedelta(days=365)).strftime("%Y-%m-%d")
            
            train_data = raw[(raw["date"] <= TRAIN_END) & raw["home_score"].notna()].copy()
            test_data = raw[(raw["date"] > TRAIN_END) & raw["home_score"].notna()].copy()
            
            if len(test_data) < 10:
                st.warning(f"⚠️ Solo {len(test_data)} partidos disponibles para validación.")
                if st.button("Cerrar", key="close_no_data_btn"):
                    st.session_state.show_validation = False
                    st.rerun()
                st.markdown("</div></div>", unsafe_allow_html=True)
                st.stop()
            
            # Entrenar modelo base
            with st.spinner("⚙️ Entrenando modelo..."):
                hist = raw[(raw.date <= TRAIN_END) & raw.home_score.notna()].sort_values("date").reset_index(drop=True)
                hist = hist[hist.date >= "2018-01-01"].copy()
                
                def entrenar_modelo_validacion(df):
                    import xgboost as xgb
                    K_ELO = 20.0
                    ELO_INIT = 1500.0
                    
                    ratings = {}
                    elo_h, elo_a = [], []
                    for _, row in df.iterrows():
                        rh = ratings.get(row.home_team, ELO_INIT)
                        ra = ratings.get(row.away_team, ELO_INIT)
                        elo_h.append(rh); elo_a.append(ra)
                        
                        exp_h = 1.0 / (1.0 + 10 ** ((ra - rh) / 400.0))
                        if row.home_score > row.away_score: score = 1.0
                        elif row.home_score == row.away_score: score = 0.5
                        else: score = 0.0
                        
                        margin = abs(row.home_score - row.away_score)
                        delta = K_ELO * (np.log(margin + 1) + 1.0) * (score - exp_h)
                        ratings[row.home_team] = rh + delta
                        ratings[row.away_team] = ra - delta
                    
                    df = df.copy()
                    df["elo_home"], df["elo_away"] = elo_h, elo_a
                    
                    records = {}
                    gf10_h, ga10_h, form5_h = [], [], []
                    gf10_a, ga10_a, form5_a = [], [], []
                    
                    for _, row in df.iterrows():
                        h_rec = records.get(row.home_team, [])
                        a_rec = records.get(row.away_team, [])
                        
                        def summarize(hist_rec):
                            last10, last5 = hist_rec[-10:], hist_rec[-5:]
                            gf = np.mean([x[1] for x in last10]) if last10 else np.nan
                            ga = np.mean([x[2] for x in last10]) if last10 else np.nan
                            pts = sum(x[3] for x in last5) if last5 else np.nan
                            return gf, ga, pts
                        
                        hgf, hga, hpts = summarize(h_rec)
                        agf, aga, apts = summarize(a_rec)
                        gf10_h.append(hgf); ga10_h.append(hga); form5_h.append(hpts)
                        gf10_a.append(agf); ga10_a.append(aga); form5_a.append(apts)
                        
                        h_pts = 3 if row.home_score > row.away_score else (1 if row.home_score == row.away_score else 0)
                        a_pts = 3 if row.away_score > row.home_score else (1 if row.home_score == row.away_score else 0)
                        records.setdefault(row.home_team, []).append((row.date, row.home_score, row.away_score, h_pts))
                        records.setdefault(row.away_team, []).append((row.date, row.away_score, row.home_score, a_pts))
                    
                    df["gf10_h"], df["ga10_h"], df["form5_h"] = gf10_h, ga10_h, form5_h
                    df["gf10_a"], df["ga10_a"], df["form5_a"] = gf10_a, ga10_a, form5_a
                    final_form = records
                    final_elo = ratings
                    
                    def to_long(dff):
                        dff["tournament_weight"] = 1.0
                        home_rows = pd.DataFrame({
                            "team": dff.home_team, "goals": dff.home_score, "is_home": 1,
                            "elo_team": dff.elo_home, "elo_opponent": dff.elo_away,
                            "gf10": dff.gf10_h, "ga10": dff.ga10_h, "form5": dff.form5_h,
                            "tournament_weight": dff.tournament_weight,
                        })
                        away_rows = pd.DataFrame({
                            "team": dff.away_team, "goals": dff.away_score, "is_home": 0,
                            "elo_team": dff.elo_away, "elo_opponent": dff.elo_home,
                            "gf10": dff.gf10_a, "ga10": dff.ga10_a, "form5": dff.form5_a,
                            "tournament_weight": dff.tournament_weight,
                        })
                        long = pd.concat([home_rows, away_rows], ignore_index=True)
                        long["elo_diff"] = long.elo_team - long.elo_opponent
                        return long.dropna(subset=["gf10", "ga10", "form5"])
                    
                    long_df = to_long(df)
                    FEATURES = ["elo_team", "elo_opponent", "elo_diff", "is_home", "gf10", "ga10", "form5", "tournament_weight"]
                    
                    xgb_model = xgb.XGBRegressor(
                        objective="count:poisson", n_estimators=200, max_depth=4, learning_rate=0.03,
                        subsample=0.8, colsample_bytree=0.8, min_child_weight=5, random_state=42,
                        n_jobs=1
                    )
                    xgb_model.fit(long_df[FEATURES], long_df["goals"])
                    return xgb_model, FEATURES, final_elo, final_form
                
                xgb_model, FEATURES, final_elo, final_form = entrenar_modelo_validacion(hist)
            
            # Evaluar sin pausas
            aciertos_a = 0
            for _, row in test_data.iterrows():
                def get_snapshot(team):
                    hist_team = final_form.get(team, [])
                    last10, last5 = hist_team[-10:], hist_team[-5:]
                    gf = np.mean([x[1] for x in last10]) if last10 else 0.0
                    ga = np.mean([x[2] for x in last10]) if last10 else 0.0
                    pts = sum(x[3] for x in last5) if last5 else 0.0
                    elo = final_elo.get(team, 1500.0)
                    return elo, gf, ga, pts
                
                elo_h, gf_h, ga_h, pts_h = get_snapshot(row['home_team'])
                elo_a, gf_a, ga_a, pts_a = get_snapshot(row['away_team'])
                
                row_home = {"elo_team": elo_h, "elo_opponent": elo_a, "elo_diff": elo_h - elo_a,
                            "is_home": 1, "gf10": gf_h, "ga10": ga_h, "form5": pts_h,
                            "tournament_weight": 4.0}
                row_away = {"elo_team": elo_a, "elo_opponent": elo_h, "elo_diff": elo_a - elo_h,
                            "is_home": 0, "gf10": gf_a, "ga10": ga_a, "form5": pts_a,
                            "tournament_weight": 4.0}
                
                feat_df = pd.DataFrame([row_home, row_away])[FEATURES]
                lam_h, lam_a = xgb_model.predict(feat_df)
                
                goals = np.arange(0, 8 + 1)
                sm = np.outer(poisson.pmf(goals, lam_h), poisson.pmf(goals, lam_a))
                sm = sm / sm.sum()
                
                pred = np.argmax([np.sum(sm[:3, :3]), np.sum(np.diag(sm[:3, :3])), np.sum(sm[:3, 1:])])
                real = 0 if row.home_score > row.away_score else 1 if row.home_score == row.away_score else 2
                if pred == real:
                    aciertos_a += 1
            acc_a = aciertos_a / len(test_data)
            
            # Evaluar con pausas de hidratación
            aciertos_b = 0
            for _, row in test_data.iterrows():
                def get_snapshot(team):
                    hist_team = final_form.get(team, [])
                    last10, last5 = hist_team[-10:], hist_team[-5:]
                    gf = np.mean([x[1] for x in last10]) if last10 else 0.0
                    ga = np.mean([x[2] for x in last10]) if last10 else 0.0
                    pts = sum(x[3] for x in last5) if last5 else 0.0
                    elo = final_elo.get(team, 1500.0)
                    return elo, gf, ga, pts
                
                elo_h, gf_h, ga_h, pts_h = get_snapshot(row['home_team'])
                elo_a, gf_a, ga_a, pts_a = get_snapshot(row['away_team'])
                
                row_home = {"elo_team": elo_h, "elo_opponent": elo_a, "elo_diff": elo_h - elo_a,
                            "is_home": 1, "gf10": gf_h, "ga10": ga_h, "form5": pts_h,
                            "tournament_weight": 4.0}
                row_away = {"elo_team": elo_a, "elo_opponent": elo_h, "elo_diff": elo_a - elo_h,
                            "is_home": 0, "gf10": gf_a, "ga10": ga_a, "form5": pts_a,
                            "tournament_weight": 4.0}
                
                feat_df = pd.DataFrame([row_home, row_away])[FEATURES]
                lam_h, lam_a = xgb_model.predict(feat_df)
                
                lam_h, lam_a = ajustar_por_pausas_hidratacion(lam_h, lam_a, elo_h, elo_a)
                
                goals = np.arange(0, 8 + 1)
                sm = np.outer(poisson.pmf(goals, lam_h), poisson.pmf(goals, lam_a))
                sm = sm / sm.sum()
                
                pred = np.argmax([np.sum(sm[:3, :3]), np.sum(np.diag(sm[:3, :3])), np.sum(sm[:3, 1:])])
                real = 0 if row.home_score > row.away_score else 1 if row.home_score == row.away_score else 2
                if pred == real:
                    aciertos_b += 1
            acc_b = aciertos_b / len(test_data)
            
            # Evaluar con pausas + momentum
            aciertos_c = 0
            for _, row in test_data.iterrows():
                def get_snapshot(team):
                    hist_team = final_form.get(team, [])
                    last10, last5 = hist_team[-10:], hist_team[-5:]
                    gf = np.mean([x[1] for x in last10]) if last10 else 0.0
                    ga = np.mean([x[2] for x in last10]) if last10 else 0.0
                    pts = sum(x[3] for x in last5) if last5 else 0.0
                    elo = final_elo.get(team, 1500.0)
                    return elo, gf, ga, pts
                
                elo_h, gf_h, ga_h, pts_h = get_snapshot(row['home_team'])
                elo_a, gf_a, ga_a, pts_a = get_snapshot(row['away_team'])
                
                row_home = {"elo_team": elo_h, "elo_opponent": elo_a, "elo_diff": elo_h - elo_a,
                            "is_home": 1, "gf10": gf_h, "ga10": ga_h, "form5": pts_h,
                            "tournament_weight": 4.0}
                row_away = {"elo_team": elo_a, "elo_opponent": elo_h, "elo_diff": elo_a - elo_h,
                            "is_home": 0, "gf10": gf_a, "ga10": ga_a, "form5": pts_a,
                            "tournament_weight": 4.0}
                
                feat_df = pd.DataFrame([row_home, row_away])[FEATURES]
                lam_h, lam_a = xgb_model.predict(feat_df)
                
                lam_h, lam_a = ajustar_por_pausas_hidratacion(lam_h, lam_a, elo_h, elo_a)
                
                es_favorito_local = elo_h > elo_a
                lam_h, lam_a = ajustar_por_momentum(
                    lam_h, lam_a, 
                    home_team=row['home_team'],
                    away_team=row['away_team'],
                    minuto_gol=85,
                    es_favorito_local=es_favorito_local,
                    marcador_actual={'home': row.home_score, 'away': row.away_score}
                )
                
                goals = np.arange(0, 8 + 1)
                sm = np.outer(poisson.pmf(goals, lam_h), poisson.pmf(goals, lam_a))
                sm = sm / sm.sum()
                
                pred = np.argmax([np.sum(sm[:3, :3]), np.sum(np.diag(sm[:3, :3])), np.sum(sm[:3, 1:])])
                real = 0 if row.home_score > row.away_score else 1 if row.home_score == row.away_score else 2
                if pred == real:
                    aciertos_c += 1
            acc_c = aciertos_c / len(test_data)
            
            # Mostrar resultados
            st.success(f"✅ Validación completada con {len(test_data)} partidos!")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📊 Partidos", len(test_data))
            with col2:
                st.metric("🔵 Base", f"{acc_a*100:.1f}%")
            with col3:
                st.metric("🟢 Con pausas", f"{acc_b*100:.1f}%", 
                         delta=f"{(acc_b - acc_a)*100:+.1f} pp")
            
            st.markdown("---")
            
            benchmark = 0.593
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📊 Benchmark", "59.3%")
            with col2:
                st.metric("🟣 + Momentum", f"{acc_c*100:.1f}%", 
                         delta=f"{(acc_c - acc_b)*100:+.1f} pp")
            with col3:
                st.metric("📈 vs benchmark", f"{(acc_c - benchmark)*100:+.1f} pp")
            
            st.markdown("---")
            
            comp_df = pd.DataFrame([
                {"Modelo": "XGBoost (copiado)", "Accuracy": "59.3%", "Mejora": "—"},
                {"Modelo": "XGBoost (base)", "Accuracy": f"{acc_a*100:.1f}%", "Mejora": "—"},
                {"Modelo": "+ Pausas", "Accuracy": f"{acc_b*100:.1f}%", "Mejora": f"{(acc_b - acc_a)*100:+.1f} pp"},
                {"Modelo": "+ Pausas + Momentum", "Accuracy": f"{acc_c*100:.1f}%", "Mejora": f"{(acc_c - acc_b)*100:+.1f} pp"}
            ])
            st.dataframe(comp_df, hide_index=True, use_container_width=True)
            
            if st.button("✅ Cerrar validación", use_container_width=True, key="close_validation_btn"):
                st.session_state.show_validation = False
                st.rerun()
            
        except Exception as e:
            st.error(f"❌ Error en la validación: {str(e)}")
            st.info("💡 La validación requiere datos históricos. Asegúrate de que el dataset esté disponible.")
            if st.button("Cerrar", key="close_error_btn"):
                st.session_state.show_validation = False
                st.rerun()
    
    # Cerrar los divs del modal
    st.markdown("</div></div>", unsafe_allow_html=True)
    
# ============================================================================
# FUNCIONES DE PREDICCIÓN
# ============================================================================

def train_bayesian_model(train, teams, team_idx, home_team, away_team, max_goals=8,
                         use_hydration=True, use_dixon_coles=True, neutral_venue=False):
    if not PYMC_AVAILABLE:
        return None, None, None, None, None
    try:
        home_idx = train.home_team.map(team_idx).values
        away_idx = train.away_team.map(team_idx).values
        home_goals = train.home_score.values
        away_goals = train.away_score.values
        
        coords = {"team": teams}
        with pm.Model(coords=coords) as bayes_model:
            sigma_att = pm.HalfNormal("sigma_att", sigma=1.0)
            sigma_def = pm.HalfNormal("sigma_def", sigma=1.0)
            
            attack_raw = pm.Normal("attack_raw", mu=0.0, sigma=sigma_att, dims="team")
            defense_raw = pm.Normal("defense_raw", mu=0.0, sigma=sigma_def, dims="team")
            
            attack = pm.Deterministic("attack", attack_raw - attack_raw.mean(), dims="team")
            defense = pm.Deterministic("defense", defense_raw - defense_raw.mean(), dims="team")
            
            # ✅ Neutral venue fijo a cero
            if neutral_venue:
                home_adv = pm.Deterministic("home_adv", 0.0)
            else:
                home_adv = pm.Normal("home_adv", mu=0.3, sigma=0.5)
            
            intercept = pm.Normal("intercept", mu=0.0, sigma=1.0)
            
            log_theta_home = intercept + home_adv + attack[home_idx] - defense[away_idx]
            log_theta_away = intercept + attack[away_idx] - defense[home_idx]
            
            pm.Poisson("home_goals_obs", mu=pm.math.exp(log_theta_home), observed=home_goals)
            pm.Poisson("away_goals_obs", mu=pm.math.exp(log_theta_away), observed=away_goals)
            
            idata = pm.sample(draws=500, tune=500, chains=2, cores=1,
                            random_seed=42, target_accept=0.85,
                            progressbar=False, return_inferencedata=True)
        
        post = idata.posterior
        intercept_vals = post["intercept"].values.flatten()
        home_adv_vals = post["home_adv"].values.flatten()
        attack_vals = post["attack"].values.reshape(-1, post["attack"].shape[-1])
        defense_vals = post["defense"].values.reshape(-1, post["defense"].shape[-1])
        
        hi, ai = team_idx[home_team], team_idx[away_team]
        log_th = intercept_vals + home_adv_vals + attack_vals[:, hi] - defense_vals[:, ai]
        log_ta = intercept_vals + attack_vals[:, ai] - defense_vals[:, hi]
        lam_h = np.exp(log_th).mean()
        lam_a = np.exp(log_ta).mean()
        
        # ✅ Obtener Elos para contracción adaptativa
        elo_h = get_espn_team_stats(home_team).get('elo', 1750)
        elo_a = get_espn_team_stats(away_team).get('elo', 1750)
        
        if use_hydration:
            lam_h, lam_a = ajustar_por_pausas_hidratacion(lam_h, lam_a, elo_h, elo_a)
        
        goals = np.arange(0, max_goals + 1)
        pmf_h = poisson.pmf(goals, lam_h)
        pmf_a = poisson.pmf(goals, lam_a)
        score_matrix = np.outer(pmf_h, pmf_a)
        
        if use_dixon_coles:
            score_matrix = aplicar_dixon_coles(score_matrix, lam_h, lam_a)
        else:
            suma = score_matrix.sum()
            if suma > 0:
                score_matrix = score_matrix / suma
        
        att_ratings = {team: post["attack"].sel(team=team).mean().item() for team in teams}
        def_ratings = {team: post["defense"].sel(team=team).mean().item() for team in teams}
        
        return score_matrix, lam_h, lam_a, att_ratings, def_ratings
    except Exception as e:
        st.warning(f"⚠️ Bayesiano: {str(e)}")
        return None, None, None, None, None

def train_xgboost_model(hist, raw_data, home_team, away_team, max_goals=8,
                        use_hydration=True, use_dixon_coles=True, neutral_venue=False):
    try:
        import xgboost as xgb
        K_ELO = 20.0
        ELO_INIT = 1500.0
        TOURNAMENT_WEIGHTS = {"FIFA World Cup": 4.0, "FIFA World Cup qualification": 2.0, "Friendly": 0.5}
        DEFAULT_WEIGHT = 1.0
        stats_h = get_espn_team_stats(home_team)
        stats_a = get_espn_team_stats(away_team)
        elo_h = stats_h.get('elo', 1750)
        elo_a = stats_a.get('elo', 1750)
        
        ratings = {}
        elo_h_hist, elo_a_hist = [], []
        for _, row in hist.iterrows():
            rh = ratings.get(row.home_team, ELO_INIT)
            ra = ratings.get(row.away_team, ELO_INIT)
            elo_h_hist.append(rh); elo_a_hist.append(ra)
            exp_h = 1.0 / (1.0 + 10 ** ((ra - rh) / 400.0))
            if row.home_score > row.away_score: score = 1.0
            elif row.home_score == row.away_score: score = 0.5
            else: score = 0.0
            margin = abs(row.home_score - row.away_score)
            delta = K_ELO * (np.log(margin + 1) + 1.0) * (score - exp_h)
            ratings[row.home_team] = rh + delta
            ratings[row.away_team] = ra - delta
        
        hist = hist.copy()
        hist["elo_home"], hist["elo_away"] = elo_h_hist, elo_a_hist
        final_elo = ratings
        
        records = {}
        gf10_h, ga10_h, form5_h = [], [], []
        gf10_a, ga10_a, form5_a = [], [], []
        
        for _, row in hist.iterrows():
            h_rec = records.get(row.home_team, [])
            a_rec = records.get(row.away_team, [])
            
            def summarize(hist_rec):
                last10, last5 = hist_rec[-10:], hist_rec[-5:]
                gf = np.mean([x[1] for x in last10]) if last10 else np.nan
                ga = np.mean([x[2] for x in last10]) if last10 else np.nan
                pts = sum(x[3] for x in last5) if last5 else np.nan
                return gf, ga, pts
            
            hgf, hga, hpts = summarize(h_rec)
            agf, aga, apts = summarize(a_rec)
            gf10_h.append(hgf); ga10_h.append(hga); form5_h.append(hpts)
            gf10_a.append(agf); ga10_a.append(aga); form5_a.append(apts)
            
            h_pts = 3 if row.home_score > row.away_score else (1 if row.home_score == row.away_score else 0)
            a_pts = 3 if row.away_score > row.home_score else (1 if row.home_score == row.away_score else 0)
            records.setdefault(row.home_team, []).append((row.date, row.home_score, row.away_score, h_pts))
            records.setdefault(row.away_team, []).append((row.date, row.away_score, row.home_score, a_pts))
        
        hist["gf10_h"], hist["ga10_h"], hist["form5_h"] = gf10_h, ga10_h, form5_h
        hist["gf10_a"], hist["ga10_a"], hist["form5_a"] = gf10_a, ga10_a, form5_a
        final_form = records
        
        def to_long(df):
            df["tournament_weight"] = df.tournament.map(TOURNAMENT_WEIGHTS).fillna(DEFAULT_WEIGHT)
            
            is_home_value = 0 if neutral_venue else 1
            
            home_rows = pd.DataFrame({
                "team": df.home_team, "goals": df.home_score, "is_home": is_home_value,
                "elo_team": df.elo_home, "elo_opponent": df.elo_away,
                "gf10": df.gf10_h, "ga10": df.ga10_h, "form5": df.form5_h,
                "tournament_weight": df.tournament_weight,
            })
            away_rows = pd.DataFrame({
                "team": df.away_team, "goals": df.away_score, "is_home": 0,
                "elo_team": df.elo_away, "elo_opponent": df.elo_home,
                "gf10": df.gf10_a, "ga10": df.ga10_a, "form5": df.form5_a,
                "tournament_weight": df.tournament_weight,
            })
            long = pd.concat([home_rows, away_rows], ignore_index=True)
            long["elo_diff"] = long.elo_team - long.elo_opponent
            return long.dropna(subset=["gf10", "ga10", "form5"])
        
        long_df = to_long(hist)
        FEATURES = ["elo_team", "elo_opponent", "elo_diff", "is_home", "gf10", "ga10", "form5", "tournament_weight"]
        
        xgb_model = xgb.XGBRegressor(
            objective="count:poisson", n_estimators=200, max_depth=4, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=5, random_state=42,
            n_jobs=1
        )
        xgb_model.fit(long_df[FEATURES], long_df["goals"])
        
        def get_snapshot(team):
            hist_team = final_form.get(team, [])
            last10, last5 = hist_team[-10:], hist_team[-5:]
            gf = np.mean([x[1] for x in last10]) if last10 else 0.0
            ga = np.mean([x[2] for x in last10]) if last10 else 0.0
            pts = sum(x[3] for x in last5) if last5 else 0.0
            elo = final_elo.get(team, ELO_INIT)
            return elo, gf, ga, pts
        
        elo_h_snap, gf_h, ga_h, pts_h = get_snapshot(home_team)
        elo_a_snap, gf_a, ga_a, pts_a = get_snapshot(away_team)
        
        is_home_value = 0 if neutral_venue else 1
        
        row_home = {"elo_team": elo_h_snap, "elo_opponent": elo_a_snap, "elo_diff": elo_h_snap - elo_a_snap,
                    "is_home": is_home_value, "gf10": gf_h, "ga10": ga_h, "form5": pts_h,
                    "tournament_weight": 4.0}
        row_away = {"elo_team": elo_a_snap, "elo_opponent": elo_h_snap, "elo_diff": elo_a_snap - elo_h_snap,
                    "is_home": 0, "gf10": gf_a, "ga10": ga_a, "form5": pts_a,
                    "tournament_weight": 4.0}
        
        feat_df = pd.DataFrame([row_home, row_away])[FEATURES]
        lam_h, lam_a = xgb_model.predict(feat_df)
        
        if use_hydration:
            lam_h, lam_a = ajustar_por_pausas_hidratacion(lam_h, lam_a, elo_h, elo_a)
        
        goals = np.arange(0, max_goals + 1)
        score_matrix = np.outer(poisson.pmf(goals, lam_h), poisson.pmf(goals, lam_a))
        
        if use_dixon_coles:
            score_matrix = aplicar_dixon_coles(score_matrix, lam_h, lam_a)
        else:
            suma = score_matrix.sum()
            if suma > 0:
                score_matrix = score_matrix / suma
        
        team_stats = {
            home_team: {"elo": elo_h, "attack": stats_h.get('attack', 1.5), "defense": stats_h.get('defense', 1.3)},
            away_team: {"elo": elo_a, "attack": stats_a.get('attack', 1.5), "defense": stats_a.get('defense', 1.3)}
        }
        
        return score_matrix, lam_h, lam_a, team_stats
    except Exception as e:
        st.error(f"❌ XGBoost: {str(e)}")
        return None, None, None, None

# ============================================================================
# PLOT REDISEÑADO
# ============================================================================
def plot_results(sm, home_team, away_team, title, max_display=7):
    if sm.shape[0] < max_display + 1 or sm.shape[1] < max_display + 1:
        sm_full = np.zeros((max_display + 1, max_display + 1))
        sm_full[:sm.shape[0], :sm.shape[1]] = sm
        sm_disp = sm_full
    else:
        sm_disp = sm[:max_display + 1, :max_display + 1]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    fig.patch.set_facecolor('#0a0e1a')
    fig.suptitle(f"{home_team} vs {away_team} — {title}", fontsize=16, fontweight="bold", color='white', y=0.98)

    from matplotlib.colors import LinearSegmentedColormap
    colors = ['#0a0e1a', '#1a3a6b', '#2d5a9a', '#00d4ff', '#ffffff']
    custom_cmap = LinearSegmentedColormap.from_list('custom', colors)

    ax0 = axes[0]
    ax0.set_facecolor('#111827')
    im = ax0.imshow(sm_disp, cmap=custom_cmap, vmin=0, origin="upper")
    for i in range(sm_disp.shape[0]):
        for j in range(sm_disp.shape[1]):
            val = sm_disp[i, j]
            if val > 0.001:
                color = "white" if val > sm_disp.max() * 0.5 else "#94a3b8"
                ax0.text(j, i, f"{val:.3f}", ha="center", va="center",
                        fontsize=8, color=color, fontweight='bold' if val == sm_disp.max() else 'normal')
    ax0.set_xticks(range(sm_disp.shape[1]))
    ax0.set_yticks(range(sm_disp.shape[0]))
    ax0.set_xlabel(f"Goles {away_team}", color='#94a3b8', fontsize=11)
    ax0.set_ylabel(f"Goles {home_team}", color='#94a3b8', fontsize=11)
    ax0.set_title("Heatmap de Marcadores", fontsize=12, color='white', pad=10)
    ax0.tick_params(colors='#94a3b8')
    
    for spine in ax0.spines.values():
        spine.set_color((1, 1, 1, 0.1))
    cbar = plt.colorbar(im, ax=ax0, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(colors='#94a3b8')

    ax1 = axes[1]
    ax1.set_facecolor('#111827')
    home_win = np.sum(np.tril(sm_disp, k=-1))
    draw = np.sum(np.diag(sm_disp))
    away_win = np.sum(np.triu(sm_disp, k=1))
    
    bars = ax1.bar([home_team[:10], "Empate", away_team[:10]],
                   [home_win, draw, away_win],
                   color=["#22c55e", "#94a3b8", "#ef4444"], 
                   width=0.5, edgecolor='none', linewidth=0)
    
    for b, v in zip(bars, [home_win, draw, away_win]):
        ax1.text(b.get_x() + b.get_width()/2, v + 0.02,
                f"{v*100:.1f}%", ha="center", fontsize=12, fontweight="bold", color='white')
    
    ax1.set_ylim(0, max(home_win, draw, away_win) * 1.3)
    ax1.set_title("Probabilidad de Resultado (1X2)", fontsize=12, color='white', pad=10)
    ax1.set_ylabel("Probabilidad", color='#94a3b8')
    ax1.tick_params(colors='#94a3b8')
    
    for spine in ax1.spines.values():
        spine.set_color((1, 1, 1, 0.1))
    ax1.spines[["top", "right"]].set_visible(False)

    ax2 = axes[2]
    ax2.set_facecolor('#111827')
    flat = np.argsort(sm.ravel())[::-1][:10]
    rows, cols = np.unravel_index(flat, sm.shape)
    probs = sm[rows, cols]
    labels = [f"{r}-{c}" for r, c in zip(rows, cols)]
    
    colors_bar = ["#ffc107"] + ["#00d4ff"] * min(9, len(labels)-1)
    bars2 = ax2.barh(np.arange(len(labels))[::-1], probs[:len(labels)], 
                     color=colors_bar[:len(labels)], edgecolor='none', height=0.7)
    
    ax2.set_yticks(np.arange(len(labels))[::-1])
    ax2.set_yticklabels(labels, fontsize=11, color='#94a3b8', fontweight='600' if len(labels) > 0 else 'normal')
    
    for y, p in zip(np.arange(len(labels))[::-1], probs[:len(labels)]):
        ax2.text(p + max(probs)*0.015, y, f"{p*100:.1f}%", va="center", fontsize=10, color='white')
    
    ax2.set_xlim(0, max(probs) * 1.25)
    ax2.set_title("Top 10 Marcadores Exactos", fontsize=12, color='white', pad=10)
    ax2.set_xlabel("Probabilidad", color='#94a3b8')
    ax2.tick_params(colors='#94a3b8')
    
    for spine in ax2.spines.values():
        spine.set_color((1, 1, 1, 0.1))
    ax2.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    return fig

# ============================================================================
# EJECUTAR PREDICCIÓN
# ============================================================================
if predict_btn:
    if home_team == away_team:
        st.error("❌ Los equipos deben ser diferentes")
        st.stop()

    # ✅ Inicializar variables de momentum antes de usarlas
    marcador_actual_h = 0
    marcador_actual_a = 0
    llegadas_previas_h = None
    llegadas_previas_a = None
    minuto_gol_favorito = None

    with st.spinner("🔄 Preparando datos..."):
        CUTOFF = pd.Timestamp(match_date) - pd.Timedelta(days=1)
        mask = (raw["date"] >= train_start) & (raw["date"] <= CUTOFF) & raw["home_score"].notna()
        train = raw.loc[mask].copy()
        train["home_score"] = train["home_score"].astype(int)
        train["away_score"] = train["away_score"].astype(int)
        train["neutral"] = train["neutral"].astype(str).str.upper().eq("TRUE")

        teams = sorted(pd.concat([train.home_team, train.away_team]).unique())
        team_idx = {t: i for i, t in enumerate(teams)}

        if home_team not in teams:
            st.error(f"❌ {home_team} no tiene suficientes partidos")
            st.stop()
        if away_team not in teams:
            st.error(f"❌ {away_team} no tiene suficientes partidos")
            st.stop()

    stats_h = get_espn_team_stats(home_team)
    stats_a = get_espn_team_stats(away_team)
    elo_h = stats_h.get('elo', 1750)
    elo_a = stats_a.get('elo', 1750)

    if elo_h > elo_a:
        favorito_elo = elo_h
        underdog_elo = elo_a
    else:
        favorito_elo = elo_a
        underdog_elo = elo_h

    results = {}
    errores = []

    if use_xgboost:
        try:
            with st.spinner("⚙️ Entrenando XGBoost..."):
                hist = raw[(raw.date <= CUTOFF) & raw.home_score.notna()].sort_values("date").reset_index(drop=True)
                hist = hist[hist.date >= train_start].copy()
                sm_xgb, lam_h_xgb, lam_a_xgb, team_stats = train_xgboost_model(
                    hist, raw, home_team, away_team, max_goals_display,
                    use_hydration_adjustment, use_dixon_coles, neutral_venue
                )
                if sm_xgb is not None:
                    # ✅ ORDEN CORRECTO:
                    # 1. Aplicar momentum (modifica lambdas)
                    if use_momentum_adjustment:
                        es_favorito_local = elo_h > elo_a
                        lam_h_xgb, lam_a_xgb = ajustar_por_momentum(
                            lam_h_xgb, lam_a_xgb,
                            home_team, away_team,
                            minuto_gol=minuto_gol_favorito,
                            es_favorito_local=es_favorito_local,
                            llegadas_previas_h=llegadas_previas_h,
                            llegadas_previas_a=llegadas_previas_a,
                            marcador_actual={'home': marcador_actual_h, 'away': marcador_actual_a}
                        )
                        # Recalcular matriz
                        goals = np.arange(0, max_goals_display + 1)
                        sm_xgb = np.outer(poisson.pmf(goals, lam_h_xgb), poisson.pmf(goals, lam_a_xgb))
                        if use_dixon_coles:
                            sm_xgb = aplicar_dixon_coles(sm_xgb, lam_h_xgb, lam_a_xgb)
                        else:
                            sm_xgb = sm_xgb / sm_xgb.sum()
                    
                    # 2. Aplicar gol temprano del underdog (modifica matriz)
                    if use_dynamic_adjustment:
                        sm_xgb = ajustar_por_gol_temprano(
                            sm_xgb, lam_h_xgb, lam_a_xgb,
                            home_team, away_team,
                            underdog_scored_first, minuto_gol,
                            favorito_elo, underdog_elo
                        )
                    
                    results['xgb'] = {
                        'score_matrix': sm_xgb,
                        'lam_h': lam_h_xgb,
                        'lam_a': lam_a_xgb,
                        'team_stats': team_stats
                    }
                else:
                    errores.append("XGBoost falló")
        except Exception as e:
            errores.append(f"XGBoost: {str(e)[:100]}")

    if use_bayesian and PYMC_AVAILABLE:
        try:
            with st.spinner("⚙️ Entrenando Bayesiano (1-2 min)..."):
                sm_bayes, lam_h_bayes, lam_a_bayes, att_ratings, def_ratings = train_bayesian_model(
                    train, teams, team_idx, home_team, away_team, max_goals_display,
                    use_hydration_adjustment, use_dixon_coles, neutral_venue
                )
                if sm_bayes is not None:
                    # ✅ ORDEN CORRECTO para Bayesiano también
                    if use_momentum_adjustment:
                        es_favorito_local = elo_h > elo_a
                        lam_h_bayes, lam_a_bayes = ajustar_por_momentum(
                            lam_h_bayes, lam_a_bayes,
                            home_team, away_team,
                            minuto_gol=minuto_gol_favorito,
                            es_favorito_local=es_favorito_local,
                            llegadas_previas_h=llegadas_previas_h,
                            llegadas_previas_a=llegadas_previas_a,
                            marcador_actual={'home': marcador_actual_h, 'away': marcador_actual_a}
                        )
                        goals = np.arange(0, max_goals_display + 1)
                        sm_bayes = np.outer(poisson.pmf(goals, lam_h_bayes), poisson.pmf(goals, lam_a_bayes))
                        if use_dixon_coles:
                            sm_bayes = aplicar_dixon_coles(sm_bayes, lam_h_bayes, lam_a_bayes)
                        else:
                            sm_bayes = sm_bayes / sm_bayes.sum()
                    
                    if use_dynamic_adjustment:
                        sm_bayes = ajustar_por_gol_temprano(
                            sm_bayes, lam_h_bayes, lam_a_bayes,
                            home_team, away_team,
                            underdog_scored_first, minuto_gol,
                            favorito_elo, underdog_elo
                        )
                    
                    results['bayes'] = {
                        'score_matrix': sm_bayes,
                        'lam_h': lam_h_bayes,
                        'lam_a': lam_a_bayes,
                        'att_ratings': att_ratings,
                        'def_ratings': def_ratings
                    }
                else:
                    errores.append("Bayesiano falló")
        except Exception as e:
            errores.append(f"Bayesiano: {str(e)[:100]}")

    if results:
        results['teams'] = (home_team, away_team)
        results['elo'] = {'home': elo_h, 'away': elo_a}
        st.session_state.results = results
        st.success("✅ Predicción completada!")
        if errores:
            st.warning(f"⚠️ Algunos fallaron: {', '.join(errores)}")
    else:
        st.error(f"❌ No se pudo completar. Errores: {', '.join(errores)}")

# ============================================================================
# MOSTRAR RESULTADOS
# ============================================================================
if 'results' in st.session_state and st.session_state.results:
    results = st.session_state.results
    home_team, away_team = results['teams']
    elo_h = results['elo']['home']
    elo_a = results['elo']['away']

    st.markdown("---")
    st.subheader("📊 Resumen de Predicción")

    model_count = len([m for m in results.keys() if m not in ['teams', 'elo']])
    cols = st.columns(min(model_count, 4))

    col_idx = 0
    for model_name, model_data in results.items():
        if model_name in ['teams', 'elo']:
            continue
        with cols[col_idx % len(cols)]:
            display_name = "Bayesiano" if model_name == 'bayes' else "XGBoost"
            st.metric(
                f"🏠 {home_team[:10]} ({display_name})",
                f"{model_data['lam_h']:.2f}",
                delta=f"vs {away_team[:10]} {model_data['lam_a']:.2f}",
                delta_color="off"
            )
        col_idx += 1

    st.markdown("---")

    model_cols = st.columns(model_count)
    col_idx = 0

    for model_name, model_data in results.items():
        if model_name in ['teams', 'elo']:
            continue

        with model_cols[col_idx % len(model_cols)]:
            display_name = "🔵 Bayesiano" if model_name == 'bayes' else "🟢 XGBoost"
            st.subheader(display_name)

            fig = plot_results(model_data['score_matrix'], home_team, away_team, display_name, max_display=7)
            st.pyplot(fig)
            plt.close(fig)

            sm = model_data['score_matrix'][:7, :7]
            home_win = np.sum(np.tril(sm, k=-1))
            draw = np.sum(np.diag(sm))
            away_win = np.sum(np.triu(sm, k=1))

            prob_cols = st.columns(3)
            with prob_cols[0]:
                st.metric(f"🏠 {home_team[:8]}", f"{home_win:.1%}")
                st.markdown(f'<div class="prob-bar"><div class="prob-bar-fill" style="width:{home_win*100}%; background: linear-gradient(90deg, #16a34a, #22c55e);"></div></div>', unsafe_allow_html=True)
            with prob_cols[1]:
                st.metric("🤝 Empate", f"{draw:.1%}")
                st.markdown(f'<div class="prob-bar"><div class="prob-bar-fill" style="width:{draw*100}%; background: linear-gradient(90deg, #64748b, #94a3b8);"></div></div>', unsafe_allow_html=True)
            with prob_cols[2]:
                st.metric(f"✈️ {away_team[:8]}", f"{away_win:.1%}")
                st.markdown(f'<div class="prob-bar"><div class="prob-bar-fill" style="width:{away_win*100}%; background: linear-gradient(90deg, #dc2626, #ef4444);"></div></div>', unsafe_allow_html=True)

            top_idx = np.unravel_index(model_data['score_matrix'][:7,:7].argmax(), (7,7))
            st.info(f"🎯 Marcador más probable: **{top_idx[0]}-{top_idx[1]}**")

            correcciones = []
            if use_dixon_coles:
                correcciones.append("🔧 DC")
            if use_hydration_adjustment:
                correcciones.append("💧 4 tiempos")
            if use_dynamic_adjustment:
                correcciones.append("⚡ Gol temprano")
            if use_momentum_adjustment:
                correcciones.append("⚡ Momentum")
            if neutral_venue:
                correcciones.append("🏟️ Neutral")
            if correcciones:
                st.caption(f"📌 Ajustes: {' · '.join(correcciones)}")

            if model_name == 'xgb' and 'team_stats' in model_data:
                with st.expander("📈 Estadísticas de los equipos (ESPN)"):
                    stats_data = []
                    for team, stats in model_data['team_stats'].items():
                        stats_data.append({
                            "Equipo": team,
                            "Elo": f"{stats.get('elo', 1750):.0f}",
                            "Ataque": f"{stats.get('attack', 1.5):.2f}",
                            "Defensa": f"{stats.get('defense', 1.3):.2f}"
                        })
                    st.dataframe(pd.DataFrame(stats_data), hide_index=True, use_container_width=True)

            if model_name == 'bayes' and 'att_ratings' in model_data:
                with st.expander("📊 Ratings Ataque/Defensa (Bayesiano)"):
                    ratings_data = []
                    for team in [home_team, away_team]:
                        ratings_data.append({
                            "Equipo": team,
                            "Ataque": f"{model_data['att_ratings'][team]:.3f}",
                            "Defensa": f"{model_data['def_ratings'][team]:.3f}"
                        })
                    st.dataframe(pd.DataFrame(ratings_data), hide_index=True, use_container_width=True)

        col_idx += 1

    if model_count > 1:
        st.markdown("---")
        st.subheader("📋 Comparativa de Modelos")

        comp_data = []
        for model_name, model_data in results.items():
            if model_name in ['teams', 'elo']:
                continue
            sm = model_data['score_matrix'][:7, :7]
            top_idx = np.unravel_index(sm.argmax(), sm.shape)
            comp_data.append({
                "Modelo": "🔵 Bayesiano" if model_name == 'bayes' else "🟢 XGBoost",
                f"Goles {home_team[:10]}": f"{model_data['lam_h']:.2f}",
                f"Goles {away_team[:10]}": f"{model_data['lam_a']:.2f}",
                "Top marcador": f"{top_idx[0]}-{top_idx[1]}",
                "Certeza del top": f"{sm[top_idx]:.1%}"
            })

        comp_df = pd.DataFrame(comp_data)
        st.dataframe(comp_df, use_container_width=True, hide_index=True)

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.markdown("""
<div class="footer">
    <p>
        ⚽ Datos: martj42/international_results
        <span class="separator">·</span>
        🔧 Dixon-Coles (ρ=-0.13)
        <span class="separator">·</span>
        💧 Ajuste por 4 tiempos
        <span class="separator">·</span>
        ⚡ Gol temprano del underdog
        <span class="separator">·</span>
        ⚡ Momentum
    </p>
    <p style="margin-top: 8px;">
        <a href="https://rpalafoxfalcoproject.streamlit.app" target="_blank">🔗 Abrir en nueva ventana</a>
        <span class="separator">·</span>
        <a href="https://satohachi.rpalafox.com/" target="_blank">🐝 rpalafox.com</a>
    </p>
</div>
""", unsafe_allow_html=True)

# ============================================================================
# INFO DEL SISTEMA
# ============================================================================
with st.expander("ℹ️ Información del sistema"):
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**Python:** `{sys.version}`")
        st.write(f"**PyMC:** {'✅ Disponible' if PYMC_AVAILABLE else '❌ No disponible'}")
        st.write(f"**SKLearn:** {'✅ Disponible' if SKLEARN_AVAILABLE else '❌ No disponible'}")
    with col2:
        st.write(f"**Dixon-Coles ρ:** `{DIXON_COLES_RHO:.3f}`")
        st.write(f"**Equipos clasificados:** `{len(wc_teams_in_data)}`")
        if PYMC_AVAILABLE:
            st.write(f"**PyMC versión:** `{pm.__version__}`")
            st.write(f"**ArviZ versión:** `{az.__version__}`")
