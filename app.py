# app.py - Predicción Mundial 2026 - Versión Completa con Dixon-Coles + Gol Temprano
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson
import warnings
import sys
import platform
import requests
import json
from datetime import datetime, timedelta

# ⚠️ IMPORTANTE: set_page_config DEBE SER LA PRIMERA INSTRUCCIÓN DE STREAMLIT
st.set_page_config(
    page_title="Predicción Mundial 2026",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Suprimir warnings
warnings.filterwarnings('ignore')

# ============================================================================
# LISTA DE SELECCIONES CALIFICADAS AL MUNDIAL 2026 (ACTUALIZADA)
# ============================================================================
WORLD_CUP_2026_TEAMS = [
    # UEFA (Europa) - 16 equipos
    "Spain", "Portugal", "France", "Belgium", "Netherlands", "Italy", "Germany",
    "England", "Croatia", "Switzerland", "Denmark", "Austria", "Ukraine",
    "Sweden", "Wales", "Turkey", "Scotland", "Norway", "Serbia", "Bosnia and Herzegovina",
    "Poland", "Czech Republic", "Hungary", "Slovakia",

    # CONMEBOL (Sudamérica) - 7 equipos
    "Brazil", "Argentina", "Uruguay", "Ecuador", "Colombia", "Chile", "Paraguay",

    # CONCACAF (Norteamérica) - 8 equipos (incluyendo anfitriones)
    "Mexico", "USA", "Canada", "Costa Rica", "Jamaica", "Panama", "Honduras", "El Salvador",
    "Haiti", "Curacao",

    # CAF (África) - 10 equipos
    "Morocco", "Senegal", "Nigeria", "Egypt", "Ghana", "Cameroon", "Algeria",
    "Tunisia", "Mali", "Ivory Coast", "Cabo Verde", "South Africa", "Congo DR",

    # AFC (Asia) - 8 equipos
    "Japan", "South Korea", "Iran", "Saudi Arabia", "Australia", "Qatar",
    "UAE", "Iraq", "Jordan", "Uzbekistan",

    # OFC (Oceanía) - 1 equipo
    "New Zealand"
]

# Normalizar nombres
WORLD_CUP_2026_TEAMS = sorted([team for team in WORLD_CUP_2026_TEAMS])

# ============================================================================
# PARÁMETROS DIXON-COLES (estimados vía MCMC)
# ============================================================================
DIXON_COLES_RHO = -0.039

# ============================================================================
# FUNCIÓN DE AJUSTE HÍBRIDO (4 tiempos con pausas de hidratación)
# ============================================================================
def ajustar_por_pausas_hidratacion(lam_h, lam_a):
    """
    Ajuste híbrido para el formato de 4 tiempos con pausas de hidratación.
    """
    factor_general = 0.92
    media_h_general = 1.35
    media_a_general = 1.05
    
    lam_h_ajustado = lam_h * factor_general
    lam_a_ajustado = lam_a * factor_general
    
    factor_contraccion = 0.15
    lam_h_ajustado = lam_h_ajustado * (1 - factor_contraccion) + media_h_general * factor_contraccion
    lam_a_ajustado = lam_a_ajustado * (1 - factor_contraccion) + media_a_general * factor_contraccion
    
    return lam_h_ajustado, lam_a_ajustado

# ============================================================================
# FUNCIÓN DIXON-COLES
# ============================================================================
def dixon_coles_factor(x, y, lam_h, lam_a, rho):
    """
    Factor de corrección de Dixon-Coles para la probabilidad conjunta.
    """
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
    """
    Aplica la corrección de Dixon-Coles a una matriz de marcadores.
    """
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
    """
    Ajusta la matriz de marcadores si el underdog anota primero.
    """
    if not underdog_scored_first:
        return score_matrix
    
    # Determinar favorito basado en λ y Elo combinados
    if lam_h > lam_a:
        favorito = 'home'
        underdog = 'away'
        diff_lam = lam_h - lam_a
    else:
        favorito = 'away'
        underdog = 'home'
        diff_lam = lam_a - lam_h
    
    # Factores de ajuste
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
    
    # Reducir marcadores bajos (0-0, 1-0, 0-1, 2-0, 0-2)
    for i in range(min(3, max_g + 1)):
        for j in range(min(3, max_g + 1)):
            if i + j <= 2:
                ajuste[i, j] = 1 - factor_ajuste * 0.3
    
    # Aumentar marcadores de remontada (2-1, 1-2, 2-2, 3-1, 1-3, 3-2, 2-3)
    for i in range(1, min(5, max_g + 1)):
        for j in range(1, min(5, max_g + 1)):
            if 3 <= i + j <= 6:
                ajuste[i, j] = 1 + factor_ajuste * 0.25
    
    # Si el underdog es visitante: favorecer 1-1, 2-1, 3-1
    if underdog == 'away':
        for i in range(1, min(5, max_g + 1)):
            for j in range(1, min(3, max_g + 1)):
                if i >= j:
                    ajuste[i, j] = 1 + factor_ajuste * 0.2
    
    # Si el underdog es local: favorecer 1-1, 1-2, 1-3
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
# CONEXIÓN A ESPN
# ============================================================================
@st.cache_data(ttl=3600)
def get_espn_fixture():
    """Obtiene el fixture del Mundial 2026 desde la API de ESPN"""
    try:
        url = "https://site.web.api.espn.com/apis/site/v2/sports/soccer/fifa.worldcup/scoreboard"
        params = {"dates": "2026-06-11", "region": "us", "lang": "en", "contentorigin": "espn"}
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        
        response = requests.get(url, params=params, headers=headers, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            events = data.get('events', [])
            fixture_data = []
            
            for event in events:
                competitions = event.get('competitions', [])
                for comp in competitions:
                    competitors = comp.get('competitors', [])
                    if len(competitors) >= 2:
                        home_team = competitors[0].get('team', {}).get('displayName', '')
                        away_team = competitors[1].get('team', {}).get('displayName', '')
                        date_str = event.get('date', '')
                        try:
                            match_date = datetime.fromisoformat(date_str.replace('Z', '+00:00')).date()
                        except:
                            match_date = datetime.strptime("2026-06-15", "%Y-%m-%d").date()
                        
                        fixture_data.append({
                            'date': match_date,
                            'home_team': home_team,
                            'away_team': away_team
                        })
            
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
    """Obtiene estadísticas de un equipo desde ESPN (simulado)"""
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
    
    synonyms = {
        'Cape Verde': 'Cabo Verde',
        'USA': 'USA',
        'United States': 'USA',
        'Czechia': 'Czech Republic',
        'Côte d\'Ivoire': 'Ivory Coast'
    }
    
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

# Título de la app
st.title("⚽ Predicción de Marcadores - Mundial FIFA 2026")
st.markdown(f"🐍 Python {sys.version.split()[0]} | PyMC: {'✅' if PYMC_AVAILABLE else '❌'} | SKLearn: {'✅' if SKLEARN_AVAILABLE else '❌'}")
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
# SIDEBAR - Selectores de equipos
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
    
    match_date = st.date_input("📅 Fecha del Partido", pd.to_datetime("2026-06-24"))
    train_start = st.selectbox("📊 Ventana de entrenamiento", ["2018-01-01", "2016-01-01", "2014-01-01", "2010-01-01"], index=0)
    
    st.markdown("---")
    st.subheader("🤖 Modelos a usar")
    
    use_xgboost = st.checkbox("✅ XGBoost", value=True)
    use_bayesian = st.checkbox("✅ Bayesiano" if PYMC_AVAILABLE else "❌ Bayesiano (no disponible)", 
                               value=PYMC_AVAILABLE, disabled=not PYMC_AVAILABLE)
    
    st.markdown("---")
    st.subheader("🔧 Correcciones")
    
    use_dixon_coles = st.checkbox("🔧 Dixon-Coles (ρ=-0.039)", value=True, help="Corrige la subestimación de empates")
    use_hydration_adjustment = st.checkbox("💧 Pausas de hidratación (4 tiempos)", value=True)
    
    st.markdown("---")
    st.subheader("⚡ Ajuste por Gol Temprano del Underdog")
    
    use_dynamic_adjustment = st.checkbox("⚡ Activar ajuste dinámico", value=False)
    
    if use_dynamic_adjustment:
        underdog_scored_first = st.checkbox("🏃 El underdog anotó primero", value=False)
        minuto_gol = st.slider("⏱️ Minuto del primer gol", 1, 90, 15, help="Minuto en que el underdog anotó")
        
        st.caption("💡 Si el underdog anota primero, el partido se vuelve más abierto. Este ajuste aumenta la probabilidad de marcadores con más goles.")
    
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
# FUNCIONES DE PREDICCIÓN
# ============================================================================
def train_bayesian_model(train, teams, team_idx, home_team, away_team, max_goals=8,
                         use_hydration=True, use_dixon_coles=True):
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
        
        if use_hydration:
            lam_h, lam_a = ajustar_por_pausas_hidratacion(lam_h, lam_a)
        
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
                        use_hydration=True, use_dixon_coles=True):
    try:
        import xgboost as xgb
        
        K_ELO = 20.0
        ELO_INIT = 1500.0
        TOURNAMENT_WEIGHTS = {"FIFA World Cup": 4.0, "FIFA World Cup qualification": 2.0, "Friendly": 0.5}
        DEFAULT_WEIGHT = 1.0
        
        stats_h = get_espn_team_stats(home_team)
        stats_a = get_espn_team_stats(away_team)
        
        ratings = {}
        elo_h, elo_a = [], []
        for _, row in hist.iterrows():
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
        
        hist = hist.copy()
        hist["elo_home"], hist["elo_away"] = elo_h, elo_a
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
            home_rows = pd.DataFrame({
                "team": df.home_team, "goals": df.home_score, "is_home": 1,
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
        
        elo_h, gf_h, ga_h, pts_h = get_snapshot(home_team)
        elo_a, gf_a, ga_a, pts_a = get_snapshot(away_team)
        
        row_home = {"elo_team": elo_h, "elo_opponent": elo_a, "elo_diff": elo_h - elo_a,
                    "is_home": 1, "gf10": gf_h, "ga10": ga_h, "form5": pts_h,
                    "tournament_weight": 4.0}
        row_away = {"elo_team": elo_a, "elo_opponent": elo_h, "elo_diff": elo_a - elo_h,
                    "is_home": 0, "gf10": gf_a, "ga10": ga_a, "form5": pts_a,
                    "tournament_weight": 4.0}
        
        feat_df = pd.DataFrame([row_home, row_away])[FEATURES]
        lam_h, lam_a = xgb_model.predict(feat_df)
        
        if use_hydration:
            lam_h, lam_a = ajustar_por_pausas_hidratacion(lam_h, lam_a)
        
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

def plot_results(sm, home_team, away_team, title, max_display=7):
    if sm.shape[0] < max_display + 1 or sm.shape[1] < max_display + 1:
        sm_full = np.zeros((max_display + 1, max_display + 1))
        sm_full[:sm.shape[0], :sm.shape[1]] = sm
        sm_disp = sm_full
    else:
        sm_disp = sm[:max_display + 1, :max_display + 1]
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.suptitle(f"{home_team} vs {away_team} — {title}", fontsize=14, fontweight="bold")
    
    im = axes[0].imshow(sm_disp, cmap="Blues", vmin=0, origin="upper")
    for i in range(sm_disp.shape[0]):
        for j in range(sm_disp.shape[1]):
            val = sm_disp[i, j]
            if val > 0.001:
                color = "white" if val > sm_disp.max() * 0.5 else "black"
                axes[0].text(j, i, f"{val:.3f}", ha="center", va="center",
                            fontsize=7, color=color, fontweight='bold' if val == sm_disp.max() else 'normal')
    axes[0].set_xticks(range(sm_disp.shape[1]))
    axes[0].set_yticks(range(sm_disp.shape[0]))
    axes[0].set_xlabel(f"Goles {away_team}")
    axes[0].set_ylabel(f"Goles {home_team}")
    axes[0].set_title("Heatmap de Marcadores", fontsize=11)
    plt.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)
    
    home_win = np.sum(np.tril(sm_disp, k=-1))
    draw = np.sum(np.diag(sm_disp))
    away_win = np.sum(np.triu(sm_disp, k=1))
    
    bars = axes[1].bar([home_team[:10], "Empate", away_team[:10]],
                       [home_win, draw, away_win],
                       color=["#2e7d32", "#9e9e9e", "#c62828"], width=0.6)
    for b, v in zip(bars, [home_win, draw, away_win]):
        axes[1].text(b.get_x() + b.get_width()/2, v + 0.01,
                    f"{v*100:.1f}%", ha="center", fontsize=10, fontweight="bold")
    axes[1].set_ylim(0, max(home_win, draw, away_win) * 1.3)
    axes[1].set_title("Probabilidad de Resultado (1X2)", fontsize=11)
    axes[1].set_ylabel("Probabilidad")
    axes[1].spines[["top", "right"]].set_visible(False)
    
    flat = np.argsort(sm.ravel())[::-1][:10]
    rows, cols = np.unravel_index(flat, sm.shape)
    probs = sm[rows, cols]
    labels = [f"{r}-{c}" for r, c in zip(rows, cols)]
    
    colors = ["#e94f37"] + ["#3b6fb6"] * min(9, len(labels)-1)
    axes[2].barh(np.arange(len(labels))[::-1], probs[:len(labels)], color=colors[:len(labels)])
    axes[2].set_yticks(np.arange(len(labels))[::-1])
    axes[2].set_yticklabels(labels, fontsize=9.5)
    for y, p in zip(np.arange(len(labels))[::-1], probs[:len(labels)]):
        axes[2].text(p + max(probs)*0.02, y, f"{p*100:.1f}%", va="center", fontsize=8.5)
    axes[2].set_xlim(0, max(probs) * 1.2)
    axes[2].set_title("Top 10 Marcadores Exactos", fontsize=11)
    axes[2].set_xlabel("Probabilidad")
    axes[2].spines[["top", "right"]].set_visible(False)
    
    plt.tight_layout()
    return fig

# ============================================================================
# EJECUTAR PREDICCIÓN
# ============================================================================
if predict_btn:
    if home_team == away_team:
        st.error("❌ Los equipos deben ser diferentes")
        st.stop()
    
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
    
    # Obtener Elos para el ajuste dinámico
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
                    use_hydration_adjustment, use_dixon_coles
                )
                if sm_xgb is not None:
                    # Aplicar ajuste dinámico si está activado
                    if use_dynamic_adjustment:
                        sm_xgb_original = sm_xgb.copy()
                        sm_xgb = ajustar_por_gol_temprano(
                            sm_xgb, lam_h_xgb, lam_a_xgb,
                            home_team, away_team,
                            underdog_scored_first, minuto_gol,
                            favorito_elo, underdog_elo
                        )
                    else:
                        sm_xgb_original = None
                    
                    results['xgb'] = {
                        'score_matrix': sm_xgb,
                        'lam_h': lam_h_xgb,
                        'lam_a': lam_a_xgb,
                        'team_stats': team_stats,
                        'original_matrix': sm_xgb_original
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
                    use_hydration_adjustment, use_dixon_coles
                )
                if sm_bayes is not None:
                    if use_dynamic_adjustment:
                        sm_bayes_original = sm_bayes.copy()
                        sm_bayes = ajustar_por_gol_temprano(
                            sm_bayes, lam_h_bayes, lam_a_bayes,
                            home_team, away_team,
                            underdog_scored_first, minuto_gol,
                            favorito_elo, underdog_elo
                        )
                    else:
                        sm_bayes_original = None
                    
                    results['bayes'] = {
                        'score_matrix': sm_bayes,
                        'lam_h': lam_h_bayes,
                        'lam_a': lam_a_bayes,
                        'att_ratings': att_ratings,
                        'def_ratings': def_ratings,
                        'original_matrix': sm_bayes_original
                    }
                else:
                    errores.append("Bayesiano falló")
        except Exception as e:
            errores.append(f"Bayesiano: {str(e)[:100]}")
    
    if results:
        results['teams'] = (home_team, away_team)
        results['dynamic_adjustment'] = use_dynamic_adjustment
        if use_dynamic_adjustment:
            results['dynamic_info'] = {
                'underdog_scored': underdog_scored_first,
                'minuto': minuto_gol,
                'favorito_elo': favorito_elo,
                'underdog_elo': underdog_elo
            }
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
    use_dynamic_adjustment = results.get('dynamic_adjustment', False)
    
    # Mostrar información del ajuste dinámico si está activo
    if use_dynamic_adjustment and results.get('dynamic_info', {}).get('underdog_scored', False):
        info = results['dynamic_info']
        underdog = 'local' if elo_h < elo_a else 'visitante'
        st.info(
            f"⚡ **Ajuste por gol temprano activado**\n\n"
            f"El {underdog} anotó en el minuto **{info['minuto']}**. "
            f"Esto ha aumentado la probabilidad de marcadores con más goles.\n\n"
            f"📊 Diferencia de Elo: {abs(info['favorito_elo'] - info['underdog_elo']):.0f} puntos"
        )
    
    st.markdown("---")
    st.subheader("📊 Resumen de Predicción")
    
    model_count = len([m for m in results.keys() if m not in ['teams', 'dynamic_adjustment', 'dynamic_info']])
    cols = st.columns(min(model_count, 4))
    
    col_idx = 0
    for model_name, model_data in results.items():
        if model_name in ['teams', 'dynamic_adjustment', 'dynamic_info']:
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
        if model_name in ['teams', 'dynamic_adjustment', 'dynamic_info']:
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
            
            col1, col2, col3 = st.columns(3)
            col1.metric(f"🏠 {home_team[:8]}", f"{home_win:.1%}")
            col2.metric("🤝 Empate", f"{draw:.1%}")
            col3.metric(f"✈️ {away_team[:8]}", f"{away_win:.1%}")
            
            top_idx = np.unravel_index(model_data['score_matrix'][:7,:7].argmax(), (7,7))
            st.info(f"🎯 Marcador más probable: **{top_idx[0]}-{top_idx[1]}**")
            
            # Mostrar info de correcciones
            correcciones = []
            if use_dixon_coles:
                correcciones.append("DC (ρ=-0.039)")
            if use_hydration_adjustment:
                correcciones.append("4 tiempos")
            if use_dynamic_adjustment and results.get('dynamic_info', {}).get('underdog_scored', False):
                correcciones.append("⚡ Gol temprano")
            
            if correcciones:
                st.caption(f"🔧 {', '.join(correcciones)}")
            
            # Comparación con ajuste dinámico
            if use_dynamic_adjustment and model_data.get('original_matrix') is not None:
                with st.expander("📊 Efecto del ajuste por gol temprano"):
                    orig_sm = model_data['original_matrix'][:7, :7]
                    adj_sm = model_data['score_matrix'][:7, :7]
                    
                    # Mostrar cambio en top 5
                    orig_top = np.argsort(orig_sm.ravel())[::-1][:5]
                    adj_top = np.argsort(adj_sm.ravel())[::-1][:5]
                    orig_labels = [f"{r}-{c}" for r, c in np.unravel_index(orig_top, orig_sm.shape)]
                    adj_labels = [f"{r}-{c}" for r, c in np.unravel_index(adj_top, adj_sm.shape)]
                    orig_probs = orig_sm.ravel()[orig_top]
                    adj_probs = adj_sm.ravel()[adj_top]
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.caption("Antes del ajuste")
                        for label, prob in zip(orig_labels, orig_probs):
                            st.write(f"{label}: {prob*100:.1f}%")
                    with col2:
                        st.caption("Después del ajuste")
                        for label, prob in zip(adj_labels, adj_probs):
                            st.write(f"{label}: {prob*100:.1f}%")
            
            if model_name == 'xgb' and 'team_stats' in model_data:
                with st.expander("📈 Estadísticas de los equipos"):
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
            if model_name in ['teams', 'dynamic_adjustment', 'dynamic_info']:
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
st.caption(" Creado con ⚽ por [rpalafox.com](https://satohachi.rpalafox.com/) ")

with st.expander("ℹ️ Información del sistema"):
    st.write(f"**Python:** {sys.version}")
    st.write(f"**PyMC:** {'✅ Sí' if PYMC_AVAILABLE else '❌ No'}")
    st.write(f"**SKLearn:** {'✅ Sí' if SKLEARN_AVAILABLE else '❌ No'}")
    st.write(f"**Dixon-Coles ρ:** {DIXON_COLES_RHO:.3f}")
    st.write(f"**Equipos clasificados:** {len(wc_teams_in_data)}")
    if PYMC_AVAILABLE:
        st.write(f"**PyMC versión:** {pm.__version__}")
        st.write(f"**ArviZ versión:** {az.__version__}")
