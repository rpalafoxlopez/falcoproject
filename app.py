# app.py - Predicción Mundial 2026 con Streamlit
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Predicción Mundial 2026", layout="wide")
st.title("⚽ Predicción de Marcadores - Mundial FIFA 2026")
st.markdown("---")

# ============================================================================
# SIDEBAR - Configuración
# ============================================================================
st.sidebar.header("⚙️ Configuración del Partido")

# Inicializar variables de sesión
if 'teams_list' not in st.session_state:
    st.session_state.teams_list = []
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False

# Cargar datos
@st.cache_data
def load_data():
    RESULTS_URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
    raw = pd.read_csv(RESULTS_URL, parse_dates=["date"])
    return raw

try:
    raw = load_data()
    teams_list = sorted(pd.concat([raw.home_team, raw.away_team]).unique())
    st.session_state.teams_list = teams_list
    st.session_state.data_loaded = True
except Exception as e:
    st.sidebar.error(f"Error cargando datos: {e}")
    st.stop()

# Selectores de equipos
with st.sidebar:
    st.subheader("Selecciona los equipos")
    
    # Selector de equipo local
    home_team = st.selectbox(
        "🏠 Equipo Local",
        options=teams_list,
        index=teams_list.index("Mexico") if "Mexico" in teams_list else 0
    )
    
    # Selector de equipo visitante
    away_team = st.selectbox(
        "✈️ Equipo Visitante",
        options=teams_list,
        index=teams_list.index("Czech Republic") if "Czech Republic" in teams_list else 1
    )
    
    # Validar que sean diferentes
    if home_team == away_team:
        st.warning("⚠️ Los equipos deben ser diferentes")
        st.stop()
    
    # Fecha del partido
    match_date = st.date_input(
        "📅 Fecha del Partido",
        pd.to_datetime("2026-06-24")
    )
    
    # Ventana de entrenamiento
    train_start = st.selectbox(
        "📊 Ventana de entrenamiento",
        options=["2018-01-01", "2016-01-01", "2014-01-01", "2010-01-01"],
        index=0
    )
    
    st.markdown("---")
    
    # Botón de predicción
    predict_btn = st.button("🔮 Predecir", type="primary", use_container_width=True)

# ============================================================================
# FUNCIONES DE PREDICCIÓN
# ============================================================================
def train_bayesian_model(train, teams, team_idx, home_team, away_team):
    """Entrena el modelo Bayesiano y retorna predicciones"""
    import pymc as pm
    import arviz as az
    
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
        
        idata = pm.sample(draws=1000, tune=1000, chains=2, cores=1, 
                         random_seed=42, target_accept=0.9, progressbar=False)
    
    # Extraer predicción
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
    
    max_g = 8
    goals = np.arange(0, max_g + 1)
    pmf_h = poisson.pmf(goals[:, None], np.exp(log_th)[None, :])
    pmf_a = poisson.pmf(goals[:, None], np.exp(log_ta)[None, :])
    score_matrix = np.einsum("gs,as->ga", pmf_h, pmf_a) / pmf_h.shape[1]
    
    # Datos de rating para mostrar
    att_ratings = {team: post["attack"].sel(team=team).mean().item() for team in teams}
    def_ratings = {team: post["defense"].sel(team=team).mean().item() for team in teams}
    
    return score_matrix, lam_h, lam_a, att_ratings, def_ratings

def train_xgboost_model(hist, raw_data, home_team, away_team):
    """Entrena el modelo XGBoost y retorna predicciones"""
    import xgboost as xgb
    
    K_ELO = 20.0
    ELO_INIT = 1500.0
    TOURNAMENT_WEIGHTS = {"FIFA World Cup": 4.0, "FIFA World Cup qualification": 2.0, "Friendly": 0.5}
    DEFAULT_WEIGHT = 1.0
    
    # Calcular Elo
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
    
    # Calcular forma reciente
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
    
    # Crear formato largo
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
        objective="count:poisson", n_estimators=300, max_depth=4, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5, random_state=42,
    )
    xgb_model.fit(long_df[FEATURES], long_df["goals"])
    
    # Obtener snapshot para los equipos
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
    
    max_g = 8
    goals = np.arange(0, max_g + 1)
    score_matrix = np.outer(poisson.pmf(goals, lam_h), poisson.pmf(goals, lam_a))
    
    # Crear diccionario con información de Elo y forma
    team_stats = {
        home_team: {"elo": elo_h, "gf10": gf_h, "ga10": ga_h, "form5": pts_h},
        away_team: {"elo": elo_a, "gf10": gf_a, "ga10": ga_a, "form5": pts_a}
    }
    
    return score_matrix, lam_h, lam_a, team_stats

def plot_results(sm, home_team, away_team, title):
    """Genera los gráficos de resultados"""
    sm_disp = sm[:7, :7]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.suptitle(f"{home_team} vs {away_team} — {title}", fontsize=14, fontweight="bold")
    
    # Heatmap
    im = axes[0].imshow(sm_disp, cmap="Blues", vmin=0, origin="upper")
    for i in range(sm_disp.shape[0]):
        for j in range(sm_disp.shape[1]):
            val = sm_disp[i, j]
            color = "white" if val > sm_disp.max() * 0.5 else "black"
            axes[0].text(j, i, f"{val:.3f}", ha="center", va="center", 
                        fontsize=7, color=color)
    axes[0].set_xticks(range(sm_disp.shape[1]))
    axes[0].set_yticks(range(sm_disp.shape[0]))
    axes[0].set_xlabel(f"Goles {away_team}")
    axes[0].set_ylabel(f"Goles {home_team}")
    axes[0].set_title("Heatmap de Marcadores", fontsize=11)
    plt.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)
    
    # 1X2
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
    
    # Top 10
    flat = np.argsort(sm.ravel())[::-1][:10]
    rows, cols = np.unravel_index(flat, sm.shape)
    probs = sm[rows, cols]
    labels = [f"{r}-{c}" for r, c in zip(rows, cols)]
    
    colors = ["#e94f37"] + ["#3b6fb6"] * min(9, len(labels)-1)
    axes[2].barh(np.arange(len(labels))[::-1], probs[:len(labels)], color=colors)
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
    
    with st.spinner("🔄 Cargando datos y entrenando modelos..."):
        # Preparar datos de entrenamiento
        CUTOFF = pd.Timestamp(match_date) - pd.Timedelta(days=1)
        mask = (raw["date"] >= train_start) & (raw["date"] <= CUTOFF) & raw["home_score"].notna()
        train = raw.loc[mask].copy()
        train["home_score"] = train["home_score"].astype(int)
        train["away_score"] = train["away_score"].astype(int)
        train["neutral"] = train["neutral"].astype(str).str.upper().eq("TRUE")
        
        teams = sorted(pd.concat([train.home_team, train.away_team]).unique())
        team_idx = {t: i for i, t in enumerate(teams)}
        
        # Verificar que los equipos existen
        if home_team not in teams:
            st.error(f"❌ {home_team} no tiene suficientes partidos en el histórico")
            st.stop()
        if away_team not in teams:
            st.error(f"❌ {away_team} no tiene suficientes partidos en el histórico")
            st.stop()
        
        # Entrenar modelos
        st.info("⚙️ Entrenando modelo Bayesiano (puede tomar 1-2 minutos)...")
        sm_bayes, lam_h_bayes, lam_a_bayes, att_ratings, def_ratings = train_bayesian_model(
            train, teams, team_idx, home_team, away_team
        )
        
        st.info("⚙️ Entrenando modelo XGBoost...")
        hist = raw[(raw.date <= CUTOFF) & raw.home_score.notna()].sort_values("date").reset_index(drop=True)
        hist = hist[hist.date >= train_start].copy()
        sm_xgb, lam_h_xgb, lam_a_xgb, team_stats = train_xgboost_model(
            hist, raw, home_team, away_team
        )
        
        # Guardar en session_state para mostrar
        st.session_state.results = {
            'bayes': {
                'score_matrix': sm_bayes,
                'lam_h': lam_h_bayes,
                'lam_a': lam_a_bayes,
                'att_ratings': att_ratings,
                'def_ratings': def_ratings
            },
            'xgb': {
                'score_matrix': sm_xgb,
                'lam_h': lam_h_xgb,
                'lam_a': lam_a_xgb,
                'team_stats': team_stats
            },
            'teams': (home_team, away_team)
        }
        st.success("✅ Predicción completada!")

# ============================================================================
# MOSTRAR RESULTADOS
# ============================================================================
if 'results' in st.session_state:
    results = st.session_state.results
    home_team, away_team = results['teams']
    
    # Métricas principales
    st.markdown("---")
    st.subheader("📊 Resumen de Predicción")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            f"🏠 {home_team[:10]}",
            f"{results['bayes']['lam_h']:.2f}",
            delta="Bayesiano",
            delta_color="off"
        )
    with col2:
        st.metric(
            f"✈️ {away_team[:10]}",
            f"{results['bayes']['lam_a']:.2f}",
            delta="Bayesiano",
            delta_color="off"
        )
    with col3:
        st.metric(
            f"🏠 {home_team[:10]}",
            f"{results['xgb']['lam_h']:.2f}",
            delta="XGBoost",
            delta_color="off"
        )
    with col4:
        st.metric(
            f"✈️ {away_team[:10]}",
            f"{results['xgb']['lam_a']:.2f}",
            delta="XGBoost",
            delta_color="off"
        )
    
    # Panel de estadísticas XGBoost
    with st.expander("📈 Estadísticas de los equipos (XGBoost)"):
        stats_df = pd.DataFrame([
            {
                "Equipo": team,
                "Elo": stats["elo"],
                "Goles a favor (últ 10)": f"{stats['gf10']:.2f}",
                "Goles en contra (últ 10)": f"{stats['ga10']:.2f}",
                "Puntos (últ 5)": f"{stats['form5']:.0f}"
            }
            for team, stats in results['xgb']['team_stats'].items()
        ])
        st.dataframe(stats_df, use_container_width=True, hide_index=True)
    
    # Panel de ratings Bayesianos
    with st.expander("📊 Ratings de Ataque y Defensa (Bayesiano)"):
        ratings_data = []
        for team in [home_team, away_team]:
            ratings_data.append({
                "Equipo": team,
                "Ataque": f"{results['bayes']['att_ratings'][team]:.3f}",
                "Defensa": f"{results['bayes']['def_ratings'][team]:.3f}"
            })
        ratings_df = pd.DataFrame(ratings_data)
        st.dataframe(ratings_df, use_container_width=True, hide_index=True)
    
    # Gráficos
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔵 Modelo Bayesiano")
        fig = plot_results(
            results['bayes']['score_matrix'],
            home_team, away_team,
            "Bayesiano Jerárquico"
        )
        st.pyplot(fig)
        plt.close(fig)
        
        # Probabilidades
        sm = results['bayes']['score_matrix'][:7, :7]
        home_win = np.sum(np.tril(sm, k=-1))
        draw = np.sum(np.diag(sm))
        away_win = np.sum(np.triu(sm, k=1))
        st.info(
            f"**Probabilidades:** {home_team} {home_win:.1%} | "
            f"Empate {draw:.1%} | {away_team} {away_win:.1%}"
        )
    
    with col2:
        st.subheader("🟢 XGBoost")
        fig = plot_results(
            results['xgb']['score_matrix'],
            home_team, away_team,
            "XGBoost (count:poisson)"
        )
        st.pyplot(fig)
        plt.close(fig)
        
        # Probabilidades
        sm = results['xgb']['score_matrix'][:7, :7]
        home_win = np.sum(np.tril(sm, k=-1))
        draw = np.sum(np.diag(sm))
        away_win = np.sum(np.triu(sm, k=1))
        st.info(
            f"**Probabilidades:** {home_team} {home_win:.1%} | "
            f"Empate {draw:.1%} | {away_team} {away_win:.1%}"
        )
    
    # Comparativa
    st.markdown("---")
    st.subheader("📋 Comparativa de Modelos")
    
    # Crear tabla comparativa
    bayes_sm = results['bayes']['score_matrix'][:7, :7]
    xgb_sm = results['xgb']['score_matrix'][:7, :7]
    
    bayes_top = np.unravel_index(bayes_sm.argmax(), bayes_sm.shape)
    xgb_top = np.unravel_index(xgb_sm.argmax(), xgb_sm.shape)
    
    comp_df = pd.DataFrame([
        {
            "Modelo": "Bayesiano",
            f"Goles {home_team[:10]}": f"{results['bayes']['lam_h']:.2f}",
            f"Goles {away_team[:10]}": f"{results['bayes']['lam_a']:.2f}",
            "Top marcador": f"{bayes_top[0]}-{bayes_top[1]}",
        },
        {
            "Modelo": "XGBoost",
            f"Goles {home_team[:10]}": f"{results['xgb']['lam_h']:.2f}",
            f"Goles {away_team[:10]}": f"{results['xgb']['lam_a']:.2f}",
            "Top marcador": f"{xgb_top[0]}-{xgb_top[1]}",
        }
    ])
    st.dataframe(comp_df, use_container_width=True, hide_index=True)
    
    st.caption("🔴 Los marcadores con borde rojo en los heatmaps indican el resultado más probable")

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.caption("⚽ Datos: martj42/international_results | Modelos: Bayesiano Jerárquico (Baio-Blangiardo) & XGBoost")
