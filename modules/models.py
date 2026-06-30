# modules/models.py - Modelos de predicción
import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import poisson

from . import config
from . import data_loader
from . import corrections

# ============================================================================
# MODELO BAYESIANO
# ============================================================================

def train_bayesian_model(train, teams, team_idx, home_team, away_team, max_goals=8,
                         use_hydration=True, use_dixon_coles=True, neutral_venue=False,
                         use_high_scoring=True):
    """Entrena el modelo Bayesiano - VERSIÓN SIMPLIFICADA Y ROBUSTA"""
    if not config.PYMC_AVAILABLE:
        return None, None, None, None, None

    try:
        import pymc as pm
        import arviz as az

        # Filtrar datos
        train_filtered = train.dropna(subset=['home_score', 'away_score'])
        train_filtered = train_filtered[(train_filtered['home_score'] >= 0) & (train_filtered['away_score'] >= 0)]

        if len(train_filtered) < 20:
            st.warning(f"⚠️ Bayesiano: Datos insuficientes ({len(train_filtered)} partidos)")
            return None, None, None, None, None

        # Preparar datos
        home_idx = train_filtered.home_team.map(team_idx).values
        away_idx = train_filtered.away_team.map(team_idx).values
        home_goals = train_filtered.home_score.values.astype(int)
        away_goals = train_filtered.away_score.values.astype(int)

        # Verificar equipos
        if home_team not in team_idx or away_team not in team_idx:
            return None, None, None, None, None

        coords = {"team": teams}

        with pm.Model(coords=coords) as bayes_model:
            sigma_att = pm.HalfNormal("sigma_att", sigma=0.3)
            sigma_def = pm.HalfNormal("sigma_def", sigma=0.3)

            attack = pm.Normal("attack", mu=0.0, sigma=sigma_att, dims="team")
            defense = pm.Normal("defense", mu=0.0, sigma=sigma_def, dims="team")

            attack_centered = attack - pm.math.mean(attack)
            defense_centered = defense - pm.math.mean(defense)

            if neutral_venue:
                home_adv = 0.0
            else:
                home_adv = pm.Normal("home_adv", mu=0.15, sigma=0.2)

            intercept = pm.Normal("intercept", mu=0.0, sigma=0.3)

            log_theta_home = intercept + home_adv + attack_centered[home_idx] - defense_centered[away_idx]
            log_theta_away = intercept + attack_centered[away_idx] - defense_centered[home_idx]

            theta_home = pm.math.exp(log_theta_home)
            theta_away = pm.math.exp(log_theta_away)

            pm.Poisson("home_goals_obs", mu=theta_home, observed=home_goals)
            pm.Poisson("away_goals_obs", mu=theta_away, observed=away_goals)

            idata = pm.sample(
                draws=600,
                tune=600,
                chains=2,
                cores=1,
                random_seed=42,
                target_accept=0.85,
                progressbar=False,
                return_inferencedata=True
            )

        post = idata.posterior

        intercept_mean = float(post["intercept"].mean().values)
        home_adv_mean = float(post["home_adv"].mean().values) if not neutral_venue else 0.0

        attack_vals = post["attack"].mean(dim=["chain", "draw"]).values
        defense_vals = post["defense"].mean(dim=["chain", "draw"]).values

        attack_mean = np.mean(attack_vals)
        defense_mean = np.mean(defense_vals)
        attack_centered_vals = attack_vals - attack_mean
        defense_centered_vals = defense_vals - defense_mean

        hi = team_idx[home_team]
        ai = team_idx[away_team]

        lam_h = np.exp(intercept_mean + home_adv_mean + attack_centered_vals[hi] - defense_centered_vals[ai])
        lam_a = np.exp(intercept_mean + attack_centered_vals[ai] - defense_centered_vals[hi])

        lam_h = max(lam_h, 0.1)
        lam_a = max(lam_a, 0.1)

        stats_h = data_loader.get_espn_team_stats(home_team)
        stats_a = data_loader.get_espn_team_stats(away_team)
        elo_h = stats_h.get('elo', 1750)
        elo_a = stats_a.get('elo', 1750)

        if use_hydration:
            lam_h, lam_a = corrections.ajustar_por_pausas_hidratacion(lam_h, lam_a, elo_h, elo_a)

        if use_high_scoring:
            lam_h, lam_a = corrections.ajuste_completo_alta_anotacion(
                lam_h, lam_a, home_team, away_team, stats_h, stats_a
            )

        goals = np.arange(0, max_goals + 1)
        pmf_h = poisson.pmf(goals, lam_h)
        pmf_a = poisson.pmf(goals, lam_a)
        score_matrix = np.outer(pmf_h, pmf_a)

        if use_high_scoring:
            score_matrix = corrections.ajustar_matriz_alta_anotacion(score_matrix, lam_h, lam_a, max_goals)

        if use_dixon_coles:
            score_matrix = corrections.aplicar_dixon_coles(score_matrix, lam_h, lam_a)
        else:
            suma = score_matrix.sum()
            if suma > 0:
                score_matrix = score_matrix / suma

        att_ratings = {team: float(attack_centered_vals[team_idx[team]]) for team in teams}
        def_ratings = {team: float(defense_centered_vals[team_idx[team]]) for team in teams}

        return score_matrix, lam_h, lam_a, att_ratings, def_ratings

    except Exception as e:
        st.warning(f"⚠️ Bayesiano: {str(e)}")
        return None, None, None, None, None

# ============================================================================
# MODELO XGBOOST
# ============================================================================

def train_xgboost_model(hist, raw_data, home_team, away_team, max_goals=8,
                        use_hydration=True, use_dixon_coles=True, neutral_venue=False,
                        use_high_scoring=True):
    """Entrena el modelo XGBoost - MODELO COMPARATIVO"""
    try:
        import xgboost as xgb
        K_ELO = 20.0
        ELO_INIT = 1500.0
        TOURNAMENT_WEIGHTS = {"FIFA World Cup": 4.0, "FIFA World Cup qualification": 2.0, "Friendly": 0.5}
        DEFAULT_WEIGHT = 1.0
        stats_h = data_loader.get_espn_team_stats(home_team)
        stats_a = data_loader.get_espn_team_stats(away_team)
        elo_h = stats_h.get('elo', 1750)
        elo_a = stats_a.get('elo', 1750)

        ratings = {}
        elo_h_list, elo_a_list = [], []
        for _, row in hist.iterrows():
            rh = ratings.get(row.home_team, ELO_INIT)
            ra = ratings.get(row.away_team, ELO_INIT)
            elo_h_list.append(rh); elo_a_list.append(ra)
            exp_h = 1.0 / (1.0 + 10 ** ((ra - rh) / 400.0))
            if row.home_score > row.away_score: score = 1.0
            elif row.home_score == row.away_score: score = 0.5
            else: score = 0.0
            margin = abs(row.home_score - row.away_score)
            delta = K_ELO * (np.log(margin + 1) + 1.0) * (score - exp_h)
            ratings[row.home_team] = rh + delta
            ratings[row.away_team] = ra - delta

        hist = hist.copy()
        hist["elo_home"] = elo_h_list
        hist["elo_away"] = elo_a_list
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
            gf10_h.append(hgf)
            ga10_h.append(hga)
            form5_h.append(hpts)
            gf10_a.append(agf)
            ga10_a.append(aga)
            form5_a.append(apts)

            h_pts = 3 if row.home_score > row.away_score else (1 if row.home_score == row.away_score else 0)
            a_pts = 3 if row.away_score > row.home_score else (1 if row.home_score == row.away_score else 0)
            records.setdefault(row.home_team, []).append((row.date, row.home_score, row.away_score, h_pts))
            records.setdefault(row.away_team, []).append((row.date, row.away_score, row.home_score, a_pts))

        hist["gf10_h"] = gf10_h
        hist["ga10_h"] = ga10_h
        hist["form5_h"] = form5_h
        hist["gf10_a"] = gf10_a
        hist["ga10_a"] = ga10_a
        hist["form5_a"] = form5_a
        final_form = records

        def to_long(df):
            df["tournament_weight"] = df.tournament.map(TOURNAMENT_WEIGHTS).fillna(DEFAULT_WEIGHT)

            is_home_value = 0 if neutral_venue else 1

            home_rows = pd.DataFrame({
                "team": df["home_team"],
                "goals": df["home_score"],
                "is_home": is_home_value,
                "elo_team": df["elo_home"],
                "elo_opponent": df["elo_away"],
                "gf10": df["gf10_h"],
                "ga10": df["ga10_h"],
                "form5": df["form5_h"],
                "tournament_weight": df["tournament_weight"],
            })
            away_rows = pd.DataFrame({
                "team": df["away_team"],
                "goals": df["away_score"],
                "is_home": 0,
                "elo_team": df["elo_away"],
                "elo_opponent": df["elo_home"],
                "gf10": df["gf10_a"],
                "ga10": df["ga10_a"],
                "form5": df["form5_a"],
                "tournament_weight": df["tournament_weight"],
            })
            long = pd.concat([home_rows, away_rows], ignore_index=True)
            long["elo_diff"] = long["elo_team"] - long["elo_opponent"]
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
            lam_h, lam_a = corrections.ajustar_por_pausas_hidratacion(lam_h, lam_a, elo_h, elo_a)

        if use_high_scoring:
            lam_h, lam_a = corrections.ajuste_completo_alta_anotacion(
                lam_h, lam_a, home_team, away_team, stats_h, stats_a
            )

        goals = np.arange(0, max_goals + 1)
        score_matrix = np.outer(poisson.pmf(goals, lam_h), poisson.pmf(goals, lam_a))

        if use_high_scoring:
            score_matrix = corrections.ajustar_matriz_alta_anotacion(score_matrix, lam_h, lam_a, max_goals)

        if use_dixon_coles:
            score_matrix = corrections.aplicar_dixon_coles(score_matrix, lam_h, lam_a)
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
# UTILIDADES PARA PREDICCIÓN EN VIVO
# ============================================================================

def _ajustar_lambda_tiempo(lam, minuto_actual, factor_presion=1.0):
    """Ajusta lambda por tiempo restante y presión del partido.

    Args:
        lam: Lambda original (goles esperados en 90 min)
        minuto_actual: Minuto actual del partido (1-90)
        factor_presion: Multiplicador por presión (ej: 1.2 si va perdiendo)

    Returns:
        Lambda ajustada para goles restantes
    """
    minutos_restantes = max(90 - minuto_actual, 1)
    factor_tiempo = minutos_restantes / 90.0
    return lam * factor_tiempo * factor_presion


def _calcular_factor_presion(marcador_h, marcador_a, es_local=True):
    """Calcula factor de presión basado en el estado del partido.

    Si va perdiendo → presiona más (lambda aumenta)
    Si va ganando → se defiende (lambda disminuye)
    """
    diff = marcador_h - marcador_a
    if es_local:
        if diff < 0:    # Local va perdiendo
            return 1.15
        elif diff > 0:  # Local va ganando
            return 0.92
        else:           # Empate
            return 1.0
    else:
        if diff > 0:    # Visitante va perdiendo
            return 1.15
        elif diff < 0:  # Visitante va ganando
            return 0.92
        else:
            return 1.0


def _shift_matrix_to_total(sm_restantes, marcador_h, marcador_a, max_goals):
    """Convierte matriz de goles RESTANTES a matriz de goles TOTALES.

    Desplaza la matriz para que sm_total[i+marcador_h, j+marcador_a] = sm_restantes[i,j]
    """
    max_restantes = sm_restantes.shape[0]
    sm_total = np.zeros((max_goals + 1, max_goals + 1))

    for i in range(max_restantes):
        for j in range(max_restantes):
            total_i = min(i + marcador_h, max_goals)
            total_j = min(j + marcador_a, max_goals)
            sm_total[total_i, total_j] += sm_restantes[i, j]

    # Normalizar
    suma = sm_total.sum()
    if suma > 0:
        sm_total = sm_total / suma

    return sm_total


def _moda_poisson(lam):
    """Devuelve la moda de una distribución Poisson(λ).
    La moda es floor(λ) o floor(λ)-1 cuando λ es entero.
    """
    if lam < 0.1:
        return 0
    return int(np.floor(lam))


# ============================================================================
# CALCULAR RESULTADO PROXIMAL
# ============================================================================

def calcular_proximal(sm, lam_h, lam_a, marcador_h, marcador_a,
                      use_dynamic, underdog_scored_first,
                      use_contextual, minuto_primer_gol):
    """Calcula el resultado proximal (más probable) con contexto.

    sm: matriz de goles TOTALES (ya desplazada si aplica)
    lam_h, lam_a: lambdas de goles RESTANTES
    marcador_h, marcador_a: goles ya anotados
    """
    # Resultado base (de la matriz de goles totales)
    top_idx = np.unravel_index(sm.argmax(), sm.shape)
    base = (int(top_idx[0]), int(top_idx[1]))

    # Verificar si empate es el más probable
    draw_prob = np.sum(np.diag(sm[:7, :7]))
    home_win_prob = np.sum(np.tril(sm[:7, :7], k=-1))
    away_win_prob = np.sum(np.triu(sm[:7, :7], k=1))
    es_empate = draw_prob > home_win_prob and draw_prob > away_win_prob

    # Calcular goles restantes esperados
    goles_rest_h = _moda_poisson(lam_h)
    goles_rest_a = _moda_poisson(lam_a)

    # Resultado proximal base = marcador actual + goles restantes esperados
    proximal_result = (marcador_h + goles_rest_h, marcador_a + goles_rest_a)

    underdog_first = None
    partido_roto = None

    # Ajuste por gol del underdog
    if use_dynamic and underdog_scored_first:
        # Partido más abierto → +1 gol para cada equipo
        underdog_first = (proximal_result[0] + 1, proximal_result[1] + 1)
        proximal_result = underdog_first

    # Ajuste por partido roto
    if use_contextual:
        diff = abs(marcador_h - marcador_a)
        if minuto_primer_gol <= 15 and diff >= 2:
            # Partido ya roto → favorito sigue marcando
            if marcador_h > marcador_a:
                partido_roto = (marcador_h + 1, marcador_a)
            else:
                partido_roto = (marcador_h, marcador_a + 1)
            proximal_result = partido_roto

    return {
        'base': base,
        'proximal': proximal_result,
        'underdog_first': underdog_first,
        'partido_roto': partido_roto,
        'es_empate': es_empate
    }


# ============================================================================
# EJECUTAR PREDICCIÓN
# ============================================================================

def run_prediction(raw, home_team, away_team, match_date, train_start, neutral_venue,
                   use_xgboost, use_bayesian, use_dixon_coles, use_hydration,
                   use_dynamic, underdog_scored_first, minuto_gol,
                   use_momentum, minuto_gol_favorito, llegadas_previas_h, llegadas_previas_a,
                   marcador_actual_h, marcador_actual_a, max_goals_display,
                   roster_factors, home_team_roster, away_team_roster,
                   use_high_scoring,
                   use_contextual=False, minuto_primer_gol=10,
                   marcador_actual_h_ctx=0, marcador_actual_a_ctx=0,
                   fase='Fase de Grupos'):
    """Ejecuta la predicción completa con todos los ajustes.

    Lógica de marcador en vivo:
    - Lambda representa goles ESPERADOS EN EL TIEMPO RESTANTE
    - La matriz de goles RESTANTES se desplaza para mostrar goles TOTALES
    - El resultado proximal = marcador_actual + moda(goles_restantes)
    """

    results = {'teams': (home_team, away_team)}
    errores = []

    # Preparar datos de entrenamiento
    CUTOFF = pd.to_datetime(match_date)
    train = raw[(raw.date <= CUTOFF) & raw.home_score.notna()].sort_values("date").reset_index(drop=True)
    train = train[train.date >= train_start].copy()

    # Obtener ELO para retorno
    stats_h = data_loader.get_espn_team_stats(home_team)
    stats_a = data_loader.get_espn_team_stats(away_team)
    elo_h = stats_h.get('elo', 1750)
    elo_a = stats_a.get('elo', 1750)

    # =====================================================================
    # BAYESIANO
    # =====================================================================
    if use_bayesian:
        try:
            with st.spinner("⚙️ Entrenando Bayesiano..."):
                teams = sorted(set(train.home_team) | set(train.away_team))
                team_idx = {t: i for i, t in enumerate(teams)}

                sm_bayes, lam_h_orig, lam_a_orig, att_ratings, def_ratings = train_bayesian_model(
                    train, teams, team_idx, home_team, away_team, max_goals_display,
                    use_hydration, use_dixon_coles, neutral_venue, use_high_scoring
                )

                if sm_bayes is not None:
                    # Guardar lambdas originales para referencia
                    lam_h_bayes = float(lam_h_orig)
                    lam_a_bayes = float(lam_a_orig)

                    # Aplicar ajuste por fase del torneo
                    lam_h_bayes, lam_a_bayes, nuevo_rho, factor_hs = corrections.ajuste_por_fase(
                        lam_h_bayes, lam_a_bayes, fase
                    )

                    # --- AJUSTE POR MOMENTUM (gol tardío del favorito) ---
                    if use_momentum and minuto_gol_favorito is not None:
                        # Lambda ajustada por tiempo restante y presión
                        factor_presion_h = _calcular_factor_presion(
                            marcador_actual_h, marcador_actual_a, es_local=True
                        )
                        factor_presion_a = _calcular_factor_presion(
                            marcador_actual_h, marcador_actual_a, es_local=False
                        )

                        lam_h_restante = _ajustar_lambda_tiempo(
                            lam_h_bayes, minuto_gol_favorito, factor_presion_h
                        )
                        lam_a_restante = _ajustar_lambda_tiempo(
                            lam_a_bayes, minuto_gol_favorito, factor_presion_a
                        )

                        # Recalcular matriz de goles RESTANTES
                        goals = np.arange(0, max_goals_display + 1)
                        sm_restantes = np.outer(poisson.pmf(goals, lam_h_restante), 
                                                 poisson.pmf(goals, lam_a_restante))

                        if use_dixon_coles:
                            sm_restantes = corrections.aplicar_dixon_coles(
                                sm_restantes, lam_h_restante, lam_a_restante, nuevo_rho
                            )
                        else:
                            suma = sm_restantes.sum()
                            if suma > 0:
                                sm_restantes = sm_restantes / suma

                        # Convertir a matriz de goles TOTALES
                        sm_bayes = _shift_matrix_to_total(
                            sm_restantes, marcador_actual_h, marcador_actual_a, max_goals_display
                        )

                        # Guardar lambdas de goles restantes para proximal
                        lam_h_bayes = lam_h_restante
                        lam_a_bayes = lam_a_restante

                    # --- AJUSTE CONTEXTUAL (partido roto) ---
                    elif use_contextual and (marcador_actual_h_ctx > 0 or marcador_actual_a_ctx > 0):
                        diff_ctx = abs(marcador_actual_h_ctx - marcador_actual_a_ctx)

                        if minuto_primer_gol <= 15 and diff_ctx >= 2:
                            # Partido roto: favorito ya gana por 2+ al min 15
                            factor_roto = 1.4
                            if marcador_actual_h_ctx > marcador_actual_a_ctx:
                                lam_h_bayes = lam_h_bayes * factor_roto
                            else:
                                lam_a_bayes = lam_a_bayes * factor_roto

                        # Calcular tiempo restante (asumimos minuto_primer_gol como referencia)
                        minuto_ref = minuto_primer_gol if minuto_primer_gol > 0 else 45
                        lam_h_restante = _ajustar_lambda_tiempo(lam_h_bayes, minuto_ref)
                        lam_a_restante = _ajustar_lambda_tiempo(lam_a_bayes, minuto_ref)

                        goals = np.arange(0, max_goals_display + 1)
                        sm_restantes = np.outer(poisson.pmf(goals, lam_h_restante), 
                                                 poisson.pmf(goals, lam_a_restante))

                        if use_dixon_coles:
                            sm_restantes = corrections.aplicar_dixon_coles(
                                sm_restantes, lam_h_restante, lam_a_restante, nuevo_rho
                            )
                        else:
                            suma = sm_restantes.sum()
                            if suma > 0:
                                sm_restantes = sm_restantes / suma

                        sm_bayes = _shift_matrix_to_total(
                            sm_restantes, marcador_actual_h_ctx, marcador_actual_a_ctx, max_goals_display
                        )

                        lam_h_bayes = lam_h_restante
                        lam_a_bayes = lam_a_restante

                    # --- AJUSTE DINÁMICO (gol temprano del underdog) ---
                    elif use_dynamic and underdog_scored_first:
                        factor_underdog = max(1.0, 1.3 - (minuto_gol / 100))

                        # El underdog anotó primero → partido más abierto
                        if elo_h > elo_a:  # Local era favorito, underdog es visitante
                            lam_h_bayes = lam_h_bayes * factor_underdog
                            lam_a_bayes = lam_a_bayes * (1 + 0.2 * factor_underdog)
                        else:  # Visitante era favorito
                            lam_a_bayes = lam_a_bayes * factor_underdog
                            lam_h_bayes = lam_h_bayes * (1 + 0.2 * factor_underdog)

                        # Recalcular matriz (sin desplazamiento, es predicción pre-partido)
                        goals = np.arange(0, max_goals_display + 1)
                        sm_bayes = np.outer(poisson.pmf(goals, lam_h_bayes), poisson.pmf(goals, lam_a_bayes))

                        if use_dixon_coles:
                            sm_bayes = corrections.aplicar_dixon_coles(sm_bayes, lam_h_bayes, lam_a_bayes, nuevo_rho)
                        else:
                            suma = sm_bayes.sum()
                            if suma > 0:
                                sm_bayes = sm_bayes / suma

                    # --- SIN AJUSTES EN VIVO: aplicar fase y listo ---
                    else:
                        goals = np.arange(0, max_goals_display + 1)
                        sm_bayes = np.outer(poisson.pmf(goals, lam_h_bayes), poisson.pmf(goals, lam_a_bayes))

                        if use_dixon_coles:
                            sm_bayes = corrections.aplicar_dixon_coles(sm_bayes, lam_h_bayes, lam_a_bayes, nuevo_rho)
                        else:
                            suma = sm_bayes.sum()
                            if suma > 0:
                                sm_bayes = sm_bayes / suma

                        sm_bayes = corrections.ajustar_matriz_por_fase(
                            sm_bayes, lam_h_bayes, lam_a_bayes, fase, max_goals_display
                        )

                    # Determinar marcador a usar para proximal
                    if use_momentum:
                        marc_h = marcador_actual_h
                        marc_a = marcador_actual_a
                    elif use_contextual:
                        marc_h = marcador_actual_h_ctx
                        marc_a = marcador_actual_a_ctx
                    else:
                        marc_h = 0
                        marc_a = 0

                    # Calcular resultados proximales
                    proximal = calcular_proximal(
                        sm_bayes, lam_h_bayes, lam_a_bayes, marc_h, marc_a,
                        use_dynamic, underdog_scored_first,
                        use_contextual, minuto_primer_gol
                    )

                    results['bayes'] = {
                        'score_matrix': sm_bayes,
                        'lam_h': lam_h_bayes,
                        'lam_a': lam_a_bayes,
                        'att_ratings': att_ratings,
                        'def_ratings': def_ratings,
                        'proximal': proximal
                    }
                else:
                    errores.append("Bayesiano falló")
        except Exception as e:
            errores.append(f"Bayesiano: {str(e)[:100]}")

    # =====================================================================
    # XGBOOST
    # =====================================================================
    if use_xgboost:
        try:
            with st.spinner("⚙️ Entrenando XGBoost..."):
                hist = raw[(raw.date <= CUTOFF) & raw.home_score.notna()].sort_values("date").reset_index(drop=True)
                hist = hist[hist.date >= train_start].copy()

                sm_xgb, lam_h_orig, lam_a_orig, team_stats = train_xgboost_model(
                    hist, raw, home_team, away_team, max_goals_display,
                    use_hydration, use_dixon_coles, neutral_venue, use_high_scoring
                )

                if sm_xgb is not None:
                    lam_h_xgb = float(lam_h_orig)
                    lam_a_xgb = float(lam_a_orig)

                    # Aplicar ajuste por fase del torneo
                    lam_h_xgb, lam_a_xgb, nuevo_rho, factor_hs = corrections.ajuste_por_fase(
                        lam_h_xgb, lam_a_xgb, fase
                    )

                    # --- AJUSTE POR MOMENTUM ---
                    if use_momentum and minuto_gol_favorito is not None:
                        factor_presion_h = _calcular_factor_presion(
                            marcador_actual_h, marcador_actual_a, es_local=True
                        )
                        factor_presion_a = _calcular_factor_presion(
                            marcador_actual_h, marcador_actual_a, es_local=False
                        )

                        lam_h_restante = _ajustar_lambda_tiempo(
                            lam_h_xgb, minuto_gol_favorito, factor_presion_h
                        )
                        lam_a_restante = _ajustar_lambda_tiempo(
                            lam_a_xgb, minuto_gol_favorito, factor_presion_a
                        )

                        goals = np.arange(0, max_goals_display + 1)
                        sm_restantes = np.outer(poisson.pmf(goals, lam_h_restante), 
                                                 poisson.pmf(goals, lam_a_restante))

                        if use_dixon_coles:
                            sm_restantes = corrections.aplicar_dixon_coles(
                                sm_restantes, lam_h_restante, lam_a_restante, nuevo_rho
                            )
                        else:
                            suma = sm_restantes.sum()
                            if suma > 0:
                                sm_restantes = sm_restantes / suma

                        sm_xgb = _shift_matrix_to_total(
                            sm_restantes, marcador_actual_h, marcador_actual_a, max_goals_display
                        )

                        lam_h_xgb = lam_h_restante
                        lam_a_xgb = lam_a_restante

                    # --- AJUSTE CONTEXTUAL ---
                    elif use_contextual and (marcador_actual_h_ctx > 0 or marcador_actual_a_ctx > 0):
                        diff_ctx = abs(marcador_actual_h_ctx - marcador_actual_a_ctx)

                        if minuto_primer_gol <= 15 and diff_ctx >= 2:
                            factor_roto = 1.4
                            if marcador_actual_h_ctx > marcador_actual_a_ctx:
                                lam_h_xgb = lam_h_xgb * factor_roto
                            else:
                                lam_a_xgb = lam_a_xgb * factor_roto

                        minuto_ref = minuto_primer_gol if minuto_primer_gol > 0 else 45
                        lam_h_restante = _ajustar_lambda_tiempo(lam_h_xgb, minuto_ref)
                        lam_a_restante = _ajustar_lambda_tiempo(lam_a_xgb, minuto_ref)

                        goals = np.arange(0, max_goals_display + 1)
                        sm_restantes = np.outer(poisson.pmf(goals, lam_h_restante), 
                                                 poisson.pmf(goals, lam_a_restante))

                        if use_dixon_coles:
                            sm_restantes = corrections.aplicar_dixon_coles(
                                sm_restantes, lam_h_restante, lam_a_restante, nuevo_rho
                            )
                        else:
                            suma = sm_restantes.sum()
                            if suma > 0:
                                sm_restantes = sm_restantes / suma

                        sm_xgb = _shift_matrix_to_total(
                            sm_restantes, marcador_actual_h_ctx, marcador_actual_a_ctx, max_goals_display
                        )

                        lam_h_xgb = lam_h_restante
                        lam_a_xgb = lam_a_restante

                    # --- AJUSTE DINÁMICO ---
                    elif use_dynamic and underdog_scored_first:
                        factor_underdog = max(1.0, 1.3 - (minuto_gol / 100))

                        if elo_h > elo_a:
                            lam_h_xgb = lam_h_xgb * factor_underdog
                            lam_a_xgb = lam_a_xgb * (1 + 0.2 * factor_underdog)
                        else:
                            lam_a_xgb = lam_a_xgb * factor_underdog
                            lam_h_xgb = lam_h_xgb * (1 + 0.2 * factor_underdog)

                        goals = np.arange(0, max_goals_display + 1)
                        sm_xgb = np.outer(poisson.pmf(goals, lam_h_xgb), poisson.pmf(goals, lam_a_xgb))

                        if use_dixon_coles:
                            sm_xgb = corrections.aplicar_dixon_coles(sm_xgb, lam_h_xgb, lam_a_xgb, nuevo_rho)
                        else:
                            suma = sm_xgb.sum()
                            if suma > 0:
                                sm_xgb = sm_xgb / suma

                    # --- SIN AJUSTES EN VIVO ---
                    else:
                        goals = np.arange(0, max_goals_display + 1)
                        sm_xgb = np.outer(poisson.pmf(goals, lam_h_xgb), poisson.pmf(goals, lam_a_xgb))

                        if use_dixon_coles:
                            sm_xgb = corrections.aplicar_dixon_coles(sm_xgb, lam_h_xgb, lam_a_xgb, nuevo_rho)
                        else:
                            suma = sm_xgb.sum()
                            if suma > 0:
                                sm_xgb = sm_xgb / suma

                        sm_xgb = corrections.ajustar_matriz_por_fase(
                            sm_xgb, lam_h_xgb, lam_a_xgb, fase, max_goals_display
                        )

                    # Determinar marcador para proximal
                    if use_momentum:
                        marc_h = marcador_actual_h
                        marc_a = marcador_actual_a
                    elif use_contextual:
                        marc_h = marcador_actual_h_ctx
                        marc_a = marcador_actual_a_ctx
                    else:
                        marc_h = 0
                        marc_a = 0

                    proximal = calcular_proximal(
                        sm_xgb, lam_h_xgb, lam_a_xgb, marc_h, marc_a,
                        use_dynamic, underdog_scored_first,
                        use_contextual, minuto_primer_gol
                    )

                    results['xgb'] = {
                        'score_matrix': sm_xgb,
                        'lam_h': lam_h_xgb,
                        'lam_a': lam_a_xgb,
                        'team_stats': team_stats,
                        'proximal': proximal
                    }
                else:
                    errores.append("XGBoost falló")
        except Exception as e:
            errores.append(f"XGBoost: {str(e)[:100]}")

    return results, errores, elo_h, elo_a
