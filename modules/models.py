# modules/models.py - Modelos de predicción
import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import poisson

from . import config
from . import data_loader
from . import corrections
from . import predictors

def train_xgboost_model(hist, raw_data, home_team, away_team, max_goals=8,
                        use_hydration=True, use_dixon_coles=True, neutral_venue=False,
                        roster_factors=None):
    """Entrena el modelo XGBoost y retorna predicciones"""
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
        hist["elo_home"], hist["elo_away"] = elo_h_list, elo_a_list
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
            lam_h, lam_a = corrections.ajustar_por_pausas_hidratacion(lam_h, lam_a, elo_h, elo_a)

        # 🔥 Aplicar factores de alineación si están disponibles
        if roster_factors:
            lam_h, lam_a = predictors.adjust_by_roster_factors(
                lam_h, lam_a, home_team, away_team, roster_factors
            )

        goals = np.arange(0, max_goals + 1)
        score_matrix = np.outer(poisson.pmf(goals, lam_h), poisson.pmf(goals, lam_a))

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

def train_bayesian_model(train, teams, team_idx, home_team, away_team, max_goals=8,
                         use_hydration=True, use_dixon_coles=True, neutral_venue=False,
                         roster_factors=None):
    """Entrena el modelo Bayesiano"""
    if not config.PYMC_AVAILABLE:
        return None, None, None, None, None
    
    try:
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

        if use_hydration:
            lam_h, lam_a = corrections.ajustar_por_pausas_hidratacion(lam_h, lam_a)

        # 🔥 Aplicar factores de alineación si están disponibles
        if roster_factors:
            lam_h, lam_a = predictors.adjust_by_roster_factors(
                lam_h, lam_a, home_team, away_team, roster_factors
            )

        goals = np.arange(0, max_goals + 1)
        pmf_h = poisson.pmf(goals, lam_h)
        pmf_a = poisson.pmf(goals, lam_a)
        score_matrix = np.outer(pmf_h, pmf_a)

        if use_dixon_coles:
            score_matrix = corrections.aplicar_dixon_coles(score_matrix, lam_h, lam_a)
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

def run_prediction(raw, home_team, away_team, match_date, train_start, neutral_venue,
                   use_xgboost, use_bayesian, use_dixon_coles, use_hydration,
                   use_dynamic, underdog_scored_first, minuto_gol,
                   use_momentum, minuto_gol_favorito, llegadas_previas_h, llegadas_previas_a,
                   marcador_actual_h, marcador_actual_a, max_goals_display,
                   roster_factors, home_team_roster, away_team_roster):
    """Ejecuta la predicción completa"""
    
    # Preparar datos
    CUTOFF = pd.Timestamp(match_date) - pd.Timedelta(days=1)
    mask = (raw["date"] >= train_start) & (raw["date"] <= CUTOFF) & raw["home_score"].notna()
    train = raw.loc[mask].copy()
    train["home_score"] = train["home_score"].astype(int)
    train["away_score"] = train["away_score"].astype(int)
    train["neutral"] = train["neutral"].astype(str).str.upper().eq("TRUE")

    teams = sorted(pd.concat([train.home_team, train.away_team]).unique())
    team_idx = {t: i for i, t in enumerate(teams)}

    if home_team not in teams or away_team not in teams:
        return None, [f"{home_team if home_team not in teams else away_team} no tiene suficientes partidos"], 0, 0

    # Obtener Elos
    stats_h = data_loader.get_espn_team_stats(home_team)
    stats_a = data_loader.get_espn_team_stats(away_team)
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

    # Preparar factores de alineación
    roster_factors_dict = None
    if roster_factors:
        roster_factors_dict = {
            'home': home_team_roster,
            'away': away_team_roster
        }

    # XGBoost
    if use_xgboost:
        try:
            with st.spinner("⚙️ Entrenando XGBoost..."):
                hist = raw[(raw.date <= CUTOFF) & raw.home_score.notna()].sort_values("date").reset_index(drop=True)
                hist = hist[hist.date >= train_start].copy()
                sm_xgb, lam_h_xgb, lam_a_xgb, team_stats = train_xgboost_model(
                    hist, raw, home_team, away_team, max_goals_display,
                    use_hydration, use_dixon_coles, neutral_venue, roster_factors_dict
                )
                if sm_xgb is not None:
                    if use_momentum:
                        es_favorito_local = elo_h > elo_a
                        lam_h_xgb, lam_a_xgb = corrections.ajustar_por_momentum(
                            lam_h_xgb, lam_a_xgb,
                            home_team, away_team,
                            minuto_gol=minuto_gol_favorito,
                            es_favorito_local=es_favorito_local,
                            llegadas_previas_h=llegadas_previas_h,
                            llegadas_previas_a=llegadas_previas_a,
                            marcador_actual={'home': marcador_actual_h, 'away': marcador_actual_a}
                        )
                        goals = np.arange(0, max_goals_display + 1)
                        sm_xgb = np.outer(poisson.pmf(goals, lam_h_xgb), poisson.pmf(goals, lam_a_xgb))
                        if use_dixon_coles:
                            sm_xgb = corrections.aplicar_dixon_coles(sm_xgb, lam_h_xgb, lam_a_xgb)
                        else:
                            suma = sm_xgb.sum()
                            if suma > 0:
                                sm_xgb = sm_xgb / suma

                    if use_dynamic:
                        sm_xgb = corrections.ajustar_por_gol_temprano(
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

    # Bayesiano
    if use_bayesian and config.PYMC_AVAILABLE:
        try:
            with st.spinner("⚙️ Entrenando Bayesiano (1-2 min)..."):
                sm_bayes, lam_h_bayes, lam_a_bayes, att_ratings, def_ratings = train_bayesian_model(
                    train, teams, team_idx, home_team, away_team, max_goals_display,
                    use_hydration, use_dixon_coles, neutral_venue, roster_factors_dict
                )
                if sm_bayes is not None:
                    if use_momentum:
                        es_favorito_local = elo_h > elo_a
                        lam_h_bayes, lam_a_bayes = corrections.ajustar_por_momentum(
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
                            sm_bayes = corrections.aplicar_dixon_coles(sm_bayes, lam_h_bayes, lam_a_bayes)
                        else:
                            suma = sm_bayes.sum()
                            if suma > 0:
                                sm_bayes = sm_bayes / suma

                    if use_dynamic:
                        sm_bayes = corrections.ajustar_por_gol_temprano(
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

    return results, errores, elo_h, elo_a