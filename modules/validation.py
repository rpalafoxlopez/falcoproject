# modules/validation.py - Validación del modelo
import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson

def render_validation_modal(raw, corrections):
    """Renderiza el modal de validación"""
    # CSS personalizado para el modal
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
    .modal-content::-webkit-scrollbar {
        width: 8px;
    }
    .modal-content::-webkit-scrollbar-track {
        background: rgba(255,255,255,0.05);
        border-radius: 4px;
    }
    .modal-content::-webkit-scrollbar-thumb {
        background: rgba(0, 212, 255, 0.3);
        border-radius: 4px;
    }
    .modal-content::-webkit-scrollbar-thumb:hover {
        background: rgba(0, 212, 255, 0.5);
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

    # Botón para cerrar
    if st.button("✕ Cerrar", key="close_modal_btn_validation"):
        st.session_state.show_validation = False
        st.rerun()

    st.markdown("---")

    with st.spinner("🔄 Ejecutando validación (puede tomar 1-2 minutos)..."):
        try:
            # Usar los últimos 12 meses de datos disponibles
            max_date = raw["date"].max()
            train_end = (max_date - pd.Timedelta(days=365)).strftime("%Y-%m-%d")

            train_data = raw[(raw["date"] <= train_end) & raw.home_score.notna()].copy()
            test_data = raw[(raw["date"] > train_end) & raw.home_score.notna()].copy()

            if len(test_data) < 10:
                st.warning(f"⚠️ Solo {len(test_data)} partidos disponibles para validación.")
                if st.button("Cerrar", key="close_no_data_btn_validation"):
                    st.session_state.show_validation = False
                    st.rerun()
                st.markdown("</div></div>", unsafe_allow_html=True)
                st.stop()

            # Entrenar modelo base (simplificado)
            def entrenar_modelo_validacion(df):
                import xgboost as xgb
                K_ELO = 20.0
                ELO_INIT = 1500.0

                ratings = {}
                elo_h_list, elo_a_list = [], []
                for _, row in df.iterrows():
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

                df = df.copy()
                df["elo_home"], df["elo_away"] = elo_h_list, elo_a_list

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

            xgb_model, FEATURES, final_elo, final_form = entrenar_modelo_validacion(
                raw[(raw.date <= train_end) & raw.home_score.notna()].sort_values("date").reset_index(drop=True)
            )

            # Evaluar sin pausas (simplificado)
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

            # Evaluar con pausas (simplificado)
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

                lam_h, lam_a = corrections.ajustar_por_pausas_hidratacion(lam_h, lam_a, elo_h, elo_a)

                goals = np.arange(0, 8 + 1)
                sm = np.outer(poisson.pmf(goals, lam_h), poisson.pmf(goals, lam_a))
                sm = sm / sm.sum()

                pred = np.argmax([np.sum(sm[:3, :3]), np.sum(np.diag(sm[:3, :3])), np.sum(sm[:3, 1:])])
                real = 0 if row.home_score > row.away_score else 1 if row.home_score == row.away_score else 2
                if pred == real:
                    aciertos_b += 1
            acc_b = aciertos_b / len(test_data)

            st.success("✅ Validación completada!")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📊 Partidos", len(test_data))
            with col2:
                st.metric("🔵 Base", f"{acc_a*100:.1f}%")
            with col3:
                st.metric("🟢 Con pausas", f"{acc_b*100:.1f}%", 
                         delta=f"{(acc_b - acc_a)*100:+.1f} pp")

            st.markdown("---")

            comp_df = pd.DataFrame([
                {"Modelo": "XGBoost (base)", "Accuracy": f"{acc_a*100:.1f}%"},
                {"Modelo": "XGBoost + pausas", "Accuracy": f"{acc_b*100:.1f}%"}
            ])
            st.dataframe(comp_df, hide_index=True, use_container_width=True)

            if st.button("✅ Cerrar validación", use_container_width=True, key="close_validation_btn_final"):
                st.session_state.show_validation = False
                st.rerun()

        except Exception as e:
            st.error(f"❌ Error en la validación: {str(e)}")
            st.info("💡 La validación requiere datos históricos. Asegúrate de que el dataset esté disponible.")
            if st.button("Cerrar", key="close_error_btn_validation"):
                st.session_state.show_validation = False
                st.rerun()

    # Cerrar los divs del modal
    st.markdown("</div></div>", unsafe_allow_html=True)