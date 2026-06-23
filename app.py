# app.py - Predicción Mundial 2026 - Versión Modular
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

st.set_page_config(
    page_title="Predicción Mundial 2026",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CSS PERSONALIZADO (aquí va tu CSS)
# ============================================================================
# ... (tu CSS aquí) ...

# ============================================================================
# INICIALIZACIÓN
# ============================================================================

def main():
    raw = data_loader.load_match_data()
    if raw is None:
        st.error("❌ No se pudieron cargar los datos.")
        st.stop()
    
    fixture_data = data_loader.get_espn_fixture()
    wc_teams = data_loader.filter_world_cup_teams(raw)
    
    ui.render_header()
    ui.render_status_bar()
    ui.render_disclaimer()
    
    with st.sidebar:
        st.sidebar.header("⚙️ Configuración del Partido")
        
        home_team, away_team, match_date, train_start = ui.render_match_selector(wc_teams)
        
        st.markdown("---")
        st.subheader("🏟️ Configuración del Partido")
        neutral_venue = st.checkbox("🏟️ Partido en sede neutral", value=False)
        
        st.markdown("---")
        st.subheader("🤖 Modelos a usar")
        use_bayesian = st.checkbox("🔵 Bayesiano (recomendado)", value=True)
        use_xgboost = st.checkbox("🟢 XGBoost (comparativo)", value=False)
        
        st.markdown("---")
        st.subheader("🔧 Correcciones")
        use_dixon_coles, use_hydration = ui.render_corrections()
        
        st.markdown("---")
        st.subheader("⚡ Ajustes Dinámicos")
        use_dynamic, underdog_scored_first, minuto_gol = ui.render_dynamic_adjustments()
        use_momentum, minuto_gol_favorito, llegadas_previas_h, llegadas_previas_a, marcador_actual_h, marcador_actual_a = ui.render_momentum_adjustments()
        
        st.markdown("---")
        use_high_scoring = ui.render_high_scoring_adjustment()
        
        # 🔥 NUEVO: Ajustes Contextuales
        use_contextual, minuto_primer_gol, marcador_actual_h_ctx, marcador_actual_a_ctx = ui.render_contextual_adjustments()
        
        max_goals_display = st.slider("📊 Máximo de goles a mostrar", 4, 10, 7)
        
        if fixture_data:
            with st.expander("📅 Fixture del Mundial 2026"):
                fixture_df = pd.DataFrame(fixture_data)
                fixture_df['date'] = fixture_df['date'].astype(str)
                st.dataframe(fixture_df, hide_index=True, use_container_width=True)
        
        st.markdown("---")
        predict_btn = st.button("🔮 Predecir", type="primary", use_container_width=True)
        
        st.markdown("---")
        st.subheader("🔬 Validación del Modelo")
        if st.button("📊 Validar modelo", use_container_width=True):
            st.session_state.show_validation = True
    
    if st.session_state.get('show_validation', False):
        validation.render_validation_modal(raw, corrections)
    
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
            use_high_scoring=use_high_scoring,
            use_contextual=use_contextual,
            minuto_primer_gol=minuto_primer_gol,
            marcador_actual_h_ctx=marcador_actual_h_ctx,
            marcador_actual_a_ctx=marcador_actual_a_ctx
        )
        
        if results:
            st.session_state.results = results
            st.session_state.elo = {'home': elo_h, 'away': elo_a}
            st.success("✅ Predicción completada!")
            if errores:
                st.warning(f"⚠️ Algunos fallaron: {', '.join(errores)}")
        else:
            st.error(f"❌ No se pudo completar. Errores: {', '.join(errores)}")
    
    if 'results' in st.session_state:
        ui.render_results(
            st.session_state.results, 
            st.session_state.get('elo', {'home': 0, 'away': 0}),
            config.DIXON_COLES_RHO
        )
    
    ui.render_footer()
    
    with st.expander("ℹ️ Información del modelo"):
        ui.render_system_info(len(wc_teams), raw)

if __name__ == "__main__":
    main()