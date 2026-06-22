# app.py - Predicción Mundial 2026 - Versión Modular
import streamlit as st
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
    
    # 4. Sidebar
    with st.sidebar:
        st.sidebar.header("⚙️ Configuración del Partido")
        
        # Selectores de equipos y fecha
        home_team, away_team, match_date, train_start = ui.render_match_selector(wc_teams)
        
        st.markdown("---")
        st.subheader("🏟️ Configuración del Partido")
        
        neutral_venue = st.checkbox("🏟️ Partido en sede neutral", value=False, 
                                   help="Anula la ventaja de localía")
        
        st.markdown("---")
        st.subheader("📋 Factores del Partido")
        
        # 🔥 NUEVO: Factores dinámicos de alineación
        use_roster_factors = st.checkbox("👥 Incluir factores de alineación", value=False, 
                                         help="Ajusta predicción según alineación y estado de jugadores")
        
        if use_roster_factors:
            # Disclaimer de precisión
            ui.render_disclaimer()
            
            # Mostrar alineaciones estimadas
            with st.expander("👥 Alineación estimada"):
                team_roster.display_roster_preview(home_team, away_team)
            
            # Factores de jugadores
            st.subheader("🎯 Factores de Jugadores")
            top_scorers = predictors.get_top_scorers_summary(home_team, away_team)
            st.caption(f"⚽ Máximos goleadores: {top_scorers}")
        
        st.markdown("---")
        
        # 5. Modelos y correcciones
        use_xgboost, use_bayesian = ui.render_model_selectors()
        use_dixon_coles, use_hydration = ui.render_corrections()
        use_dynamic, underdog_scored_first, minuto_gol = ui.render_dynamic_adjustments()
        use_momentum, minuto_gol_favorito, llegadas_previas_h, llegadas_previas_a, marcador_actual_h, marcador_actual_a = ui.render_momentum_adjustments()
        
        max_goals_display = st.slider("📊 Máximo de goles a mostrar", 4, 10, 7)
        
        # 6. Fixture
        if fixture_data:
            with st.expander("📅 Fixture del Mundial 2026"):
                fixture_df = pd.DataFrame(fixture_data)
                fixture_df['date'] = fixture_df['date'].astype(str)
                st.dataframe(fixture_df, hide_index=True, use_container_width=True)
        
        # 7. Botón de predicción
        predict_btn = st.button("🔮 Predecir", type="primary", use_container_width=True)
        
        # 8. Validación
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
            roster_factors=use_roster_factors,
            home_team_roster=team_roster.get_team_roster(home_team) if use_roster_factors else None,
            away_team_roster=team_roster.get_team_roster(away_team) if use_roster_factors else None
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
    with st.expander("ℹ️ Información del sistema"):
        ui.render_system_info(len(wc_teams))


if __name__ == "__main__":
    main()