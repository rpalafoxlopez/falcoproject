# modules/ui.py - Componentes de interfaz de usuario
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import sys
import time

from . import config
from . import data_loader

def render_header():
    """Renderiza el header de la aplicación"""
    st.markdown("""
    <div class="title-bar">
        <h1>⚽ PREDICCIÓN DE MARCADORES</h1>
        <div>Mundial FIFA 2026 — Modelo Bayesiano Jerárquico</div>
        <div class="badge-row">
            <span class="badge">Dixon-Coles</span>
            <span class="badge">Bayesiano</span>
            <span class="badge gold">Mundial 2026</span>
            <span class="badge">4 Tiempos</span>
            <span class="badge">Alta Anotación</span>
            <span class="badge">Contextual</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_status_bar():
    """Renderiza la barra de estado (sin tecnologías)"""
    st.markdown('<div class="status-bar">', unsafe_allow_html=True)
    status_cols = st.columns(3)
    with status_cols[0]:
        st.markdown(f"""
        <div class="status-item">
            <div class="label">Dixon-Coles ρ</div>
            <div class="value gold">{config.DIXON_COLES_RHO:.3f}</div>
        </div>
        """, unsafe_allow_html=True)
    with status_cols[1]:
        st.markdown(f"""
        <div class="status-item">
            <div class="label">Modelo Principal</div>
            <div class="value cyan">Bayesiano</div>
        </div>
        """, unsafe_allow_html=True)
    with status_cols[2]:
        st.markdown(f"""
        <div class="status-item">
            <div class="label">Ajustes Activos</div>
            <div class="value cyan">4 Tiempos + DC + Contextual</div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown("---")

def render_disclaimer():
    """Muestra el disclaimer de uso responsable"""
    st.markdown("""
    <div style="
        background: rgba(239, 68, 68, 0.08);
        border: 1px solid rgba(239, 68, 68, 0.2);
        border-radius: 12px;
        padding: 16px 20px;
        margin: 12px 0 24px 0;
    ">
        <p style="
            color: #94a3b8;
            font-size: 0.85rem;
            margin: 0;
            line-height: 1.5;
        ">
            <strong style="color: #ef4444;">⚠️ Aviso importante:</strong> 
            Esta herramienta es un modelo estadístico con fines <strong>educativos y de entretenimiento</strong>. 
            No constituye asesoramiento financiero, deportivo ni de apuestas. 
            Las predicciones son estimaciones basadas en datos históricos y no garantizan resultados reales.
            <br><br>
            <span style="color: #64748b; font-size: 0.7rem;">
                🔞 Prohibido su uso para menores de edad o para actividades de apuestas ilegales.
            </span>
        </p>
    </div>
    """, unsafe_allow_html=True)

def render_match_selector(wc_teams):
    """Renderiza los selectores de equipos y fecha"""
    home_team = st.selectbox(
        "🏠 Equipo Local",
        options=wc_teams,
        index=wc_teams.index("Mexico") if "Mexico" in wc_teams else 0
    )

    away_team = st.selectbox(
        "✈️ Equipo Visitante",
        options=wc_teams,
        index=wc_teams.index("Cabo Verde") if "Cabo Verde" in wc_teams else min(1, len(wc_teams)-1)
    )

    if home_team == away_team:
        st.warning("⚠️ Los equipos deben ser diferentes")
        if len(wc_teams) > 1:
            away_idx = 1 if wc_teams[1] != home_team else 0
            away_team = wc_teams[away_idx]

    match_date = st.date_input("📅 Fecha del Partido", data_loader.get_current_date_mexico())
    train_start = st.selectbox("📊 Ventana de entrenamiento", config.TRAIN_WINDOWS, index=0)
    
    return home_team, away_team, match_date, train_start

def render_model_selectors():
    """Renderiza los selectores de modelos"""
    use_bayesian = st.checkbox("🔵 Bayesiano (recomendado)", value=True)
    use_xgboost = st.checkbox("🟢 XGBoost (comparativo)", value=False)
    return use_xgboost, use_bayesian

def render_corrections():
    """Renderiza las correcciones"""
    use_dixon_coles = st.checkbox("🔧 Dixon-Coles", value=True, help="Corrige la subestimación de empates")
    use_hydration = st.checkbox("💧 Pausas de hidratación (4 tiempos)", value=True)
    return use_dixon_coles, use_hydration

def render_dynamic_adjustments():
    """Renderiza los ajustes dinámicos"""
    use_dynamic = st.checkbox("⚡ Ajuste por gol temprano del underdog", value=False)
    
    underdog_scored_first = False
    minuto_gol = 15
    
    if use_dynamic:
        underdog_scored_first = st.checkbox("🏃 El underdog anotó primero", value=False)
        minuto_gol = st.slider("⏱️ Minuto del primer gol", 1, 90, 15, help="Minuto en que el underdog anotó")
        st.caption("💡 Si el underdog anota primero, el partido se vuelve más abierto.")
    
    return use_dynamic, underdog_scored_first, minuto_gol

def render_momentum_adjustments():
    """Renderiza los ajustes por momentum"""
    use_momentum = st.checkbox("⚡ Ajuste por momentum (gol tardío del favorito)", value=False)
    
    minuto_gol_favorito = None
    llegadas_previas_h = None
    llegadas_previas_a = None
    marcador_actual_h = 0
    marcador_actual_a = 0
    
    if use_momentum:
        minuto_gol_favorito = st.slider("⏱️ Minuto del gol del favorito", 1, 90, 85)
        llegadas_previas_h = st.number_input("Llegadas del local en el cuarto anterior", 0, 20, 5)
        llegadas_previas_a = st.number_input("Llegadas del visitante en el cuarto anterior", 0, 20, 3)
        st.markdown("**Marcador actual:**")
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            marcador_actual_h = st.number_input("Goles local", 0, 10, 0, key="marc_h")
        with col_m2:
            marcador_actual_a = st.number_input("Goles visitante", 0, 10, 0, key="marc_a")
    
    return use_momentum, minuto_gol_favorito, llegadas_previas_h, llegadas_previas_a, marcador_actual_h, marcador_actual_a

def render_high_scoring_adjustment():
    """Renderiza el ajuste de alta anotación"""
    return st.checkbox("⚽ Ajuste por alta anotación", value=True, 
                       help="Aumenta probabilidad de partidos con +3 goles (ej: Noruega 3-2)")

# ============================================================================
# NUEVO: AJUSTES CONTEXTUALES
# ============================================================================

def render_contextual_adjustments():
    """Renderiza los ajustes contextuales avanzados (partidos rotos)"""
    st.markdown("---")
    st.subheader("⚡ Ajustes Contextuales")
    
    use_contextual = st.checkbox("✅ Activar ajustes contextuales", value=True,
                                 help="Ajusta predicciones para partidos que se 'rompen'")
    
    minuto_primer_gol = 10
    marcador_actual_h_ctx = 0
    marcador_actual_a_ctx = 0
    
    if use_contextual:
        st.caption("⚽ Si el favorito anota temprano (min 1-15), el partido se rompe")
        minuto_primer_gol = st.slider("⏱️ Minuto del primer gol del favorito", 1, 90, 10,
                                      help="Minuto en que el favorito anotó el primer gol")
        
        st.markdown("**Marcador actual:**")
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            marcador_actual_h_ctx = st.number_input("Goles local", 0, 10, 0, key="ctx_h")
        with col_m2:
            marcador_actual_a_ctx = st.number_input("Goles visitante", 0, 10, 0, key="ctx_a")
        
        st.caption("💡 Si hay 3+ goles de diferencia en el 1T, el partido se considera 'roto'")
    
    return use_contextual, minuto_primer_gol, marcador_actual_h_ctx, marcador_actual_a_ctx

def plot_results(sm, home_team, away_team, title, max_display=7):
    """Genera los gráficos de resultados"""
    if sm.shape[0] < max_display + 1 or sm.shape[1] < max_display + 1:
        sm_full = np.zeros((max_display + 1, max_display + 1))
        sm_full[:sm.shape[0], :sm.shape[1]] = sm
        sm_disp = sm_full
    else:
        sm_disp = sm[:max_display + 1, :max_display + 1]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    fig.patch.set_facecolor('#0a0e1a')
    fig.suptitle(f"{home_team} vs {away_team} — {title}", fontsize=16, fontweight="bold", color='white', y=0.98)

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

def render_results(results, elo, dixon_coles_rho):
    """Renderiza los resultados de la predicción con resultados proximales"""
    if 'teams' not in results:
        st.error("❌ No hay resultados para mostrar.")
        return
    
    home_team, away_team = results['teams']
    elo_h = elo.get('home', 0)
    elo_a = elo.get('away', 0)

    st.markdown("---")
    
    # ============================================================
    # RESULTADOS PROXIMALES (DESTACADOS)
    # ============================================================
    st.subheader("🎯 RESULTADO PROXIMAL")
    st.caption("Predicción ajustada según el contexto del partido")

    # Mostrar resultados proximales en una fila
    prox_cols = st.columns(3)
    
    col_idx = 0
    for model_name, model_data in results.items():
        if model_name in ['teams', 'elo']:
            continue
        if 'proximal' not in model_data:
            continue
        
        prox = model_data['proximal']
        display_name = "🔵 Bayesiano" if model_name == 'bayes' else "🟢 XGBoost"
        
        # ✅ Verificar si es empate
        es_empate = prox.get('es_empate', False)
        empate_tag = " 📊 Empate más probable" if es_empate else ""
        border_color = 'rgba(255, 193, 7, 0.3)' if es_empate else 'rgba(0, 212, 255, 0.3)'
        text_color = '#ffc107' if es_empate else '#00d4ff'
        
        with prox_cols[col_idx % 3]:
            st.markdown(f"""
            <div style="
                background: linear-gradient(135deg, #1a1f2e, #0f172a);
                border: 2px solid {border_color};
                border-radius: 16px;
                padding: 20px;
                text-align: center;
            ">
                <div style="color: #94a3b8; font-size: 0.75rem; margin: 0;">{display_name}</div>
                <div style="
                    font-family: 'Bebas Neue', sans-serif;
                    font-size: 3rem;
                    color: {text_color};
                    margin: 8px 0;
                ">{prox['proximal'][0]} - {prox['proximal'][1]}</div>
                <div style="color: #64748b; font-size: 0.7rem; margin: 0;">
                    📊 Base: {prox['base'][0]}-{prox['base'][1]}
                    {empate_tag}
                </div>
            </div>
            """, unsafe_allow_html=True)

        col_idx += 1
    
    # Mostrar resultados específicos por contexto
    st.markdown("---")
    st.subheader("📊 Resultados por Contexto")
    
    context_cols = st.columns(2)
    
    # Columna 1: Gol del underdog
    with context_cols[0]:
        st.markdown("#### ⚡ Por Gol del Underdog")
        for model_name, model_data in results.items():
            if model_name in ['teams', 'elo']:
                continue
            if 'proximal' not in model_data:
                continue
            prox = model_data['proximal']
            display_name = "Bayesiano" if model_name == 'bayes' else "XGBoost"
            if prox.get('es_empate', False):
                st.write(f"**{display_name}:** Empate ya es lo más probable")
            elif prox['underdog_first'] is not None and prox['underdog_first'] != prox['base']:
                st.write(f"**{display_name}:** {prox['underdog_first'][0]}-{prox['underdog_first'][1]}")
            else:
                st.write(f"**{display_name}:** Sin cambio")
    
    # Columna 2: Partido roto
    with context_cols[1]:
        st.markdown("#### 💥 Por Partido Roto")
        for model_name, model_data in results.items():
            if model_name in ['teams', 'elo']:
                continue
            if 'proximal' not in model_data:
                continue
            prox = model_data['proximal']
            display_name = "Bayesiano" if model_name == 'bayes' else "XGBoost"
            if prox.get('es_empate', False):
                st.write(f"**{display_name}:** Empate ya es lo más probable")
            elif prox['partido_roto'] is not None and prox['partido_roto'] != prox['base']:
                st.write(f"**{display_name}:** {prox['partido_roto'][0]}-{prox['partido_roto'][1]}")
            else:
                st.write(f"**{display_name}:** Sin cambio")
    
    # ============================================================
    # RESTO DE RESULTADOS
    # ============================================================
    st.markdown("---")
    st.subheader("📊 Resumen de Predicción")

    valid_models = []
    for key, value in results.items():
        if key not in ['teams', 'elo'] and value is not None:
            if isinstance(value, dict) and 'score_matrix' in value:
                valid_models.append(key)
    
    if not valid_models:
        st.warning("⚠️ No hay modelos con predicciones válidas.")
        return

    cols = st.columns(min(len(valid_models), 4))
    
    col_idx = 0
    for model_name in valid_models:
        model_data = results[model_name]
        with cols[col_idx % len(cols)]:
            display_name = "🔵 Bayesiano" if model_name == 'bayes' else "🟢 XGBoost"
            st.metric(
                f"🏠 {home_team[:10]} ({display_name})",
                f"{model_data['lam_h']:.2f}",
                delta=f"vs {away_team[:10]} {model_data['lam_a']:.2f}",
                delta_color="off"
            )
        col_idx += 1

    st.markdown("---")

    model_cols = st.columns(len(valid_models))
    col_idx = 0

    for model_name in valid_models:
        model_data = results[model_name]

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

    if len(valid_models) > 1:
        st.markdown("---")
        st.subheader("📋 Comparativa de Modelos")

        comp_data = []
        for model_name in valid_models:
            model_data = results[model_name]
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

def render_footer():
    """Renderiza el footer"""
    st.markdown("---")
    st.markdown("""
    <div class="footer">
        <div>
            ⚽ Modelo Bayesiano Jerárquico — Mundial FIFA 2026
            <span class="separator">·</span>
            🔧 Dixon-Coles (ρ=-0.13)
            <span class="separator">·</span>
            💧 Ajuste por 4 tiempos
            <span class="separator">·</span>
            ⚽ Alta anotación
            <span class="separator">·</span>
            ⚡ Contextual
        </div>
        <div style="margin-top: 8px; color: #64748b; font-size: 0.7rem;">
            <a href="https://satohachi.rpalafox.com/" target="_blank" style="color: #00d4ff;">🐝 rpalafox.com</a>
            <span class="separator">·</span>
            <span style="color: #64748b;">Uso exclusivamente educativo y de entretenimiento</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_system_info(num_teams, raw):
    """Renderiza la información del sistema"""
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**Modelo principal:** Bayesiano Jerárquico")
        st.write(f"**Dixon-Coles ρ:** `{config.DIXON_COLES_RHO:.3f}`")
        st.write(f"**Equipos clasificados:** `{num_teams}`")
    with col2:
        st.write(f"**Ajustes activos:** Pausas hidratación, alta anotación, contextual")
        st.write(f"**Ventana de datos:** 2018-2026")
        st.write(f"**Partidos históricos:** `{len(raw):,}`")