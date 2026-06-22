# modules/ui.py - Componentes de interfaz de usuario
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import sys

from . import config
from . import data_loader

def render_header():
    """Renderiza el header de la aplicación"""
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

def render_status_bar():
    """Renderiza la barra de estado"""
    st.markdown('<div class="status-bar">', unsafe_allow_html=True)
    status_cols = st.columns(4)
    with status_cols[0]:
        st.markdown(f"""
        <div class="status-item">
            <div class="label">Python</div>
            <div class="value cyan">{config.PYTHON_VERSION}</div>
        </div>
        """, unsafe_allow_html=True)
    with status_cols[1]:
        st.markdown(f"""
        <div class="status-item">
            <div class="label">PyMC</div>
            <div class="value {'green' if config.PYMC_AVAILABLE else 'red'}">{'✅ Disponible' if config.PYMC_AVAILABLE else '❌ No disponible'}</div>
        </div>
        """, unsafe_allow_html=True)
    with status_cols[2]:
        st.markdown(f"""
        <div class="status-item">
            <div class="label">SKLearn</div>
            <div class="value {'green' if config.SKLEARN_AVAILABLE else 'red'}">{'✅ Disponible' if config.SKLEARN_AVAILABLE else '❌ No disponible'}</div>
        </div>
        """, unsafe_allow_html=True)
    with status_cols[3]:
        st.markdown(f"""
        <div class="status-item">
            <div class="label">Dixon-Coles ρ</div>
            <div class="value gold">{config.DIXON_COLES_RHO:.3f}</div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown("---")
    
    if not config.PYMC_AVAILABLE:
        st.info("ℹ️ El modelo Bayesiano no está disponible. Solo se usará XGBoost.")

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
    use_xgboost = st.checkbox("✅ XGBoost", value=True)
    use_bayesian = st.checkbox("✅ Bayesiano" if config.PYMC_AVAILABLE else "❌ Bayesiano (no disponible)",
                               value=config.PYMC_AVAILABLE, disabled=not config.PYMC_AVAILABLE)
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

def render_disclaimer():
    """Muestra el disclaimer sobre precisión antes del partido"""
    st.markdown("""
    <div style="
        background: rgba(0, 212, 255, 0.08);
        border: 1px solid rgba(0, 212, 255, 0.15);
        border-radius: 12px;
        padding: 16px 20px;
        margin: 12px 0;
    ">
        <p style="
            color: #94a3b8;
            font-size: 0.85rem;
            margin: 0;
            display: flex;
            align-items: center;
            gap: 8px;
        ">
            <span style="font-size: 1.2rem;">⚡</span>
            <span>
                <strong style="color: #00d4ff;">Precisión mejorada:</strong> 
                Revisa la alineación 
                <strong style="color: #ffffff;">1 hora antes del partido</strong> 
                para obtener la predicción más precisa. Los factores de alineación 
                pueden modificar significativamente el resultado esperado.
            </span>
        </p>
    </div>
    """, unsafe_allow_html=True)

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

def render_results(results, elo, dixon_coles_rho):
    """Renderiza los resultados de la predicción"""
    # ✅ Verificar que 'teams' existe
    if 'teams' not in results:
        st.error("❌ No hay resultados para mostrar. Los modelos no pudieron generar predicciones.")
        return
    
    home_team, away_team = results['teams']
    elo_h = elo.get('home', 0)
    elo_a = elo.get('away', 0)

    st.markdown("---")
    st.subheader("📊 Resumen de Predicción")

    # Filtrar solo los modelos que tienen datos válidos
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
            display_name = "Bayesiano" if model_name == 'bayes' else "XGBoost"
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

def render_system_info(num_teams):
    """Renderiza la información del sistema"""
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**Python:** `{sys.version}`")
        st.write(f"**PyMC:** {'✅ Disponible' if config.PYMC_AVAILABLE else '❌ No disponible'}")
        st.write(f"**SKLearn:** {'✅ Disponible' if config.SKLEARN_AVAILABLE else '❌ No disponible'}")
    with col2:
        st.write(f"**Dixon-Coles ρ:** `{config.DIXON_COLES_RHO:.3f}`")
        st.write(f"**Equipos clasificados:** `{num_teams}`")
        if config.PYMC_AVAILABLE:
            import pymc as pm
            import arviz as az
            st.write(f"**PyMC versión:** `{pm.__version__}`")
            st.write(f"**ArviZ versión:** `{az.__version__}`")