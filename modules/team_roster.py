# modules/team_roster.py - Alineaciones y jugadores (COMPLETO)
import streamlit as st
import json
import os

# Base de datos de jugadores
PLAYER_DATABASE = {
    'Argentina': {
        'attackers': ['Messi', 'Martínez', 'Di María', 'Álvarez'],
        'midfielders': ['De Paul', 'Mac Allister', 'Paredes', 'Fernández'],
        'defenders': ['Romero', 'Otamendi', 'Tagliafico', 'Molina'],
        'goalkeeper': ['Martínez'],
        'top_scorer': {'Messi': 8, 'Martínez': 4, 'Álvarez': 3}
    },
    'Spain': {
        'attackers': ['Morata', 'Ferran Torres', 'Yamal'],
        'midfielders': ['Pedri', 'Gavi', 'Rodri', 'Fabián Ruiz'],
        'defenders': ['Carvajal', 'Le Normand', 'Laporte', 'Grimaldo'],
        'goalkeeper': ['Simón'],
        'top_scorer': {'Morata': 5, 'Yamal': 3, 'Ferran Torres': 3}
    },
    'Mexico': {
        'attackers': ['Jiménez', 'Martín', 'Vega'],
        'midfielders': ['Álvarez', 'Chávez', 'Rodríguez'],
        'defenders': ['Araujo', 'Montes', 'Vásquez', 'Gallardo'],
        'goalkeeper': ['Ochoa'],
        'top_scorer': {'Jiménez': 6, 'Vega': 4, 'Martín': 3}
    },
    'Brazil': {
        'attackers': ['Vinicius Jr', 'Rodrygo', 'Raphinha', 'Endrick'],
        'midfielders': ['Casemiro', 'Paquetá', 'Guimarães', 'Gomes'],
        'defenders': ['Marquinhos', 'Gabriel', 'Danilo', 'Lodi'],
        'goalkeeper': ['Alisson'],
        'top_scorer': {'Vinicius Jr': 7, 'Rodrygo': 5, 'Raphinha': 4}
    },
    'France': {
        'attackers': ['Mbappé', 'Griezmann', 'Thuram', 'Dembélé'],
        'midfielders': ['Tchouaméni', 'Camavinga', 'Rabiot', 'Fofana'],
        'defenders': ['Saliba', 'Upamecano', 'Hernández', 'Koundé'],
        'goalkeeper': ['Maignan'],
        'top_scorer': {'Mbappé': 9, 'Griezmann': 5, 'Thuram': 4}
    },
    'Norway': {
        'attackers': ['Haaland', 'Sørloth', 'Ødegaard'],
        'midfielders': ['Berge', 'Thorsby', 'Aursnes'],
        'defenders': ['Ajer', 'Strandberg', 'Meling', 'Pedersen'],
        'goalkeeper': ['Nyland'],
        'top_scorer': {'Haaland': 10, 'Sørloth': 4, 'Ødegaard': 3}
    }
}

# Estados de forma de jugadores
PLAYER_FORM = {
    'Messi': {'form': 'Excelente', 'goals_last_5': 7, 'factor': 1.15},
    'Martínez': {'form': 'Buena', 'goals_last_5': 4, 'factor': 1.08},
    'Álvarez': {'form': 'Excelente', 'goals_last_5': 5, 'factor': 1.12},
    'Di María': {'form': 'Buena', 'goals_last_5': 3, 'factor': 1.05},
    'Morata': {'form': 'Regular', 'goals_last_5': 2, 'factor': 0.95},
    'Yamal': {'form': 'Excelente', 'goals_last_5': 4, 'factor': 1.12},
    'Jiménez': {'form': 'Buena', 'goals_last_5': 3, 'factor': 1.05},
    'Vinicius Jr': {'form': 'Excelente', 'goals_last_5': 6, 'factor': 1.18},
    'Mbappé': {'form': 'Excelente', 'goals_last_5': 8, 'factor': 1.20},
    'Ochoa': {'form': 'Regular', 'goals_last_5': 0, 'factor': 0.90},
    'Haaland': {'form': 'Excelente', 'goals_last_5': 9, 'factor': 1.25},
    'Ødegaard': {'form': 'Buena', 'goals_last_5': 4, 'factor': 1.10},
}

def get_team_roster(team_name):
    """Obtiene la alineación estimada de un equipo"""
    return PLAYER_DATABASE.get(team_name, {
        'attackers': ['Sin datos'],
        'midfielders': ['Sin datos'],
        'defenders': ['Sin datos'],
        'goalkeeper': ['Sin datos'],
        'top_scorer': {}
    })

def get_top_scorers(team_name, limit=3):
    """Obtiene los máximos goleadores de un equipo"""
    roster = get_team_roster(team_name)
    scorers = roster.get('top_scorer', {})
    sorted_scorers = sorted(scorers.items(), key=lambda x: x[1], reverse=True)
    return [f"{player} ({goals})" for player, goals in sorted_scorers[:limit]]

def get_player_form(player_name):
    """Obtiene el estado de forma de un jugador"""
    return PLAYER_FORM.get(player_name, {'form': 'Sin datos', 'goals_last_5': 0, 'factor': 1.0})

def get_player_momentum_factor(player_name):
    """Factor de momentum para un jugador"""
    return PLAYER_FORM.get(player_name, {'factor': 1.0}).get('factor', 1.0)

def display_roster_preview(home_team, away_team):
    """Muestra un preview de las alineaciones"""
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"### 🏠 {home_team}")
        roster_h = get_team_roster(home_team)
        st.write(f"⚽ **Delanteros:** {', '.join(roster_h['attackers'])}")
        st.write(f"🔄 **Mediocampistas:** {', '.join(roster_h['midfielders'])}")
        st.write(f"🛡️ **Defensas:** {', '.join(roster_h['defenders'])}")
        st.write(f"🧤 **Portero:** {', '.join(roster_h['goalkeeper'])}")
        
        scorers = get_top_scorers(home_team, 3)
        if scorers and scorers[0] != 'Sin datos':
            st.write(f"⚽ **Máximos goleadores:** {', '.join(scorers)}")
        
    with col2:
        st.markdown(f"### ✈️ {away_team}")
        roster_a = get_team_roster(away_team)
        st.write(f"⚽ **Delanteros:** {', '.join(roster_a['attackers'])}")
        st.write(f"🔄 **Mediocampistas:** {', '.join(roster_a['midfielders'])}")
        st.write(f"🛡️ **Defensas:** {', '.join(roster_a['defenders'])}")
        st.write(f"🧤 **Portero:** {', '.join(roster_a['goalkeeper'])}")
        
        scorers = get_top_scorers(away_team, 3)
        if scorers and scorers[0] != 'Sin datos':
            st.write(f"⚽ **Máximos goleadores:** {', '.join(scorers)}")
    
    # Mostrar estado de forma de jugadores clave
    st.markdown("---")
    st.subheader("📊 Estado de Jugadores Clave")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"**{home_team}**")
        top_scorers_h = list(PLAYER_DATABASE.get(home_team, {}).get('top_scorer', {}).keys())
        for player in top_scorers_h[:2]:
            form = get_player_form(player)
            if form['form'] != 'Sin datos':
                emoji = "🟢" if form['factor'] > 1.05 else "🟡" if form['factor'] > 0.95 else "🔴"
                st.write(f"{emoji} {player}: {form['form']} ({form['goals_last_5']} goles últimos 5)")
    
    with col2:
        st.markdown(f"**{away_team}**")
        top_scorers_a = list(PLAYER_DATABASE.get(away_team, {}).get('top_scorer', {}).keys())
        for player in top_scorers_a[:2]:
            form = get_player_form(player)
            if form['form'] != 'Sin datos':
                emoji = "🟢" if form['factor'] > 1.05 else "🟡" if form['factor'] > 0.95 else "🔴"
                st.write(f"{emoji} {player}: {form['form']} ({form['goals_last_5']} goles últimos 5)")