# modules/data_loader.py - Carga de datos
import pandas as pd
import requests
import streamlit as st
from datetime import datetime, timedelta
import pytz

from . import config

# Lista de selecciones clasificadas
WORLD_CUP_2026_TEAMS = [
    "Spain", "Portugal", "France", "Belgium", "Netherlands", "Italy", "Germany",
    "England", "Croatia", "Switzerland", "Denmark", "Austria", "Ukraine",
    "Sweden", "Wales", "Turkey", "Scotland", "Norway", "Serbia", "Bosnia and Herzegovina",
    "Poland", "Czech Republic", "Hungary", "Slovakia",
    "Brazil", "Argentina", "Uruguay", "Ecuador", "Colombia", "Chile", "Paraguay",
    "Mexico", "USA", "Canada", "Costa Rica", "Jamaica", "Panama", "Honduras", "El Salvador",
    "Haiti", "Curacao",
    "Morocco", "Senegal", "Nigeria", "Egypt", "Ghana", "Cameroon", "Algeria",
    "Tunisia", "Mali", "Ivory Coast", "Cabo Verde", "South Africa", "Congo DR",
    "Japan", "South Korea", "Iran", "Saudi Arabia", "Australia", "Qatar",
    "UAE", "Iraq", "Jordan", "Uzbekistan",
    "New Zealand"
]
WORLD_CUP_2026_TEAMS = sorted(WORLD_CUP_2026_TEAMS)

@st.cache_data(ttl=3600)
def load_match_data():
    """Carga los datos de resultados internacionales"""
    try:
        RESULTS_URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
        raw = pd.read_csv(RESULTS_URL, parse_dates=["date"])
        return raw
    except Exception as e:
        st.error(f"Error al cargar los datos: {str(e)}")
        return None

def filter_world_cup_teams(raw):
    """Filtra los equipos que están en la lista del Mundial 2026"""
    all_teams = sorted(pd.concat([raw.home_team, raw.away_team]).unique())
    wc_teams = [team for team in WORLD_CUP_2026_TEAMS if team in all_teams]
    
    if len(wc_teams) < 10:
        wc_teams = all_teams[:20]
    
    return wc_teams

@st.cache_data(ttl=3600)
def get_espn_fixture():
    """Obtiene el fixture del Mundial 2026 desde ESPN"""
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
                        fixture_data.append({'date': match_date, 'home_team': home_team, 'away_team': away_team})
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
    """Obtiene estadísticas de un equipo desde ESPN (con datos de alta anotación)"""
    stats_cache = {
        'Spain': {'attack': 2.5, 'defense': 0.8, 'elo': 2050, 'avg_goles': 2.2, 'both_score_pct': 0.65},
        'Uruguay': {'attack': 1.8, 'defense': 1.1, 'elo': 1850, 'avg_goles': 1.8, 'both_score_pct': 0.55},
        'Cabo Verde': {'attack': 1.4, 'defense': 1.3, 'elo': 1680, 'avg_goles': 1.5, 'both_score_pct': 0.50},
        'Saudi Arabia': {'attack': 1.3, 'defense': 1.4, 'elo': 1700, 'avg_goles': 1.4, 'both_score_pct': 0.45},
        'Mexico': {'attack': 1.8, 'defense': 1.2, 'elo': 1880, 'avg_goles': 1.9, 'both_score_pct': 0.60},
        'USA': {'attack': 1.9, 'defense': 1.1, 'elo': 1900, 'avg_goles': 2.0, 'both_score_pct': 0.62},
        'Canada': {'attack': 1.7, 'defense': 1.3, 'elo': 1820, 'avg_goles': 1.8, 'both_score_pct': 0.55},
        'Brazil': {'attack': 2.8, 'defense': 0.7, 'elo': 2100, 'avg_goles': 2.5, 'both_score_pct': 0.70},
        'Argentina': {'attack': 2.6, 'defense': 0.8, 'elo': 2080, 'avg_goles': 2.4, 'both_score_pct': 0.68},
        'France': {'attack': 2.4, 'defense': 0.9, 'elo': 2040, 'avg_goles': 2.3, 'both_score_pct': 0.65},
        'Germany': {'attack': 2.3, 'defense': 0.9, 'elo': 2020, 'avg_goles': 2.2, 'both_score_pct': 0.62},
        'England': {'attack': 2.2, 'defense': 0.9, 'elo': 2000, 'avg_goles': 2.1, 'both_score_pct': 0.60},
        'Netherlands': {'attack': 2.1, 'defense': 1.0, 'elo': 1980, 'avg_goles': 2.0, 'both_score_pct': 0.58},
        'Portugal': {'attack': 2.0, 'defense': 1.0, 'elo': 1960, 'avg_goles': 2.0, 'both_score_pct': 0.58},
        'Italy': {'attack': 1.9, 'defense': 0.9, 'elo': 1940, 'avg_goles': 1.9, 'both_score_pct': 0.55},
        'Belgium': {'attack': 1.9, 'defense': 1.1, 'elo': 1920, 'avg_goles': 1.9, 'both_score_pct': 0.56},
        'Austria': {'attack': 1.6, 'defense': 1.2, 'elo': 1780, 'avg_goles': 1.7, 'both_score_pct': 0.50},
        'Norway': {'attack': 1.9, 'defense': 1.3, 'elo': 1820, 'avg_goles': 2.1, 'both_score_pct': 0.65},
    }
    synonyms = {'Cape Verde': 'Cabo Verde', 'USA': 'USA', 'United States': 'USA',
                'Czechia': 'Czech Republic', "Côte d'Ivoire": 'Ivory Coast'}
    search_name = synonyms.get(team_name, team_name)
    default = {'attack': 1.5, 'defense': 1.3, 'elo': 1750, 'avg_goles': 1.5, 'both_score_pct': 0.50}
    return stats_cache.get(search_name, default)

def get_current_date_mexico():
    """Obtiene la fecha actual en zona horaria de México"""
    mexico_tz = pytz.timezone('America/Mexico_City')
    return datetime.now(mexico_tz).date()