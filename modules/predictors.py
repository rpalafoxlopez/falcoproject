# modules/predictors.py - Factores predictivos dinámicos (COMPLETO)
import numpy as np
from . import team_roster

def adjust_by_roster_factors(lam_h, lam_a, home_team, away_team, roster_factors):
    """
    Ajusta los goles esperados (λ) basado en factores de alineación
    """
    if not roster_factors:
        return lam_h, lam_a
    
    # 1. Factores de goleadores
    h_scorers = team_roster.get_top_scorers(home_team, 3)
    a_scorers = team_roster.get_top_scorers(away_team, 3)
    
    h_scorer_factor = 1.0
    a_scorer_factor = 1.0
    
    for scorer in h_scorers:
        player_name = scorer.split('(')[0].strip()
        momentum = team_roster.get_player_momentum_factor(player_name)
        h_scorer_factor = max(h_scorer_factor, momentum)
    
    for scorer in a_scorers:
        player_name = scorer.split('(')[0].strip()
        momentum = team_roster.get_player_momentum_factor(player_name)
        a_scorer_factor = max(a_scorer_factor, momentum)
    
    # 2. Factor de profundidad de banquillo
    bench_factor_h = 1.05
    bench_factor_a = 1.05
    
    if roster_factors:
        home_roster = roster_factors.get('home', {})
        away_roster = roster_factors.get('away', {})
        
        if len(home_roster.get('attackers', [])) > 3:
            bench_factor_h = 1.08
        if len(away_roster.get('attackers', [])) > 3:
            bench_factor_a = 1.08
    
    # 3. Factor de confianza (basado en goleadores en racha)
    confidence_h = 1.0
    confidence_a = 1.0
    
    for scorer in h_scorers:
        player_name = scorer.split('(')[0].strip()
        if team_roster.get_player_momentum_factor(player_name) > 1.1:
            confidence_h += 0.03
    for scorer in a_scorers:
        player_name = scorer.split('(')[0].strip()
        if team_roster.get_player_momentum_factor(player_name) > 1.1:
            confidence_a += 0.03
    
    # 4. Aplicar ajustes
    lam_h_ajustado = lam_h * h_scorer_factor * bench_factor_h * confidence_h
    lam_a_ajustado = lam_a * a_scorer_factor * bench_factor_a * confidence_a
    
    return lam_h_ajustado, lam_a_ajustado

def get_top_scorers_summary(home_team, away_team):
    """Obtiene un resumen de los máximos goleadores"""
    h_scorers = team_roster.get_top_scorers(home_team, 2)
    a_scorers = team_roster.get_top_scorers(away_team, 2)
    
    h_str = f"🏠 {home_team}: {', '.join(h_scorers) if h_scorers else 'Sin datos'}"
    a_str = f"✈️ {away_team}: {', '.join(a_scorers) if a_scorers else 'Sin datos'}"
    
    return f"{h_str} | {a_str}"

def get_match_factors(home_team, away_team):
    """
    Obtiene todos los factores relevantes para el partido
    """
    factors = {
        'home_team': home_team,
        'away_team': away_team,
        'home_scorers': team_roster.get_top_scorers(home_team, 3),
        'away_scorers': team_roster.get_top_scorers(away_team, 3),
        'momentum': {
            'home': 1.0,
            'away': 1.0
        },
        'bench_depth': {
            'home': 1.05,
            'away': 1.05
        },
        'confidence': {
            'home': 0.85,
            'away': 0.80
        }
    }
    return factors

def get_predictive_advantage(home_team, away_team):
    """
    Calcula una ventaja predictiva basada en factores de alineación
    Retorna: (factor_home, factor_away)
    """
    h_scorers = team_roster.get_top_scorers(home_team, 3)
    a_scorers = team_roster.get_top_scorers(away_team, 3)
    
    # Calcular calidad ofensiva basada en goleadores
    h_quality = len(h_scorers) * 0.05
    a_quality = len(a_scorers) * 0.05
    
    # Factores de momentum de jugadores
    h_momentum = 1.0
    a_momentum = 1.0
    
    for scorer in h_scorers:
        player = scorer.split('(')[0].strip()
        h_momentum = max(h_momentum, team_roster.get_player_momentum_factor(player))
    
    for scorer in a_scorers:
        player = scorer.split('(')[0].strip()
        a_momentum = max(a_momentum, team_roster.get_player_momentum_factor(player))
    
    # Factores combinados
    factor_h = 1.0 + h_quality + (h_momentum - 1.0)
    factor_a = 1.0 + a_quality + (a_momentum - 1.0)
    
    return factor_h, factor_a