# modules/predictors.py - Factores predictivos dinámicos
from .team_roster import get_top_scorers, get_player_momentum_factor

def adjust_by_roster_factors(lam_h, lam_a, home_team, away_team, roster_factors):
    """
    Ajusta los goles esperados (λ) basado en factores de alineación
    """
    if not roster_factors:
        return lam_h, lam_a
    
    # 1. Factores de goleadores
    h_scorers = get_top_scorers(home_team, 3)
    a_scorers = get_top_scorers(away_team, 3)
    
    h_scorer_factor = 1.0
    a_scorer_factor = 1.0
    
    for scorer in h_scorers:
        player_name = scorer.split('(')[0].strip()
        momentum = get_player_momentum_factor(player_name)
        h_scorer_factor = max(h_scorer_factor, momentum)
    
    for scorer in a_scorers:
        player_name = scorer.split('(')[0].strip()
        momentum = get_player_momentum_factor(player_name)
        a_scorer_factor = max(a_scorer_factor, momentum)
    
    # 2. Factor de profundidad de banquillo
    bench_factor_h = 1.05
    bench_factor_a = 1.05
    
    # Si hay más de 3 delanteros en la plantilla, hay más opciones de gol
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
        if get_player_momentum_factor(player_name) > 1.1:
            confidence_h += 0.03
    for scorer in a_scorers:
        player_name = scorer.split('(')[0].strip()
        if get_player_momentum_factor(player_name) > 1.1:
            confidence_a += 0.03
    
    # 4. Aplicar ajustes
    lam_h_ajustado = lam_h * h_scorer_factor * bench_factor_h * confidence_h
    lam_a_ajustado = lam_a * a_scorer_factor * bench_factor_a * confidence_a
    
    return lam_h_ajustado, lam_a_ajustado

def get_top_scorers_summary(home_team, away_team):
    """Obtiene un resumen de los máximos goleadores"""
    h_scorers = get_top_scorers(home_team, 2)
    a_scorers = get_top_scorers(away_team, 2)
    
    h_str = f"🏠 {home_team}: {', '.join(h_scorers) if h_scorers else 'Sin datos'}"
    a_str = f"✈️ {away_team}: {', '.join(a_scorers) if a_scorers else 'Sin datos'}"
    
    return f"{h_str} | {a_str}"