# modules/corrections.py - Correcciones del modelo
import numpy as np
from scipy.stats import poisson

from . import config

def ajustar_por_pausas_hidratacion(lam_h, lam_a, elo_h=None, elo_a=None):
    """Ajuste para el formato de 4 tiempos con pausas de hidratación"""
    factor_general = 0.95
    media_h_general = 1.35
    media_a_general = 1.05

    lam_h_ajustado = lam_h * factor_general
    lam_a_ajustado = lam_a * factor_general

    if elo_h is not None and elo_a is not None:
        diff_elo = abs(elo_h - elo_a)
        if diff_elo > 200:
            factor_contraccion = 0.05
        elif diff_elo > 100:
            factor_contraccion = 0.08
        else:
            factor_contraccion = 0.12
    else:
        factor_contraccion = 0.12

    lam_h_ajustado = lam_h_ajustado * (1 - factor_contraccion) + media_h_general * factor_contraccion
    lam_a_ajustado = lam_a_ajustado * (1 - factor_contraccion) + media_a_general * factor_contraccion

    return lam_h_ajustado, lam_a_ajustado

def dixon_coles_factor(x, y, lam_h, lam_a, rho):
    """Factor de corrección de Dixon-Coles"""
    mask_00 = (x == 0) & (y == 0)
    mask_01 = (x == 0) & (y == 1)
    mask_10 = (x == 1) & (y == 0)
    mask_11 = (x == 1) & (y == 1)
    tau = np.ones_like(x, dtype=float)
    tau[mask_00] = 1 - lam_h * lam_a * rho
    tau[mask_01] = 1 + lam_h * rho
    tau[mask_10] = 1 + lam_a * rho
    tau[mask_11] = 1 - rho
    tau = np.maximum(tau, 0.01)
    return tau

def aplicar_dixon_coles(score_matrix, lam_h, lam_a, rho=None):
    """Aplica la corrección de Dixon-Coles a una matriz de marcadores"""
    if rho is None:
        rho = config.DIXON_COLES_RHO
    max_g = score_matrix.shape[0] - 1
    goals = np.arange(0, max_g + 1)
    H, A = np.meshgrid(goals, goals, indexing='ij')
    tau = dixon_coles_factor(H, A, lam_h, lam_a, rho)
    score_matrix_corregida = score_matrix * tau
    suma = score_matrix_corregida.sum()
    if suma > 0:
        score_matrix_corregida = score_matrix_corregida / suma
    return score_matrix_corregida

def ajustar_por_gol_temprano(score_matrix, lam_h, lam_a, home_team, away_team,
                             underdog_scored_first, minuto_gol, favorito_elo, underdog_elo):
    """Ajuste por gol temprano del underdog"""
    if not underdog_scored_first:
        return score_matrix
    if lam_h > lam_a:
        favorito = 'home'; underdog = 'away'; diff_lam = lam_h - lam_a
    else:
        favorito = 'away'; underdog = 'home'; diff_lam = lam_a - lam_h
    if minuto_gol <= 15:
        factor_tiempo = 1.0
    elif minuto_gol <= 30:
        factor_tiempo = 0.8
    elif minuto_gol <= 45:
        factor_tiempo = 0.6
    else:
        factor_tiempo = 0.3
    factor_sorpresa = min(1.0, diff_lam * 0.5)
    factor_ajuste = factor_tiempo * factor_sorpresa
    if factor_ajuste < 0.1:
        return score_matrix
    max_g = score_matrix.shape[0] - 1
    ajuste = np.ones_like(score_matrix)
    for i in range(min(3, max_g + 1)):
        for j in range(min(3, max_g + 1)):
            if i + j <= 2:
                ajuste[i, j] = 1 - factor_ajuste * 0.3
    for i in range(1, min(5, max_g + 1)):
        for j in range(1, min(5, max_g + 1)):
            if 3 <= i + j <= 6:
                ajuste[i, j] = 1 + factor_ajuste * 0.25
    if underdog == 'away':
        for i in range(1, min(5, max_g + 1)):
            for j in range(1, min(3, max_g + 1)):
                if i >= j:
                    ajuste[i, j] = 1 + factor_ajuste * 0.2
    if underdog == 'home':
        for i in range(1, min(3, max_g + 1)):
            for j in range(1, min(5, max_g + 1)):
                if j >= i:
                    ajuste[i, j] = 1 + factor_ajuste * 0.2
    score_matrix_ajustada = score_matrix * ajuste
    suma = score_matrix_ajustada.sum()
    if suma > 0:
        score_matrix_ajustada = score_matrix_ajustada / suma
    return score_matrix_ajustada

def ajustar_por_momentum(lam_h, lam_a, home_team, away_team, 
                         minuto_gol=None, es_favorito_local=None,
                         llegadas_previas_h=None, llegadas_previas_a=None,
                         marcador_actual=None):
    """Ajuste dinámico por momentum en tiempo real"""
    if minuto_gol is not None and minuto_gol >= 80 and es_favorito_local is not None:
        if es_favorito_local:
            lam_h *= 1.12
            lam_a *= 0.95
        else:
            lam_a *= 1.12
            lam_h *= 0.95

    if marcador_actual is not None:
        diff = marcador_actual.get('home', 0) - marcador_actual.get('away', 0)
        if diff >= 2:
            lam_a *= 1.08
        elif diff <= -2:
            lam_h *= 1.08

    if (llegadas_previas_h is not None and llegadas_previas_a is not None 
        and llegadas_previas_a > 0 and llegadas_previas_h > 0):
        ratio = llegadas_previas_h / llegadas_previas_a
        if ratio > 1.5:
            lam_h *= 1.06
        elif ratio < 0.67:
            lam_a *= 1.06

    return lam_h, lam_a

# ============================================================================
# NUEVOS AJUSTES PARA ALTA ANOTACIÓN
# ============================================================================

def ajuste_completo_alta_anotacion(lam_h, lam_a, home_team, away_team,
                                    stats_h=None, stats_a=None):
    """
    Combina todos los ajustes para partidos de alta anotación
    """
    # 1. Factor de alta anotación histórica
    if stats_h is not None and stats_a is not None:
        avg_goles_h = stats_h.get('avg_goles', 1.5)
        avg_goles_a = stats_a.get('avg_goles', 1.5)
        
        if avg_goles_h > 2.0 and avg_goles_a > 1.8:
            lam_h *= 1.12
            lam_a *= 1.12
        elif avg_goles_h > 2.0:
            lam_h *= 1.08
        elif avg_goles_a > 2.0:
            lam_a *= 1.08
    
    # 2. Factor de partidos con ambos anotan
    both_score_h = stats_h.get('both_score_pct', 0.5) if stats_h else 0.5
    both_score_a = stats_a.get('both_score_pct', 0.5) if stats_a else 0.5
    
    if both_score_h > 0.65 and both_score_a > 0.65:
        lam_h *= 1.06
        lam_a *= 1.06
    
    # 3. Factor de goleadores en racha (basado en Elo ofensivo)
    attack_h = stats_h.get('attack', 1.5) if stats_h else 1.5
    attack_a = stats_a.get('attack', 1.5) if stats_a else 1.5
    
    if attack_h > 2.0 and attack_a > 1.8:
        lam_h *= 1.05
        lam_a *= 1.05
    
    return lam_h, lam_a

def ajustar_matriz_alta_anotacion(score_matrix, lam_h, lam_a, max_g=8):
    """
    Ajusta la matriz para dar más peso a marcadores de alta anotación
    """
    ajuste = np.ones_like(score_matrix)
    
    for i in range(max_g + 1):
        for j in range(max_g + 1):
            total_goles = i + j
            if total_goles >= 4:
                ajuste[i, j] = 1.15
            elif total_goles >= 3:
                ajuste[i, j] = 1.08
    
    score_matrix_ajustada = score_matrix * ajuste
    suma = score_matrix_ajustada.sum()
    if suma > 0:
        score_matrix_ajustada = score_matrix_ajustada / suma
    
    return score_matrix_ajustada