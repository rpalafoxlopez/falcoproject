# modules/corrections.py - Correcciones del modelo (VERSIÓN MEJORADA)
import numpy as np
from scipy.stats import poisson

from . import config

# ============================================================================
# AJUSTES BASE
# ============================================================================

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
# AJUSTES DE ALTA ANOTACIÓN
# ============================================================================

def ajuste_completo_alta_anotacion(lam_h, lam_a, home_team, away_team,
                                    stats_h=None, stats_a=None):
    """
    Combina todos los ajustes para partidos de alta anotación
    """
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
    
    both_score_h = stats_h.get('both_score_pct', 0.5) if stats_h else 0.5
    both_score_a = stats_a.get('both_score_pct', 0.5) if stats_a else 0.5
    
    if both_score_h > 0.65 and both_score_a > 0.65:
        lam_h *= 1.06
        lam_a *= 1.06
    
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

# ============================================================================
# AJUSTES CONTEXTUALES (Partidos "rotos") - MEJORADOS
# ============================================================================

def ajustar_por_diferencia_elo(lam_h, lam_a, elo_h, elo_a):
    """
    Si la diferencia de Elo es muy grande, el favorito tiene más ventaja
    """
    diff_elo = abs(elo_h - elo_a)
    
    # Si la diferencia es de +200 Elo (muy significativa)
    if diff_elo > 200:
        if elo_h > elo_a:
            lam_h *= 1.10
            lam_a *= 0.90
        else:
            lam_a *= 1.10
            lam_h *= 0.90
    
    # Si la diferencia es de +400 Elo (abismal)
    if diff_elo > 400:
        if elo_h > elo_a:
            lam_h *= 1.20
            lam_a *= 0.80
        else:
            lam_a *= 1.20
            lam_h *= 0.80
    
    return lam_h, lam_a

def ajustar_por_gol_temprano_favorito(lam_h, lam_a, es_favorito_local, minuto_gol, marcador_actual):
    """
    Si el favorito anota en los primeros 15 minutos, el partido tiende a romperse
    """
    if minuto_gol <= 15:
        if es_favorito_local:
            lam_h *= 1.30
            lam_a *= 0.75
        else:
            lam_a *= 1.30
            lam_h *= 0.75
    
    if marcador_actual is not None:
        diff = marcador_actual.get('home', 0) - marcador_actual.get('away', 0)
        if diff >= 2:
            lam_h *= 1.15
            lam_a *= 0.85
        elif diff <= -2:
            lam_a *= 1.15
            lam_h *= 0.85
    
    return lam_h, lam_a

def ajustar_por_partido_roto(lam_h, lam_a, goles_h, goles_a, minuto):
    """
    Si hay 3+ goles de diferencia en el primer tiempo, el partido se rompe
    """
    diff = abs(goles_h - goles_a)
    
    if diff >= 4 and minuto <= 45:
        if goles_h > goles_a:
            lam_h *= 1.35
            lam_a *= 0.60
        else:
            lam_a *= 1.35
            lam_h *= 0.60
    
    elif diff >= 3 and minuto <= 45:
        if goles_h > goles_a:
            lam_h *= 1.25
            lam_a *= 0.70
        else:
            lam_a *= 1.25
            lam_h *= 0.70
    
    return lam_h, lam_a

def ajustar_por_motivacion(lam_h, lam_a, goles_h, goles_a):
    """
    Si un equipo va ganando cómodo, se relaja; si va perdiendo, se desespera
    """
    diff = goles_h - goles_a
    
    if diff >= 2:
        lam_h *= 1.08
        lam_a *= 1.05
    elif diff <= -2:
        lam_a *= 1.08
        lam_h *= 1.05
    elif diff >= 1:
        lam_h *= 1.03
        lam_a *= 0.97
    elif diff <= -1:
        lam_a *= 1.03
        lam_h *= 0.97
    
    return lam_h, lam_a

def ajuste_completo_contextual(lam_h, lam_a, home_team, away_team,
                                es_favorito_local, minuto_primer_gol=None,
                                marcador_actual=None, use_early_goal=True,
                                use_partido_roto=True, use_motivacion=True,
                                elo_h=None, elo_a=None):
    """
    Combina todos los ajustes contextuales para partidos "rotos"
    """
    if elo_h is not None and elo_a is not None:
        lam_h, lam_a = ajustar_por_diferencia_elo(lam_h, lam_a, elo_h, elo_a)
    
    if use_early_goal and minuto_primer_gol is not None and minuto_primer_gol <= 15:
        lam_h, lam_a = ajustar_por_gol_temprano_favorito(
            lam_h, lam_a, es_favorito_local, minuto_primer_gol, marcador_actual
        )
    
    if use_partido_roto and marcador_actual is not None:
        lam_h, lam_a = ajustar_por_partido_roto(
            lam_h, lam_a, 
            marcador_actual.get('home', 0), 
            marcador_actual.get('away', 0),
            minuto_primer_gol if minuto_primer_gol is not None else 45
        )
    
    if use_motivacion and marcador_actual is not None:
        lam_h, lam_a = ajustar_por_motivacion(
            lam_h, lam_a,
            marcador_actual.get('home', 0),
            marcador_actual.get('away', 0)
        )
    
    return lam_h, lam_a

# ============================================================================
# SISTEMA DE RESULTADO PROXIMAL
# ============================================================================

def calcular_resultado_proximal(score_matrix, lam_h, lam_a, 
                                marcador_actual_h, marcador_actual_a,
                                es_favorito_local, underdog_scored_first=False,
                                minuto_gol=0, partido_roto=False):
    """
    Calcula el resultado más probable considerando el contexto del partido.
    """
    resultados = {
        'base': None,
        'underdog_first': None,
        'partido_roto': None,
        'proximal': None,
        'es_empate': False
    }
    
    # 1. Resultado base
    flat_idx = np.argmax(score_matrix)
    base_h, base_a = np.unravel_index(flat_idx, score_matrix.shape)
    resultados['base'] = (base_h, base_a)
    
    # 2. Verificar si el empate es el resultado más probable
    home_win_prob = np.sum(np.tril(score_matrix, k=-1))
    draw_prob = np.sum(np.diag(score_matrix))
    away_win_prob = np.sum(np.triu(score_matrix, k=1))
    
    if draw_prob >= home_win_prob and draw_prob >= away_win_prob:
        resultados['proximal'] = resultados['base']
        resultados['es_empate'] = True
        resultados['underdog_first'] = resultados['base']
        resultados['partido_roto'] = resultados['base']
        return resultados
    
    # 3. Si el underdog anotó primero
    if underdog_scored_first:
        if es_favorito_local:
            underdog_h = False
        else:
            underdog_h = True
        
        if underdog_h:
            if score_matrix.shape[0] > 1:
                prob_underdog = score_matrix[1:, :]
                if prob_underdog.size > 0:
                    flat_idx_ud = np.argmax(prob_underdog)
                    ud_h, ud_a = np.unravel_index(flat_idx_ud, prob_underdog.shape)
                    resultados['underdog_first'] = (ud_h + 1, ud_a)
                else:
                    resultados['underdog_first'] = resultados['base']
            else:
                resultados['underdog_first'] = resultados['base']
        else:
            if score_matrix.shape[1] > 1:
                prob_underdog = score_matrix[:, 1:]
                if prob_underdog.size > 0:
                    flat_idx_ud = np.argmax(prob_underdog)
                    ud_h, ud_a = np.unravel_index(flat_idx_ud, prob_underdog.shape)
                    resultados['underdog_first'] = (ud_h, ud_a + 1)
                else:
                    resultados['underdog_first'] = resultados['base']
            else:
                resultados['underdog_first'] = resultados['base']
    
    # 4. Si el partido está roto
    if partido_roto:
        diff = abs(marcador_actual_h - marcador_actual_a)
        
        if marcador_actual_h > marcador_actual_a:
            min_h = marcador_actual_h
            if min_h < score_matrix.shape[0]:
                prob_roto = score_matrix[min_h:, :]
                if prob_roto.size > 0:
                    flat_idx_roto = np.argmax(prob_roto)
                    r_h, r_a = np.unravel_index(flat_idx_roto, prob_roto.shape)
                    if (r_h + min_h) - r_a >= diff:
                        resultados['partido_roto'] = (r_h + min_h, r_a)
                    else:
                        for i in range(min_h, score_matrix.shape[0]):
                            for j in range(score_matrix.shape[1]):
                                if i - j >= diff:
                                    resultados['partido_roto'] = (i, j)
                                    break
                            if resultados['partido_roto'] is not None:
                                break
                else:
                    resultados['partido_roto'] = resultados['base']
            else:
                resultados['partido_roto'] = resultados['base']
        else:
            min_a = marcador_actual_a
            if min_a < score_matrix.shape[1]:
                prob_roto = score_matrix[:, min_a:]
                if prob_roto.size > 0:
                    flat_idx_roto = np.argmax(prob_roto)
                    r_h, r_a = np.unravel_index(flat_idx_roto, prob_roto.shape)
                    if (r_a + min_a) - r_h >= diff:
                        resultados['partido_roto'] = (r_h, r_a + min_a)
                    else:
                        for j in range(min_a, score_matrix.shape[1]):
                            for i in range(score_matrix.shape[0]):
                                if j - i >= diff:
                                    resultados['partido_roto'] = (i, j)
                                    break
                            if resultados['partido_roto'] is not None:
                                break
                else:
                    resultados['partido_roto'] = resultados['base']
            else:
                resultados['partido_roto'] = resultados['base']
    
    # 5. Resultado proximal
    if partido_roto and resultados['partido_roto'] is not None:
        resultados['proximal'] = resultados['partido_roto']
    elif underdog_scored_first and resultados['underdog_first'] is not None:
        resultados['proximal'] = resultados['underdog_first']
    else:
        resultados['proximal'] = resultados['base']
    
    return resultados

# ============================================================================
# AJUSTE POR FASE DEL TORNEO (NUEVO)
# ============================================================================

def ajuste_por_fase(lam_h, lam_a, fase):
    """
    Aplica ajustes específicos según la fase del torneo
    """
    fase_config = config.FASE_CONFIG.get(fase, config.FASE_CONFIG['Fase de Grupos'])
    
    # 1. Factor de goles
    lam_h *= fase_config['factor_goles']
    lam_a *= fase_config['factor_goles']
    
    # 2. Obtener nuevo rho para Dixon-Coles
    nuevo_rho = fase_config['rho_dixon_coles']
    
    # 3. Factor de alta anotación
    factor_high_scoring = fase_config['factor_high_scoring']
    
    return lam_h, lam_a, nuevo_rho, factor_high_scoring

def ajustar_matriz_por_fase(score_matrix, lam_h, lam_a, fase, max_g=8):
    """
    Ajusta la matriz de marcadores según la fase del torneo
    """
    fase_config = config.FASE_CONFIG.get(fase, config.FASE_CONFIG['Fase de Grupos'])
    
    factor_goleadas = fase_config['factor_goleadas']
    
    ajuste = np.ones_like(score_matrix)
    
    for i in range(max_g + 1):
        for j in range(max_g + 1):
            if abs(i - j) >= 3:
                ajuste[i, j] = factor_goleadas
            elif abs(i - j) >= 2:
                ajuste[i, j] = 1 - (1 - factor_goleadas) * 0.5
    
    score_matrix_ajustada = score_matrix * ajuste
    suma = score_matrix_ajustada.sum()
    if suma > 0:
        score_matrix_ajustada = score_matrix_ajustada / suma
    
    return score_matrix_ajustada

def get_fase_color(fase):
    """Obtiene el color asociado a una fase"""
    fase_config = config.FASE_CONFIG.get(fase, config.FASE_CONFIG['Fase de Grupos'])
    return fase_config.get('color', '#22c55e')

def get_fase_descripcion(fase):
    """Obtiene la descripción de una fase"""
    fase_config = config.FASE_CONFIG.get(fase, config.FASE_CONFIG['Fase de Grupos'])
    return fase_config.get('descripcion', '')