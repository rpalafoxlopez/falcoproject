# modules/config.py - Configuraciones globales
import sys

# Parámetros del modelo
DIXON_COLES_RHO = -0.13
MAX_GOALS_DEFAULT = 8

# Ventanas de entrenamiento
TRAIN_WINDOWS = ["2018-01-01", "2016-01-01", "2014-01-01", "2010-01-01"]
TRAIN_WINDOW_DEFAULT = "2018-01-01"

# Dependencias (solo para uso interno, no se muestra en UI)
PYMC_AVAILABLE = False
SKLEARN_AVAILABLE = False

try:
    import pymc as pm
    import arviz as az
    PYMC_AVAILABLE = True
except ImportError:
    pass

try:
    import sklearn
    SKLEARN_AVAILABLE = True
except ImportError:
    pass

# Versión de Python (solo para referencia interna)
PYTHON_VERSION = sys.version.split()[0]


# ============================================================================
# CONFIGURACIÓN POR FASE DEL TORNEO (basado en datos históricos 2010-2022)
# ============================================================================

FASE_CONFIG = {
    'Fase de Grupos': {
        'factor_goles': 1.00,
        'factor_empates': 1.00,
        'factor_goleadas': 1.00,
        'rho_dixon_coles': -0.13,
        'factor_high_scoring': 1.00,
        'descripcion': 'Partidos de grupos, mayor variabilidad de resultados',
        'color': '#22c55e'  # Verde
    },
    'Octavos de Final': {
        'factor_goles': 0.92,
        'factor_empates': 1.15,
        'factor_goleadas': 0.75,
        'rho_dixon_coles': -0.15,
        'factor_high_scoring': 0.85,
        'descripcion': 'Eliminación directa, equipos más cautelosos, menos goleadas',
        'color': '#3b82f6'  # Azul
    },
    'Cuartos de Final': {
        'factor_goles': 0.88,
        'factor_empates': 1.20,
        'factor_goleadas': 0.65,
        'rho_dixon_coles': -0.17,
        'factor_high_scoring': 0.75,
        'descripcion': 'Partidos muy intensos, empates más probables, pocos goles',
        'color': '#8b5cf6'  # Morado
    },
    'Semifinales': {
        'factor_goles': 0.85,
        'factor_empates': 1.25,
        'factor_goleadas': 0.55,
        'rho_dixon_coles': -0.18,
        'factor_high_scoring': 0.70,
        'descripcion': 'Máxima intensidad, partidos muy cerrados, pocas goleadas',
        'color': '#ec4899'  # Rosa
    },
    'Final': {
        'factor_goles': 0.82,
        'factor_empates': 1.30,
        'factor_goleadas': 0.50,
        'rho_dixon_coles': -0.20,
        'factor_high_scoring': 0.65,
        'descripcion': 'El partido más importante, muchos empates, pocos goles',
        'color': '#f59e0b'  # Amarillo
    }
}

FASES = list(FASE_CONFIG.keys())
FASE_DEFAULT = 'Fase de Grupos'