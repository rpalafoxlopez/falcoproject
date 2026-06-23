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