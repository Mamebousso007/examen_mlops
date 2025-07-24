"""
Module de configuration du logging pour l'API House Price Prediction
"""

from .logging_config import setup_logging

# Si get_logger n'existe pas dans logging_config.py, on ne l'importe pas.
# Sinon, décommente la ligne suivante :
# from .logging_config import get_logger

from .filters import PredictionFilter, APIFilter

__all__ = [
    "setup_logging",
    # "get_logger",  # Décommente cette ligne si get_logger est bien défini dans logging_config.py
    "PredictionFilter",
    "APIFilter"
]
