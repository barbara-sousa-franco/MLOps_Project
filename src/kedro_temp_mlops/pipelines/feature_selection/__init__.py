"""Feature selection via SHAP do champion -> best_cols.pkl.

Corre DEPOIS de model_train (precisa do champion treinado). best_cols volta a alimentar
um retreino via flag `use_feature_selection` no model_train (opção A do blueprint).
"""

from .pipeline import create_pipeline

__all__ = ["create_pipeline"]
