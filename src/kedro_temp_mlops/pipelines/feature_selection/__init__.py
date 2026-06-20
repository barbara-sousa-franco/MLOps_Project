"""Feature selection via SHAP of the champion -> best_cols.pkl.

Runs AFTER model_train (needs the trained champion). best_cols feeds back into a retrain
via the `use_feature_selection` flag in model_train (blueprint option A).
"""

from .pipeline import create_pipeline

__all__ = ["create_pipeline"]
