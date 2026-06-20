"""Model selection: challengers (several models) + Optuna tuning. Picks the best."""

from .pipeline import create_pipeline

__all__ = ["create_pipeline"]
