"""Preprocessing (treino): limpeza + feature engineering + encoding. Guarda encoder.pkl.

ANTI-LEAKAGE: o encoder/scaler é `fit` SÓ aqui (treino). O batch faz só `transform`.
"""

from .pipeline import create_pipeline

__all__ = ["create_pipeline"]
