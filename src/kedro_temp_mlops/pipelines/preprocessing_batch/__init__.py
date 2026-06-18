"""Preprocessing (batch novo): aplica o MESMO encoder do treino (transform, sem refit)."""

from .pipeline import create_pipeline

__all__ = ["create_pipeline"]
