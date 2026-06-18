"""Model train: treina champion + MLflow Model Registry (champion/challenger).

O Registry é construção VOSSA — o exemplo do prof versiona só por pickle + autolog,
sem registry. É aqui que ganham pontos de criatividade.
"""

from .pipeline import create_pipeline

__all__ = ["create_pipeline"]
