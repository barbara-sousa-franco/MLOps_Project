"""Split data: train/test split. O test set é sagrado — não tocar até avaliação final."""

from .pipeline import create_pipeline

__all__ = ["create_pipeline"]
