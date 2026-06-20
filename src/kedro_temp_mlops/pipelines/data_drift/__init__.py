"""Data drift: compares a new batch (analysis) vs the reference (training). nannyml/evidently."""

from .pipeline import create_pipeline

__all__ = ["create_pipeline"]
