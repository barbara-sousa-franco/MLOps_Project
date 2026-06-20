"""Model train: trains the champion + MLflow Model Registry (champion/challenger).

The Registry is our own addition — the reference example versions only via pickle +
autolog, without a registry.
"""

from .pipeline import create_pipeline

__all__ = ["create_pipeline"]
