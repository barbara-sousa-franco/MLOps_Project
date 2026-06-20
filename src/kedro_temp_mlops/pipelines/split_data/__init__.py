"""Split data: carves the out-of-sample batch (ana_data) off the ingested data.

ref_data is the training pool; ana_data is the held-out batch for drift/inference."""

from .pipeline import create_pipeline

__all__ = ["create_pipeline"]
