"""Split data: carves the out-of-sample batch (test_data) off the ingested data.

learning_data is the training pool; test_data is the held-out batch for drift/inference."""

from .pipeline import create_pipeline

__all__ = ["create_pipeline"]
