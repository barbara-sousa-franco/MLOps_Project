"""Ingestion pipeline: raw -> (Great Expectations efémero) -> feature store -> ler de volta."""

from .pipeline import create_pipeline

__all__ = ["create_pipeline"]
