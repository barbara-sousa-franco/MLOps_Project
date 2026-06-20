"""Ingestion pipeline: raw -> (ephemeral Great Expectations) -> feature store -> read back."""

from .pipeline import create_pipeline

__all__ = ["create_pipeline"]
