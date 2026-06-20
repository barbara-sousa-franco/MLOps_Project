"""Data unit tests pipeline: quality asserts (Great Expectations) + WRITES THE TRAFFIC LIGHT.

Do NOT confuse with the `tests/` folder (pytest). This validates DATA QUALITY.
"""

from .pipeline import create_pipeline

__all__ = ["create_pipeline"]
