"""Data unit tests pipeline: asserts de qualidade (Great Expectations) + ESCREVE SEMÁFORO.

NÃO confundir com a pasta `tests/` (pytest). Aqui testa-se a QUALIDADE DOS DADOS.
"""

from .pipeline import create_pipeline

__all__ = ["create_pipeline"]
