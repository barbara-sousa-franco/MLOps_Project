"""Smoke test: a pipeline corre end-to-end na sample.

Padrão Kedro: cria um KedroSession e corre a pipeline na sample, garantindo que o
encadeamento de nós e o catalog estão coerentes (não valida qualidade do modelo).
"""

import pytest


def test_pipeline_runs_end_to_end():
    """A pipeline __default__ (ou data_prep) corre sem erros na sample.

    TODO:
      - apontar o catalog para tests/pipelines/sample/sample.csv (env de teste ou override)
      - bootstrap_project + KedroSession.create(...); session.run(pipeline_name="data_prep")
      - assert que os outputs esperados existem
    """
    pytest.skip("TODO: implementar smoke test quando os nós estiverem prontos")
