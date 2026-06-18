"""Wrapper de KedroSession para o Prefect chamar pipelines por nome.

Segue o padrão do exemplo bank_example do prof.
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def run_pipeline(pipeline_name: str):
    """Corre uma pipeline Kedro por nome dentro de uma KedroSession.

    Args:
        pipeline_name: nome registado em pipeline_registry (ex: "data_prep", "training").

    Returns:
        O output de session.run().

    TODO run_pipeline:
      - bootstrap_project(PROJECT_ROOT)
      - with KedroSession.create(project_path=PROJECT_ROOT) as session:
            return session.run(pipeline_name=pipeline_name)
      - é o que o Prefect chama (igual ao padrão bank_example)
    """
    # PROJECT_ROOT = Path(__file__).resolve().parents[2]
    # TODO: implementar
    raise NotImplementedError
