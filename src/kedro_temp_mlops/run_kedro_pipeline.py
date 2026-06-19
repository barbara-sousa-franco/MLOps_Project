"""KedroSession wrapper for Prefect to call pipelines by name.

Follows the pattern from the professor's bank_example.
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def run_pipeline(pipeline_name: str):
    """Run a Kedro pipeline by name inside a KedroSession.

    Args:
        pipeline_name: name registered in pipeline_registry (e.g. "data_prep", "training").

    Returns:
        The output of session.run().

    TODO run_pipeline:
      - bootstrap_project(PROJECT_ROOT)
      - with KedroSession.create(project_path=PROJECT_ROOT) as session:
            return session.run(pipeline_name=pipeline_name)
      - this is what Prefect calls (same pattern as bank_example)
    """
    # PROJECT_ROOT = Path(__file__).resolve().parents[2]
    # TODO: implement
    raise NotImplementedError
