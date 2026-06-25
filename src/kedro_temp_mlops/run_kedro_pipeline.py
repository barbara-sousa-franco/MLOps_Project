"""KedroSession wrapper so Prefect can run Kedro pipelines by name.

This is the bridge Prefect -> Kedro: a plain Python function that bootstraps the project
and runs a named pipeline inside a KedroSession. Follows the professor's bank_example pattern.
"""

import logging
import os
from pathlib import Path

from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project

logger = logging.getLogger(__name__)

# project root = two levels up from this file (src/kedro_temp_mlops/run_kedro_pipeline.py)
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def run_pipeline(pipeline_name: str = "__default__"):
    """Run a Kedro pipeline by name inside a KedroSession.

    Args:
        pipeline_name: a name registered in pipeline_registry (e.g. "data_prep", "training",
            "inference", "monitoring", "__default__").

    Returns:
        The output of session.run().
    """
    # run from the project root so the catalog's relative `data/...` paths resolve correctly
    os.chdir(PROJECT_ROOT)
    bootstrap_project(PROJECT_ROOT)
    logger.info("Running Kedro pipeline '%s'...", pipeline_name)
    with KedroSession.create(project_path=PROJECT_ROOT) as session:
        return session.run(pipeline_name=pipeline_name)
