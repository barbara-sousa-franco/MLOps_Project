
"""Smoke test: data_prep runs end-to-end on the sample CSV (test env)."""

from pathlib import Path

import pytest
from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project


@pytest.mark.slow
def test_data_prep_runs_end_to_end():
    """data_prep wires up and executes without errors on the sample."""
    bootstrap_project(Path.cwd())
    with KedroSession.create(env="test") as session:
        output = session.run(pipeline_name="data_prep")
    assert output is not None



