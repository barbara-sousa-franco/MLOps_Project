"""Smoke test: data_prep runs end-to-end on the sample CSV (test env)."""

from pathlib import Path

import pytest
from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project


@pytest.mark.slow
def test_data_prep_runs_end_to_end():
    """data_prep wires up and executes without errors on the sample.

    Uses env=test so the catalog points to tests/pipelines/sample/sample.csv
    instead of the full dataset.
    Verifies that the key intermediate outputs are produced.
    """
    bootstrap_project(Path.cwd())
    with KedroSession.create(env="test") as session:
        output = session.run(pipeline_name="data_prep")

    assert output is not None

    # key outputs that downstream pipelines depend on
    expected = [
        Path("data/02_intermediate/learning_data.csv"),
        Path("data/02_intermediate/test_data.csv"),
        Path("data/03_primary/X_train.csv"),
        Path("data/03_primary/X_val.csv"),
        Path("data/03_primary/y_train.csv"),
        Path("data/03_primary/y_val.csv"),
    ]
    for path in expected:
        assert path.exists(), f"Expected output not found: {path}"
