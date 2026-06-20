"""Smoke test: the pipeline runs end-to-end on the sample.

Kedro pattern: create a KedroSession and run the pipeline on the sample, ensuring the
node chaining and the catalog are consistent (does not validate model quality).
"""

import pytest


def test_pipeline_runs_end_to_end():
    """The __default__ (or data_prep) pipeline runs without errors on the sample.

    TODO:
      - point the catalog to tests/pipelines/sample/sample.csv (test env or override)
      - bootstrap_project + KedroSession.create(...); session.run(pipeline_name="data_prep")
      - assert the expected outputs exist
    """
    pytest.skip("TODO: implement smoke test once the nodes are ready")
