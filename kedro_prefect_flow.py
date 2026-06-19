"""Prefect flows that call Kedro pipelines by name.

Copies the professor's example pattern. Each flow calls run_pipeline(<name>) from
src/kedro_temp_mlops/run_kedro_pipeline.py.
"""

import logging

from prefect import flow, task

from kedro_temp_mlops.run_kedro_pipeline import run_pipeline

logger = logging.getLogger(__name__)


@task
def _run(pipeline_name: str):
    """Prefect task that runs a Kedro pipeline by name.

    TODO: handle Prefect errors/retries here (retries=, retry_delay_seconds=).
    """
    # TODO: return run_pipeline(pipeline_name)
    raise NotImplementedError


@flow(name="flow_data_unit_tests")
def flow_data_unit_tests():
    """Run data unit tests only (nightly). Gatekeeper via traffic light.

    TODO: _run("data_unit_tests")
    """
    raise NotImplementedError


@flow(name="flow_data_prep")
def flow_data_prep():
    """ingestion + data_unit_tests + preprocessing_train + split_data.

    TODO: _run("data_prep")  (only if traffic light is OK)
    """
    raise NotImplementedError


@flow(name="flow_training")
def flow_training():
    """model_selection + model_train + feature_selection.

    TODO: _run("training")
    """
    raise NotImplementedError


@flow(name="flow_inference")
def flow_inference():
    """preprocessing_batch + model_predict.

    TODO: _run("inference")
    """
    raise NotImplementedError


@flow(name="flow_monitoring")
def flow_monitoring():
    """data_drift (reference vs new batch).

    TODO: _run("monitoring")
    """
    raise NotImplementedError


@flow(name="full_pipeline")
def full_pipeline():
    """Full sequence (__default__).

    TODO: chain data_prep -> training -> inference -> monitoring -> reporting,
          respecting the traffic light between data_unit_tests and the rest.
    """
    raise NotImplementedError


if __name__ == "__main__":
    # TODO: full_pipeline()
    pass
