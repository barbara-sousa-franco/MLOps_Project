"""Prefect flows that orchestrate the Kedro pipelines.

Each flow runs one Kedro composition via run_pipeline(); `full_pipeline` chains them and
GATES the modelling steps on the data-quality traffic light (only proceeds if green).
Mirrors the professor's bank_example pattern.
"""

import sys
from pathlib import Path

from prefect import flow, get_run_logger, task

# make the src-layout package importable when running this file directly
PROJECT_ROOT = Path(__file__).parent
sys.path.append(str(PROJECT_ROOT / "src"))

from kedro_temp_mlops.run_kedro_pipeline import run_pipeline  # noqa: E402

REPORTING = PROJECT_ROOT / "data" / "08_reporting"


# ----------------------------------------------------------------------------
# task: run one Kedro pipeline
# ----------------------------------------------------------------------------
@task(retries=1, retry_delay_seconds=10)
def run_kedro_task(pipeline_name: str):
    """Run a Kedro pipeline by name (with a retry on transient failure)."""
    logger = get_run_logger()
    logger.info("Running Kedro pipeline: %s", pipeline_name)
    run_pipeline(pipeline_name)
    logger.info("Kedro pipeline '%s' finished successfully.", pipeline_name)


# ----------------------------------------------------------------------------
# traffic-light gate (data quality gatekeeping)
# ----------------------------------------------------------------------------
def _clear_flags():
    """Remove stale traffic-light flags before a fresh data_prep run."""
    for f in REPORTING.glob("*.flag"):
        f.unlink(missing_ok=True)


def _traffic_light_is_green() -> bool:
    """Green when no *_FAIL.flag exists (data_unit_tests passed all suites)."""
    return not any(REPORTING.glob("*_FAIL.flag"))


# ----------------------------------------------------------------------------
# one flow per Kedro composition
# ----------------------------------------------------------------------------
@flow(name="data_unit_tests")
def flow_data_unit_tests():
    run_kedro_task("data_unit_tests")


@flow(name="data_prep")
def flow_data_prep():
    run_kedro_task("data_prep")


@flow(name="training")
def flow_training():
    run_kedro_task("training")


@flow(name="inference")
def flow_inference():
    run_kedro_task("inference")


@flow(name="monitoring")
def flow_monitoring():
    run_kedro_task("monitoring")


# ----------------------------------------------------------------------------
# orchestration flow: data_prep -> [traffic-light gate] -> training -> inference -> monitoring
# ----------------------------------------------------------------------------
@flow(name="full_pipeline")
def full_pipeline():
    logger = get_run_logger()

    _clear_flags()
    run_kedro_task("data_prep")  # includes data_unit_tests + writes the traffic-light flags

    if not _traffic_light_is_green():
        logger.error("Traffic light is RED — data quality gate failed; stopping before training.")
        raise RuntimeError("Data quality gate failed (red traffic light).")
    logger.info("Traffic light is GREEN — proceeding to training.")

    run_kedro_task("training")
    run_kedro_task("inference")
    run_kedro_task("monitoring")
    logger.info("Full pipeline finished.")


if __name__ == "__main__":
    # quick manual run (use full_pipeline() for the whole chain)
    flow_data_prep()
