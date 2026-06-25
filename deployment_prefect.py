"""Scheduled Prefect deployments (cron). Mirrors the professor's bank_example pattern.

Run this file to register + serve the deployments; Prefect then triggers each flow on its
schedule. Stop with Ctrl+C.

    uv run python deployment_prefect.py
"""

from prefect import serve
from prefect.client.schemas.schedules import CronSchedule

from kedro_prefect_flow import (
    flow_data_unit_tests,
    flow_monitoring,
    flow_training,
    full_pipeline,
)

if __name__ == "__main__":
    print("Building deployments...")

    # nightly data-quality tests (22:00)
    dep_tests = flow_data_unit_tests.to_deployment(
        name="data-tests-nightly",
        schedule=CronSchedule(cron="0 22 * * *", timezone="Europe/Lisbon"),
        tags=["data-quality"],
    )

    # daily drift monitoring (06:00)
    dep_drift = flow_monitoring.to_deployment(
        name="drift-daily",
        schedule=CronSchedule(cron="0 6 * * *", timezone="Europe/Lisbon"),
        tags=["monitoring"],
    )

    # weekly retraining (Mondays 06:00)
    dep_train = flow_training.to_deployment(
        name="training-weekly",
        schedule=CronSchedule(cron="0 6 * * 1", timezone="Europe/Lisbon"),
        tags=["training"],
    )

    # full pipeline — on demand (no schedule), with the traffic-light gate
    dep_full = full_pipeline.to_deployment(name="full-pipeline-on-demand")

    print("Serving deployments (Ctrl+C to stop)...")
    serve(dep_tests, dep_drift, dep_train, dep_full)
