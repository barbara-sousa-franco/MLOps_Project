"""Scheduled Prefect deployments (cron). Copies the professor's example pattern.

TODO deployments:
  - daily drift, weekly training, nightly data tests (see blueprint).
  - use flow.to_deployment(name=..., cron=...) / serve(...) depending on the Prefect version.
"""

import logging

from kedro_prefect_flow import (
    flow_data_unit_tests,
    flow_monitoring,
    flow_training,
    full_pipeline,
)

logger = logging.getLogger(__name__)


def deploy():
    """Create/serve the scheduled deployments.

    TODO deploy:
      - nightly data tests:  cron "0 2 * * *"   -> flow_data_unit_tests
      - daily drift:         cron "0 6 * * *"   -> flow_monitoring
      - weekly training:     cron "0 3 * * 1"   -> flow_training
      - full pipeline:       manual/on-demand    -> full_pipeline
    """
    # TODO: implement (flow.to_deployment(...) + serve(...))
    raise NotImplementedError


if __name__ == "__main__":
    deploy()
