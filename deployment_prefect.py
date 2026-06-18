"""Deployments Prefect agendados (cron). Copia o padrão do exemplo do prof.

TODO deployments:
  - drift diário, treino semanal, data tests nightly (ver blueprint).
  - usar flow.to_deployment(name=..., cron=...) / serve(...) conforme a versão do Prefect.
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
    """Cria/serve os deployments agendados.

    TODO deploy:
      - data tests nightly:   cron "0 2 * * *"   -> flow_data_unit_tests
      - drift diário:         cron "0 6 * * *"   -> flow_monitoring
      - treino semanal:       cron "0 3 * * 1"   -> flow_training
      - full pipeline:        manual/on-demand   -> full_pipeline
    """
    # TODO: implementar (flow.to_deployment(...) + serve(...))
    raise NotImplementedError


if __name__ == "__main__":
    deploy()
