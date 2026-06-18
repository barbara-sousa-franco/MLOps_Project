"""Flows Prefect que chamam as pipelines Kedro por nome.

Copia o padrão do exemplo do prof. Cada flow chama run_pipeline(<nome>) do
src/kedro_temp_mlops/run_kedro_pipeline.py.
"""

import logging

from prefect import flow, task

from kedro_temp_mlops.run_kedro_pipeline import run_pipeline

logger = logging.getLogger(__name__)


@task
def _run(pipeline_name: str):
    """Task Prefect que corre uma pipeline Kedro por nome.

    TODO: tratar erros/retries do Prefect aqui (retries=, retry_delay_seconds=).
    """
    # TODO: return run_pipeline(pipeline_name)
    raise NotImplementedError


@flow(name="flow_data_unit_tests")
def flow_data_unit_tests():
    """Corre só os data unit tests (nightly). Gatekeeper via semáforo.

    TODO: _run("data_unit_tests")
    """
    raise NotImplementedError


@flow(name="flow_data_prep")
def flow_data_prep():
    """ingestion + data_unit_tests + preprocessing_train + split_data.

    TODO: _run("data_prep")  (só se semáforo OK)
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
    """data_drift (referência vs batch novo).

    TODO: _run("monitoring")
    """
    raise NotImplementedError


@flow(name="full_pipeline")
def full_pipeline():
    """Sequência completa (__default__).

    TODO: encadear data_prep -> training -> inference -> monitoring -> reporting,
          respeitando o semáforo entre data_unit_tests e o resto.
    """
    raise NotImplementedError


if __name__ == "__main__":
    # TODO: full_pipeline()
    pass
