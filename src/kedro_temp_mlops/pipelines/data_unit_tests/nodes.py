"""Nodes da pipeline `data_unit_tests`.

Revalida o que sai da feature store com Great Expectations 1.x e ESCREVE O SEMÁFORO
(melhoria vossa, não está no exemplo do prof). As próximas pipelines verificam a flag
antes de correr (gatekeeping da dica do prof).
"""

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def get_validation_results(validation_results) -> pd.DataFrame:
    """Faz parse do objeto de resultados do GX para um DataFrame legível.

    Args:
        validation_results: objeto devolvido pela validação Great Expectations.

    Returns:
        DataFrame com colunas: Success, Expectation Type, Column,
        Unexpected Count, Unexpected Percent, Observed Value.
        (vai para `reporting_tests` -> data_tests.csv)

    TODO get_validation_results:
      - copiar/adaptar do exemplo: iterar results["results"]
      - extrair expectation_config.type, kwargs["column"], result counts
      - devolver DataFrame tabular para reporting
    """
    # TODO: implementar
    raise NotImplementedError


def unit_test(ingested_data: pd.DataFrame, parameters: dict) -> pd.DataFrame:
    """Corre os asserts de qualidade das casas com GX 1.x (contexto efémero).

    Args:
        ingested_data: dataset vindo da ingestion/feature store.
        parameters: limites/sets válidos (parameters_data_unit_tests.yml).

    Returns:
        df_validation — DataFrame de resultados (via get_validation_results).

    TODO unit_test:
      - GX 1.x: contexto efémero, Data Source pandas, ValidationDefinition,
        ExpectationSuite
      - VÁRIOS asserts das casas (do EDA):
          * ExpectColumnValuesToBeOfType: Price float, áreas float, ConstructionYear int
          * ExpectColumnMinToBeBetween: Price min > 0; áreas min >= 0
          * ConstructionYear entre 1800 e 2026
          * ExpectColumnDistinctValuesToBeInSet: Type, EnergyCertificate em sets válidos
      - se results.success == False: log de CADA expectativa falhada + raise ValueError
      - devolver df_validation (get_validation_results(results))
    """
    # TODO: implementar
    raise NotImplementedError


def write_traffic_light(df_validation: pd.DataFrame, parameters: dict) -> str:
    """Escreve o ficheiro semáforo se todos os asserts críticos passarem (MELHORIA vossa).

    Args:
        df_validation: resultados de `unit_test`.
        parameters: config do semáforo (path da flag, lista de asserts críticos).

    Returns:
        Caminho/conteúdo do ficheiro semáforo escrito (ex: PIPELINE_OK.flag).

    TODO write_traffic_light:   # não existe no exemplo do prof — é onde ganham pontos
      - se todos os asserts críticos passam -> escrever ficheiro semáforo
        (ex: data/08_reporting/PIPELINE_OK.flag com timestamp)
      - próximas pipelines verificam a flag antes de correr (gatekeeping)
      - se algum crítico falha -> NÃO escrever (ou escrever PIPELINE_FAIL.flag)
    """
    # TODO: implementar
    raise NotImplementedError
