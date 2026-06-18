"""Nodes da pipeline `ingestion`.

Segue de perto o padrão do exemplo do prof (Hopsworks + Great Expectations inline),
adaptado ao caso das casas (`portugal_listings`). A validação GX aqui é EFÉMERA
(em memória, antes do upload à feature store) — a revalidação "oficial" + semáforo
acontece na pipeline `data_unit_tests`.
"""

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def build_expectation_suite(suite_name: str, feature_group: str, parameters: dict):
    """Constrói uma Great Expectations ExpectationSuite (GX 1.x) por grupo de features.

    Args:
        suite_name: Nome da suite a criar.
        feature_group: Um de {"numerical", "categorical", "target"} — define que
            expectativas aplicar.
        parameters: Limites/conjuntos válidos vindos do EDA (parameters_data_unit_tests.yml).

    Returns:
        ExpectationSuite pronta a usar numa ValidationDefinition.

    TODO build_expectation_suite:
      - usar Great Expectations 1.x: ExpectColumnValuesToBeOfType,
        ExpectColumnMinToBeBetween, ExpectColumnDistinctValuesToBeInSet
        (ferramenta da aula, pedida pelo enunciado)
      - suites por grupo, regras vindas do EDA:
          * numéricas: GrossArea/TotalArea/LivingArea >= 0, ConstructionYear entre
            1800 e 2026, Price > 0
          * categóricas: District/City/Type/EnergyCertificate em conjuntos válidos
          * target: Price do tipo float, > 0
      - ler limites/sets de `parameters` (nada hard-coded)
    """
    # TODO: implementar
    raise NotImplementedError


def to_feature_store(
    data: pd.DataFrame,
    group_name: str,
    version: int,
    description: str,
    feature_descriptions: dict,
    credentials: dict,
):
    """Faz upload de um grupo de features para a Hopsworks Feature Store.

    Args:
        data: DataFrame do grupo (numérico / categórico / target) com a primary key.
        group_name: Nome do feature group em Hopsworks.
        version: Versão do feature group.
        description: Descrição do feature group.
        feature_descriptions: Dict coluna->descrição (update_feature_description).
        credentials: {"api_key": ..., "project": ...} vindo de conf/local/credentials.yml.

    Returns:
        O objeto feature group criado/atualizado.

    TODO to_feature_store:
      - hopsworks.login(api_key=credentials["api_key"], project=credentials["project"])
      - fs = project.get_feature_store()
      - get_or_create_feature_group(primary_key=["index"], event_time="PublishDate")
      - feature_group.insert(data)
      - update_feature_description(...) por coluna
      - compute_statistics()
    """
    # TODO: implementar
    raise NotImplementedError


def ingestion(df_raw: pd.DataFrame, parameters: dict, credentials: dict) -> pd.DataFrame:
    """Ingestão dos dados crus das casas + validação GX efémera + upload opcional à FS.

    Args:
        df_raw: `raw_house_data` (portugal_listings).
        parameters: parâmetros de ingestão (inclui flag `to_feature_store`).
        credentials: credenciais Hopsworks.

    Returns:
        df_full (ingested_data) — dataset completo para as pipelines seguintes.

    TODO ingestion:
      - (se houver dataset adicional, merge; senão usar só portugal_listings)
      - reset_index(names="index"); usar PublishDate como event_time
        (vantagem vs exemplo: já existe PublishDate, não é preciso inventar datetime)
      - separar em 3 grupos: numéricos, categóricos, target (Price)
      - validação GX EFÉMERA em memória ANTES do upload (build_expectation_suite +
        ValidationDefinition); se falhar -> raise (como o prof)
      - se parameters["to_feature_store"]: to_feature_store() dos 3 grupos
      - devolver df_full
    """
    # TODO: implementar
    raise NotImplementedError


def read_from_feature_store(parameters: dict, credentials: dict) -> pd.DataFrame:
    """Lê os dados de volta da Feature Store (demonstra o ciclo write->read).

    Args:
        parameters: parâmetros de ingestão (nomes/versões dos feature groups).
        credentials: credenciais Hopsworks.

    Returns:
        DataFrame reconstruído a partir dos feature groups (ingested_data).

    TODO read_from_feature_store:
      - hopsworks.login(...); fs.get_feature_group(...)
      - juntar os 3 grupos pela primary key "index"
      - demonstra o ciclo write->read da dica do prof
    """
    # TODO: implementar
    raise NotImplementedError
