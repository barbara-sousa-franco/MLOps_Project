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


def ingestion(df_raw: pd.DataFrame, parameters: dict) -> pd.DataFrame:
    """Ingestão dos dados crus das casas.

    Versão MÍNIMA funcional (Fase 0): cria a primary key `index`, descarta linhas sem
    target e devolve o dataset. A validação GX efémera, a separação em 3 grupos e o
    upload opcional à Hopsworks ficam por implementar (Fase 1/5).

    Args:
        df_raw: `raw_house_data` (portugal_listings).
        parameters: parâmetros de ingestão (target_col, primary_key, to_feature_store...).

    Returns:
        df_full (ingested_data) — dataset completo para as pipelines seguintes.

    TODO ingestion (Fase 1/5):
      - separar em 3 grupos: numéricos, categóricos, target (Price)
      - validação GX EFÉMERA em memória ANTES do upload (build_expectation_suite +
        ValidationDefinition); se falhar -> raise (como o prof)
      - usar PublishDate como event_time (NOTA: ~78% nulos — ver ASSUMPTIONS)
      - se parameters["to_feature_store"]: to_feature_store() dos 3 grupos (precisa de
        credentials Hopsworks — re-adicionar o input "credentials" ao nó nessa altura)
    """
    target_col = parameters["target_col"]
    primary_key = parameters.get("primary_key", "index")

    df = df_raw.copy()

    # primary key estável para a feature store / joins (event_time = PublishDate, Fase 5)
    if primary_key not in df.columns:
        df = df.reset_index(names=primary_key)

    # o target tem de existir (Price tem ~0.2% nulos no raw)
    n_before = len(df)
    df = df.dropna(subset=[target_col]).reset_index(drop=True)
    logger.info(
        "Ingestion: %d linhas (descartadas %d sem '%s').",
        len(df),
        n_before - len(df),
        target_col,
    )

    if parameters.get("to_feature_store", False):
        # TODO (Fase 5): upload dos grupos para Hopsworks via to_feature_store().
        logger.warning("to_feature_store=True mas o upload ainda não está implementado.")

    return df


def split_reference_analysis(ingested_data: pd.DataFrame, parameters: dict):
    """Parte o dataset ingerido em referência (baseline) e batch de análise.

    `ref_data` representa a distribuição de referência (treino) e `ana_data` o "batch
    novo" que alimenta o drift e a inferência.

    NOTA: `PublishDate` está ~78% nula, por isso um split TEMPORAL é inviável — usa-se um
    split ALEATÓRIO reprodutível (por `seed`). Ver ASSUMPTIONS.md.

    Args:
        ingested_data: saída de `ingestion`.
        parameters: usa `reference_fraction` (fração para referência) e `seed`.

    Returns:
        Tuple (ref_data, ana_data).

    TODO (extra criatividade, Fase 3): injetar drift artificial em `ana_data` para
    demonstrar a deteção de drift (sugestão do prof).
    """
    ref_fraction = parameters.get("reference_fraction", 0.8)
    seed = parameters["seed"]

    ref_data = ingested_data.sample(frac=ref_fraction, random_state=seed)
    ana_data = ingested_data.drop(index=ref_data.index)

    ref_data = ref_data.reset_index(drop=True)
    ana_data = ana_data.reset_index(drop=True)
    logger.info(
        "Split ref/ana: ref_data=%d linhas, ana_data=%d linhas (frac=%.2f, seed=%d).",
        len(ref_data),
        len(ana_data),
        ref_fraction,
        seed,
    )
    return ref_data, ana_data


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
