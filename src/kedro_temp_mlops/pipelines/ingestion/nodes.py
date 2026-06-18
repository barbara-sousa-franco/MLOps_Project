"""Nodes da pipeline `ingestion`.

Segue de perto o padrão do exemplo do prof (Hopsworks + Great Expectations inline),
adaptado ao caso das casas (`portugal_listings`). A validação GX aqui é EFÉMERA
(em memória, antes do upload à feature store) — a revalidação "oficial" + semáforo
acontece na pipeline `data_unit_tests`.
"""

import logging

import great_expectations as gx
import pandas as pd
from great_expectations import expectations as gxe

logger = logging.getLogger(__name__)


def _build_between(column: str, rng: dict) -> gxe.ExpectColumnValuesToBeBetween:
    """Constrói um ExpectColumnValuesToBeBetween a partir de {min,max,strict_min,strict_max}.

    NOTA: ignora nulos por defeito (GX só avalia os valores não-nulos).
    """
    return gxe.ExpectColumnValuesToBeBetween(
        column=column,
        min_value=rng.get("min"),
        max_value=rng.get("max"),
        strict_min=rng.get("strict_min", False),
        strict_max=rng.get("strict_max", False),
    )


def build_expectation_suite(
    suite_name: str, feature_group: str, parameters: dict
) -> gx.ExpectationSuite:
    """Constrói uma Great Expectations ExpectationSuite (GX 1.x) por grupo de features.

    As regras vêm todas de `parameters` (parameters_data_unit_tests.yml) — nada hard-coded.

    Args:
        suite_name: Nome da suite a criar.
        feature_group: Um de {"numerical", "categorical", "target"} — define que
            expectativas aplicar.
        parameters: dict de `data_unit_tests` (target_col, column_types, ranges, valid_sets).

    Returns:
        ExpectationSuite (ainda não adicionada a um contexto) pronta a validar.

    Raises:
        ValueError: se `feature_group` não for reconhecido.
    """
    target_col = parameters.get("target_col")
    column_types = parameters.get("column_types", {})
    ranges = parameters.get("ranges", {})
    valid_sets = parameters.get("valid_sets", {})

    expectations: list = []

    if feature_group == "numerical":
        # tipos + intervalos das numéricas (exclui o target, que vai no grupo "target")
        for col, dtype in column_types.items():
            if col == target_col:
                continue
            expectations.append(gxe.ExpectColumnValuesToBeOfType(column=col, type_=dtype))
        for col, rng in ranges.items():
            if col == target_col:
                continue
            expectations.append(_build_between(col, rng))
    elif feature_group == "categorical":
        for col, value_set in valid_sets.items():
            expectations.append(
                gxe.ExpectColumnDistinctValuesToBeInSet(column=col, value_set=list(value_set))
            )
    elif feature_group == "target":
        if target_col in column_types:
            expectations.append(
                gxe.ExpectColumnValuesToBeOfType(column=target_col, type_=column_types[target_col])
            )
        if target_col in ranges:
            expectations.append(_build_between(target_col, ranges[target_col]))
    else:
        raise ValueError(
            f"feature_group desconhecido: {feature_group!r} "
            "(esperado: 'numerical', 'categorical' ou 'target')"
        )

    return gx.ExpectationSuite(name=suite_name, expectations=expectations)


def _run_ephemeral_validation(df: pd.DataFrame, parameters: dict) -> None:
    """Valida `df` em memória (GX efémero) contra as 3 suites; faz `raise` se falhar.

    Proteção ANTES do upload à feature store (padrão do prof). A revalidação "oficial"
    + semáforo é feita depois na pipeline `data_unit_tests`.

    Args:
        df: dataset a validar.
        parameters: dict de `data_unit_tests` (regras vindas do EDA).

    Raises:
        ValueError: se alguma expectation falhar (lista as que falharam no log).
    """
    context = gx.get_context(mode="ephemeral")
    # silenciar as barras de progresso do GX nos logs
    context.variables.progress_bars = {"globally": False, "metric_calculations": False}

    data_source = context.data_sources.add_pandas("ingestion_source")
    asset = data_source.add_dataframe_asset(name="houses")
    batch_definition = asset.add_batch_definition_whole_dataframe("batch")

    failed: list[str] = []
    for group in ("numerical", "categorical", "target"):
        suite = build_expectation_suite(f"ingestion_{group}", group, parameters)
        if not suite.expectations:
            continue
        suite = context.suites.add(suite)
        validation_definition = context.validation_definitions.add(
            gx.ValidationDefinition(data=batch_definition, suite=suite, name=f"vd_{group}")
        )
        result = validation_definition.run(batch_parameters={"dataframe": df})
        if not result.success:
            for r in result["results"]:
                if not r["success"]:
                    cfg = r["expectation_config"]
                    failed.append(f"{cfg['type']}(column={cfg['kwargs'].get('column')})")

    if failed:
        for f in failed:
            logger.error("Validação GX falhou: %s", f)
        raise ValueError(f"Validação GX efémera falhou em {len(failed)} expectation(s): {failed}")

    logger.info("Validação GX efémera: todas as expectations passaram.")


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


def _split_feature_groups(df: pd.DataFrame, primary_key: str, target_col: str) -> dict:
    """Separa o dataset em 3 grupos (numérico/categórico/target) com a primary key.

    Usado pelo upload à feature store (3 feature groups). Cada grupo leva a primary key
    para permitir o join no read_from_feature_store.
    """
    target_df = df[[primary_key, target_col]]
    feature_cols = [c for c in df.columns if c not in (primary_key, target_col)]
    numeric_cols = df[feature_cols].select_dtypes(include="number").columns.tolist()
    categorical_cols = [c for c in feature_cols if c not in numeric_cols]
    return {
        "numerical": df[[primary_key, *numeric_cols]],
        "categorical": df[[primary_key, *categorical_cols]],
        "target": target_df,
    }


def ingestion(df_raw: pd.DataFrame, parameters: dict, validation_params: dict) -> pd.DataFrame:
    """Ingestão dos dados crus das casas + validação GX efémera (+ upload opcional à FS).

    Args:
        df_raw: `raw_house_data` (portugal_listings).
        parameters: parâmetros de ingestão (target_col, primary_key, run_validation,
            to_feature_store...).
        validation_params: regras GX (`data_unit_tests`: column_types, ranges, valid_sets).

    Returns:
        df_full (ingested_data) — dataset completo para as pipelines seguintes.

    Raises:
        ValueError: se a validação GX efémera falhar (proteção antes do upload).

    TODO ingestion (Fase 5 — Hopsworks):
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

    # validação GX EFÉMERA em memória ANTES do upload (proteção; raise se falhar)
    if parameters.get("run_validation", True):
        _run_ephemeral_validation(df, validation_params)

    if parameters.get("to_feature_store", False):
        # separar em 3 grupos para os 3 feature groups da Hopsworks
        groups = _split_feature_groups(df, primary_key, target_col)
        logger.info(
            "Grupos p/ feature store: numerical=%d cols, categorical=%d cols, target=%d cols.",
            groups["numerical"].shape[1],
            groups["categorical"].shape[1],
            groups["target"].shape[1],
        )
        # TODO (Fase 5): upload de cada grupo via to_feature_store(...).
        logger.warning("to_feature_store=True mas o upload Hopsworks ainda não está implementado.")

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
