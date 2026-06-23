"""Nodes for the `ingestion` pipeline.

Closely follows the professor's example pattern (Hopsworks + Great Expectations inline),
adapted to the housing use case (`portugal_listings`). GX validation here is EPHEMERAL
(in memory, before upload to the feature store) — the "official" revalidation + traffic
light happens in the `data_unit_tests` pipeline.
"""

import logging

import great_expectations as gx
import pandas as pd
from great_expectations import expectations as gxe
from kedro_temp_mlops.utils import build_expectation_suite, _build_between

logger = logging.getLogger(__name__)



def _run_ephemeral_validation(df: pd.DataFrame, parameters: dict) -> None:
    """Validate `df` in memory (ephemeral GX) against the 3 suites; raises if it fails.

    Protection BEFORE upload to the feature store (professor's pattern). The "official"
    revalidation + traffic light is performed afterwards in the `data_unit_tests` pipeline.

    Args:
        df: dataset to validate.
        parameters: dict from `data_unit_tests` (rules derived from EDA).

    Raises:
        ValueError: if any expectation fails (logs the failing ones).
    """
    context = gx.get_context(mode="ephemeral")
    # suppress GX progress bars in logs
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
            logger.error("GX validation failed: %s", f)
        raise ValueError(f"Ephemeral GX validation failed on {len(failed)} expectation(s): {failed}")

    logger.info("Ephemeral GX validation: all expectations passed.")


def to_feature_store(
    data: pd.DataFrame,
    group_name: str,
    version: int,
    description: str,
    feature_descriptions: dict,
    credentials: dict,
):
    """Upload a feature group to the Hopsworks Feature Store.

    Args:
        data: DataFrame for the group (numerical / categorical / target) with the primary key.
        group_name: Name of the feature group in Hopsworks.
        version: Feature group version.
        description: Feature group description.
        feature_descriptions: Dict column->description (update_feature_description).
        credentials: {"api_key": ..., "project": ...} from conf/local/credentials.yml.

    Returns:
        The created/updated feature group object.

    TODO to_feature_store:
      - hopsworks.login(api_key=credentials["api_key"], project=credentials["project"])
      - fs = project.get_feature_store()
      - get_or_create_feature_group(primary_key=["index"], event_time="PublishDate")
      - feature_group.insert(data)
      - update_feature_description(...) per column
      - compute_statistics()
    """
    # TODO: implement
    raise NotImplementedError


def _split_feature_groups(df: pd.DataFrame, primary_key: str, target_col: str) -> dict:
    """Split the dataset into 3 groups (numerical/categorical/target) with the primary key.

    Used for the feature store upload (3 feature groups). Each group carries the primary key
    to allow joins in read_from_feature_store.
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
    """Ingest raw housing data + ephemeral GX validation (+ optional feature store upload).

    Args:
        df_raw: `raw_house_data` (portugal_listings).
        parameters: ingestion parameters (target_col, primary_key, run_validation,
            to_feature_store...).
        validation_params: GX rules (`data_unit_tests`: column_types, ranges, valid_sets).

    Returns:
        df_full (ingested_data) — complete dataset for downstream pipelines.

    Raises:
        ValueError: if ephemeral GX validation fails (protection before upload).

    TODO ingestion (Phase 5 — Hopsworks):
      - use PublishDate as event_time (NOTE: ~78% nulls — see ASSUMPTIONS)
      - if parameters["to_feature_store"]: call to_feature_store() for the 3 groups
        (requires Hopsworks credentials — re-add the "credentials" input to the node then)
    """
    target_col = parameters["target_col"]
    primary_key = parameters.get("primary_key", "index")

    df = df_raw.copy()

    # stable primary key for the feature store / joins (event_time = PublishDate, Phase 5)
    if primary_key not in df.columns:
        df = df.reset_index(names=primary_key)

    # target must exist (Price has ~0.2% nulls in raw)
    n_before = len(df)
    df = df.dropna(subset=[target_col]).reset_index(drop=True)
    logger.info(
        "Ingestion: %d rows (dropped %d without '%s').",
        len(df),
        n_before - len(df),
        target_col,
    )

    # EPHEMERAL GX validation in memory BEFORE upload (protection; raises if it fails)
    if parameters.get("run_validation", True):
        _run_ephemeral_validation(df, validation_params)

    if parameters.get("to_feature_store", False):
        # split into 3 groups for the 3 Hopsworks feature groups
        groups = _split_feature_groups(df, primary_key, target_col)
        logger.info(
            "Feature store groups: numerical=%d cols, categorical=%d cols, target=%d cols.",
            groups["numerical"].shape[1],
            groups["categorical"].shape[1],
            groups["target"].shape[1],
        )
        # TODO (Phase 5): upload each group via to_feature_store(...).
        logger.warning("to_feature_store=True but Hopsworks upload is not yet implemented.")

    return df


# NOTE: the ref/ana split was moved to the `split_data` pipeline (split_out_of_sample),
# which is the canonical version (supports the 'biased' strategy for the drift demo). The old
# `split_reference_analysis` was removed from here to eliminate the duplication.


def read_from_feature_store(parameters: dict, credentials: dict) -> pd.DataFrame:
    """Read data back from the Feature Store (demonstrates the write->read cycle).

    Args:
        parameters: ingestion parameters (feature group names/versions).
        credentials: Hopsworks credentials.

    Returns:
        DataFrame reconstructed from the feature groups (ingested_data).

    TODO read_from_feature_store:
      - hopsworks.login(...); fs.get_feature_group(...)
      - join the 3 groups by primary key "index"
      - demonstrates the write->read cycle from the professor's tip
    """
    # TODO: implement
    raise NotImplementedError
