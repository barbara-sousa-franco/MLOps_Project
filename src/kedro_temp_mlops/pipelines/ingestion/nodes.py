"""Nodes for the `ingestion` pipeline.

Closely follows the professor's example pattern (Hopsworks + Great Expectations inline),
adapted to the housing use case (`portugal_listings`). GX validation here is EPHEMERAL
(in memory, before upload to the feature store) — the "official" revalidation + traffic
light happens in the `data_unit_tests` pipeline.
"""


import logging
from pathlib import Path
from typing import Any, Dict

import great_expectations as gx
import hopsworks
import pandas as pd

from kedro.config import OmegaConfigLoader
from kedro.framework.project import settings

# module-level credentials load (his proven pattern)
conf_loader = OmegaConfigLoader(conf_source=str(Path("") / settings.CONF_SOURCE))
credentials = conf_loader["credentials"]

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



def to_feature_store(data, group_name, feature_group_version,
                     description, group_description, credentials_input):
    """Upload one feature group to Hopsworks. Data already validated upstream."""
    project = hopsworks.login(
        api_key_value=credentials_input["api_key"],
        project=credentials_input["project"],
    )
    feature_store = project.get_feature_store()

    fg = feature_store.get_or_create_feature_group(
        name=group_name,
        version=feature_group_version,
        description=description,
        primary_key=["index"],
        online_enabled=False,
        # no event_time — static listings snapshot, no point-in-time joins needed
    )

    fg.insert(data, overwrite=False, write_options={"wait_for_job": True})

    if group_description:
        for desc in group_description:
            fg.update_feature_description(desc["name"], desc["description"])

    fg.compute_statistics()
    logger.info("Feature group '%s' v%d: inserted %d rows.",
                group_name, feature_group_version, len(data))
    return fg


def read_from_feature_store(parameters: dict, credentials: dict) -> pd.DataFrame:
    """Read the 3 feature groups back and join on the primary key (write->read demo)."""
    project = hopsworks.login(
        api_key_value=credentials["api_key"],
        project=credentials["project"],
    )
    fs = project.get_feature_store()

    fg_cfg = parameters["feature_groups"]   # {numerical: {name, version}, categorical: {...}, target: {...}}
    pk = parameters.get("primary_key", "index")

    groups = {}
    for key in ("numerical", "categorical", "target"):
        fg = fs.get_feature_group(name=fg_cfg[key]["name"], version=fg_cfg[key]["version"])
        groups[key] = fg.read()

    df = (groups["numerical"]
          .merge(groups["categorical"], on=pk, how="inner")
          .merge(groups["target"], on=pk, how="inner"))

    logger.info("Read from feature store: %s rows, %s cols.", df.shape[0], df.shape[1])
    return df


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


def ingestion(df_raw: pd.DataFrame, parameters: Dict[str, Any],
              validation_params: Dict[str, Any]) -> pd.DataFrame:
    """Ingest raw housing data, validate in memory (ephemeral GX), optionally upload."""
    target_col = parameters["target_col"]
    primary_key = parameters.get("primary_key", "index")

    df = df_raw.copy()

    # stable synthetic primary key for the 3-group join
    if primary_key not in df.columns:
        df = df.reset_index(names=primary_key)

    # target must exist
    n_before = len(df)
    df = df.dropna(subset=[target_col]).reset_index(drop=True)
    logger.info("Ingestion: %d rows (dropped %d without '%s').",
                len(df), n_before - len(df), target_col)

    # split into the 3 feature groups (each carries the primary key for joins)
    feature_cols = [c for c in df.columns if c not in (primary_key, target_col)]
    numeric_cols = df[feature_cols].select_dtypes(include="number").columns.tolist()
    categorical_cols = [c for c in feature_cols if c not in numeric_cols]

    df_numeric     = df[[primary_key, *numeric_cols]]
    df_categorical = df[[primary_key, *categorical_cols]]
    df_target      = df[[primary_key, target_col]]

    # ephemeral GX validation BEFORE upload (his validate_slice pattern)
    if parameters.get("run_validation", True):
        context = gx.get_context(mode="ephemeral")
        context.variables.progress_bars = {"globally": False, "metric_calculations": False}
        data_source = context.data_sources.add_pandas("ingestion_source")

        def validate_slice(df_slice, asset_name, group):
            suite = build_expectation_suite(context, f"{asset_name}_suite", group, validation_params)
            if not suite.expectations:
                return None
            asset = data_source.add_dataframe_asset(name=asset_name)
            batch_def = asset.add_batch_definition_whole_dataframe("batch_def")
            batch = batch_def.get_batch(batch_parameters={"dataframe": df_slice})
            return batch.validate(suite)

        results = [
            validate_slice(df_numeric,     "numeric",     "numerical"),
            validate_slice(df_categorical, "categorical", "categorical"),
            validate_slice(df_target,      "target",      "target"),
        ]
        if not all(r.success for r in results if r is not None):
            logger.error("Ephemeral GX validation failed — halting before upload.")
            raise ValueError("Data did not pass Great Expectations validation.")
        logger.info("Ephemeral GX validation passed.")

    # upload to Hopsworks
    if parameters.get("to_feature_store", False):
        creds = credentials["hopsworks"]          # matches your credentials.yml key
        for data, name, desc in [
            (df_numeric,     "house_numerical",   "Numerical housing features"),
            (df_categorical, "house_categorical", "Categorical housing features"),
            (df_target,      "house_target",      "Target (Price)"),
        ]:
            logger.info("Uploading %s to Hopsworks...", name)
            to_feature_store(
                data=data, group_name=name, feature_group_version=1,
                description=desc, group_description=[],
                credentials_input=creds,
            )

    return df


# NOTE: the ref/ana split was moved to the `split_data` pipeline (split_out_of_sample),
# which is the canonical version (supports the 'biased' strategy for the drift demo). The old
# `split_reference_analysis` was removed from here to eliminate the duplication.


def read_from_feature_store(parameters: dict, credentials: dict) -> pd.DataFrame:
    """Read the 3 feature groups back and join on the primary key (write->read demo)."""
    project = hopsworks.login(
        api_key_value=credentials["api_key"],
        project=credentials["project"],
    )
    fs = project.get_feature_store()

    fg_cfg = parameters["feature_groups"]   # {numerical: {name, version}, categorical: {...}, target: {...}}
    pk = parameters.get("primary_key", "index")

    groups = {}
    for key in ("numerical", "categorical", "target"):
        fg = fs.get_feature_group(name=fg_cfg[key]["name"], version=fg_cfg[key]["version"])
        groups[key] = fg.read()

    df = (groups["numerical"]
          .merge(groups["categorical"], on=pk, how="inner")
          .merge(groups["target"], on=pk, how="inner"))

    logger.info("Read from feature store: %s rows, %s cols.", df.shape[0], df.shape[1])
    return df
