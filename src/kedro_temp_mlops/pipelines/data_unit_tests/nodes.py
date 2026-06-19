"""Nodes for the `data_unit_tests` pipeline.

Two validation suites:
  1. `cleaned_data`   — validates the preprocessing output (pre-split)
  2. `model_input`    — validates X_train/X_test (ready for modelling)
"""

import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
import great_expectations as gx
from great_expectations import expectations as gxe

logger = logging.getLogger(__name__)


# =============================================================================
# HELPERS
# =============================================================================

def _build_between(column: str, rng: dict) -> gxe.ExpectColumnValuesToBeBetween:
    return gxe.ExpectColumnValuesToBeBetween(
        column=column,
        min_value=rng.get("min"),
        max_value=rng.get("max"),
        strict_min=rng.get("strict_min", False),
        strict_max=rng.get("strict_max", False),
    )


def get_validation_results(validation_results) -> pd.DataFrame:
    """Extract validation results into a pandas DataFrame."""
    vd = (validation_results.to_json_dict()
          if hasattr(validation_results, "to_json_dict") else validation_results)
    rows = []
    for result in vd.get("results", []):
        cfg = result.get("expectation_config", {})
        kwargs = cfg.get("kwargs", {})
        res = result.get("result", {})
        value_set = kwargs.get("value_set", "")
        observed = res.get("observed_value", "")
        unexpected_value = (
            [v for v in observed if v not in value_set]
            if isinstance(observed, list) and isinstance(value_set, list) else []
        )
        rows.append({
            "Success":            result.get("success", False),
            "Expectation Type":   cfg.get("type") or cfg.get("expectation_type", ""),
            "Column":             kwargs.get("column", ""),
            "Min Value":          kwargs.get("min_value", ""),
            "Max Value":          kwargs.get("max_value", ""),
            "Value Set":          value_set,
            "Element Count":      res.get("element_count", ""),
            "Unexpected Count":   res.get("unexpected_count", ""),
            "Unexpected Percent": res.get("unexpected_percent", ""),
            "Observed Value":     observed,
        })
    return pd.DataFrame(rows)


# =============================================================================
# SUITE BUILDERS
# =============================================================================

def build_cleaned_data_suite(parameters: dict) -> gx.ExpectationSuite:
    """Build expectations for cleaned_data (post-preprocessing, pre-split).

    Validates that:
    - Dropped columns no longer exist
    - No negative values in numeric columns
    - District does not contain 'Z - Fora de Portugal'
    - Type only contains known property types
    - EnergyCertificate is ordinal encoded (0-9)
    - ConstructionYear is within valid range
    - Price_log > 0
    - Row count is within expected range
    """
    expectations = []

    # Dropped columns must not exist 
    for col in parameters["dropped_columns"]:
        expectations.append(
            gxe.ExpectColumnToNotExist(column=col)
        )

    # Row count 
    expectations.append(
        gxe.ExpectTableRowCountToBeBetween(
            min_value=parameters["row_count"]["min"],
            max_value=parameters["row_count"]["max"],
        )
    )

    # No nulls in critical columns 
    for col in parameters["not_null_columns"]:
        expectations.append(
            gxe.ExpectColumnValuesToNotBeNull(column=col)
        )

    #  Column types
    for col, dtype in parameters["column_types_cleaned"].items():
        expectations.append(
            gxe.ExpectColumnValuesToBeOfType(column=col, type_=dtype)
        )

    # Numeric ranges
    for col, rng in parameters["ranges"].items():
        expectations.append(_build_between(col, rng))

    # Categorical valid sets 
    for col, value_set in parameters["valid_sets"].items():
        expectations.append(
            gxe.ExpectColumnDistinctValuesToBeInSet(
                column=col,
                value_set=list(value_set)
            )
        )

    # District must NOT contain 'Z - Fora de Portugal'
    expectations.append(
        gxe.ExpectColumnValuesToNotBeInSet(
            column="District",
            value_set=["Z - Fora de Portugal"]
        )
    )

    return gx.ExpectationSuite(
        name="cleaned_data_suite",
        expectations=expectations
    )


def build_model_input_suite(parameters: dict) -> gx.ExpectationSuite:
    """Build expectations for X_train / X_test (ready for modelling).

    Validates that:
    - No missing values in any column
    - All columns are numeric
    - Numeric ranges are within expected bounds after imputation and capping
    - Row count is consistent with the train/test split
    """
    expectations = []

    # No missing values anywhere
    for col in parameters["model_input_columns"]:
        expectations.append(
            gxe.ExpectColumnValuesToNotBeNull(column=col)
        )

    # Column types 
    for col, dtype in parameters["column_types_model_input"].items():
        expectations.append(
            gxe.ExpectColumnValuesToBeOfType(column=col, type_=dtype)
        )

    # Row count consistent with split
    expectations.append(
        gxe.ExpectTableRowCountToBeBetween(
            min_value=parameters["model_input_row_count"]["min"],
            max_value=parameters["model_input_row_count"]["max"],
        )
    )

    # Numeric ranges after pipeline
    for col, rng in parameters["model_input_ranges"].items():
        expectations.append(_build_between(col, rng))

    return gx.ExpectationSuite(
        name="model_input_suite",
        expectations=expectations
    )


# =============================================================================
# SHARED RULEBOOK (kept for ingestion pipeline compatibility)
# =============================================================================

def build_expectation_suite(
    suite_name: str, feature_group: str, parameters: dict
) -> gx.ExpectationSuite:
    """Build a GX ExpectationSuite for a given feature group (ingestion pipeline).

    Kept for compatibility with the ingestion pipeline.
    """
    target_col = parameters.get("target_col")
    column_types = parameters.get("column_types", {})
    ranges = parameters.get("ranges", {})
    valid_sets = parameters.get("valid_sets", {})

    expectations = []

    if feature_group == "numerical":
        for col, dtype in column_types.items():
            if col == target_col:
                continue
            expectations.append(
                gxe.ExpectColumnValuesToBeOfType(column=col, type_=dtype)
            )
        for col, rng in ranges.items():
            if col == target_col:
                continue
            expectations.append(_build_between(col, rng))
    elif feature_group == "categorical":
        for col, value_set in valid_sets.items():
            expectations.append(
                gxe.ExpectColumnDistinctValuesToBeInSet(
                    column=col,
                    value_set=list(value_set)
                )
            )
    elif feature_group == "target":
        if target_col in column_types:
            expectations.append(
                gxe.ExpectColumnValuesToBeOfType(
                    column=target_col,
                    type_=column_types[target_col]
                )
            )
        if target_col in ranges:
            expectations.append(_build_between(target_col, ranges[target_col]))
    else:
        raise ValueError(
            f"Unknown feature_group: {feature_group!r} "
            "(expected: 'numerical', 'categorical' or 'target')"
        )

    return gx.ExpectationSuite(name=suite_name, expectations=expectations)


# =============================================================================
# VALIDATION RUNNERS
# =============================================================================

def _run_validation(
    df: pd.DataFrame,
    suite: gx.ExpectationSuite,
    suite_name: str,
) -> pd.DataFrame:
    """Run a GX validation suite against a DataFrame and return results."""
    context = gx.get_context(mode="ephemeral")
    context.variables.progress_bars = {"globally": False, "metric_calculations": False}

    data_source = context.data_sources.add_pandas(f"{suite_name}_source")
    asset = data_source.add_dataframe_asset(suite_name)
    batch_def = asset.add_batch_definition_whole_dataframe("batch")

    suite = context.suites.add(suite)
    vd = context.validation_definitions.add(
        gx.ValidationDefinition(data=batch_def, suite=suite, name=f"vd_{suite_name}")
    )
    results = vd.run(batch_parameters={"dataframe": df})
    return get_validation_results(results)


def unit_test_cleaned_data(
    cleaned_data: pd.DataFrame,
    parameters: dict,
) -> pd.DataFrame:
    """Validate cleaned_data (post-preprocessing, pre-split)."""
    logger.info("Running data unit tests on cleaned_data...")

    suite = build_cleaned_data_suite(parameters)
    df_validation = _run_validation(cleaned_data, suite, "cleaned_data")

    n_failed = int((~df_validation["Success"].astype(bool)).sum())
    if n_failed:
        logger.warning("%d expectation(s) failed on cleaned_data.", n_failed)
        for _, r in df_validation[~df_validation["Success"].astype(bool)].iterrows():
            logger.warning("FAILED: %s on column '%s'", r["Expectation Type"], r["Column"])
        if parameters.get("hard_fail", False):
            raise ValueError(f"cleaned_data validation failed on {n_failed} expectation(s).")
    else:
        logger.info("All cleaned_data expectations passed.")

    return df_validation


def unit_test_model_input(
    X_train: pd.DataFrame,
    parameters: dict,
) -> pd.DataFrame:
    """Validate X_train (ready for modelling — no nulls, all numeric)."""
    logger.info("Running data unit tests on model input (X_train)...")

    suite = build_model_input_suite(parameters)
    df_validation = _run_validation(X_train, suite, "model_input")

    n_failed = int((~df_validation["Success"].astype(bool)).sum())
    if n_failed:
        logger.warning("%d expectation(s) failed on model input.", n_failed)
        for _, r in df_validation[~df_validation["Success"].astype(bool)].iterrows():
            logger.warning("FAILED: %s on column '%s'", r["Expectation Type"], r["Column"])
        if parameters.get("hard_fail", False):
            raise ValueError(f"Model input validation failed on {n_failed} expectation(s).")
    else:
        logger.info("All model input expectations passed.")

    return df_validation


def unit_test(
    ingested_data: pd.DataFrame,
    parameters: dict,
) -> pd.DataFrame:
    """Run data unit tests on ingested data (kept for ingestion pipeline compatibility)."""
    logger.info("Starting official data validation (strict)...")

    context = gx.get_context(mode="ephemeral")
    context.variables.progress_bars = {"globally": False, "metric_calculations": False}
    data_source = context.data_sources.add_pandas("dut_source")
    asset = data_source.add_dataframe_asset("houses")
    batch_def = asset.add_batch_definition_whole_dataframe("batch")

    frames = []
    for group in ("numerical", "categorical", "target"):
        suite = build_expectation_suite(f"dut_{group}", group, parameters)
        if not suite.expectations:
            continue
        suite = context.suites.add(suite)
        vd = context.validation_definitions.add(
            gx.ValidationDefinition(
                data=batch_def, suite=suite, name=f"dut_vd_{group}"
            )
        )
        results = vd.run(batch_parameters={"dataframe": ingested_data})
        frames.append(get_validation_results(results))

    df_validation = (
        pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    )

    n_failed = (
        int((~df_validation["Success"].astype(bool)).sum()) if len(df_validation) else 0
    )
    if n_failed:
        logger.warning(
            "%d expectation(s) failed — see reporting_tests / traffic light.", n_failed
        )
        for _, r in df_validation[~df_validation["Success"].astype(bool)].iterrows():
            logger.warning("Failed: %s on %s", r["Expectation Type"], r["Column"])
        if parameters.get("hard_fail", False):
            raise ValueError(f"Data validation failed on {n_failed} expectation(s).")
    else:
        logger.info("All data unit tests passed.")

    return df_validation


def write_traffic_light(df_validation: pd.DataFrame, parameters: dict) -> str:
    """Write traffic light flag based on validation results."""
    tl = parameters.get("traffic_light", {})
    ok_path   = Path(tl.get("flag_path", "data/08_reporting/PIPELINE_OK.flag"))
    fail_path = Path(tl.get("fail_flag_path", "data/08_reporting/PIPELINE_FAIL.flag"))
    critical_cols = parameters.get("critical_columns")

    if len(df_validation):
        relevant = (
            df_validation[df_validation["Column"].isin(critical_cols)]
            if critical_cols else df_validation
        )
        success_mask = relevant["Success"].astype(bool)
        n_failed = int((~success_mask).sum())
        passed = bool(success_mask.all()) and len(relevant) > 0
    else:
        n_failed, passed = -1, False

    for p in (ok_path, fail_path):
        if p.exists():
            p.unlink()

    target = ok_path if passed else fail_path
    target.parent.mkdir(parents=True, exist_ok=True)
    status = "OK" if passed else "FAIL"
    target.write_text(
        f"{status} @ {datetime.now().isoformat()} | failed_critical={n_failed}\n"
    )
    logger.info(
        "Traffic light: %s (failed critical=%s) -> %s", status, n_failed, target
    )
    return str(target)