"""Nodes for the `data_unit_tests` pipeline.

Three validation suites:
  1. `cleaned_data`   — validates the preprocessing output (pre-split)
  2. `model_input`    — validates X_train/X_val (ready for modelling)
  3. ingested_data     — validates raw ingested_data (strict, group-based)
"""

import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
import great_expectations as gx
from great_expectations import expectations as gxe

logger = logging.getLogger(__name__)
from kedro_temp_mlops.utils import build_expectation_suite, _build_between, get_validation_results



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

    # NOTE: the "dropped columns must not exist" check is NOT done here — GX 1.x has no
    # ExpectColumnToNotExist. It is done in Python in `unit_test_cleaned_data`
    # (see _check_dropped_columns), which returns rows in the same report format.

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
    for col, dtype in parameters["column_types"].items():
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
    """Build expectations for X_train / X_val (ready for modelling).

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
    for col, dtype in parameters["column_types"].items():
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


def _check_dropped_columns(df: pd.DataFrame, dropped_columns: list) -> pd.DataFrame:
    """Check in Python that the dropped columns do not exist (GX 1.x lacks this expectation).

    Returns a DataFrame in the same format as `get_validation_results` (one row per column).
    """
    rows = []
    for col in dropped_columns:
        present = col in df.columns
        rows.append({
            "Success": not present,
            "Expectation Type": "expect_column_to_not_exist",
            "Column": col,
            "Min Value": "", "Max Value": "", "Value Set": "",
            "Element Count": "", "Unexpected Count": "", "Unexpected Percent": "",
            "Observed Value": "present" if present else "absent",
        })
    return pd.DataFrame(rows)


def unit_test_cleaned_data(
    cleaned_data: pd.DataFrame,
    parameters: dict,
) -> pd.DataFrame:
    """Validate cleaned_data (post-preprocessing, pre-split)."""
    logger.info("Running data unit tests on cleaned_data...")

    suite = build_cleaned_data_suite(parameters)
    df_validation = _run_validation(cleaned_data, suite, "cleaned_data")

    # Python check for dropped columns (replaces ExpectColumnToNotExist)
    dropped = _check_dropped_columns(cleaned_data, parameters.get("dropped_columns", []))
    if not dropped.empty:
        df_validation = pd.concat([dropped, df_validation], ignore_index=True)

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


def _run_and_report(df: pd.DataFrame, parameters: dict, label: str) -> pd.DataFrame:
    """Run model_input suite on one split and return results with a 'Split' column."""
    suite = build_model_input_suite(parameters)
    df_val = _run_validation(df, suite, f"model_input_{label}")
    df_val.insert(0, "Split", label)
    return df_val


def unit_test_model_input(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    parameters: dict,
) -> pd.DataFrame:
    """Validate X_train and X_val (ready for modelling — no nulls, all numeric)."""
    logger.info("Running data unit tests on model input (X_train + X_val)...")

    df_train = _run_and_report(X_train, parameters, "train")
    df_val   = _run_and_report(X_val,   parameters, "val")
    df_validation = pd.concat([df_train, df_val], ignore_index=True)

    n_failed = int((~df_validation["Success"].astype(bool)).sum())
    if n_failed:
        logger.warning("%d expectation(s) failed on model input.", n_failed)
        for _, r in df_validation[~df_validation["Success"].astype(bool)].iterrows():
            logger.warning("FAILED [%s]: %s on column '%s'", r["Split"], r["Expectation Type"], r["Column"])
        if parameters.get("hard_fail", False):
            raise ValueError(f"Model input validation failed on {n_failed} expectation(s).")
    else:
        logger.info("All model input expectations passed (train + val).")

    return df_validation


def unit_test_raw(
    ingested_data: pd.DataFrame,
    parameters: dict,
) -> pd.DataFrame:
    """Validate raw ingested_data (group-based strict checks)."""
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