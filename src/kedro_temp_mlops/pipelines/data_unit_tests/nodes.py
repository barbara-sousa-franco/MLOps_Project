"""Nodes for the `data_unit_tests` pipeline (final version).

Reuses build_expectation_suite from ingestion (shared rulebook), validates with
GX 1.x and writes the traffic light that gates downstream pipelines.
STRICT (no mostly) -> assumes data received is ALREADY CLEAN.
"""

import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
import great_expectations as gx

from ..ingestion.nodes import build_expectation_suite  # shared rulebook for both checkpoints

logger = logging.getLogger(__name__)


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
        unexpected_value = ([v for v in observed if v not in value_set]
                            if isinstance(observed, list) and isinstance(value_set, list) else [])
        rows.append({
            "Success": result.get("success", False),
            "Expectation Type": cfg.get("type") or cfg.get("expectation_type", ""),
            "Column": kwargs.get("column", ""),
            "Min Value": kwargs.get("min_value", ""),
            "Max Value": kwargs.get("max_value", ""),
            "Value Set": value_set,
            "Element Count": res.get("element_count", ""),
            "Unexpected Count": res.get("unexpected_count", ""),
            "Unexpected Percent": res.get("unexpected_percent", ""),
            "Observed Value": observed,
        })
    return pd.DataFrame(rows)


def unit_test(ingested_data: pd.DataFrame, parameters: dict) -> pd.DataFrame:
    """Run data unit tests on ingested data."""

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
            gx.ValidationDefinition(data=batch_def, suite=suite, name=f"dut_vd_{group}")
        )
        results = vd.run(batch_parameters={"dataframe": ingested_data})
        frames.append(get_validation_results(results))

    df_validation = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    n_failed = int((~df_validation["Success"].astype(bool)).sum()) if len(df_validation) else 0
    if n_failed:
        logger.warning("%d expectation(s) failed — see reporting_tests / traffic light.", n_failed)
        for _, r in df_validation[~df_validation["Success"].astype(bool)].iterrows():
            logger.warning("Failed: %s on %s", r["Expectation Type"], r["Column"])
        if parameters.get("hard_fail", False):
            raise ValueError(f"Data validation failed on {n_failed} expectation(s).")
    else:
        logger.info("All data unit tests passed.")

    return df_validation


def write_traffic_light(df_validation: pd.DataFrame, parameters: dict) -> str:
    """Write traffic light flag based on validation results.
    Returns the path to the traffic light file.
    """
    
    tl = parameters.get("traffic_light", {})
    ok_path   = Path(tl.get("flag_path", "data/08_reporting/PIPELINE_OK.flag"))
    fail_path = Path(tl.get("fail_flag_path", "data/08_reporting/PIPELINE_FAIL.flag"))
    critical_cols = parameters.get("critical_columns")

    if len(df_validation):
        relevant = (df_validation[df_validation["Column"].isin(critical_cols)]
                    if critical_cols else df_validation)
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
    target.write_text(f"{status} @ {datetime.now().isoformat()} | failed_critical={n_failed}\n")
    logger.info("Traffic light: %s (failed critical=%s) -> %s", status, n_failed, target)
    return str(target)
