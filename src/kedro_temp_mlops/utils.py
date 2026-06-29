"""Shared helpers used across pipelines.

Great Expectations:
  - build_expectation_suite, _build_between, get_validation_results
    (used by ingestion + data_unit_tests)

Hopsworks Feature Store:
  - to_feature_store, upload_cleaned_to_fs, upload_engineered_to_fs
    (used by ingestion + preprocessing_train + feature_engineering)

Also loads the module-level `credentials` from conf (resolved via OmegaConf /
.env), shared by all feature-store helpers.
""" 
import pandas as pd
import great_expectations as gx
from great_expectations import expectations as gxe
from pathlib import Path
import logging
import hopsworks

from kedro.config import OmegaConfigLoader
from kedro.framework.project import settings

logger = logging.getLogger(__name__)

conf_loader = OmegaConfigLoader(conf_source=str(Path("") / settings.CONF_SOURCE))
credentials = conf_loader["credentials"]


def _build_between(column: str, rng: dict) -> gxe.ExpectColumnValuesToBeBetween:
    return gxe.ExpectColumnValuesToBeBetween(
        column=column,
        min_value=rng.get("min"),
        max_value=rng.get("max"),
        strict_min=rng.get("strict_min", False),
        strict_max=rng.get("strict_max", False),
    )


def build_expectation_suite(suite_name: str, feature_group: str, parameters: dict) -> gx.ExpectationSuite:
    target_col = parameters.get("target_col")
    column_types = parameters.get("column_types", {})
    ranges = parameters.get("ranges", {})
    valid_sets = parameters.get("valid_sets", {})
    expectations = []
    if feature_group == "numerical":
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
            expectations.append(gxe.ExpectColumnDistinctValuesToBeInSet(column=col, value_set=list(value_set)))
    elif feature_group == "target":
        if target_col in column_types:
            expectations.append(gxe.ExpectColumnValuesToBeOfType(column=target_col, type_=column_types[target_col]))
        if target_col in ranges:
            expectations.append(_build_between(target_col, ranges[target_col]))
    else:
        raise ValueError(f"Unknown feature_group: {feature_group!r}")
    return gx.ExpectationSuite(name=suite_name, expectations=expectations)


def get_validation_results(validation_results) -> pd.DataFrame:
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
        time_travel_format="HUDI",
    )
    # Avro (Hopsworks insert) needs real nulls, not pandas NaN, for string columns.
    data = data.copy()
    for col in data.select_dtypes(include="object").columns:
        data[col] = data[col].where(data[col].notna(), None)
    
    fg.insert(data, overwrite=False, write_options={"wait_for_job": False, "compute_statistics":False})

    if group_description:
        for desc in group_description:
            fg.update_feature_description(desc["name"], desc["description"])
    return fg


def upload_cleaned_to_fs(cleaned_data, parameters):
    """Store cleaned (pre-split, statistics-free) data. Pass-through."""
    if not parameters.get("to_feature_store", False):
        return cleaned_data
    df = cleaned_data.copy()
    if "index" not in df.columns:
        df = df.reset_index(names="index")
    to_feature_store(
        data=df, group_name="house_cleaned", feature_group_version=4,
        description="Cleaned housing data (pre-split, statistics-free)",
        group_description=[], credentials_input=credentials["hopsworks"],
    )
    return cleaned_data


def upload_engineered_to_fs(X_train_scaled, y_train_data, parameters):
    """Store feature-engineered + scaled TRAIN data (fit-on-train; cycle demo). Pass-through."""
    if not parameters.get("to_feature_store", False):
        return X_train_scaled
    df = X_train_scaled.copy().reset_index(names="index")
    df["Price_log"] = y_train_data.values
    to_feature_store(
        data=df, group_name="cleaned_after_split", feature_group_version=4,
        description="Feature-engineered + scaled TRAIN data (fitted on train split)",
        group_description=[], credentials_input=credentials["hopsworks"],
    )
    return X_train_scaled