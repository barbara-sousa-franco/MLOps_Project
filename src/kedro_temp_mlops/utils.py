"""Shared Great Expectations helpers (used by ingestion + data_unit_tests)."""
import pandas as pd
import great_expectations as gx
from great_expectations import expectations as gxe


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