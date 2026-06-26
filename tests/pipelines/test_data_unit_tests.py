


"""Pytest tests that verify GX assertions CATCH bad data.

Raw checkpoint (unit_test_raw): first line of defence — bad incoming data is flagged.
Cleaned checkpoint (unit_test_cleaned_data): verifies clean_data removed/fixed the bad values.
"""

import pandas as pd
import pytest

from kedro_temp_mlops.pipelines.data_unit_tests.nodes import (
    unit_test_raw,
    unit_test_cleaned_data,
    write_traffic_light,
)

# Mirrors the raw `data_unit_tests` block (raw column names: Price, ConstructionYear).
RAW_PARAMS = {
    "target_col": "Price",
    "column_types": {"Price": "float", "ConstructionYear": "float"},
    "ranges": {
        "Price": {"min": 0, "strict_min": True},
        "ConstructionYear": {"min": 1800, "max": 2026},
    },
    "valid_sets": {
        "Type": ["House", "Land", "Apartment"],
        "EnergyCertificate": ["A", "B", "C", "No Certificate"],
    },
    "hard_fail": False,
}


def _good_raw_row():
    """A raw row that passes every raw expectation."""
    return {
        "Price": 250000.0,
        "ConstructionYear": 2000.0,
        "Type": "House",
        "EnergyCertificate": "B",
    }


def _failed_cols(report: pd.DataFrame) -> set:
    return set(report[~report["Success"].astype(bool)]["Column"])


# ---------------------------------------------------------------------------
# RAW checkpoint — catches bad incoming data
# ---------------------------------------------------------------------------

def test_raw_catches_negative_price():
    """Raw Price = -1 must be flagged (strict_min: true on Price)."""
    bad = pd.DataFrame([{**_good_raw_row(), "Price": -1.0}])
    report = unit_test_raw(bad, RAW_PARAMS)
    assert "Price" in _failed_cols(report)


def test_raw_catches_invalid_construction_year():
    """ConstructionYear = 1500 (outside [1800, 2026]) must be flagged."""
    bad = pd.DataFrame([{**_good_raw_row(), "ConstructionYear": 1500.0}])
    report = unit_test_raw(bad, RAW_PARAMS)
    assert "ConstructionYear" in _failed_cols(report)


def test_raw_catches_unknown_type():
    """A Type outside the valid set must be flagged."""
    bad = pd.DataFrame([{**_good_raw_row(), "Type": "Spaceship"}])
    report = unit_test_raw(bad, RAW_PARAMS)
    assert "Type" in _failed_cols(report)


def test_raw_good_data_passes():
    """Clean raw data should produce no failures."""
    good = pd.DataFrame([_good_raw_row()])
    report = unit_test_raw(good, RAW_PARAMS)
    assert report[~report["Success"].astype(bool)].empty


# ---------------------------------------------------------------------------
# CLEANED checkpoint — verifies clean_data removed the bad values
# ---------------------------------------------------------------------------

CLEANED_PARAMS = {
    "dropped_columns": [],
    "row_count": {"min": 1, "max": 1000},
    "not_null_columns": ["Price_log"],
    "column_types": {"Price_log": "float64", "EnergyCertificate": "int64"},
    "ranges": {
        "Price_log": {"min": 0, "strict_min": True},
        "ConstructionYear": {"min": 1800, "max": 2026},
        "EnergyCertificate": {"min": 0, "max": 9},
    },
    "valid_sets": {"Type": ["House", "Land", "Apartment"]},
    "hard_fail": False,
}

def _good_cleaned_row():
    return {"Price_log": 12.0, "ConstructionYear": 2000.0,
            "EnergyCertificate": 5, "Type": "House"}

def test_cleaned_flags_nonpositive_price_log():
    """If a non-positive Price_log somehow survives, the cleaned suite flags it."""
    bad = pd.DataFrame([{**_good_cleaned_row(), "Price_log": -1.0}])
    report = unit_test_cleaned_data(bad, CLEANED_PARAMS)
    assert "Price_log" in _failed_cols(report)


# ---------------------------------------------------------------------------
# Traffic light
# ---------------------------------------------------------------------------

def test_traffic_light_ok_on_success(tmp_path):
    ok, fail = tmp_path / "OK.flag", tmp_path / "FAIL.flag"
    params = {"traffic_light": {"flag_path": str(ok), "fail_flag_path": str(fail)},
              "critical_columns": None}
    report = pd.DataFrame([{"Success": True, "Column": "Price", "Expectation Type": "x"}])
    write_traffic_light(report, params)
    assert ok.exists() and not fail.exists()

def test_traffic_light_fail_on_failure(tmp_path):
    ok, fail = tmp_path / "OK.flag", tmp_path / "FAIL.flag"
    params = {"traffic_light": {"flag_path": str(ok), "fail_flag_path": str(fail)},
              "critical_columns": None}
    report = pd.DataFrame([{"Success": False, "Column": "Price", "Expectation Type": "x"}])
    write_traffic_light(report, params)
    assert fail.exists() and not ok.exists()