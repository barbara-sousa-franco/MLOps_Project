"""Pytest tests that verify GX assertions CATCH bad data."""

import pytest


def test_asserts_catch_negative_price():
    """Injecting Price = -1 should make unit_test fail (raise ValueError).

    TODO:
      - build a bad df (one row with Price=-1)
      - assert that unit_test(df_bad, parameters) raises ValueError
    """
    pytest.skip("TODO: implement when unit_test is ready")


def test_asserts_catch_invalid_construction_year():
    """ConstructionYear outside [1800, 2026] should fail.

    TODO: df with ConstructionYear=1500 -> expect failure.
    """
    pytest.skip("TODO: implement when unit_test is ready")


def test_traffic_light_written_on_success():
    """With good data, write_traffic_light creates the traffic light file.

    TODO: run unit_test + write_traffic_light on valid data and assert the flag exists.
    """
    pytest.skip("TODO: implement when write_traffic_light is ready")
