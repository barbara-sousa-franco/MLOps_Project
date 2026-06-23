"""Nodes for the `data_drift` pipeline (valued by the professor)."""

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def compute_drift(learning_data: pd.DataFrame, test_data: pd.DataFrame, parameters: dict):
    """Compare reference (training) vs new batch distributions and detect drift.

    Args:
        learning_data: reference data (02_intermediate/learning_data.csv).
        test_data: batch to analyse (02_intermediate/test_data.csv).
        parameters: features to monitor, thresholds, engine (nannyml/evidently).

    Returns:
        Tuple (drift_result, drift_report):
          - drift_result: CSV with drift per feature (08_reporting/drift_result.csv).
          - drift_report: HTML (08_reporting/data_drift_report.html).

    TODO compute_drift:
      - use nannyml OR evidently
      - compare feature distributions ref vs analysis; univariate drift per feature
      - generate data_drift_report.html + drift_result.csv -> 08_reporting
      - (extra creativity: inject artificial drift in a sample to demonstrate detection —
        suggested by the professor)
    """
    # TODO: implement
    raise NotImplementedError
