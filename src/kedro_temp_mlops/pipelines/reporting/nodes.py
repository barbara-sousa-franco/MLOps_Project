"""Nodes for the `reporting` pipeline."""

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def build_report(
    metrics: dict,
    shap_plot,
    drift_result: pd.DataFrame,
    test_results: pd.DataFrame,
    parameters: dict,
):
    """Consolidate everything that goes into the 6-page report.

    Args:
        metrics: champion metrics (production_model_metrics).
        shap_plot: feature importance (SHAP).
        drift_result: drift detection result.
        test_results: data unit test results (reporting_tests).
        parameters: reporting config.

    Returns:
        Consolidated artifact(s) for the report (08_reporting).

    TODO build_report:
      - consolidate champion metrics, feature importance (SHAP),
        data test results and drift into a single report
    """
    # TODO: implement
    raise NotImplementedError
