"""Nodes da pipeline `reporting`."""

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
    """Consolida tudo o que vai para o relatório de 6 páginas.

    Args:
        metrics: métricas do champion (production_model_metrics).
        shap_plot: feature importance (SHAP).
        drift_result: resultado do drift.
        test_results: resultado dos data unit tests (reporting_tests).
        parameters: config de reporting.

    Returns:
        Artefacto(s) consolidado(s) para o relatório (08_reporting).

    TODO build_report:
      - consolidar métricas do champion, feature importance (SHAP),
        resultado dos data tests e drift num único report
    """
    # TODO: implementar
    raise NotImplementedError
