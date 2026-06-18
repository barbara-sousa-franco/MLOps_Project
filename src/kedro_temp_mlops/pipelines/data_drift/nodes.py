"""Nodes da pipeline `data_drift` (valorizado pelo prof)."""

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def compute_drift(ref_data: pd.DataFrame, ana_data: pd.DataFrame, parameters: dict):
    """Compara distribuições de referência (treino) vs batch novo e deteta drift.

    Args:
        ref_data: dados de referência (02_intermediate/ref_data.csv).
        ana_data: batch a analisar (02_intermediate/ana_data.csv).
        parameters: features a monitorizar, thresholds, motor (nannyml/evidently).

    Returns:
        Tuple (drift_result, drift_report):
          - drift_result: CSV com drift por feature (08_reporting/drift_result.csv).
          - drift_report: HTML (08_reporting/data_drift_report.html).

    TODO compute_drift:
      - usar nannyml OU evidently
      - comparar distribuição das features ref vs analysis; drift univariado por feature
      - gerar data_drift_report.html + drift_result.csv -> 08_reporting
      - (extra criatividade: injetar drift artificial num sample p/ mostrar deteção —
        o prof sugeriu)
    """
    # TODO: implementar
    raise NotImplementedError
