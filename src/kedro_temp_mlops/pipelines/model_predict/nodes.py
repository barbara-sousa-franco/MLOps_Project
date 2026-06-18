"""Nodes da pipeline `model_predict`."""

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def predict(
    production_model,
    preprocessed_batch_data: pd.DataFrame,
    best_columns,
    parameters: dict,
) -> pd.DataFrame:
    """Prevê o Price no batch novo com o champion + best_cols.

    Args:
        production_model: champion treinado.
        preprocessed_batch_data: batch pré-processado (preprocessing_batch).
        best_columns: colunas selecionadas por SHAP.
        parameters: config (inclui use_log_target para inverter log1p).

    Returns:
        DataFrame com as predições (07_model_output).

    TODO predict:
      - selecionar best_columns no batch; model.predict(...)
      - se parameters["use_log_target"]: aplicar expm1 às predições (inverter log1p)
      - guardar em 07_model_output
    """
    # TODO: implementar
    raise NotImplementedError
