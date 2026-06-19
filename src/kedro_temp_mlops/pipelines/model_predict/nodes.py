"""Nodes for the `model_predict` pipeline."""

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def predict(
    production_model,
    preprocessed_batch_data: pd.DataFrame,
    best_columns,
    parameters: dict,
) -> pd.DataFrame:
    """Predict Price on the new batch using the champion model + best_cols.

    Args:
        production_model: trained champion.
        preprocessed_batch_data: pre-processed batch (preprocessing_batch).
        best_columns: columns selected by SHAP.
        parameters: config (includes use_log_target to invert log1p).

    Returns:
        DataFrame with predictions (07_model_output).

    TODO predict:
      - select best_columns in the batch; model.predict(...)
      - if parameters["use_log_target"]: apply expm1 to predictions (invert log1p)
      - save to 07_model_output
    """
    # TODO: implement
    raise NotImplementedError
