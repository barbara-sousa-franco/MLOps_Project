"""Nodes for the `preprocessing_batch` pipeline.

Reuses the `encoder_transform` from training: applies `transform` (NEVER `fit`) to the
new batch. Output feeds drift detection and prediction.
"""

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def preprocess_batch(ana_data: pd.DataFrame, encoder, parameters: dict) -> pd.DataFrame:
    """Pre-process a new batch using the already-fitted encoder.

    Args:
        ana_data: new batch to analyse (02_intermediate/ana_data.csv).
        encoder: `encoder_transform` fitted during training.
        parameters: same cleaning/encoding config as training (parameters_preprocessing.yml).

    Returns:
        preprocessed_batch_data — ready for drift detection and inference.

    TODO preprocess_batch:
      - apply the SAME cleaning/FE as training (no refit)
      - encoder.transform(...) — NOT encoder.fit(...)  (anti-leakage)
      - do NOT apply log1p to target if the batch has no target (inference)
      - output preprocessed_batch_data
    """
    # TODO: implement
    raise NotImplementedError
