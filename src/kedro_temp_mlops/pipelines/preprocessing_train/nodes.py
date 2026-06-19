"""Nodes for the `preprocessing_train` pipeline.

The encoder/scaler is trained (`fit`) HERE and ONLY here (anti-leakage). It is returned
to be saved in encoder.pkl (MLflow artifact) and reused in the batch pipeline (`transform` only).
"""

import logging

import pandas as pd

from .utils import bin_area, energy_to_ordinal, property_age  # noqa: F401  (FE used in clean/encode)

logger = logging.getLogger(__name__)


def clean_data(ingested_data: pd.DataFrame, parameters: dict) -> pd.DataFrame:
    """Clean raw housing data (impossible values, outliers, missing).

    Args:
        ingested_data: dataset from ingestion.
        parameters: cleaning/imputation/outlier strategy (parameters_preprocessing.yml).

    Returns:
        Cleaned DataFrame (not yet encoded).

    TODO clean_data:
      - remove/fix impossible values identified in EDA
      - handle extreme outliers (~1.38 billion EUR price, negative areas)
      - impute missing values (strategy in parameters: median/mode/constant)
      - apply FE functions from utils.py (property_age, bin_area, energy_to_ordinal)
    """
    # TODO: implement
    raise NotImplementedError


def encode_features(clean_df: pd.DataFrame, parameters: dict):
    """Fit the encoder/scaler on training data and return transformed data + encoder.

    Args:
        clean_df: output of `clean_data`.
        parameters: encoding/scaling config and `use_log_target` flag.

    Returns:
        Tuple (preprocessed_df, encoder):
          - preprocessed_df: encoded features + target (log1p if use_log_target).
          - encoder: fitted object to be saved in encoder.pkl.

    TODO encode_features:
      - FIT the encoder/scaler HERE (training only) — anti-leakage
      - return encoder to save in encoder.pkl (MLflow artifact)
      - apply log1p to target (Price) if parameters["use_log_target"]
        (EDA shows strong skew — test and document in ASSUMPTIONS.md)
    """
    # TODO: implement
    raise NotImplementedError
