"""Replays the SAME cleaning as training (no refit), then applies the
train-fitted imputers, capper and target encoder via transform only.
Defaults to inference (no target); `batch_has_target` opts into a labelled batch.
"""

import logging
from typing import Any, Dict

import pandas as pd

from ..preprocessing_train.nodes import clean_data  # single source of truth

logger = logging.getLogger(__name__)


def preprocess_batch(test_data, imputer, capper, target_encoder, scaler, parameters):
    has_target = parameters.get("batch_has_target", False)

    df, _ = clean_data(test_data, parameters,
                       has_target=has_target, drop_missing_target=False)

    df = imputer.transform(df)
    df = capper.transform(df)

    enc_cols = list(target_encoder.feature_names_in_)
    df[enc_cols] = target_encoder.transform(df[enc_cols])

    scale_cols = list(scaler.feature_names_in_)
    df[scale_cols] = scaler.transform(df[scale_cols])

    logger.info("Preprocessed batch shape: %s", df.shape)
    return df