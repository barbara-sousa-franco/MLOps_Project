"""Replays the SAME cleaning as training (no refit), then applies the
train-fitted imputers, capper and target encoder via transform only.
Defaults to inference (no target); `batch_has_target` opts into a labelled batch.
"""

import logging
from typing import Any, Dict

import pandas as pd

from ..preprocessing_train.nodes import clean_data  # single source of truth

logger = logging.getLogger(__name__)


def preprocess_batch(ana_data, num_imputer, cat_imputer, capper, target_encoder, parameters):
    has_target = parameters.get("batch_has_target", False)

    df, _ = clean_data(ana_data, parameters,
                       has_target=has_target, drop_missing_target=False)

    num_cols = list(num_imputer.feature_names_in_)
    df[num_cols] = num_imputer.transform(df[num_cols])
    if cat_imputer is not None:
        cat_cols = list(cat_imputer.feature_names_in_)
        df[cat_cols] = cat_imputer.transform(df[cat_cols])

    df = capper.transform(df)

    enc_cols = list(target_encoder.feature_names_in_)
    df[enc_cols] = target_encoder.transform(df[enc_cols])

    logger.info("Preprocessed batch shape: %s", df.shape)
    return df