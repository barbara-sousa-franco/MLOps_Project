"""Replays the SAME cleaning as training (no refit), then applies the
train-fitted imputers, capper and target encoder via transform only.
Defaults to inference (no target); `batch_has_target` opts into a labelled batch.
"""

import logging


from ..preprocessing_train.nodes import clean_data  # single source of truth

logger = logging.getLogger(__name__)


def preprocess_batch(test_data, imputer, capper, target_encoder, scaler, parameters):
    has_target = parameters.get("batch_has_target", False)

    df, _ = clean_data(test_data, parameters,
                       has_target=has_target, drop_missing_target=False)

    # Train/serve dtype skew: the training cleaned_data round-trips through CSV, turning
    # pandas nullable Int64 columns into float64. The batch is cleaned in-memory and keeps
    # Int64, which breaks the float-median imputer. Normalise to match the fitted transformers.
    nullable_int = [c for c in df.columns if str(df[c].dtype).startswith(("Int", "UInt"))]
    if nullable_int:
        df[nullable_int] = df[nullable_int].astype("float64")

    df = imputer.transform(df)
    df = capper.transform(df)

    enc_cols = list(target_encoder.feature_names_in_)
    df[enc_cols] = target_encoder.transform(df[enc_cols])

    scale_cols = list(scaler.feature_names_in_)
    df[scale_cols] = scaler.transform(df[scale_cols])

    logger.info("Preprocessed batch shape: %s", df.shape)
    return df