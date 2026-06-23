"""Carves the out-of-sample batch (`ana_data`) off the full ingested data,
BEFORE any cleaning. `ref_data` is the training pool; `ana_data` is the
held-out batch that later flows through `preprocessing_batch`.

ROLE NOTE (professor's bank-example scheme): `ana_data` is the true out-of-sample
**TEST** set — it never enters training/tuning, so the honest final metric is computed
on it (inference / Phase 3). The `X_test` produced by `split_train` is, despite its name,
the VALIDATION set used for tuning/selection. Keep `strategy: random` so `ana_data` stays
representative as a test set ('biased' is only for the drift demo with injected drift).
"""

import logging
from typing import Any, Dict, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


def split_out_of_sample(
    ingested_data: pd.DataFrame,
    parameters: Dict[str, Any],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """ref_data = training pool, ana_data = out-of-sample batch.

    strategy='random' -> ana_data ~ same distribution as ref_data (drift baseline).
    strategy='biased' -> ana_data oversamples one district (drift demo).
    """
    strategy = parameters.get("strategy", "random")
    seed = parameters["random_state"]
    ref_frac = parameters["ref_frac"]

    ref_data = ingested_data.sample(frac=ref_frac, random_state=seed)
    remaining = ingested_data.drop(ref_data.index)

    if strategy == "biased":
        col = parameters.get("bias_column", "District")
        district = parameters["bias_district"]
        ana_data = remaining[remaining[col] == district]
        if ana_data.empty:
            logger.warning("No rows for %s=%s — falling back to random ana_data", col, district)
            ana_data = remaining
    else:
        ana_data = remaining

    logger.info("strategy=%s | ref_data=%s, ana_data=%s",
                strategy, ref_data.shape, ana_data.shape)
    return ref_data, ana_data