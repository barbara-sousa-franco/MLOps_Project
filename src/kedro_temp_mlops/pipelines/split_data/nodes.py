"""Carves the out-of-sample batch (`test_data`) off the full ingested data,
BEFORE any cleaning. `learning_data` is the training pool; `test_data` is the
held-out batch that later flows through `preprocessing_batch`.

ROLE NOTE: `test_data` is the true out-of-sample TEST set — it never enters
training/tuning, so the honest final metric is computed on it (inference / Phase 3).
The `X_val` produced by `split_train` is the VALIDATION set
used for tuning/selection. Keep `strategy: random` so `test_data` stays
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
    """learning_data = training pool, test_data = out-of-sample batch.

    strategy='random' -> test_data ~ same distribution as learning_data (drift baseline).
    strategy='biased' -> test_data oversamples one district (drift demo).
    """
    strategy = parameters.get("strategy", "random")
    seed = parameters["random_state"]
    ref_frac = parameters["ref_frac"]

    learning_data = ingested_data.sample(frac=ref_frac, random_state=seed)
    remaining = ingested_data.drop(learning_data.index)

    if strategy == "biased":
        col = parameters.get("bias_column", "District")
        district = parameters["bias_district"]
        test_data = remaining[remaining[col] == district]
        if test_data.empty:
            logger.warning("No rows for %s=%s — falling back to random test_data", col, district)
            test_data = remaining
    else:
        test_data = remaining

    logger.info("strategy=%s | learning_data=%s, test_data=%s",
                strategy, learning_data.shape, test_data.shape)
    return learning_data, test_data
