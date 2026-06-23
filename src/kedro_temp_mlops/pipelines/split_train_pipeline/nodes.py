"""
This is a boilerplate pipeline 'split_train_pipeline'
generated using Kedro 1.3.1
"""
"""Train/validation split of the cleaned (pre-transform) data. Runs BEFORE the
fit-on-train transforms so those fit on the training split only.

Outputs:
  - X_train / y_train: training set — transformers FIT here.
  - X_val   / y_val  : validation set — used for tuning/selection, LEAK-FREE.

The true out-of-sample TEST set is `test_data` (from the `split_data` pipeline).
"""

import logging
from typing import Any, Dict, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


def split_train(
    cleaned_data: pd.DataFrame,
    parameters: Dict[str, Any],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, list]:
    """Separate target/features and split (regression-safe)."""
    target_col = parameters["target_col"]                 # 'Price_log'
    assert cleaned_data[target_col].notnull().all(), "target has nulls"

    y = cleaned_data[target_col]
    X = cleaned_data.drop(columns=[target_col])
    if "index" in X.columns:
        X = X.drop(columns="index")

    # NEVER stratify=y on a continuous target — bin it instead.
    strat = None
    n_bins = parameters.get("stratify_bins")
    if n_bins:
        strat = pd.qcut(y, q=n_bins, labels=False, duplicates="drop")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=parameters["test_size"],
        random_state=parameters["random_state"],
        stratify=strat,
    )
    logger.info("Split: X_train=%s, X_val=%s", X_train.shape, X_val.shape)
    return X_train, X_val, y_train, y_val, list(X_train.columns)
