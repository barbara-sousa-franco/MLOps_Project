"""Nodes for the `split_data` pipeline.

WARNING — BUG IN THE EXAMPLE NOT TO COPY: `stratify=y` only works for classification;
it breaks for regression. See TODO below.
"""

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def split_data(preprocessed_training_data: pd.DataFrame, parameters: dict):
    """Separate features/target and perform a train/test split (regression).

    Args:
        preprocessed_training_data: output of `preprocessing_train`.
        parameters: test_size, seed, strategy (parameters_split.yml).

    Returns:
        Tuple (X_train, X_test, y_train, y_test, columns):
          - columns: X_train.columns (maps to best_columns in catalog, as in the example).

    TODO split_data:
      - assert no nulls (as in the example)
      - separate target (Price) from features; drop the "index" column
      - train_test_split with seed and test_size from parameters
      - WARNING: do NOT use stratify=y (bug in the example — only works for classification).
        Regression: either no stratify, OR create Price bins (pd.qcut) and stratify by
        bin to ensure the test set covers the full price range (recommended).
      - also return X_train.columns
      - test set is sacred: do not touch until final evaluation
    """
    # TODO: implement
    raise NotImplementedError
