"""
This is a boilerplate pipeline 'feature_engineering'
generated using Kedro 1.3.1
"""
"""Post-split transforms, all fitted on the training split only:
impute -> cap -> target-encode.
"""

import logging
from typing import Any, Dict, Tuple

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import TargetEncoder

logger = logging.getLogger(__name__)

TARGET_ENC_COLS = ['District', 'Type']


class PercentileCapper(BaseEstimator, TransformerMixin):
    """Caps numeric columns at given percentile thresholds (fitted on train)."""

    def __init__(self, thresholds: Dict[str, float]):
        self.thresholds = thresholds

    def fit(self, X: pd.DataFrame, y=None):
        self.caps_ = {col: X[col].quantile(pct)
                      for col, pct in self.thresholds.items() if col in X.columns}
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        X = X.copy()
        for col, cap in self.caps_.items():
            X[col] = X[col].clip(upper=cap)
        return X


def impute_missing(X_train, X_test, parameters):
    """Median (numeric) / most-frequent (categorical). Fitted on train."""
    numeric_cols = X_train.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()

    num_imputer = SimpleImputer(strategy='median').fit(X_train[numeric_cols])
    X_train_imp, X_test_imp = X_train.copy(), X_test.copy()
    X_train_imp[numeric_cols] = num_imputer.transform(X_train[numeric_cols])
    X_test_imp[numeric_cols] = num_imputer.transform(X_test[numeric_cols])

    if categorical_cols:
        cat_imputer = SimpleImputer(strategy='most_frequent').fit(X_train[categorical_cols])
        X_train_imp[categorical_cols] = cat_imputer.transform(X_train[categorical_cols])
        X_test_imp[categorical_cols] = cat_imputer.transform(X_test[categorical_cols])
    else:
        cat_imputer = None

    logger.info("Imputation done. Missing train=%d test=%d",
                X_train_imp.isnull().sum().sum(), X_test_imp.isnull().sum().sum())
    return X_train_imp, X_test_imp, num_imputer, cat_imputer


def cap_outliers(X_train, X_test, parameters):
    """Percentile capping (fitted on train)."""
    capper = PercentileCapper(thresholds=parameters['percentile_thresholds']).fit(X_train)
    logger.info("Capping thresholds: %s", capper.caps_)
    return capper.transform(X_train), capper.transform(X_test), capper


def encode_categoricals(X_train, X_test, y_train, parameters):
    """Target-encode high-cardinality nominals. fit_transform on train
    (cross-fitted, leakage-safe); transform on test.
    """
    cols = [c for c in TARGET_ENC_COLS if c in X_train.columns]
    encoder = TargetEncoder(random_state=parameters['random_state'])

    X_train_enc, X_test_enc = X_train.copy(), X_test.copy()
    X_train_enc[cols] = encoder.fit_transform(X_train[cols], y_train)
    X_test_enc[cols] = encoder.transform(X_test[cols])

    logger.info("Target-encoded %s", cols)
    return X_train_enc, X_test_enc, encoder