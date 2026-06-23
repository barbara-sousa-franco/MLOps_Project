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
from sklearn.preprocessing import StandardScaler

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
    
class GroupImputer(BaseEstimator, TransformerMixin):
    """Impute by group (e.g. per Type): numeric -> group median, categorical -> group mode.
    Falls back to the global median/mode when a group is unseen or has no value.
    Fitted on train only.
    """

    def __init__(self, group_col, numeric_cols, categorical_cols):
        self.group_col = group_col
        self.numeric_cols = numeric_cols
        self.categorical_cols = categorical_cols

    def fit(self, X, y=None):
        g = X.groupby(self.group_col)
        # per-group statistics
        self.num_by_group_ = g[self.numeric_cols].median() if self.numeric_cols else None
        self.cat_by_group_ = (
            g[self.categorical_cols].agg(lambda s: s.mode().iloc[0] if not s.mode().empty else None)
            if self.categorical_cols else None
        )
        # global fallbacks (for unseen groups / all-NaN groups)
        self.num_global_ = X[self.numeric_cols].median() if self.numeric_cols else None
        self.cat_global_ = (
            X[self.categorical_cols].agg(lambda s: s.mode().iloc[0] if not s.mode().empty else None)
            if self.categorical_cols else None
        )
        self.feature_names_in_ = list(X.columns)   # for batch column alignment
        return self

    def transform(self, X, y=None):
        X = X.copy()
        for col in self.numeric_cols:
            # map each row's group to that group's median; unseen group -> NaN -> global
            filled = X[self.group_col].map(self.num_by_group_[col])
            X[col] = X[col].fillna(filled).fillna(self.num_global_[col])
        for col in self.categorical_cols:
            filled = X[self.group_col].map(self.cat_by_group_[col])
            X[col] = X[col].fillna(filled).fillna(self.cat_global_[col])
        return X


def impute_missing(X_train, X_test, parameters):
    """Impute numeric (group median) and categorical (group mode) by Type.
    Fitted on train; global fallback for rare/unseen types."""
    group_col = parameters.get("impute_group_col", "Type")

    numeric_cols = X_train.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = [c for c in X_train.select_dtypes(include=['object']).columns
                        if c != group_col]   # don't impute the group key itself with itself

    imputer = GroupImputer(group_col, numeric_cols, categorical_cols).fit(X_train)
    X_train_imp = imputer.transform(X_train)
    X_test_imp = imputer.transform(X_test)

    logger.info("Group imputation by %s done. Missing train=%d test=%d",
                group_col, X_train_imp.isnull().sum().sum(), X_test_imp.isnull().sum().sum())
    return X_train_imp, X_test_imp, imputer


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


def scale_features(X_train, X_test, parameters):
    """Standardise numeric features. Fitted on train only; scaler reused on batch."""
    scale_cols = X_train.select_dtypes(include=['number']).columns.tolist()

    scaler = StandardScaler().fit(X_train[scale_cols])
    X_train_scaled, X_test_scaled = X_train.copy(), X_test.copy()
    X_train_scaled[scale_cols] = scaler.transform(X_train[scale_cols])
    X_test_scaled[scale_cols] = scaler.transform(X_test[scale_cols])

    logger.info("Scaled %d numeric columns", len(scale_cols))
    return X_train_scaled, X_test_scaled, scaler