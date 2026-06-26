"""Pytest tests for preprocessing — covers both pipelines:
  - preprocessing_train.nodes  (clean_data)
  - preproc_after_split.nodes  (GroupImputer, PercentileCapper, encode, scale)

Runs on the sample CSV, NOT to be confused with the data_unit_tests GX pipeline.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from kedro_temp_mlops.pipelines.preprocessing_train.nodes import clean_data
from kedro_temp_mlops.pipelines.preproc_after_split.nodes import (
    GroupImputer,
    PercentileCapper,
    encode_categoricals,
    scale_features,
)

SAMPLE_PATH = Path(__file__).parent / "sample" / "sample.csv"

PARAMS = {
    "max_construction_year": 2026, "min_construction_year": 1800,
    "max_bedrooms": 20, "max_bathrooms": 20, "max_wc": 20, "max_total_rooms": 50,
}


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """~200 representative rows from portugal_listings for fast tests."""
    return pd.read_csv(SAMPLE_PATH)


# ---------------------------------------------------------------------------
# clean_data  (preprocessing_train)
# ---------------------------------------------------------------------------

def test_clean_data_returns_report(sample_df):
    _, report = clean_data(sample_df, PARAMS)
    assert isinstance(report, dict)


def test_clean_data_target_not_null(sample_df):
    df, _ = clean_data(sample_df, PARAMS, has_target=True, drop_missing_target=True)
    assert df["Price_log"].notnull().all()


def test_clean_data_batch_has_no_target(sample_df):
    df, _ = clean_data(sample_df, PARAMS, has_target=False, drop_missing_target=False)
    assert "Price_log" not in df.columns


def test_clean_data_log_transform(sample_df):
    df, _ = clean_data(sample_df, PARAMS)
    assert "LivingArea_log" in df.columns and "LivingArea" not in df.columns


def test_clean_data_energy_numeric(sample_df):
    df, _ = clean_data(sample_df, PARAMS)
    assert pd.api.types.is_numeric_dtype(df["EnergyCertificate"])


# ---------------------------------------------------------------------------
# GroupImputer  (preproc_after_split)
# ---------------------------------------------------------------------------

def test_group_imputer_group_median():
    train = pd.DataFrame({"Type": ["House", "House", "Land", "Land"],
                          "Area": [100, 200, 1000, np.nan]})
    imp = GroupImputer("Type", ["Area"], []).fit(train)
    out = imp.transform(pd.DataFrame({"Type": ["Land"], "Area": [np.nan]}))
    assert out.loc[0, "Area"] == 1000               # filled with Land's median


def test_group_imputer_global_fallback():
    train = pd.DataFrame({"Type": ["House", "House"], "Area": [100, 200]})
    imp = GroupImputer("Type", ["Area"], []).fit(train)
    out = imp.transform(pd.DataFrame({"Type": ["Manor"], "Area": [np.nan]}))
    assert out.loc[0, "Area"] == train["Area"].median()   # unseen Type -> global


def test_no_nulls_after_imputation(sample_df):
    df, _ = clean_data(sample_df, PARAMS)
    group_col = "Type"
    num = df.select_dtypes(include=["number"]).columns.tolist()
    cat = [c for c in df.select_dtypes(include=["object"]).columns if c != group_col]
    imp = GroupImputer(group_col, num, cat).fit(df)
    out = imp.transform(df)

    # Columns that are entirely empty in the sample cannot be imputed (no value to learn from).
    # The imputer must fill every numeric column that has at least one observed value.
    fillable = [c for c in num if df[c].notna().any()]
    assert out[fillable].isna().sum().sum() == 0

# ---------------------------------------------------------------------------
# PercentileCapper  (preproc_after_split)
# ---------------------------------------------------------------------------

def test_percentile_capper_caps_upper():
    train = pd.DataFrame({"A": [1, 2, 3, 4, 100]})
    capper = PercentileCapper({"A": 0.8}).fit(train)
    out = capper.transform(pd.DataFrame({"A": [1, 100]}))
    assert out["A"].max() <= train["A"].quantile(0.8)


# ---------------------------------------------------------------------------
# encode_categoricals — anti-leakage  (preproc_after_split)
# ---------------------------------------------------------------------------

def test_encoder_not_refit_on_transform():
    X_train = pd.DataFrame({
        "District": ["Lisboa","Porto","Lisboa","Faro","Porto","Lisboa","Faro","Porto","Lisboa","Faro"],
        "Type":     ["House","Land","House","Store","Land","House","Store","Land","House","Store"],
    })
    y_train = pd.Series([12.0, 11.0, 12.5, 10.5, 11.2, 12.3, 10.8, 11.1, 12.6, 10.4])
    X_val = pd.DataFrame({"District": ["Lisboa","Porto"], "Type": ["House","Land"]})

    _, _, encoder = encode_categoricals(X_train.copy(), X_val.copy(), y_train, {"random_state": 42})

    before = encoder.categories_
    _ = encoder.transform(X_val[["District","Type"]])
    after = encoder.categories_
    assert all(np.array_equal(b, a) for b, a in zip(before, after))

## Scale features (preproc_after_split)
  
def test_scaler_standardises_train():
    X_train = pd.DataFrame({"A": [10.0, 20, 30, 40, 50], "B": [1.0, 2, 3, 4, 5]})
    X_val = pd.DataFrame({"A": [25.0, 35], "B": [2.5, 3.5]})
    Xtr, Xval, scaler = scale_features(X_train.copy(), X_val.copy(), {})

    # train columns are standardised: mean ~0, std ~1
    assert np.allclose(Xtr["A"].mean(), 0, atol=1e-9)
    assert np.allclose(Xtr["A"].std(ddof=0), 1, atol=1e-9)


def test_scaler_uses_train_stats_on_val():
    X_train = pd.DataFrame({"A": [10.0, 20, 30, 40, 50]})
    X_val = pd.DataFrame({"A": [30.0]})          # equals train mean
    Xtr, Xval, scaler = scale_features(X_train.copy(), X_val.copy(), {})

    # 30 is the TRAIN mean, so it scales to ~0 using train stats (not val's own)
    assert np.allclose(Xval["A"].iloc[0], 0, atol=1e-9)