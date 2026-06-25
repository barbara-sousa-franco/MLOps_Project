"""Nodes for the `preprocessing_train` pipeline.

Follows the professor's pattern (bank example) but adapted to the housing use case.
Steps mirror the preprocessing notebook exactly:

Pre-split (no data statistics used — safe before split):
  1. drop_duplicates
  2. drop_columns
  3. remove_outside_portugal
  4. impossible_values_to_nan
  5. implausible_values_to_nan
  6. drop_missing_target
  7. consolidate_energy_certificate
  8. fix_non_residential_zeros
  9. fix_dtypes
  10. encode_booleans
  11. encode_energy_certificate (ordinal — no data statistics)
  12. log_transform (areas + target — no data statistics)
  13. split_data

Post-split (fitted on train only — applied to both):
  14. impute_missing (global median on train)
  15. cap_outliers (percentile capping on train)
"""

import logging
from typing import Dict, Tuple, Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin



logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================

NON_RESIDENTIAL_TYPES = [
    'Land', 'Garage', 'Warehouse', 'Storage',
    'Industrial', 'Store', 'Office', 'Other - Commercial'
]

ENERGY_ORDER = ['A+', 'A', 'B', 'B-', 'C', 'D', 'E', 'F', 'G', 'No Rating']
ENERGY_MAP = {v: i for i, v in enumerate(reversed(ENERGY_ORDER))}

LOG_AREA_COLS = [
    'LivingArea', 'TotalArea', 'GrossArea', 'BuiltArea', 'LotSize'
]

DISCRETE_COLS = [
    'NumberOfBedrooms', 'NumberOfBathrooms', 'NumberOfWC',
    'TotalRooms', 'Parking', 'ConstructionYear'
]

BOOL_COLS = ['Garage', 'Elevator', 'ElectricCarsCharging', 'HasParking']

BOOL_MAP = {True: 1, False: 0, 'True': 1, 'False': 0}

DROP_COLS = [
    'PublishDate', 'Floor', 'ConservationStatus',
    'EnergyEfficiencyLevel', 'City', 'Town'
]

NO_RATING_VALUES = ['NC', 'Not available', 'No Certificate']

LAND_ZERO_COLS = [
    'NumberOfBedrooms', 'NumberOfBathrooms', 'NumberOfWC',
    'TotalRooms', 'LivingArea', 'BuiltArea'
]

COMMERCIAL_ZERO_COLS = [
    'NumberOfBedrooms', 'NumberOfBathrooms', 'NumberOfWC',
    'TotalRooms', 'LivingArea'
]


# =============================================================================
# CUSTOM TRANSFORMER
# =============================================================================

class PercentileCapper(BaseEstimator, TransformerMixin):
    """Caps numeric columns at given percentile thresholds.
    
    Fitted on train only — thresholds are computed from the training set
    and applied to both train and test to avoid data leakage.
    """

    def __init__(self, thresholds: Dict[str, float]):
        self.thresholds = thresholds

    def fit(self, X: pd.DataFrame, y=None):
        self.caps_ = {
            col: X[col].quantile(pct)
            for col, pct in self.thresholds.items()
            if col in X.columns
        }
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        X = X.copy()
        for col, cap in self.caps_.items():
            X[col] = X[col].clip(upper=cap)
        return X


# =============================================================================
# PRE-SPLIT NODES
# =============================================================================

def clean_data(
    raw_data: pd.DataFrame,
    parameters: Dict[str, Any],
    has_target: bool = True,
    drop_missing_target: bool = True,
) -> Tuple[pd.DataFrame, Dict]:
    """Clean raw data before the train/test split.

    Statistics-free — safe to run on the full dataset (train) or on a new
    batch (inference). `has_target` / `drop_missing_target` default to the
    training behaviour, so train output is unchanged.
    """
    df = raw_data.copy()
    report = {}

    # 1. Remove duplicates
    n_before = len(df)
    df = df.drop_duplicates()
    report['duplicates_removed'] = n_before - len(df)
    logger.info("Duplicates removed: %d", report['duplicates_removed'])

    # 2. Drop irrelevant/unusable columns
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
    logger.info("Columns after drop: %d", df.shape[1])

    # 3. Remove rows outside Portugal
    if 'District' in df.columns:
        n_before = len(df)
        df = df[df['District'] != 'Z - Fora de Portugal']
        report['outside_portugal_removed'] = n_before - len(df)
        logger.info("Rows outside Portugal removed: %d", report['outside_portugal_removed'])

    # 4. Impossible values -> NA  (Price only when a target is present)
    impossible = {
        'GrossArea':        df['GrossArea'] < 0,
        'TotalArea':        df['TotalArea'] < 0,
        'LivingArea':       df['LivingArea'] < 0,
        'BuiltArea':        df['BuiltArea'] < 0,
        'LotSize':          df['LotSize'] < 0,
        'NumberOfWC':       df['NumberOfWC'] < 0,
        'NumberOfBathrooms':df['NumberOfBathrooms'] < 0,
        'NumberOfBedrooms': df['NumberOfBedrooms'] < 0,
        'TotalRooms':       df['TotalRooms'] < 0,
        'Parking':          df['Parking'] < 0,
        'ConstructionYear': (
            (df['ConstructionYear'] > parameters['max_construction_year']) |
            (df['ConstructionYear'] < parameters['min_construction_year'])
        ),
    }
    if has_target and 'Price' in df.columns:
        impossible['Price'] = df['Price'] <= 0

    impossible_counts = {}
    for col, mask in impossible.items():
        n = mask.fillna(False).sum()
        if n > 0:
            df.loc[mask, col] = np.nan
            impossible_counts[col] = int(n)
    report['impossible_values_to_nan'] = impossible_counts
    logger.info("Impossible values → NA: %s", impossible_counts)

    # 5. Implausible values -> NA  (Price only when a target is present)
    implausible = {
        'NumberOfBedrooms':  df['NumberOfBedrooms'] > parameters['max_bedrooms'],
        'NumberOfBathrooms': df['NumberOfBathrooms'] > parameters['max_bathrooms'],
        'NumberOfWC':        df['NumberOfWC'] > parameters['max_wc'],
        'TotalRooms':        df['TotalRooms'] > parameters['max_total_rooms'],
    }
    if has_target and 'Price' in df.columns:
        implausible['Price'] = df['Price'] == 1

    implausible_counts = {}
    for col, mask in implausible.items():
        n = mask.fillna(False).sum()
        if n > 0:
            df.loc[mask, col] = np.nan
            implausible_counts[col] = int(n)
    report['implausible_values_to_nan'] = implausible_counts
    logger.info("Implausible values → NA: %s", implausible_counts)

    # 6. Drop rows with missing target (train only)
    if drop_missing_target and 'Price' in df.columns:
        n_before = len(df)
        df = df.dropna(subset=['Price'])
        report['missing_target_removed'] = n_before - len(df)
        logger.info("Rows removed (missing Price): %d", report['missing_target_removed'])

    # 7. Consolidate EnergyCertificate
    df['EnergyCertificate'] = df['EnergyCertificate'].replace(NO_RATING_VALUES, 'No Rating')
    df['EnergyCertificate'] = df['EnergyCertificate'].fillna('No Rating')

    # 8. Non-residential types -> 0 for area/room columns
    land_mask = df['Type'] == 'Land'
    for col in LAND_ZERO_COLS:
        if col in df.columns:
            df.loc[land_mask, col] = df.loc[land_mask, col].fillna(0)

    commercial_mask = df['Type'].isin([t for t in NON_RESIDENTIAL_TYPES if t != 'Land'])
    for col in COMMERCIAL_ZERO_COLS:
        if col in df.columns:
            df.loc[commercial_mask, col] = df.loc[commercial_mask, col].fillna(0)

    # 9. Fix dtypes: discrete numerics -> nullable Int64
    for col in DISCRETE_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')

    # 10. Boolean columns -> 0/1
    for col in BOOL_COLS:
        if col in df.columns:
            df[col] = df[col].map(BOOL_MAP)

    # 11. Ordinal encoding for EnergyCertificate
    df['EnergyCertificate'] = df['EnergyCertificate'].map(ENERGY_MAP)

    # 12. log1p transform: areas always, target only when present
    for col in LOG_AREA_COLS:
        if col in df.columns:
            df[f'{col}_log'] = np.log1p(df[col])
            df = df.drop(columns=[col])

    if has_target and 'Price' in df.columns:
        df['Price_log'] = np.log1p(df['Price'])
        df = df.drop(columns=['Price'])

    report['final_shape'] = df.shape
    report['missing_values'] = df.isnull().sum()[df.isnull().sum() > 0].to_dict()
    logger.info("Cleaned data shape: %s", df.shape)

    # Normalise all string columns to clean UTF-8
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].apply(
            lambda x: x.encode('latin-1', errors='replace').decode('utf-8', errors='replace')
            if isinstance(x, str) else x
        )

    return df, report




