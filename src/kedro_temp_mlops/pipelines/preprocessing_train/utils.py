"""Feature engineering functions for housing data (equivalent to the professor's utils.py).

One function per feature group — small, pure and testable (covered in
tests/pipelines/test_data_preprocessing.py).
"""

import pandas as pd


def property_age(df: pd.DataFrame, parameters: dict) -> pd.DataFrame:
    """Create the property age feature.

    TODO property_age:
      - age = parameters["reference_year"] - ConstructionYear  (ref_year in parameters, e.g. 2026)
      - handle null ConstructionYear (leave as NaN for downstream imputation)
    """
    # TODO: implement
    raise NotImplementedError


def bin_area(df: pd.DataFrame, parameters: dict) -> pd.DataFrame:
    """Bin GrossArea/LivingArea into ranges.

    TODO bin_area:
      - pd.cut with boundaries defined in parameters (nothing hard-coded)
      - generate categorical area-range column(s)
    """
    # TODO: implement
    raise NotImplementedError


def energy_to_ordinal(df: pd.DataFrame, parameters: dict) -> pd.DataFrame:
    """Map EnergyCertificate (A+, A, B, ...) to an ordinal scale.

    TODO energy_to_ordinal:
      - use the ordinal map in parameters (e.g. {"A+": 7, "A": 6, ..., "F": 1, "NC": 0})
      - values outside the map -> NaN/unknown category
    """
    # TODO: implement
    raise NotImplementedError
