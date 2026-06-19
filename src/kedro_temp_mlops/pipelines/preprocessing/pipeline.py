"""Pipeline for `preprocessing_train`.

Follows the professor's bank example pattern, adapted to the housing use case.

Node sequence:
  1. clean_data        — pre-split cleaning (safe, no data statistics)
  2. split_data        — train/test split (80/20)
  3. impute_missing    — global median/mode imputation (fitted on train)
  4. cap_outliers      — percentile capping (fitted on train)

Outputs:
  - cleaned_data           -> 02_intermediate (pre-split, for data unit tests)
  - X_train, X_test        -> 03_primary (post-split, semi-processed)
  - y_train, y_test        -> 03_primary
  - X_train_processed,
    X_test_processed       -> 03_primary (post-imputation + capping)
  - num_imputer,
    cat_imputer, capper    -> 04_feature (artefacts for batch inference)
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import clean_data, split_data, impute_missing, cap_outliers


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([

        # Pre-split cleaning
        node(
            func=clean_data,
            inputs=["raw_house_data", "params:preprocessing"],
            outputs=["cleaned_data", "reporting_data_preprocessing"],
            name="clean_data",
        ),

        # Train/test split
        node(
            func=split_data,
            inputs=["cleaned_data", "params:preprocessing"],
            outputs=["X_train_data", "X_test_data", "y_train_data", "y_test_data"],
            name="split_data",
        ),

        # Imputation (fitted on train)
        node(
            func=impute_missing,
            inputs=["X_train_data", "X_test_data", "params:preprocessing"],
            outputs=[
                "X_train_imputed",
                "X_test_imputed",
                "num_imputer",
                "cat_imputer",
            ],
            name="impute_missing",
        ),

        # Outlier capping (fitted on train)
        node(
            func=cap_outliers,
            inputs=["X_train_imputed", "X_test_imputed", "params:preprocessing"],
            outputs=["X_train_processed", "X_test_processed", "capper"],
            name="cap_outliers",
        ),

    ])