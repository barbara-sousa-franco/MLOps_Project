"""Pipeline `preproc_after_split` — post-split fit-on-train transforms (impute/cap/encode/scale)."""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import impute_missing, cap_outliers, encode_categoricals, scale_features
from kedro_temp_mlops.utils import upload_engineered_to_fs


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=impute_missing,
            inputs=["X_train_data", "X_val_data", "params:preproc_after_split"],
            outputs=["X_train_imputed", "X_val_imputed", "imputer"],
            name="impute_missing",
        ),
        node(
            func=cap_outliers,
            inputs=["X_train_imputed", "X_val_imputed", "params:preproc_after_split"],
            outputs=["X_train_processed", "X_val_processed", "capper"],
            name="cap_outliers",
        ),
        node(
            func=encode_categoricals,
            inputs=["X_train_processed", "X_val_processed", "y_train_data", "params:preproc_after_split"],
            outputs=["X_train_encoded", "X_val_encoded", "target_encoder"],
            name="encode_categoricals",
        ),
        node(
            func=scale_features,
            inputs=["X_train_encoded", "X_val_encoded", "params:preproc_after_split"],
            outputs=["X_train_scaled", "X_val_scaled", "scaler"],
            name="scale_features",
        ),
        node(
            func=upload_engineered_to_fs,
            inputs=["X_train_scaled", "y_train_data", "params:ingestion"],
            outputs="engineered_fs_done",
            name="upload_engineered_to_fs_node",
        ),
    ])
