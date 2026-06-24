"""Nodes for the `model_predict` pipeline.

Runs the champion on the preprocessed out-of-sample batch (`test_data`). 
Predictions are made on the log scale and inverted with expm1 to euros. When the
batch carries the target (`batch_has_target: true`), this is also where the true test
metric is computed for the final evaluation.
"""

import logging

import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

logger = logging.getLogger(__name__)


def predict(production_model, preprocessed_batch_data: pd.DataFrame,
            base_target: str = "Price", use_log_target: bool = True):
    """Predict Price on the out-of-sample batch with the champion; evaluate if labelled.

    Args:
        production_model: the trained champion.
        preprocessed_batch_data: batch after `preprocessing_batch` (same transforms as
            training, applied via transform only). Carries the log target `Price_log`
            when `batch_has_target` is true.
        base_target: raw target name (e.g. "Price"); the log column is "<base>_log".
        use_log_target: whether the model was trained on log1p(target).

    Returns:
        Tuple (predictions, test_metrics):
          - predictions: DataFrame with `prediction_log` and `prediction_eur` (and the
            true target columns when available).
          - test_metrics: dict of honest test metrics (empty if the batch is unlabelled).
    """
    # clean_data builds the log target as f"{Price}_log" and drops the raw Price column,
    # so the labelled column present in the batch is e.g. "Price_log".
    target_col = f"{base_target}_log" if use_log_target else base_target

    # the champion was fit on a DataFrame, so it knows exactly which feature columns it expects
    feature_cols = list(getattr(production_model, "feature_names_in_", preprocessed_batch_data.columns))
    feature_cols = [c for c in feature_cols if c in preprocessed_batch_data.columns]
    X = preprocessed_batch_data[feature_cols]

    pred = production_model.predict(X)
    predictions = pd.DataFrame(index=preprocessed_batch_data.index)
    if use_log_target:
        predictions["prediction_log"] = pred
        predictions["prediction_eur"] = np.expm1(pred)
    else:
        predictions["prediction_eur"] = pred

    test_metrics: dict = {}
    if target_col in preprocessed_batch_data.columns:
        # honest TEST evaluation (the out-of-sample batch is labelled)
        y_log = preprocessed_batch_data[target_col].to_numpy()
        pred_log = pred if use_log_target else np.log1p(pred)
        # a few rows can have a NaN target (impossible prices set to NA in cleaning, kept
        # because drop_missing_target=False) — exclude them from the metric, keep predictions.
        # keep the full true target on the predictions output (NaN where missing)
        predictions[target_col] = y_log
        predictions["true_eur"] = np.expm1(y_log)
        # exclude rows with a NaN target from the metric only (prediction still exists)
        labelled = ~np.isnan(y_log)
        n_dropped = int((~labelled).sum())
        if n_dropped:
            logger.info("Excluding %d row(s) with missing target from the test metric.", n_dropped)
        y_log_m, pred_log_m = y_log[labelled], pred_log[labelled]
        y_eur_m, pred_eur_m = np.expm1(y_log_m), np.expm1(pred_log_m)
        test_metrics = {
            "test_rmse_log": float(root_mean_squared_error(y_log_m, pred_log_m)),
            "test_mae_log": float(mean_absolute_error(y_log_m, pred_log_m)),
            "test_r2": float(r2_score(y_log_m, pred_log_m)),
            "test_rmse_eur": float(root_mean_squared_error(y_eur_m, pred_eur_m)),
            "test_mae_eur": float(mean_absolute_error(y_eur_m, pred_eur_m)),
        }
        logger.info(
            "HONEST TEST (n=%d): RMSE(log)=%.4f, R2=%.4f | RMSE=%.0f EUR, MAE=%.0f EUR",
            int(labelled.sum()), test_metrics["test_rmse_log"], test_metrics["test_r2"],
            test_metrics["test_rmse_eur"], test_metrics["test_mae_eur"],
        )
        if mlflow.active_run() is not None:
            mlflow.log_metrics(test_metrics)
    else:
        logger.info("Predicted %d rows (unlabelled batch — no test metrics).", len(X))

    return predictions, test_metrics
