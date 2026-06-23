"""Nodes for the `model_train` pipeline.

REGRESSION metrics (RMSE, MAE, R²), never accuracy. Always compare against a baseline
(mean of Price) — the skill requires a baseline.

WARNING — BUG IN THE EXAMPLE NOT TO COPY: bare `except:` when loading the champion.
Catch `FileNotFoundError` specifically.

=============================================================================
NOTES FOR IMPLEMENTERS (decisions already taken with the team — please follow):
-----------------------------------------------------------------------------
1. CHAMPION/CHALLENGER + MLflow Model Registry now live HERE (moved out of
   feature_selection to remove the overlap). feature_selection keeps only SHAP ->
   best_columns. So `register_model` is where the promotion logic belongs.

2. MODEL INPUTS = `X_train_scaled` / `X_test_scaled` (the 05_model_input layer, after
   impute -> cap -> encode -> scale). The pipeline is already wired to these. Whatever
   features the champion is TRAINED on, SHAP in feature_selection must use the SAME ones.

3. SPLIT SEMANTICS (professor's scheme — names kept):
   - `X_train` = training data (FIT here).
   - `X_test` = the leak-free VALIDATION set (despite the name) -> use it for the
     champion-vs-challenger PROMOTION decision.
   - `ana_data` (split_data) = the true out-of-sample TEST set -> the HONEST final metric
     is computed there later (Phase 3: preprocessing_batch -> model_predict), NOT here.
   So `production_model_metrics` here are validation metrics; the test number comes from ana_data.

4. `selected_model` (from model_selection) is ALREADY tuned and fit on X_train. You can use
   it directly, or refit (e.g. on train+validation) before the final test on ana_data.

5. BASELINE = `sklearn.dummy.DummyRegressor(strategy="mean")` (the naive "predict the mean"
   model to beat). This is DIFFERENT from the fallback champion when no champion exists yet.
=============================================================================
"""

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def model_train(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    parameters: dict,
    selected_model=None,
    best_columns=None,
):
    """Train the champion, evaluate (regression) and save model + metrics + columns.

    Args:
        X_train, X_test, y_train, y_test: data from the split.
        parameters: baseline_model_params, use_feature_selection, etc. (parameters_model_train.yml).
        selected_model: best model from `model_selection` (type/hyperparameters).
        best_columns: columns selected by SHAP (feature_selection). OPTIONAL and not
            wired in the first pass (avoids a cycle with feature_selection). In the second
            pass (use_feature_selection=true) wire best_columns as a persisted input from
            the previous run.

    Returns:
        Tuple (production_model, production_columns, production_model_metrics):
          - metrics: RMSE, MAE, R² + comparison against baseline.

    TODO model_train:
      - read experiment_name from mlflow.yml; mlflow.sklearn.autolog
      - load existing champion:
          try: pickle.load(production_model.pkl)
          except FileNotFoundError:   # WARNING: specific, NOT bare except (bug in example)
              use baseline RandomForestRegressor(**parameters["baseline_model_params"])
      - if parameters["use_feature_selection"]: X_train/X_test = X[best_columns]
      - train; predict; REGRESSION metrics: RMSE, MAE, R² (not accuracy!)
      - compare against baseline (mean of Price) — the skill requires a baseline
      - save results_dict + production_model.pkl + production_cols.pkl
    """
    # TODO: implement
    raise NotImplementedError


def register_model(model, metrics: dict, parameters: dict):
    """Register the model in the MLflow Model Registry (champion/challenger). NOT in the example.

    Args:
        model: model trained in `model_train`.
        metrics: model metrics (for the promotion decision).
        parameters: registered model name + promotion rules.

    Returns:
        Registered ModelVersion (and optional promotion to champion).

    TODO register_model:   # our own construction — not in the example
      - mlflow.register_model(model_uri, name="house_price_model")
      - define stage/alias: new model enters as "challenger"
      - if it beats the champion (lower RMSE) -> promote to "champion"
      - use MlflowClient().set_registered_model_alias / transition_model_version_stage
    """
    # TODO: implement
    raise NotImplementedError
