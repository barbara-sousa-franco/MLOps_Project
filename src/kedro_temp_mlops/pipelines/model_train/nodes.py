"""Nodes da pipeline `model_train`.

Métricas de REGRESSÃO (RMSE, MAE, R²), nunca accuracy. Comparar sempre com baseline
(média do Price) — a skill exige baseline.

⚠️ BUG DO EXEMPLO A NÃO COPIAR: `except:` nu ao carregar o champion. Apanhar
`FileNotFoundError` específico.
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
    """Treina o champion, avalia (regressão) e guarda modelo + métricas + colunas.

    Args:
        X_train, X_test, y_train, y_test: dados do split.
        parameters: baseline_model_params, use_feature_selection, etc. (parameters_model_train.yml).
        selected_model: melhor modelo vindo de `model_selection` (tipo/hiperparâmetros).
        best_columns: colunas selecionadas por SHAP (feature_selection). OPCIONAL e não
            cabeado no 1º passe (evita ciclo com feature_selection). No 2º passe
            (use_feature_selection=true) cabear best_columns como input persistido do run
            anterior.

    Returns:
        Tuple (production_model, production_columns, production_model_metrics):
          - métricas: RMSE, MAE, R² + comparação com baseline.

    TODO model_train:
      - ler experiment_name do mlflow.yml; mlflow.sklearn.autolog
      - carregar champion existente:
          try: pickle.load(production_model.pkl)
          except FileNotFoundError:   # ⚠️ específico, NÃO except nu (bug do exemplo)
              usar baseline RandomForestRegressor(**parameters["baseline_model_params"])
      - se parameters["use_feature_selection"]: X_train/X_test = X[best_columns]
      - treinar; prever; métricas REGRESSÃO: RMSE, MAE, R² (não accuracy!)
      - comparar com baseline (média do Price) — a skill exige baseline
      - guardar results_dict + production_model.pkl + production_cols.pkl
    """
    # TODO: implementar
    raise NotImplementedError


def register_model(model, metrics: dict, parameters: dict):
    """Regista o modelo no MLflow Model Registry (champion/challenger). NÃO está no exemplo.

    Args:
        model: modelo treinado em `model_train`.
        metrics: métricas do modelo (para a decisão de promoção).
        parameters: nome do registered model + regras de promoção.

    Returns:
        ModelVersion registada (e eventual promoção a champion).

    TODO register_model:   # construção vossa — não existe no exemplo
      - mlflow.register_model(model_uri, name="house_price_model")
      - definir stage/alias: novo modelo entra como "challenger"
      - se bater o champion (RMSE menor) -> promover a "champion"
      - usar MlflowClient().set_registered_model_alias / transition_model_version_stage
    """
    # TODO: implementar
    raise NotImplementedError
