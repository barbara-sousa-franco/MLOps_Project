"""Nodes da pipeline `feature_selection` (SHAP).

⚠️ BUG DO EXEMPLO A NÃO COPIAR: `shap_values[:,:,1]` é índice de classe (classificação).
Em REGRESSÃO não há eixo de classes — usar `shap_values` 2D direto.
"""

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def compute_shap(production_model, X_train: pd.DataFrame, parameters: dict):
    """Calcula valores SHAP do champion e gera o summary plot.

    Args:
        production_model: champion treinado (model_train).
        X_train: features de treino.
        parameters: config SHAP (parameters_model_train.yml ou _model_selection.yml).

    Returns:
        Tuple (shap_values, shap_plot) — shap_plot vai como artifact MLflow (png).

    TODO compute_shap:
      - shap.TreeExplainer(model); shap_values = explainer(X_train)
      - ⚠️ shap.summary_plot(shap_values, X_train, ...) — 2D direto.
        NÃO usar shap_values[:,:,1] (índice de classe; bug do exemplo)
      - gerar shap_plot.png -> artifact MLflow
    """
    # TODO: implementar
    raise NotImplementedError


def select_features(shap_values, X_train: pd.DataFrame, parameters: dict):
    """Seleciona as top-N features por |SHAP| médio.

    Args:
        shap_values: saída de `compute_shap`.
        X_train: features de treino (para mapear nomes de colunas).
        parameters: N ou threshold de seleção.

    Returns:
        best_cols — lista de colunas a guardar em best_cols.pkl (artifact).

    TODO select_features:
      - top-N features por |SHAP| médio (N ou threshold em parameters)
      - guardar best_cols.pkl (artifact)
      - (dica do prof: feature selection a partir do SHAP; best_cols realimenta
        model_train via use_feature_selection)
    """
    # TODO: implementar
    raise NotImplementedError
