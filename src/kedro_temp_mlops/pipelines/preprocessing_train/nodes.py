"""Nodes da pipeline `preprocessing_train`.

O encoder/scaler é treinado (`fit`) AQUI e SÓ aqui (anti-leakage). É devolvido para
ser guardado em encoder.pkl (artifact MLflow) e reutilizado no batch (só `transform`).
"""

import logging

import pandas as pd

from .utils import bin_area, energy_to_ordinal, property_age  # noqa: F401  (FE usada em clean/encode)

logger = logging.getLogger(__name__)


def clean_data(ingested_data: pd.DataFrame, parameters: dict) -> pd.DataFrame:
    """Limpa os dados crus das casas (valores impossíveis, outliers, missing).

    Args:
        ingested_data: dataset vindo da ingestion.
        parameters: estratégia de limpeza/imputação/outliers (parameters_preprocessing.yml).

    Returns:
        DataFrame limpo (ainda sem encoding).

    TODO clean_data:
      - remover/corrigir valores impossíveis identificados no EDA
      - tratar outliers extremos (o ~1.38 bilião €, áreas negativas)
      - imputação de missing (estratégia em parameters: mediana/moda/constante)
      - aplicar funções de FE de utils.py (property_age, bin_area, energy_to_ordinal)
    """
    # TODO: implementar
    raise NotImplementedError


def encode_features(clean_df: pd.DataFrame, parameters: dict):
    """Faz fit do encoder/scaler no treino e devolve dados transformados + encoder.

    Args:
        clean_df: saída de `clean_data`.
        parameters: config de encoding/scaling e flag `use_log_target`.

    Returns:
        Tuple (preprocessed_df, encoder):
          - preprocessed_df: features codificadas + target (log1p se use_log_target).
          - encoder: objeto ajustado a guardar em encoder.pkl.

    TODO encode_features:
      - FIT do encoder/scaler AQUI (só treino) — anti-leakage
      - devolver encoder p/ guardar em encoder.pkl (artifact MLflow)
      - aplicar log1p ao target (Price) se parameters["use_log_target"]
        (EDA mostra skew forte — testar e documentar em ASSUMPTIONS.md)
    """
    # TODO: implementar
    raise NotImplementedError
