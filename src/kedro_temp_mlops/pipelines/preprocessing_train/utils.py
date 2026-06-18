"""Funções de feature engineering das casas (equivalente ao utils.py do prof).

Uma função por grupo de features — pequenas, puras e testáveis (cobertas em
tests/pipelines/test_data_preprocessing.py).
"""

import pandas as pd


def property_age(df: pd.DataFrame, parameters: dict) -> pd.DataFrame:
    """Cria a feature de idade do imóvel.

    TODO property_age:
      - idade = parameters["reference_year"] - ConstructionYear  (ref_year em parameters, p.ex. 2026)
      - tratar ConstructionYear nulo (deixar NaN para imputação a jusante)
    """
    # TODO: implementar
    raise NotImplementedError


def bin_area(df: pd.DataFrame, parameters: dict) -> pd.DataFrame:
    """Faz binning de GrossArea/LivingArea em faixas.

    TODO bin_area:
      - pd.cut com os limites definidos em parameters (nada hard-coded)
      - gerar coluna(s) categóricas de faixa de área
    """
    # TODO: implementar
    raise NotImplementedError


def energy_to_ordinal(df: pd.DataFrame, parameters: dict) -> pd.DataFrame:
    """Mapeia EnergyCertificate (A+, A, B, ...) para escala ordinal.

    TODO energy_to_ordinal:
      - usar o mapa ordinal em parameters (ex: {"A+": 7, "A": 6, ..., "F": 1, "NC": 0})
      - valores fora do mapa -> NaN/categoria desconhecida
    """
    # TODO: implementar
    raise NotImplementedError
