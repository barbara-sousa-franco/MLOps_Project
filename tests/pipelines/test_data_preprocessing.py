"""Testes pytest do preprocessing (corre na sample, NÃO confundir com data_unit_tests)."""

from pathlib import Path

import pandas as pd
import pytest

SAMPLE_PATH = Path(__file__).parent / "sample" / "sample.csv"


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """~200 linhas representativas do portugal_listings para testes rápidos."""
    return pd.read_csv(SAMPLE_PATH)


def test_clean_data_no_nulls_after_imputation(sample_df):
    """clean_data não deve deixar nulls nas colunas imputadas.

    TODO:
      - chamar clean_data(sample_df, parameters)
      - assert resultado[colunas_imputadas].isna().sum() == 0
    """
    pytest.skip("TODO: implementar quando clean_data estiver pronto")


def test_encoder_not_fit_on_test(sample_df):
    """O encoder é fit só no treino — anti-leakage.

    TODO:
      - fit encoder no treino; garantir que transform no batch NÃO altera o encoder
        (mesmas categorias/estatísticas)
    """
    pytest.skip("TODO: implementar quando encode_features estiver pronto")
