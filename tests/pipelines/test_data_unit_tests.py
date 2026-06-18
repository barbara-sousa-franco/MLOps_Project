"""Testes pytest que verificam que os asserts GX APANHAM dados maus."""

import pytest


def test_asserts_catch_negative_price():
    """Injetar Price = -1 deve fazer o unit_test falhar (raise ValueError).

    TODO:
      - construir df mau (uma linha com Price=-1)
      - assert que unit_test(df_mau, parameters) levanta ValueError
    """
    pytest.skip("TODO: implementar quando unit_test estiver pronto")


def test_asserts_catch_invalid_construction_year():
    """ConstructionYear fora de [1800, 2026] deve falhar.

    TODO: df com ConstructionYear=1500 -> espera falha.
    """
    pytest.skip("TODO: implementar quando unit_test estiver pronto")


def test_traffic_light_written_on_success():
    """Com dados bons, write_traffic_light cria o ficheiro semáforo.

    TODO: correr unit_test + write_traffic_light em dados válidos e assert que a flag existe.
    """
    pytest.skip("TODO: implementar quando write_traffic_light estiver pronto")
