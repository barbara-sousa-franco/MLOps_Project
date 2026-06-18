# MLOps_Project — Previsão de preços de imóveis (Portugal)

Pipeline de MLOps end-to-end para prever `Price` (regressão) a partir de
`portugal_listings`. Stack: **Kedro + kedro-mlflow + MLflow (tracking + registry) +
Optuna + SHAP + Great Expectations + Hopsworks (feature store) + evidently/nannyml
(drift) + Prefect + Docker**.

> Estado: **esqueleto/scaffold**. Os nós têm a assinatura + docstring + TODOs e levantam
> `NotImplementedError`. Ver [BLUEPRINT.md](BLUEPRINT.md) para o mapa completo e a
> distribuição de trabalho, e [ASSUMPTIONS.md](ASSUMPTIONS.md) para pressupostos.

## Setup

> ⚠️ **Ambiente: usar SEMPRE o `.venv` gerido pelo `uv`. NÃO criar `.venv-mlops` nem
> outro venv à mão.** O `.venv` está no `.gitignore` (não vem do git) — cada pessoa
> recria-o localmente a partir do `uv.lock` com `uv sync`. Assim todas têm exatamente as
> mesmas versões.

```bash
# 1. ambiente (uv) — cria/atualiza o .venv a partir do uv.lock
uv sync

# 2. correr comandos no ambiente do uv (uma de duas opções):
uv run kedro run            # prefixar tudo com `uv run`  (recomendado)
#   OU activar o venv na shell:
source .venv/bin/activate    # (Windows: .venv\Scripts\activate)

# 3. segredos
cp .env.example .env         # preencher HW_API_KEY etc.
#    conf/local/credentials.yml lê estas vars (NUNCA commitar segredos).
#    se credentials.yml já estiver tracked: git rm --cached conf/local/credentials.yml
```

> Se alterarem dependências, fazem-no no `pyproject.toml` e correm `uv lock` (atualiza o
> `uv.lock`, que é versionado) + `uv sync`. Para regenerar o `requirements.txt`:
> `uv export --format requirements-txt --no-emit-project -o requirements.txt`.

## Como correr

```bash
# pipelines individuais (correm isoladas):
kedro run --pipeline ingestion
kedro run --pipeline data_unit_tests
kedro run --pipeline preprocessing_train
# ... (split_data, model_selection, model_train, feature_selection,
#      preprocessing_batch, model_predict, data_drift, reporting)

# composições nomeadas:
kedro run --pipeline data_prep      # ingestion + data_unit_tests + preprocessing_train + split_data
kedro run --pipeline training       # model_selection + model_train + feature_selection
kedro run --pipeline inference      # preprocessing_batch + model_predict
kedro run --pipeline monitoring     # data_drift
kedro run                           # __default__ (sequência completa)
```

### Hopsworks (feature store)
`parameters.yml: ingestion.to_feature_store` controla o upload. Com `false`, a pipeline
corre a partir dos CSVs locais (para quem não tem API key). Com `true`, faz o ciclo
write->read na feature store.

### MLflow UI e Kedro-Viz
```bash
kedro mlflow ui        # tracking + model registry (champion/challenger)
kedro viz              # grafo das pipelines
```

### Prefect (orquestração agendada)
```bash
python deployment_prefect.py   # cria/serve deployments (drift diário, treino semanal, tests nightly)
```

### Testes (pytest — distinto dos data_unit_tests)
```bash
pytest                 # smoke test end-to-end + testes unitários na sample (tests/pipelines/sample/sample.csv)
```

### Docker
```bash
docker build -t mlops-houses .
docker run --rm mlops-houses --pipeline __default__
```

## Estrutura
Ver [BLUEPRINT.md](BLUEPRINT.md) — árvore completa, conteúdo e TODOs por ficheiro, e a
ordem de execução das pipelines (secção 3).

---

# Workflow de Git (equipa)

PASSO 1:

Antes de começares a trabalhar vê em que branch estás:

git branch

Se já estiveres no teu branch, avança para o passo 2. Se não estiveres, muda para o teu branch com:

git checkout nome-do-teu-branch

PASSO 2:

git pull origin nome-branch-comum

Agora estás pronto para trabalhar à vontade

PASSO 3:

Quando acabares o trabalho

git add . git commit -m "mensagem explicativa do commit" git push origin nome-do-teu-branch

PASSO 4:

Atualizar o branch comum

git checkout nome-branch-comum
git pull origin nome-branch-comum
git merge nome-do-teu-branch git push nome-branch-comum

Voltar ao branch pessoal: git checkout nome-do-teu-branch
