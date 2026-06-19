# MLOps_Project — House price prediction (Portugal)

End-to-end MLOps pipeline to predict `Price` (regression) from
`portugal_listings`. Stack: **Kedro + kedro-mlflow + MLflow (tracking + registry) +
Optuna + SHAP + Great Expectations + Hopsworks (feature store) + evidently/nannyml
(drift) + Prefect + Docker**.

> Status: **skeleton/scaffold**. Nodes have the signature + docstring + TODOs and raise
> `NotImplementedError`. See [BLUEPRINT.md](BLUEPRINT.md) for the full map and work
> distribution, and [ASSUMPTIONS.md](ASSUMPTIONS.md) for assumptions.

## Setup

> ⚠️ **Environment: ALWAYS use the `.venv` managed by `uv`. Do NOT create `.venv-mlops` or
> another venv manually.** `.venv` is in `.gitignore` (not tracked by git) — each person
> recreates it locally from `uv.lock` with `uv sync`. This ensures everyone has exactly the
> same versions.

```bash
# 1. environment (uv) — create/update .venv from uv.lock
uv sync

# 2. run commands in the uv environment (one of two options):
uv run kedro run            # prefix everything with `uv run`  (recommended)
#   OR activate the venv in the shell:
source .venv/bin/activate    # (Windows: .venv\Scripts\activate)

# 3. secrets
cp .env.example .env         # fill in HW_API_KEY etc.
#    conf/local/credentials.yml reads these vars (NEVER commit secrets).
#    if credentials.yml is already tracked: git rm --cached conf/local/credentials.yml
```

> If you change dependencies, do so in `pyproject.toml` and run `uv lock` (updates
> `uv.lock`, which is versioned) + `uv sync`. To regenerate `requirements.txt`:
> `uv export --format requirements-txt --no-emit-project -o requirements.txt`.

## How to run

```bash
# individual pipelines (run in isolation):
kedro run --pipeline ingestion
kedro run --pipeline data_unit_tests
kedro run --pipeline preprocessing_train
# ... (split_data, model_selection, model_train, feature_selection,
#      preprocessing_batch, model_predict, data_drift, reporting)

# named compositions:
kedro run --pipeline data_prep      # ingestion + data_unit_tests + preprocessing_train + split_data
kedro run --pipeline training       # model_selection + model_train + feature_selection
kedro run --pipeline inference      # preprocessing_batch + model_predict
kedro run --pipeline monitoring     # data_drift
kedro run                           # __default__ (full sequence)
```

### Hopsworks (feature store)
`parameters.yml: ingestion.to_feature_store` controls the upload. With `false`, the pipeline
runs from local CSVs (for those without an API key). With `true`, it performs the
write->read cycle in the feature store.

### MLflow UI and Kedro-Viz
```bash
kedro mlflow ui        # tracking + model registry (champion/challenger)
kedro viz              # pipeline graph
```

### Prefect (scheduled orchestration)
```bash
python deployment_prefect.py   # create/serve deployments (daily drift, weekly training, nightly tests)
```

### Tests (pytest — distinct from data_unit_tests)
```bash
pytest                 # smoke test end-to-end + unit tests on the sample (tests/pipelines/sample/sample.csv)
```

### Docker
```bash
docker build -t mlops-houses .
docker run --rm mlops-houses --pipeline __default__
```

## Structure
See [BLUEPRINT.md](BLUEPRINT.md) — full tree, file contents and TODOs per file, and the
pipeline execution order (section 3).

---

# Git Workflow (team)

STEP 1:

Before starting work, check which branch you are on:

```bash
git branch
```

If you are already on your branch, proceed to step 2. Otherwise, switch to your branch with:

```bash
git checkout your-branch-name
```

STEP 2:

```bash
git pull origin common-branch-name
```

Now you are ready to work freely.

STEP 3:

When you finish your work:

```bash
git add .
git commit -m "descriptive commit message"
git push origin your-branch-name
```

STEP 4:

Update the common branch:

```bash
git checkout common-branch-name
git pull origin common-branch-name
git merge your-branch-name
git push common-branch-name
```

Return to your personal branch: `git checkout your-branch-name`
