# MLOps_Project — House price prediction (Portugal)

End-to-end MLOps pipeline to predict `Price` (regression) from
`portugal_listings`. Stack: **Kedro + kedro-mlflow + MLflow (tracking + registry) +
Optuna + SHAP + Great Expectations + Hopsworks (feature store) + evidently/nannyml
(drift) + Prefect + Docker**.

> Status: **functional end-to-end.** Implemented: data prep (Great Expectations + traffic
> light), model selection (Optuna), model training (+ MLflow Model Registry), feature
> selection (RFE), explainability (SHAP), inference (honest test on out-of-sample data),
> drift monitoring (evidently), Hopsworks feature store, Docker, and Prefect orchestration.
> Remaining: `reporting` pipeline + pytest. See [ASSUMPTIONS.md](ASSUMPTIONS.md).

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
# named compositions (the usual way to run):
uv run kedro run --pipeline data_prep    # ingestion -> split_data -> preprocessing -> split_train -> preproc_after_split -> data_unit_tests (writes the traffic light)
uv run kedro run --pipeline training     # model_selection (Optuna) -> model_train (+ MLflow Registry)
uv run kedro run --pipeline inference    # preprocessing_batch -> model_predict (honest test on test_data)
uv run kedro run --pipeline monitoring   # data_drift (evidently)
uv run kedro run                         # __default__ (data_prep + training + inference + monitoring)

# individual pipelines also run in isolation, e.g.:
uv run kedro run --pipeline feature_selection   # RFE on the champion -> best_columns
uv run kedro run --pipeline explainability      # SHAP on the champion
```

> Two-pass feature selection: run `training` (all features) -> `feature_selection` (RFE) ->
> `training` again with `use_feature_selection: true` -> `explainability` (SHAP).

### Hopsworks (feature store)
`parameters.yml: ingestion.to_feature_store` controls the upload (**default `false`** so the
pipeline runs from local CSVs without a Hopsworks key — Docker / grader / CI). Set `true`
locally (with `HW_API_KEY` in `.env`) to perform the write->read cycle in the feature store.

### MLflow UI and Kedro-Viz
```bash
kedro mlflow ui        # tracking + model registry (champion/challenger)
kedro viz              # pipeline graph
```

### Prefect (orchestration)
```bash
uv run prefect server start             # UI at http://127.0.0.1:4200
uv run python kedro_prefect_flow.py     # run a flow once (flow_data_prep)
uv run python deployment_prefect.py     # register + serve the cron deployments
```
Flows wrap the Kedro pipelines; `full_pipeline` chains them and **gates** the modelling on
the data-quality traffic light (stops if any `*_FAIL.flag`). Deployments: nightly data
tests, daily drift, weekly training, on-demand full pipeline.

### Tests (pytest — distinct from data_unit_tests)
```bash
pytest                 # smoke test end-to-end + unit tests on the sample (tests/pipelines/sample/sample.csv)
```

### Docker
Data is **not** baked into the image — mount it at runtime with `-v` (don't mount `mlruns`,
its host paths break in the container). Note: pass the full `kedro run` command.
```bash
docker build -t mlops-houses .
docker run --rm -v "$(pwd)/data:/home/kedro_docker/data" \
  mlops-houses kedro run --pipeline data_prep
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
