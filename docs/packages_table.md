# Packages & Environment

**Python 3.13.13** · environment managed with **uv** (`.venv` from `uv.lock`).

| Package | Version | Purpose |
|---|---|---|
| kedro | 1.3.1 | Pipeline framework / project structure |
| kedro-datasets | 9.4.0 | Catalog datasets (CSV, Pickle, JSON, Matplotlib) |
| kedro-mlflow | 2.0.2 | Kedro ↔ MLflow integration (tracking + registry) |
| kedro-viz | 12.4.0 | Pipeline visualization (Figure 3) |
| kedro-telemetry | 0.7.0 | Kedro usage telemetry |
| mlflow | 3.11.1 | Experiment tracking + Model Registry |
| optuna | 4.8.0 | Hyperparameter tuning (TPE) |
| scikit-learn | 1.8.0 | Regression models + metrics |
| xgboost | 3.3.0 | Candidate model (gradient boosting) |
| lightgbm | 4.6.0 | Candidate model (gradient boosting) |
| shap | 0.52.0 | Explainability (feature importance) |
| great-expectations | 1.16.1 | Data validation / quality (traffic light) |
| evidently | 0.7.21 | Data drift monitoring |
| hopsworks | 5.0.0 | Feature store |
| prefect | 3.6.27 | Workflow orchestration (scheduling) |
| pandas | 2.3.3 | Data manipulation |
| numpy | 2.3.5 | Numerical computing |
| matplotlib | 3.10.0 | Plotting (SHAP / EDA figures) |
| ydata-profiling | 4.18.1 | Automated EDA profiling |
| confluent-kafka | 2.14.0 | Streaming ingestion (Hopsworks) |
| docker | 7.1.0 | Containerized reproducible runs |
