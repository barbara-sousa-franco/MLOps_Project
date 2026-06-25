# ASSUMPTIONS

Data and model assumptions and key design decisions.

## Data
- **Raw:** `data/01_raw/portugal_listings.csv` — 135,536 rows, 25 columns.
- **Target:** `Price` in **EUR** (regression). Has ~0.2% nulls -> rows without a target are
  **dropped in ingestion** (the target must exist).
- **`PublishDate`:** **~78% null** (only ~21% have a date, almost all Oct 2024). Therefore a
  **temporal** split is not feasible — `learning_data` / `test_data` are carved with a
  reproducible **random split** instead (`split_data`, seed + `ref_frac=0.8`). `PublishDate`
  is kept as the feature-store `event_time` (with the null caveat).
- **`learning_data` / `test_data`:** carved by `split_data.split_out_of_sample` from the
  ingested data BEFORE any cleaning. `learning_data` = training pool; `test_data` = the
  out-of-sample **TEST** set (the honest final metric is computed here, once) and the
  reference batch for drift. Artificial drift can be injected to demonstrate detection.
- **Outliers:** the ~1.38 billion EUR target outlier is **removed**; impossible/negative
  areas are set to NA in `clean_data`, and numeric features are **percentile-capped**
  (capper fit on train only).
- **Missing values:** very high null rates in several columns (ConservationStatus 86%,
  BuiltArea 80%, GrossArea 80%...). Imputed **after the split** with a group imputer
  (median/mode by `Type`), fit on train only.
- **Target skew:** EDA shows strong skew -> the model trains on **`log1p(Price)`**
  (`use_log_target: true`); predictions are inverted with `expm1` to report euros.

## Features
- Expected raw column set in `parameters.yml:raw_columns`; numeric/categorical split
  confirmed against the raw dtypes.
- **High cardinality:** `Town`/`City` are dropped; `District` and `Type` are
  **target-encoded** (fit on train -> anti-leakage). One-hot is avoided.
- **Booleans:** `HasParking`/`Garage`/`Elevator`/`ElectricCarsCharging` are normalised to
  numeric.
- **Inference features = training features** (without the target). The out-of-sample batch
  goes through the same cleaning + the train-fitted transformers (transform only).

## Model & evaluation
- **Candidates:** RandomForest / GradientBoosting (XGBoost/LightGBM optional — they need
  `libomp` on macOS, so they are imported lazily and disabled if absent).
- **Splits / leakage:** `learning_data` -> `split_train` -> `X_train` (fit) + `X_val` (the
  leak-free **validation** holdout used for tuning/selection). `test_data` is the honest
  **test** set. All fitted transformers (imputer, capper, target encoder, scaler) and the
  models are fit on **train only**; everything downstream is transform/predict.
- **Selection:** challengers compared, then **Optuna** tuning on the validation holdout,
  minimising **RMSE**. Baseline = `DummyRegressor(strategy="mean")`.
- **Champion/challenger** managed via the **MLflow Model Registry** (`house_price_model`).
- **Result:** champion = GradientBoosting — **validation R² ≈ 0.72**, **honest test R² ≈ 0.68**
  (on `test_data`). The EUR RMSE is inflated by heavy-tailed price outliers, so **R² and MAE**
  are the headline metrics.

## Limitations / risks (see also the report's "risks & mitigations")
- **Pandas-only:** does not scale to much larger data -> Spark/Polars as mitigation.
- **External Hopsworks:** depends on an API key + cloud service; `to_feature_store=false`
  (default) lets everything run from local CSVs.
- **Optuna `n_trials` limited by time:** quality vs duration trade-off.
- **EUR RMSE inflated** by price outliers -> report R²/MAE as the main metrics.
- **LightGBM/XGBoost** need a system dependency (`libomp`) — optional, disabled if absent.
- **Artificial drift** is used to demonstrate detection; in production it would be continuous.
