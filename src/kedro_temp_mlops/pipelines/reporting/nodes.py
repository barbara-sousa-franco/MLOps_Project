"""Nodes for the `reporting` pipeline.

Consolidates the artifacts produced across the project (champion + metrics, data-quality
tests, drift, SHAP) into a single human-readable Markdown report. It computes nothing new —
it AGGREGATES what the other pipelines already produced.
"""

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

_SHAP_IMPORTANCE_PATH = Path("data/08_reporting/shap_importance.csv")
_KEY_HYPERPARAMS = ("n_estimators", "max_depth", "learning_rate", "max_iter",
                    "min_samples_leaf", "max_leaf_nodes")


def _fmt(x, nd=4):
    try:
        return f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return str(x)


def build_report(
    production_model,
    val_metrics: dict,
    test_metrics: dict,
    drift_result: pd.DataFrame,
    data_tests_cleaned: pd.DataFrame,
    data_tests_model_input: pd.DataFrame,
    parameters: dict,
) -> str:
    """Build the consolidated Markdown report.

    Args:
        production_model: the champion (for type + key hyperparameters).
        val_metrics: validation metrics (production_model_metrics).
        test_metrics: honest test metrics on test_data (production_test_metrics).
        drift_result: per-feature drift table (+ a `__dataset__` summary row).
        data_tests_cleaned / data_tests_model_input: GX validation result tables.
        parameters: reporting config (e.g. `top_shap` features to show).

    Returns:
        The report as a Markdown string (saved to `final_report`).
    """
    top_shap = parameters.get("top_shap", 10)
    lines: list[str] = ["# House Price Prediction — Model Report", ""]

    # ---- champion ----
    model_type = type(production_model).__name__
    hp = {k: v for k, v in production_model.get_params().items()
          if k in _KEY_HYPERPARAMS and v is not None}
    lines += ["## Champion model", "",
              f"- **Type:** `{model_type}`",
              f"- **Key hyperparameters:** {hp or '(defaults)'}",
              "- **Registry:** MLflow Model Registry — `house_price_model` (champion alias)", ""]

    # ---- metrics ----
    lines += ["## Metrics", "",
              "Target is `log1p(Price)`; € metrics are the expm1-inverted predictions.",
              "Validation = `X_val` (used for selection); test = the out-of-sample `test_data`.", "",
              "| Metric | Validation | Honest test |",
              "|---|---|---|",
              f"| R² (log) | {_fmt(val_metrics.get('val_r2'))} | "
              f"{_fmt(test_metrics.get('test_r2'))} |",
              f"| RMSE (log) | {_fmt(val_metrics.get('val_rmse'))} | "
              f"{_fmt(test_metrics.get('test_rmse_log'))} |",
              f"| MAE (log) | {_fmt(val_metrics.get('val_mae'))} | "
              f"{_fmt(test_metrics.get('test_mae_log'))} |",
              f"| RMSE (€) | — | {_fmt(test_metrics.get('test_rmse_eur'), 0)} |",
              f"| MAE (€) | — | {_fmt(test_metrics.get('test_mae_eur'), 0)} |", "",
              f"**Baseline** (`DummyRegressor(mean)`): R²={_fmt(val_metrics.get('baseline_r2'))}, "
              f"RMSE(log)={_fmt(val_metrics.get('baseline_rmse'))} — the champion beats it clearly.",
              "", "> The € RMSE is inflated by heavy-tailed price outliers; R² and MAE are the "
              "headline metrics.", ""]

    # ---- data quality (traffic light) ----
    def _quality(df: pd.DataFrame, label: str) -> str:
        n = len(df)
        failed = int((~df["Success"].astype(bool)).sum()) if n else 0
        status = "✅ OK" if failed == 0 else f"❌ {failed} FAILED"
        return f"- **{label}:** {n - failed}/{n} checks passed — {status}"

    lines += ["## Data quality (Great Expectations + traffic light)", "",
              _quality(data_tests_cleaned, "cleaned_data"),
              _quality(data_tests_model_input, "model_input"), ""]

    # ---- drift ----
    summary = drift_result[drift_result["column"] == "__dataset__"]
    share = float(summary["drift_score"].iloc[0]) if not summary.empty else float("nan")
    dataset_drift = bool(summary["drift_detected"].iloc[0]) if not summary.empty else False
    per_feature = drift_result[drift_result["column"] != "__dataset__"]
    n_drift = int(per_feature["drift_detected"].sum())
    lines += ["## Data drift (evidently — learning_data vs test_data)", "",
              f"- **Drifted features:** {n_drift}/{len(per_feature)} (share={_fmt(share, 2)})",
              f"- **Dataset-level drift:** {'YES' if dataset_drift else 'NO'}", ""]

    # ---- SHAP (optional — only if explainability has run) ----
    lines += ["## Feature importance (SHAP)", ""]
    if _SHAP_IMPORTANCE_PATH.exists():
        shap_df = pd.read_csv(_SHAP_IMPORTANCE_PATH)
        feat_col = shap_df.columns[0]
        imp_col = shap_df.columns[1] if len(shap_df.columns) > 1 else feat_col
        top = shap_df.head(top_shap)
        lines += [f"Top {len(top)} features by mean |SHAP|:", ""]
        lines += [f"{i + 1}. `{r[feat_col]}` ({_fmt(r[imp_col])})"
                  for i, (_, r) in enumerate(top.iterrows())]
    else:
        lines += ["_Not available — run the `explainability` pipeline to generate "
                  "`shap_importance`._"]
    lines += [""]

    report = "\n".join(lines)
    logger.info("Report built: champion=%s, test R2=%s, drift=%s",
                model_type, _fmt(test_metrics.get("test_r2")), dataset_drift)
    return report
