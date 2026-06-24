"""Nodes for the `data_drift` pipeline (monitoring — valued by the professor).

Compares the REFERENCE distribution (`learning_data`, the training pool) against the
NEW batch (`test_data`, the out-of-sample data) with evidently 0.7.x, producing an HTML
report and a per-feature drift table. Optionally injects artificial drift into the batch
to demonstrate that detection works (professor's tip).
"""

import logging
import re

import pandas as pd
from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset

logger = logging.getLogger(__name__)

_VALUE_DRIFT = re.compile(
    r"ValueDrift\(column=(?P<col>.+?),method=(?P<method>.+?),threshold=(?P<thr>[0-9.]+)\)"
)


def _inject_artificial_drift(df: pd.DataFrame, numeric_cols: list, shift_pct: float) -> pd.DataFrame:
    """Shift numeric features by (1 + shift_pct) to simulate distribution drift."""
    df = df.copy()
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col] * (1.0 + shift_pct)
    logger.info("Artificial drift injected: numeric features x %.2f", 1.0 + shift_pct)
    return df


def _drift_detected(method: str, score: float, threshold: float) -> bool:
    """A p-value test drifts when score < threshold; a distance metric when score > threshold."""
    if "p_value" in method.lower():
        return score < threshold
    return score > threshold


def compute_drift(learning_data: pd.DataFrame, test_data: pd.DataFrame, parameters: dict):
    """Detect data drift of the new batch vs the training reference.

    Args:
        learning_data: reference data (training pool).
        test_data: batch to analyse (out-of-sample).
        parameters: `data_drift` config (features, threshold, optional injection).

    Returns:
        Tuple (drift_result, drift_report):
          - drift_result: per-feature drift DataFrame (column, method, score, threshold,
            drift_detected) plus a dataset-level summary row.
          - drift_report: the evidently HTML report (string).
    """
    numeric_cols = [c for c in parameters.get("numeric_features", []) if c in learning_data.columns]
    categorical_cols = [c for c in parameters.get("categorical_features", []) if c in learning_data.columns]
    share_threshold = parameters.get("drift_share_threshold", 0.5)

    reference = learning_data[numeric_cols + categorical_cols]
    current = test_data[numeric_cols + categorical_cols]
    if parameters.get("inject_artificial_drift", False):
        current = _inject_artificial_drift(current, numeric_cols, parameters.get("numeric_shift_pct", 0.3))

    data_def = DataDefinition(numerical_columns=numeric_cols, categorical_columns=categorical_cols)
    ref_ds = Dataset.from_pandas(reference, data_definition=data_def)
    cur_ds = Dataset.from_pandas(current, data_definition=data_def)

    snapshot = Report([DataDriftPreset()]).run(reference_data=ref_ds, current_data=cur_ds)

    # ---- per-feature drift table -------------------------------------------------
    rows = []
    for metric in snapshot.dict()["metrics"]:
        m = _VALUE_DRIFT.match(metric.get("metric_name", ""))
        if not m:
            continue
        method, score, thr = m["method"], float(metric["value"]), float(m["thr"])
        rows.append({
            "column": m["col"],
            "method": method,
            "drift_score": round(score, 4),
            "threshold": thr,
            "drift_detected": _drift_detected(method, score, thr),
        })
    drift_result = pd.DataFrame(rows)

    n_cols = len(drift_result)
    n_drifted = int(drift_result["drift_detected"].sum()) if n_cols else 0
    drift_share = n_drifted / n_cols if n_cols else 0.0
    dataset_drift = drift_share >= share_threshold

    # dataset-level summary row (prepended)
    summary = pd.DataFrame([{
        "column": "__dataset__", "method": "drifted_share", "drift_score": round(drift_share, 4),
        "threshold": share_threshold, "drift_detected": dataset_drift,
    }])
    drift_result = pd.concat([summary, drift_result], ignore_index=True)

    logger.info(
        "Data drift: %d/%d features drifted (share=%.2f) -> dataset drift = %s",
        n_drifted, n_cols, drift_share, dataset_drift,
    )

    drift_report = snapshot.get_html_str(as_iframe=False)
    return drift_result, drift_report
