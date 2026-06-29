"""Project pipelines.

Registers each pipeline individually (so they can run in isolation) and composes named
sequences. The actual node order is resolved by Kedro from the data dependencies in the
catalog — these sums only define WHICH nodes enter each composition.
"""

from kedro.pipeline import Pipeline

from kedro_temp_mlops.pipelines.data_drift import create_pipeline as data_drift
from kedro_temp_mlops.pipelines.data_unit_tests import create_pipeline as data_unit_tests
from kedro_temp_mlops.pipelines.explainability import create_pipeline as explainability
from kedro_temp_mlops.pipelines.preproc_after_split import create_pipeline as preproc_after_split
from kedro_temp_mlops.pipelines.feature_selection import create_pipeline as feature_selection
from kedro_temp_mlops.pipelines.ingestion import create_pipeline as ingestion
from kedro_temp_mlops.pipelines.model_predict import create_pipeline as model_predict
from kedro_temp_mlops.pipelines.preprocessing_batch import create_pipeline as preprocessing_batch
from kedro_temp_mlops.pipelines.model_selection import create_pipeline as model_selection
from kedro_temp_mlops.pipelines.model_train import create_pipeline as model_train
from kedro_temp_mlops.pipelines.preprocessing_train import create_pipeline as preprocessing_train
from kedro_temp_mlops.pipelines.reporting import create_pipeline as reporting
from kedro_temp_mlops.pipelines.split_data import create_pipeline as split_data
from kedro_temp_mlops.pipelines.split_train_pipeline import create_pipeline as split_train


def register_pipelines() -> dict[str, Pipeline]:
    """Register the project pipelines and named compositions.

    Data-prep flow (Phase 0/1):
        ingestion -> split_data -> preprocessing (clean) -> split_train
        -> preproc_after_split (impute/cap/encode/scale) -> data_unit_tests

    Training flow (single pass):
        compare_models → tune_model (Optuna) → feature_selection (RFE) → model_train
    """
    p_ingestion = ingestion()
    p_split_data = split_data()
    p_preprocessing_train = preprocessing_train()
    p_split_train = split_train()
    p_preproc_after_split = preproc_after_split()
    p_data_unit_tests = data_unit_tests()
    p_model_selection = model_selection()
    p_model_train = model_train()
    p_feature_selection = feature_selection()
    p_model_predict = model_predict()
    p_preprocessing_batch = preprocessing_batch()
    p_data_drift = data_drift()
    p_explainability = explainability()
    p_reporting = reporting()

    # named compositions
    data_prep = (
        p_ingestion
        + p_split_data
        + p_preprocessing_train
        + p_split_train
        + p_preproc_after_split
        + p_data_unit_tests
    )
    # single-pass: compare → feature_selection (RFE) → tune → train
    training = p_model_selection + p_feature_selection + p_model_train
    # Phase 3: preprocess the out-of-sample batch (test_data) with the train-fitted
    # transformers, then predict + evaluate the HONEST test metric with the champion.
    inference = p_preprocessing_batch + p_model_predict
    monitoring = p_data_drift

    return {
        # individual pipelines (run in isolation)
        "ingestion": p_ingestion,
        "split_data": p_split_data,
        "preprocessing_train": p_preprocessing_train,
        "split_train": p_split_train,
        "preproc_after_split": p_preproc_after_split,
        "data_unit_tests": p_data_unit_tests,
        "model_selection": p_model_selection,
        "model_train": p_model_train,
        "feature_selection": p_feature_selection,
        "explainability": p_explainability,
        "model_predict": p_model_predict,
        "data_drift": p_data_drift,
        "reporting": p_reporting,
        # named compositions
        "data_prep": data_prep,
        "training": training,
        "inference": inference,
        "monitoring": monitoring,
        "__default__": data_prep + training + inference + monitoring + p_reporting,
    }
