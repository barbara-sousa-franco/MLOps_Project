"""Project pipelines.

Registers the 11 pipelines individually (to run in isolation, as the professor requires) and
composes named sequences. Do NOT use blind find_pipelines() — we register explicitly to
control order and compositions.
"""

from kedro.pipeline import Pipeline

from kedro_temp_mlops.pipelines.data_drift import create_pipeline as data_drift
from kedro_temp_mlops.pipelines.data_unit_tests import create_pipeline as data_unit_tests
from kedro_temp_mlops.pipelines.feature_selection import create_pipeline as feature_selection
from kedro_temp_mlops.pipelines.ingestion import create_pipeline as ingestion
from kedro_temp_mlops.pipelines.model_predict import create_pipeline as model_predict
from kedro_temp_mlops.pipelines.model_selection import create_pipeline as model_selection
from kedro_temp_mlops.pipelines.model_train import create_pipeline as model_train
from kedro_temp_mlops.pipelines.preprocessing_batch import create_pipeline as preprocessing_batch
from kedro_temp_mlops.pipelines.preprocessing_train import create_pipeline as preprocessing_train
from kedro_temp_mlops.pipelines.reporting import create_pipeline as reporting
from kedro_temp_mlops.pipelines.split_data import create_pipeline as split_data


def register_pipelines() -> dict[str, Pipeline]:
    """Register project pipelines and named compositions.

    Returns:
        Map name -> Pipeline.

    TODO register_pipelines:
      - named compositions (see blueprint section 3):
          * "data_prep" = ingestion + data_unit_tests + preprocessing_train + split_data
          * "training"  = model_selection + model_train + feature_selection
          * "inference" = preprocessing_batch + model_predict
          * "monitoring"= data_drift
          * __default__ = full sequence
      - confirm order against catalog dependencies before summing.
    """
    p_ingestion = ingestion()
    p_data_unit_tests = data_unit_tests()
    p_preprocessing_train = preprocessing_train()
    p_preprocessing_batch = preprocessing_batch()
    p_split_data = split_data()
    p_model_selection = model_selection()
    p_model_train = model_train()
    p_feature_selection = feature_selection()
    p_model_predict = model_predict()
    p_data_drift = data_drift()
    p_reporting = reporting()

    data_prep = p_ingestion + p_data_unit_tests + p_preprocessing_train + p_split_data
    training = p_model_selection + p_model_train + p_feature_selection
    inference = p_preprocessing_batch + p_model_predict
    monitoring = p_data_drift

    return {
        # individual pipelines (run in isolation)
        "ingestion": p_ingestion,
        "data_unit_tests": p_data_unit_tests,
        "preprocessing_train": p_preprocessing_train,
        "preprocessing_batch": p_preprocessing_batch,
        "split_data": p_split_data,
        "model_selection": p_model_selection,
        "model_train": p_model_train,
        "feature_selection": p_feature_selection,
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
