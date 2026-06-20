"""Project pipelines."""

from kedro.pipeline import Pipeline

from kedro_temp_mlops.pipelines.data_drift import create_pipeline as data_drift
from kedro_temp_mlops.pipelines.data_unit_tests import create_pipeline as data_unit_tests
from kedro_temp_mlops.pipelines.feature_selection import create_pipeline as feature_selection
from kedro_temp_mlops.pipelines.ingestion import create_pipeline as ingestion
from kedro_temp_mlops.pipelines.model_predict import create_pipeline as model_predict
from kedro_temp_mlops.pipelines.model_selection import create_pipeline as model_selection
from kedro_temp_mlops.pipelines.model_train import create_pipeline as model_train
from kedro_temp_mlops.pipelines.preprocessing_train import create_pipeline as preprocessing


def register_pipelines() -> dict[str, Pipeline]:
    """Register project pipelines and named compositions."""

    p_ingestion            = ingestion()
    p_data_unit_tests      = data_unit_tests()
    p_preprocessing  = preprocessing()
    p_model_selection      = model_selection()
    p_model_train          = model_train()
    p_feature_selection    = feature_selection()
    p_model_predict        = model_predict()
    p_data_drift           = data_drift()

    # Named compositions
    data_prep = p_ingestion + p_preprocessing + p_data_unit_tests
    training  = p_model_selection + p_model_train + p_feature_selection
    inference = p_model_predict
    monitoring = p_data_drift

    return {
        # Individual pipelines
        "ingestion":           p_ingestion,
        "data_unit_tests":     p_data_unit_tests,
        "preprocessing":       p_preprocessing,
        "model_selection":     p_model_selection,
        "model_train":         p_model_train,
        "feature_selection":   p_feature_selection,
        "model_predict":       p_model_predict,
        "data_drift":          p_data_drift,
        # Named compositions
        "data_prep":           data_prep,
        "training":            training,
        "inference":           inference,
        "monitoring":          monitoring,
        "__default__":         data_prep + training + inference + monitoring,
    }
