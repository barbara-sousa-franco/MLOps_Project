"""Pipeline `split_data`."""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import split_out_of_sample


"""Pipeline `split_data` — out-of-sample (ref/ana) carve-off."""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import split_out_of_sample


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=split_out_of_sample,
            inputs=["ingested_data", "params:split_data"],
            outputs=["ref_data", "ana_data"],
            name="split_out_of_sample_node",
        ),
    ])
