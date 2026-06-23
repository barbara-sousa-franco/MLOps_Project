"""Pipeline `ingestion`."""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import ingestion


def create_pipeline(**kwargs) -> Pipeline:
    """Create the ingestion pipeline.

    Node(ingestion) -> `ingested_data`

    NOTE: the ref/ana split does NOT live here — it is done in the `split_data` pipeline
    (split_out_of_sample), which also supports the 'biased' strategy for the drift demo.
    """
    return pipeline(
        [
            node(
                func=ingestion,
                inputs=[
                    "raw_house_data",
                    "params:ingestion",
                    "params:data_unit_tests",
                ],
                outputs="ingested_data",
                name="ingestion_node",
            ),
        ]
    )
