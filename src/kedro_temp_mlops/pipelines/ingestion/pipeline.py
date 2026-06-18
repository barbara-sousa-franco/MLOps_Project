"""Pipeline `ingestion`."""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import ingestion


def create_pipeline(**kwargs) -> Pipeline:
    """Cria a pipeline de ingestão.

    Node(ingestion) -> outputs `ingested_data`.

    TODO:
      - se separarem o write/read da FS em nós distintos, adicionar
        Node(read_from_feature_store) -> ref_data/ana_data conforme necessário.
    """
    return pipeline(
        [
            node(
                func=ingestion,
                inputs=["raw_house_data", "params:ingestion", "credentials"],
                outputs="ingested_data",
                name="ingestion_node",
            ),
        ]
    )
