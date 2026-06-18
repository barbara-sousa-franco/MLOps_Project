"""Pipeline `ingestion`."""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import ingestion, split_reference_analysis


def create_pipeline(**kwargs) -> Pipeline:
    """Cria a pipeline de ingestão.

    Node(ingestion) -> `ingested_data`
    Node(split_reference_analysis) -> `ref_data`, `ana_data`

    TODO (Fase 5): quando o upload à feature store estiver pronto, re-adicionar o input
    "credentials" ao nó de ingestion e/ou um Node(read_from_feature_store).
    """
    return pipeline(
        [
            node(
                func=ingestion,
                inputs=["raw_house_data", "params:ingestion"],
                outputs="ingested_data",
                name="ingestion_node",
            ),
            node(
                func=split_reference_analysis,
                inputs=["ingested_data", "params:ingestion"],
                outputs=["ref_data", "ana_data"],
                name="split_reference_analysis_node",
            ),
        ]
    )
