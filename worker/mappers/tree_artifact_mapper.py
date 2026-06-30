import rf_v2_pb2 as rf_pb2
from common.contracts import TreeArtifactMetadata
"""
Mapper per convertire i metadati degli alberi addestrati dal formato
interno del worker al formato protobuf usato nella comunicazione gRPC.

Questo file non contiene logica di training o di salvataggio: si occupa
solo di adattare i dati al contratto previsto tra worker e master.
"""

def to_proto_tree_artifact(a: TreeArtifactMetadata) -> rf_pb2.TrainedTreeArtifact:
    """
    Converte i metadati di un albero addestrato nel messaggio protobuf
    restituito dal worker al master al termine del training di uno shard.
    """

    return rf_pb2.TrainedTreeArtifact(
        tree_id=a.tree_id,
        experiment_id=a.experiment_id,
        tree_index=a.tree_index,
        artifact_uri=a.artifact_uri,
        worker_id=a.worker_id,
        seed=a.seed,
        training_time_seconds=a.training_time_seconds,
        feature_importances=list(a.feature_importances),
    )