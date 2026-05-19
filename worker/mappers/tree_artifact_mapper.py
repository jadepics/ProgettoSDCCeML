import rf_v2_pb2 as rf_pb2
from common.contracts import TreeArtifactMetadata


def to_proto_tree_artifact(a: TreeArtifactMetadata) -> rf_pb2.TrainedTreeArtifact:
    return rf_pb2.TrainedTreeArtifact(
        tree_id=a.tree_id,
        experiment_id=a.experiment_id,
        tree_index=a.tree_index,
        artifact_uri=a.artifact_uri,
        worker_id=a.worker_id,
        seed=a.seed,
        training_time_seconds=a.training_time_seconds,
    )