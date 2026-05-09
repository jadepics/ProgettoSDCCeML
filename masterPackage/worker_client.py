from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import grpc
import numpy as np

import rf_v2_pb2 as rf_pb2
import rf_v2_pb2_grpc as rf_pb2_grpc

from common.contracts import (
    ShardTrainingResult,
    TrainingShard,
    TreeArtifactMetadata,
)
from common.enums import TreeStatus

GRPC_MAX_MESSAGE_LENGTH = 64 * 1024 * 1024  # 64 MB

GRPC_OPTIONS = [
    ("grpc.max_send_message_length", GRPC_MAX_MESSAGE_LENGTH),
    ("grpc.max_receive_message_length", GRPC_MAX_MESSAGE_LENGTH),
]

@dataclass(slots=True)
class PredictionShardResult:
    worker_id: str
    success: bool
    error_message: str | None
    values: np.ndarray


class WorkerClient:
    """
    Responsabilità:
    - incapsulare le chiamate RPC master -> worker
    - costruire i protobuf request a partire dai contratti del dominio
    - trasformare le protobuf response in oggetti Python del progetto

    Nota:
    questa classe NON decide scheduling, retry o orchestration.
    Quella logica deve stare nel TrainingOrchestrator / InferenceCoordinator.
    """

    def __init__(
        self,
        timeout_train_seconds: float = 600.0,
        timeout_predict_seconds: float = 120.0,
    ) -> None:
        self.timeout_train_seconds = timeout_train_seconds
        self.timeout_predict_seconds = timeout_predict_seconds

    def train_shard(
        self,
        worker_host: str,
        worker_port: int,
        shard: TrainingShard,
    ) -> ShardTrainingResult:
        request = self._build_train_shard_request(shard)
        response = self._call_train_shard(
            worker_host=worker_host,
            worker_port=worker_port,
            request=request,
        )
        return self._build_train_shard_result_from_response(
            job_id=shard.job_id,
            experiment_id=shard.experiment_id,
            task_id=shard.task_id,
            response=response,
        )

    def predict_shard(
        self,
        worker_host: str,
        worker_port: int,
        model_id: str,
        experiment_id: str,
        task_type: str,
        features: np.ndarray,
        tree_artifact_uris: Sequence[str],
        class_labels: Sequence[str] | None = None,
    ) -> PredictionShardResult:
        request = self._build_predict_shard_request(
            model_id=model_id,
            experiment_id=experiment_id,
            task_type=task_type,
            features=features,
            tree_artifact_uris=tree_artifact_uris,
            class_labels=class_labels or [],
        )
        response = self._call_predict_shard(
            worker_host=worker_host,
            worker_port=worker_port,
            request=request,
        )
        return self._build_prediction_result_from_response(response)

    def _build_train_shard_request(
        self,
        shard: TrainingShard,
    ) -> rf_pb2.TrainShardRequest:
        fc = shard.forest_config

        max_depth = 0 if fc.max_depth is None else fc.max_depth
        max_features = "none" if fc.max_features is None else str(fc.max_features)
        lease_expires_at_unix_ms = int(shard.lease_expires_at_ts * 1000)

        return rf_pb2.TrainShardRequest(
            task_id=shard.task_id,
            attempt_id=shard.attempt_id,
            job_id=shard.job_id,
            experiment_id=shard.experiment_id,
            assigned_worker_id=shard.assigned_worker_id,
            tree_start_index=shard.tree_start_index,
            tree_count=shard.tree_count,
            task_type=fc.task_type,
            n_estimators=fc.n_estimators,
            max_depth=max_depth,
            max_features=max_features,
            min_samples_split=fc.min_samples_split,
            min_samples_leaf=fc.min_samples_leaf,
            criterion=fc.criterion,
            bootstrap=fc.bootstrap,
            global_random_seed=fc.global_random_seed,
            train_features_uri=shard.train_features_uri,
            train_labels_uri=shard.train_labels_uri,
            artifact_output_dir=shard.artifact_output_dir,
            seed_base=shard.seed_base,
            lease_expires_at_unix_ms=lease_expires_at_unix_ms,
        )

    def _build_train_shard_result_from_response(
        self,
        job_id: str,
        experiment_id: str,
        task_id: str,
        response: rf_pb2.TrainShardResponse,
    ) -> ShardTrainingResult:
        tree_artifacts: list[TreeArtifactMetadata] = []

        for artifact in response.artifacts:
            tree_artifacts.append(
                TreeArtifactMetadata(
                    tree_id=artifact.tree_id,
                    job_id=job_id,
                    experiment_id=experiment_id,
                    task_id=task_id,
                    tree_index=artifact.tree_index,
                    worker_id=artifact.worker_id,
                    seed=artifact.seed,
                    artifact_uri=artifact.artifact_uri,
                    status=TreeStatus.COMPLETED if response.success else TreeStatus.FAILED,
                    training_time_seconds=0.0,
                )
            )

        return ShardTrainingResult(
            task_id=response.task_id,
            attempt_id=response.attempt_id,
            worker_id=response.worker_id,
            completed_tree_ids=list(response.completed_tree_ids),
            failed_tree_ids=list(response.failed_tree_ids),
            success=response.success,
            error_message=response.error if response.error else None,
            tree_artifacts=tree_artifacts,
            completed_tree_count=len(response.completed_tree_ids),
            failed_tree_count=len(response.failed_tree_ids),
            elapsed_time_seconds=response.elapsed_time_seconds,
        )

    def _build_predict_shard_request(
        self,
        model_id: str,
        experiment_id: str,
        task_type: str,
        features: np.ndarray,
        tree_artifact_uris: Sequence[str],
        class_labels: Sequence[str],
    ) -> rf_pb2.PredictShardRequest:
        matrix = self._matrix_to_proto(features)

        return rf_pb2.PredictShardRequest(
            model_id=model_id,
            experiment_id=experiment_id,
            task_type=task_type,
            features=matrix,
            tree_artifact_uris=list(tree_artifact_uris),
            class_labels=list(class_labels),
        )

    def _build_prediction_result_from_response(
        self,
        response: rf_pb2.PredictShardResponse,
    ) -> PredictionShardResult:
        if response.n_rows < 0 or response.n_cols < 0:
            raise ValueError("Invalid prediction response shape")

        expected_size = response.n_rows * response.n_cols
        if len(response.values) != expected_size:
            raise ValueError(
                f"PredictShardResponse shape mismatch: expected {expected_size} values, "
                f"got {len(response.values)}"
            )

        values = np.asarray(response.values, dtype=float).reshape(
            response.n_rows,
            response.n_cols,
        )

        return PredictionShardResult(
            worker_id=response.worker_id,
            success=response.success,
            error_message=response.error if response.error else None,
            values=values,
        )

    def _call_train_shard(
        self,
        worker_host: str,
        worker_port: int,
        request: rf_pb2.TrainShardRequest,
    ) -> rf_pb2.TrainShardResponse:
        address = f"{worker_host}:{worker_port}"
        with grpc.insecure_channel(address, options=GRPC_OPTIONS) as channel:
            stub = rf_pb2_grpc.WorkerServiceStub(channel)
            return stub.TrainShard(request, timeout=self.timeout_train_seconds)

    def _call_predict_shard(
        self,
        worker_host: str,
        worker_port: int,
        request: rf_pb2.PredictShardRequest,
    ) -> rf_pb2.PredictShardResponse:
        address = f"{worker_host}:{worker_port}"
        with grpc.insecure_channel(address, options=GRPC_OPTIONS) as channel:
            stub = rf_pb2_grpc.WorkerServiceStub(channel)
            return stub.PredictShard(request, timeout=self.timeout_predict_seconds)

    def _matrix_to_proto(self, arr: np.ndarray) -> rf_pb2.DenseMatrix:
        arr = np.asarray(arr, dtype=float)
        if arr.ndim != 2:
            raise ValueError("Expected a 2D feature matrix")

        return rf_pb2.DenseMatrix(
            values=arr.ravel().tolist(),
            n_rows=arr.shape[0],
            n_cols=arr.shape[1],
        )