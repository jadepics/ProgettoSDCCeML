from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, Sequence

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

from masterPackage.Metrics.Scalability_Metrics_Collector import (
    ScalabilityMetricsCollector,
)
from common.grpc_config import GRPC_OPTIONS


@dataclass(slots=True)
class PredictionShardResult:
    worker_id: str
    success: bool
    error_message: str | None

    # Vecchio percorso: predizioni dentro gRPC.
    values: np.ndarray | None = None

    # Nuovo percorso scalabile: predizioni salvate su EFS.
    prediction_uri: str | None = None
    n_rows: int = 0
    n_cols: int = 0


class WorkerClient:
    """
    Responsabilità:
    - incapsulare le chiamate RPC master -> worker
    - costruire i protobuf request a partire dai contratti del dominio
    - trasformare le protobuf response in oggetti Python del progetto
    - raccogliere metriche di comunicazione RPC quando viene passato
      uno Scalability_Metrics_Collector

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
        metrics_collector: Optional[ScalabilityMetricsCollector] = None,
    ) -> ShardTrainingResult:
        request = self._build_train_shard_request(shard)
        request_bytes = self._safe_proto_size(request)

        self._metrics_event(
            metrics_collector,
            "worker_client_train_rpc_started",
            job_id=shard.job_id,
            experiment_id=shard.experiment_id,
            task_id=shard.task_id,
            attempt_id=shard.attempt_id,
            worker_id=shard.assigned_worker_id,
            worker_host=worker_host,
            worker_port=worker_port,
            tree_start_index=shard.tree_start_index,
            tree_count=shard.tree_count,
            request_bytes=request_bytes,
            timeout_seconds=self.timeout_train_seconds,
        )

        started_at = time.time()

        try:
            response = self._call_train_shard(
                worker_host=worker_host,
                worker_port=worker_port,
                request=request,
            )

        except Exception as exc:
            latency_seconds = time.time() - started_at

            self._metrics_event(
                metrics_collector,
                "worker_client_train_rpc_failed",
                job_id=shard.job_id,
                experiment_id=shard.experiment_id,
                task_id=shard.task_id,
                attempt_id=shard.attempt_id,
                worker_id=shard.assigned_worker_id,
                worker_host=worker_host,
                worker_port=worker_port,
                tree_count=shard.tree_count,
                request_bytes=request_bytes,
                response_bytes=None,
                latency_seconds=latency_seconds,
                error_message=str(exc),
            )

            raise

        latency_seconds = time.time() - started_at
        response_bytes = self._safe_proto_size(response)

        result = self._build_train_shard_result_from_response(
            job_id=shard.job_id,
            experiment_id=shard.experiment_id,
            task_id=shard.task_id,
            response=response,
        )

        self._metrics_event(
            metrics_collector,
            "worker_client_train_rpc_completed",
            job_id=shard.job_id,
            experiment_id=shard.experiment_id,
            task_id=shard.task_id,
            attempt_id=shard.attempt_id,
            worker_id=result.worker_id or shard.assigned_worker_id,
            worker_host=worker_host,
            worker_port=worker_port,
            tree_start_index=shard.tree_start_index,
            tree_count=shard.tree_count,
            request_bytes=request_bytes,
            response_bytes=response_bytes,
            latency_seconds=latency_seconds,
            result_success=result.success,
            completed_tree_count=result.completed_tree_count,
            failed_tree_count=result.failed_tree_count,
            worker_elapsed_time_seconds=result.elapsed_time_seconds,
            error_message=result.error_message,
        )

        return result

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
        metrics_collector: Optional[ScalabilityMetricsCollector] = None,
    ) -> PredictionShardResult:
        request = self._build_predict_shard_request(
            model_id=model_id,
            experiment_id=experiment_id,
            task_type=task_type,
            features=features,
            tree_artifact_uris=tree_artifact_uris,
            class_labels=class_labels or [],
        )

        request_bytes = self._safe_proto_size(request)
        n_rows, n_cols = self._matrix_shape(features)

        self._metrics_event(
            metrics_collector,
            "worker_client_predict_rpc_started",
            model_id=model_id,
            experiment_id=experiment_id,
            worker_host=worker_host,
            worker_port=worker_port,
            task_type=task_type,
            n_rows=n_rows,
            n_features=n_cols,
            tree_artifact_count=len(tree_artifact_uris),
            class_count=len(class_labels or []),
            request_bytes=request_bytes,
            timeout_seconds=self.timeout_predict_seconds,
        )

        started_at = time.time()

        try:
            response = self._call_predict_shard(
                worker_host=worker_host,
                worker_port=worker_port,
                request=request,
            )

        except Exception as exc:
            latency_seconds = time.time() - started_at

            self._metrics_event(
                metrics_collector,
                "worker_client_predict_rpc_failed",
                model_id=model_id,
                experiment_id=experiment_id,
                worker_host=worker_host,
                worker_port=worker_port,
                task_type=task_type,
                n_rows=n_rows,
                n_features=n_cols,
                tree_artifact_count=len(tree_artifact_uris),
                request_bytes=request_bytes,
                response_bytes=None,
                latency_seconds=latency_seconds,
                error_message=str(exc),
            )

            raise

        latency_seconds = time.time() - started_at
        response_bytes = self._safe_proto_size(response)

        result = self._build_prediction_result_from_response(response)

        self._metrics_event(
            metrics_collector,
            "worker_client_predict_rpc_completed",
            model_id=model_id,
            experiment_id=experiment_id,
            worker_id=result.worker_id,
            worker_host=worker_host,
            worker_port=worker_port,
            task_type=task_type,
            n_rows=n_rows,
            n_features=n_cols,
            tree_artifact_count=len(tree_artifact_uris),
            class_count=len(class_labels or []),
            request_bytes=request_bytes,
            response_bytes=response_bytes,
            latency_seconds=latency_seconds,
            result_success=result.success,
            response_rows=result.n_rows,
            response_cols=result.n_cols,
            error_message=result.error_message,
        )

        return result

    def predict_shard_from_uri(
        self,
        worker_host: str,
        worker_port: int,
        model_id: str,
        experiment_id: str,
        task_type: str,
        features_uri: str,
        prediction_output_dir: str,
        prediction_id: str,
        tree_artifact_uris: Sequence[str],
        class_labels: Sequence[str] | None = None,
        metrics_collector: Optional[ScalabilityMetricsCollector] = None,
    ) -> PredictionShardResult:
        """
        Nuovo percorso scalabile per validation/test/inference.

        Invece di inviare una DenseMatrix dentro gRPC, invia solo:
          - features_uri
          - prediction_output_dir
          - prediction_id
          - tree_artifact_uris

        Il worker legge le feature da EFS, calcola le predizioni parziali,
        le salva su EFS e ritorna prediction_uri.
        """
        request = self._build_predict_shard_from_uri_request(
            model_id=model_id,
            experiment_id=experiment_id,
            task_type=task_type,
            features_uri=features_uri,
            prediction_output_dir=prediction_output_dir,
            prediction_id=prediction_id,
            tree_artifact_uris=tree_artifact_uris,
            class_labels=class_labels or [],
        )

        request_bytes = self._safe_proto_size(request)

        self._metrics_event(
            metrics_collector,
            "worker_client_predict_uri_rpc_started",
            model_id=model_id,
            experiment_id=experiment_id,
            worker_host=worker_host,
            worker_port=worker_port,
            task_type=task_type,
            features_uri=features_uri,
            prediction_output_dir=prediction_output_dir,
            prediction_id=prediction_id,
            tree_artifact_count=len(tree_artifact_uris),
            class_count=len(class_labels or []),
            request_bytes=request_bytes,
            timeout_seconds=self.timeout_predict_seconds,
        )

        started_at = time.time()

        try:
            response = self._call_predict_shard(
                worker_host=worker_host,
                worker_port=worker_port,
                request=request,
            )

        except Exception as exc:
            latency_seconds = time.time() - started_at

            self._metrics_event(
                metrics_collector,
                "worker_client_predict_uri_rpc_failed",
                model_id=model_id,
                experiment_id=experiment_id,
                worker_host=worker_host,
                worker_port=worker_port,
                task_type=task_type,
                features_uri=features_uri,
                prediction_output_dir=prediction_output_dir,
                prediction_id=prediction_id,
                tree_artifact_count=len(tree_artifact_uris),
                request_bytes=request_bytes,
                response_bytes=None,
                latency_seconds=latency_seconds,
                error_message=str(exc),
            )

            raise

        latency_seconds = time.time() - started_at
        response_bytes = self._safe_proto_size(response)

        result = self._build_prediction_result_from_response(response)

        self._metrics_event(
            metrics_collector,
            "worker_client_predict_uri_rpc_completed",
            model_id=model_id,
            experiment_id=experiment_id,
            worker_id=result.worker_id,
            worker_host=worker_host,
            worker_port=worker_port,
            task_type=task_type,
            features_uri=features_uri,
            prediction_output_dir=prediction_output_dir,
            prediction_id=prediction_id,
            tree_artifact_count=len(tree_artifact_uris),
            class_count=len(class_labels or []),
            request_bytes=request_bytes,
            response_bytes=response_bytes,
            latency_seconds=latency_seconds,
            result_success=result.success,
            prediction_uri=result.prediction_uri,
            response_rows=result.n_rows,
            response_cols=result.n_cols,
            error_message=result.error_message,
        )

        return result

    def _build_train_shard_request(
        self,
        shard: TrainingShard,
    ) -> rf_pb2.TrainShardRequest:
        fc = shard.forest_config

        max_depth = 0 if fc.max_depth is None else fc.max_depth
        max_features = "none" if fc.max_features is None else str(fc.max_features)

        lease_expires_at_unix_ms = 0
        if shard.lease_expires_at_ts is not None:
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
                    training_time_seconds=artifact.training_time_seconds,
                    feature_importances=list(artifact.feature_importances),
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

    def _build_predict_shard_from_uri_request(
        self,
        model_id: str,
        experiment_id: str,
        task_type: str,
        features_uri: str,
        prediction_output_dir: str,
        prediction_id: str,
        tree_artifact_uris: Sequence[str],
        class_labels: Sequence[str],
    ) -> rf_pb2.PredictShardRequest:
        if not features_uri:
            raise ValueError("features_uri cannot be empty")

        if not prediction_output_dir:
            raise ValueError("prediction_output_dir cannot be empty")

        if not prediction_id:
            raise ValueError("prediction_id cannot be empty")

        if not tree_artifact_uris:
            raise ValueError("tree_artifact_uris cannot be empty")

        return rf_pb2.PredictShardRequest(
            model_id=str(model_id),
            experiment_id=str(experiment_id),
            task_type=str(task_type),
            tree_artifact_uris=[
                str(uri)
                for uri in tree_artifact_uris
            ],
            class_labels=[
                str(label)
                for label in class_labels
            ],
            features_uri=str(features_uri),
            prediction_output_dir=str(prediction_output_dir),
            prediction_id=str(prediction_id),
        )

    def _build_prediction_result_from_response(
        self,
        response: rf_pb2.PredictShardResponse,
    ) -> PredictionShardResult:
        if response.n_rows < 0 or response.n_cols < 0:
            raise ValueError("Invalid prediction response shape")

        error_message = response.error if response.error else None
        prediction_uri = response.prediction_uri if response.prediction_uri else None

        # Nuovo percorso scalabile: il worker ritorna solo prediction_uri.
        if prediction_uri:
            return PredictionShardResult(
                worker_id=response.worker_id,
                success=response.success,
                error_message=error_message,
                values=None,
                prediction_uri=prediction_uri,
                n_rows=int(response.n_rows),
                n_cols=int(response.n_cols),
            )

        # Vecchio percorso legacy: il worker ritorna values dentro gRPC.
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
            error_message=error_message,
            values=values,
            prediction_uri=None,
            n_rows=int(response.n_rows),
            n_cols=int(response.n_cols),
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

    # --------------------------------------------------------
    # metrics helpers
    # --------------------------------------------------------

    def _metrics_event(
        self,
        metrics_collector: Optional[ScalabilityMetricsCollector],
        event: str,
        **payload,
    ) -> None:
        if metrics_collector is None:
            return

        try:
            metrics_collector.record_event(
                event,
                **payload,
            )
        except Exception as exc:
            print(
                "[WorkerClient] Failed to record scalability event "
                f"'{event}': {exc}",
                flush=True,
            )

    def _safe_proto_size(self, message) -> Optional[int]:
        try:
            return int(message.ByteSize())
        except Exception:
            return None

    def _matrix_shape(self, features: np.ndarray) -> tuple[int, int]:
        arr = np.asarray(features)

        if arr.ndim != 2:
            return 0, 0

        return int(arr.shape[0]), int(arr.shape[1])