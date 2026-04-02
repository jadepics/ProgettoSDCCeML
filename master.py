from __future__ import annotations

import os
import threading
import time
import uuid
from concurrent import futures
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import grpc
import numpy as np
import pandas as pd
from masterPackage.shard_planner import ShardPlanner
from masterPackage.worker_client import WorkerClient
import rf_v2_pb2 as rf_pb2
import rf_v2_pb2_grpc as rf_pb2_grpc
from masterPackage.data.data_preparation_service import DataPreparationService
from masterPackage.data.dataset_loader import DatasetLoader
from masterPackage.data.dataset_validator import DatasetValidator
from masterPackage.data.split_manager import SplitManager
from masterPackage.training_orchestrator import TrainingOrchestrator
from masterPackage.experiment_planner import ExperimentPlanner
from masterPackage.model_manifest_builder import ModelManifestBuilder
from masterPackage.inference_coordinator import InferenceCoordinator
from masterPackage.validation_coordinator import ValidationCoordinator

from common.enums import (
    ExperimentStatus,
    JobStatus,
    ModelStatus,
    TaskStatus,
    TreeStatus,
)
from common.contracts import (
    ExperimentRecord,
    ForestConfiguration,
    HyperparameterSpace,
    ModelManifest,
    TaskRecord,
    TrainingJobRecord,
    TrainingRequest,
    TrainingShard,
    TreeArtifactMetadata,
    ValidationMetrics,
)
from common.ids import (
    generate_experiment_id,
    generate_job_id,
    generate_task_id,
    generate_tree_id,
    tree_seed,
)
from common.repositories import (
    JobRepository,
    ModelRepository,
    SharedArtifactStore,
    TaskLedger,
)
from common.storage_layout import StorageLayout
from masterPackage.fault_tolerance import (
    InMemoryLeaderConsensusService,
    LeadershipGuard,
)

HEARTBEAT_TIMEOUT_SECONDS = 15.0
DEFAULT_RPC_TIMEOUT_SECONDS = 600.0


# ============================================================
# Utility
# ============================================================

def matrix_from_proto(msg: rf_pb2.DenseMatrix) -> np.ndarray:
    arr = np.asarray(msg.values, dtype=float)
    if msg.n_rows * msg.n_cols != arr.size:
        raise ValueError("DenseMatrix shape mismatch")
    return arr.reshape(msg.n_rows, msg.n_cols)


def matrix_to_proto(arr: np.ndarray) -> rf_pb2.DenseMatrix:
    arr = np.asarray(arr, dtype=float)
    if arr.ndim != 2:
        raise ValueError("Expected 2D matrix")
    return rf_pb2.DenseMatrix(
        values=arr.ravel().tolist(),
        n_rows=arr.shape[0],
        n_cols=arr.shape[1],
    )


def parse_dataset_url(dataset_url: str) -> str:
    if dataset_url.startswith("file://"):
        return dataset_url.replace("file://", "", 1)
    return dataset_url


def read_csv_dataset(dataset_url: str) -> pd.DataFrame:
    path_or_url = parse_dataset_url(dataset_url)
    return pd.read_csv(path_or_url)


def now_ts() -> float:
    return time.time()


# ============================================================
# Worker registry
# ============================================================

class WorkerInfo:
    def __init__(self, worker_id: str, host: str, port: int) -> None:
        self.worker_id = worker_id
        self.host = host
        self.port = port
        self.last_heartbeat = now_ts()
        self.running_tasks = 0

    @property
    def address(self) -> str:
        return f"{self.host}:{self.port}"


class WorkerRegistry:
    def __init__(self) -> None:
        self._workers: dict[str, WorkerInfo] = {}
        self._lock = threading.Lock()

    def register(self, worker_id: str, host: str, port: int) -> None:
        with self._lock:
            self._workers[worker_id] = WorkerInfo(worker_id, host, port)

    def heartbeat(self, worker_id: str, running_tasks: int) -> bool:
        with self._lock:
            worker = self._workers.get(worker_id)
            if worker is None:
                return False
            worker.last_heartbeat = now_ts()
            worker.running_tasks = running_tasks
            return True

    def alive_workers(self) -> list[WorkerInfo]:
        cutoff = now_ts() - HEARTBEAT_TIMEOUT_SECONDS
        with self._lock:
            return [w for w in self._workers.values() if w.last_heartbeat >= cutoff]

    def get_retry_candidate(self, exclude_worker_id: str | None = None) -> Optional[WorkerInfo]:
        candidates = [
            w for w in self.alive_workers()
            if exclude_worker_id is None or w.worker_id != exclude_worker_id
        ]
        if not candidates:
            return None
        return sorted(candidates, key=lambda w: w.running_tasks)[0]


# ============================================================
# Master coordinator
# ============================================================

class MasterCoordinator(rf_pb2_grpc.CoordinatorServiceServicer):
    """
    Versione ponte del masterPackage:
    - usa contratti comuni
    - usa ID deterministici
    - usa ledger e manifest persistiti
    - applica leader-only execution
    - non implementa ancora Raft reale, ma è compatibile con esso
    """

    def __init__(self, artifact_root: str = "/shared/artifacts") -> None:
        self.registry = WorkerRegistry()

        self.artifact_root = Path(artifact_root)
        self.artifact_root.mkdir(parents=True, exist_ok=True)
        # Consensus/leader guard: placeholder leader-only service.
        self.consensus = InMemoryLeaderConsensusService(is_leader=True)
        self.leadership_guard = LeadershipGuard(self.consensus)

        self.store = SharedArtifactStore(str(self.artifact_root))
        self.layout = StorageLayout(str(self.artifact_root))
        self.job_repository = JobRepository(self.store)
        self.model_repository = ModelRepository(self.store)
        self.task_ledger = TaskLedger(self.store)
        self.shard_planner = ShardPlanner(self.layout)
        self.worker_client = WorkerClient(
            timeout_train_seconds=DEFAULT_RPC_TIMEOUT_SECONDS,
            timeout_predict_seconds=DEFAULT_RPC_TIMEOUT_SECONDS,
        )
        self.training_orchestrator = TrainingOrchestrator(
            leadership_guard=self.leadership_guard,
            worker_registry=self.registry,
            task_ledger=self.task_ledger,
            job_repository=self.job_repository,
            shard_planner=self.shard_planner,
            worker_client=self.worker_client,
        )
        self.data_preparation_service = DataPreparationService(
            dataset_loader=DatasetLoader(),
            dataset_validator=DatasetValidator(),
            split_manager=SplitManager(),
            artifact_store=self.store,
        )
        self.inference_coordinator = InferenceCoordinator(
            leadership_guard=self.leadership_guard,
            worker_registry=self.registry,
            worker_client=self.worker_client,
            model_repository=self.model_repository,
        )
        self.validation_coordinator = ValidationCoordinator(
            leadership_guard=self.leadership_guard,
            worker_registry=self.registry,
            worker_client=self.worker_client,
        )
        self.model_manifest_builder = ModelManifestBuilder()
        self.experiment_planner = ExperimentPlanner()

        self._lock = threading.Lock()
    # --------------------------------------------------------
    # RPC: worker lifecycle
    # --------------------------------------------------------

    def RegisterWorker(self, request, context):
        self.registry.register(
            worker_id=request.worker_id,
            host=request.host,
            port=request.port,
        )
        return rf_pb2.RegisterWorkerResponse(
            accepted=True,
            message=f"Worker {request.worker_id} registered",
        )

    def Heartbeat(self, request, context):
        ok = self.registry.heartbeat(
            worker_id=request.worker_id,
            running_tasks=request.running_tasks,
        )
        return rf_pb2.HeartbeatResponse(ok=ok)

    # --------------------------------------------------------
    # RPC: submit training
    # --------------------------------------------------------

    def SubmitTraining(self, request, context):
        try:
            self.leadership_guard.require_leader()
        except Exception as exc:
            return rf_pb2.SubmitTrainingResponse(
                job_id="",
                status=rf_pb2.FAILED,
                message=f"Not leader: {exc}",
            )

        task_type = request.task_type.strip().lower()
        if task_type not in {"classification", "regression"}:
            return rf_pb2.SubmitTrainingResponse(
                job_id="",
                status=rf_pb2.FAILED,
                message="task_type must be 'classification' or 'regression'",
            )

        if request.n_estimators_total <= 0:
            return rf_pb2.SubmitTrainingResponse(
                job_id="",
                status=rf_pb2.FAILED,
                message="n_estimators_total must be > 0",
            )

        if not request.dataset_url.strip():
            return rf_pb2.SubmitTrainingResponse(
                job_id="",
                status=rf_pb2.FAILED,
                message="dataset_url must be non-empty",
            )

        if not request.target_column.strip():
            return rf_pb2.SubmitTrainingResponse(
                job_id="",
                status=rf_pb2.FAILED,
                message="target_column must be non-empty",
            )

        if request.validation_ratio < 0.0 or request.test_ratio < 0.0:
            return rf_pb2.SubmitTrainingResponse(
                job_id="",
                status=rf_pb2.FAILED,
                message="validation_ratio and test_ratio must be >= 0",
            )

        if request.validation_ratio + request.test_ratio >= 1.0:
            return rf_pb2.SubmitTrainingResponse(
                job_id="",
                status=rf_pb2.FAILED,
                message="validation_ratio + test_ratio must be < 1.0",
            )

        alive_workers = self.registry.alive_workers()
        if not alive_workers:
            return rf_pb2.SubmitTrainingResponse(
                job_id="",
                status=rf_pb2.FAILED,
                message="No alive workers available",
            )

        job_id = generate_job_id()
        model_id = str(uuid.uuid4())

        max_depth_candidates = [
            value if value > 0 else None
            for value in request.max_depth_candidates
        ]
        if not max_depth_candidates:
            max_depth_candidates = [None]

        max_features_candidates = list(request.max_features_candidates)
        if not max_features_candidates:
            max_features_candidates = ["sqrt" if task_type == "classification" else "1.0"]

        min_samples_split_candidates = list(request.min_samples_split_candidates)
        if not min_samples_split_candidates:
            min_samples_split_candidates = [2]

        min_samples_leaf_candidates = list(request.min_samples_leaf_candidates)
        if not min_samples_leaf_candidates:
            min_samples_leaf_candidates = [1]

        criterion_candidates = list(request.criterion_candidates)
        if not criterion_candidates:
            criterion_candidates = (
                ["gini"] if task_type == "classification" else ["squared_error"]
            )

        training_request = TrainingRequest(
            job_id=job_id,
            dataset_uri=request.dataset_url,
            target_column=request.target_column,
            task_type=task_type,
            hyperparameter_space=HyperparameterSpace(
                n_estimators_candidates=[request.n_estimators_total],
                max_depth_candidates=max_depth_candidates,
                max_features_candidates=max_features_candidates,
                min_samples_split_candidates=min_samples_split_candidates,
                min_samples_leaf_candidates=min_samples_leaf_candidates,
                criterion_candidates=criterion_candidates,
                bootstrap=request.bootstrap,
                global_random_seed=request.global_random_seed,
            ),
            n_estimators_total=request.n_estimators_total,
            validation_ratio=request.validation_ratio,
            test_ratio=request.test_ratio,
            global_random_seed=request.global_random_seed,
            bootstrap=request.bootstrap,
        )

        initial_experiment = self.experiment_planner.select_initial_experiment(training_request)
        experiment_id = initial_experiment.experiment_id

        job_record = TrainingJobRecord(
            job_id=job_id,
            status=JobStatus.PENDING,
            training_request=training_request,
            prepared_dataset=None,
            experiment_ids=[experiment_id],
            selected_experiment_id=None,
            model_id=model_id,
            message="Training queued",
            created_at=now_ts(),
            updated_at=now_ts(),
        )
        self.job_repository.save(job_record)
        self.job_repository.save_experiment(job_id, initial_experiment)

        threading.Thread(
            target=self._run_training_job,
            args=(job_id, model_id, experiment_id),
            daemon=True,
        ).start()

        return rf_pb2.SubmitTrainingResponse(
            job_id=job_id,
            status=rf_pb2.PENDING,
            message="Training started",
        )

    def GetTrainingStatus(self, request, context):
        record = self.job_repository.load(request.job_id)
        if record is None:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details("Job not found")
            return rf_pb2.GetTrainingStatusResponse()

        completed_trees = self.task_ledger.count_completed_trees(request.job_id)
        total_trees = record.training_request.n_estimators_total

        return rf_pb2.GetTrainingStatusResponse(
            job_id=record.job_id,
            model_id=record.model_id or "",
            status=self._job_status_to_proto(record.status),
            total_trees=total_trees,
            completed_trees=completed_trees,
            message=record.message,
            workers=[w.worker_id for w in self.registry.alive_workers()],
        )

    # --------------------------------------------------------
    # Internal training pipeline
    # --------------------------------------------------------

    def _run_training_job(self, job_id: str, model_id: str, experiment_id: str) -> None:
        record = self.job_repository.load(job_id)
        if record is None:
            return

        try:
            self.leadership_guard.require_leader()

            record.status = JobStatus.RUNNING
            record.message = "Training in progress"
            record.updated_at = now_ts()
            self.job_repository.save(record)

            req = record.training_request
            model_type = req.task_type.strip().lower()
            if model_type not in {"classification", "regression"}:
                raise ValueError("task_type must be 'classification' or 'regression'")

            prepared_dataset = self.data_preparation_service.prepare(
                job_id=req.job_id,
                dataset_uri=req.dataset_uri,
                target_column=req.target_column,
                task_type=req.task_type,
                validation_ratio=req.validation_ratio,
                test_ratio=req.test_ratio,
                random_seed=req.global_random_seed,
            )

            record.prepared_dataset = prepared_dataset
            record.updated_at = now_ts()
            self.job_repository.save(record)


            experiment = self.job_repository.load_experiment(job_id, experiment_id)
            if experiment is None:
                raise RuntimeError(f"Experiment '{experiment_id}' not found")

            forest_config = experiment.forest_config

            collected_artifacts = self.training_orchestrator.run_experiment(
                job_id=job_id,
                experiment_id=experiment_id,
                forest_config=forest_config,
            )

            validation_result = self.validation_coordinator.validate_experiment(
                experiment_id=experiment_id,
                task_type=model_type,
                validation_features_uri=prepared_dataset.validation_features_uri,
                validation_labels_uri=prepared_dataset.validation_labels_uri,
                tree_artifacts=collected_artifacts,
                class_labels=prepared_dataset.class_labels or [],
            )

            validation_metrics = validation_result.metrics

            experiment = self.job_repository.load_experiment(job_id, experiment_id)
            if experiment is None:
                raise RuntimeError(
                    f"Experiment '{experiment_id}' not found after training orchestration"
                )

            experiment.status = ExperimentStatus.COMPLETED
            experiment.completed_tree_count = len(collected_artifacts)
            experiment.validation_metrics = validation_metrics
            self.job_repository.save_experiment(job_id, experiment)

            manifest = self.model_manifest_builder.build(
                model_id=model_id,
                job_id=job_id,
                experiment_id=experiment_id,
                model_type=model_type,
                forest_config=forest_config,
                prepared_dataset=prepared_dataset,
                tree_artifacts=collected_artifacts,
                validation_metrics=validation_metrics,
                test_metrics=None,
                status=ModelStatus.READY,
            )

            self.model_repository.save(manifest)
            record = self.job_repository.load(job_id)
            if record is None:
                raise RuntimeError(f"Job '{job_id}' disappeared before completion update")

            record.status = JobStatus.COMPLETED
            record.message = "Training completed successfully"
            record.selected_experiment_id = experiment_id
            record.updated_at = now_ts()
            self.job_repository.save(record)

        except Exception as exc:
            record = self.job_repository.load(job_id)
            if record is not None:
                record.status = JobStatus.FAILED
                record.message = str(exc)
                record.updated_at = now_ts()
                self.job_repository.save(record)

    def _split_trees(
            self,
            n_trees: int,
            workers: list[WorkerInfo],
    ) -> list[tuple[WorkerInfo, int, int]]:
        if n_trees <= 0:
            raise ValueError("n_trees must be > 0")
        if not workers:
            return []

        chunks: list[tuple[WorkerInfo, int, int]] = []
        base = n_trees // len(workers)
        rem = n_trees % len(workers)

        start = 0
        for index, worker in enumerate(workers):
            tree_count = base + (1 if index < rem else 0)
            if tree_count <= 0:
                continue

            chunks.append((worker, start, tree_count))
            start += tree_count

        return chunks

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
    def _call_train_shard(
        self,
        worker: WorkerInfo,
        request: rf_pb2.TrainShardRequest,
    ) -> rf_pb2.TrainShardResponse:
        with grpc.insecure_channel(worker.address) as channel:
            stub = rf_pb2_grpc.WorkerServiceStub(channel)
            return stub.TrainShard(request, timeout=DEFAULT_RPC_TIMEOUT_SECONDS)

    # --------------------------------------------------------
    # RPC: inference
    # --------------------------------------------------------

    def SubmitInference(self, request, context):
        try:
            X = matrix_from_proto(request.features)

            result = self.inference_coordinator.run_inference(
                model_id=request.model_id,
                features=X,
            )

            if result.task_type == "classification":
                return rf_pb2.SubmitInferenceResponse(
                    status=rf_pb2.COMPLETED,
                    predicted_labels=result.predicted_labels or [],
                    message="Inference completed",
                )

            return rf_pb2.SubmitInferenceResponse(
                status=rf_pb2.COMPLETED,
                predicted_values=result.predicted_values or [],
                message="Inference completed",
            )

        except Exception as exc:
            return rf_pb2.SubmitInferenceResponse(
                status=rf_pb2.FAILED,
                message=str(exc),
            )
    def _call_predict_shard(
        self,
        worker: WorkerInfo,
        request: rf_pb2.PredictShardRequest,
    ) -> rf_pb2.PredictShardResponse:
        with grpc.insecure_channel(worker.address) as channel:
            stub = rf_pb2_grpc.WorkerServiceStub(channel)
            return stub.PredictShard(request, timeout=DEFAULT_RPC_TIMEOUT_SECONDS)

    def _split_tree_uris(self, tree_uris: list[str], n_parts: int) -> list[list[str]]:
        if n_parts <= 0:
            raise ValueError("n_parts must be > 0")
        shards = [[] for _ in range(min(n_parts, len(tree_uris)))]
        for i, uri in enumerate(tree_uris):
            shards[i % len(shards)].append(uri)
        return [s for s in shards if s]

    # --------------------------------------------------------
    # Helpers
    # --------------------------------------------------------

    def _job_status_to_proto(self, status: str) -> int:
        mapping = {
            "PENDING": rf_pb2.PENDING,
            "RUNNING": rf_pb2.RUNNING,
            "COMPLETED": rf_pb2.COMPLETED,
            "FAILED": rf_pb2.FAILED,
        }
        return mapping.get(status, rf_pb2.FAILED)


# ============================================================
# Server bootstrap
# ============================================================

def serve(host: str = "0.0.0.0", port: int = 50051, artifact_root: str = "/shared/artifacts"):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=32))
    rf_pb2_grpc.add_CoordinatorServiceServicer_to_server(
        MasterCoordinator(artifact_root=artifact_root),
        server,
    )
    server.add_insecure_port(f"{host}:{port}")
    server.start()
    print(f"[MASTER] listening on {host}:{port}")
    server.wait_for_termination()


if __name__ == "__main__":
    serve(
        host=os.getenv("MASTER_HOST", "0.0.0.0"),
        port=int(os.getenv("MASTER_PORT", "50051")),
        artifact_root=os.getenv("ARTIFACT_ROOT", "/shared/artifacts"),
    )