from __future__ import annotations

import os
import threading
import time
from concurrent import futures
from pathlib import Path
from typing import Optional

import grpc
import numpy as np

import rf_v2_pb2 as rf_pb2
import rf_v2_pb2_grpc as rf_pb2_grpc

from masterPackage.data.data_preparation_service import DataPreparationService
from masterPackage.data.dataset_loader import DatasetLoader
from masterPackage.data.dataset_validator import DatasetValidator
from masterPackage.data.split_manager import SplitManager
from masterPackage.experiment_planner import ExperimentPlanner
from masterPackage.fault_tolerance import (
    InMemoryLeaderConsensusService,
    LeadershipGuard,
)
from masterPackage.inference_coordinator import InferenceCoordinator
from masterPackage.model_manifest_builder import ModelManifestBuilder
from masterPackage.model_selector import ModelSelector
from masterPackage.shard_planner import ShardPlanner
from masterPackage.training_job_service import TrainingJobService
from masterPackage.training_orchestrator import TrainingOrchestrator
from masterPackage.validation_coordinator import ValidationCoordinator
from masterPackage.worker_client import WorkerClient

from common.contracts import HyperparameterSpace, TrainingRequest
from common.repositories import (
    JobRepository,
    ModelRepository,
    SharedArtifactStore,
    TaskLedger,
)
from common.storage_layout import StorageLayout

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
            return [worker for worker in self._workers.values() if worker.last_heartbeat >= cutoff]

    def get_retry_candidate(self, exclude_worker_id: str | None = None) -> Optional[WorkerInfo]:
        candidates = [
            worker
            for worker in self.alive_workers()
            if exclude_worker_id is None or worker.worker_id != exclude_worker_id
        ]
        if not candidates:
            return None
        return sorted(candidates, key=lambda worker: worker.running_tasks)[0]


# ============================================================
# Master coordinator
# ============================================================

class MasterCoordinator(rf_pb2_grpc.CoordinatorServiceServicer):
    """
    Facciata RPC del master.

    Responsabilità:
    - ricevere RPC
    - validare input minimi
    - applicare leader-only execution
    - delegare ai servizi applicativi del control plane
    """

    def __init__(self, artifact_root: str = "/shared/artifacts") -> None:
        self.registry = WorkerRegistry()

        self.artifact_root = Path(artifact_root)
        self.artifact_root.mkdir(parents=True, exist_ok=True)

        self.consensus = InMemoryLeaderConsensusService(is_leader=True)
        self.leadership_guard = LeadershipGuard(self.consensus)

        self.store = SharedArtifactStore(str(self.artifact_root))
        self.layout = StorageLayout(str(self.artifact_root))

        self.job_repository = JobRepository(self.store)
        self.model_repository = ModelRepository(self.store)
        self.task_ledger = TaskLedger(self.store)

        self.data_preparation_service = DataPreparationService(
            dataset_loader=DatasetLoader(),
            dataset_validator=DatasetValidator(),
            split_manager=SplitManager(),
            artifact_store=self.store,
        )

        self.experiment_planner = ExperimentPlanner()
        self.model_selector = ModelSelector(selection_metric="accuracy")
        self.model_manifest_builder = ModelManifestBuilder()

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

        self.validation_coordinator = ValidationCoordinator(
            leadership_guard=self.leadership_guard,
            worker_registry=self.registry,
            worker_client=self.worker_client,
        )

        self.inference_coordinator = InferenceCoordinator(
            leadership_guard=self.leadership_guard,
            worker_registry=self.registry,
            worker_client=self.worker_client,
            model_repository=self.model_repository,
        )

        self.training_job_service = TrainingJobService(
            leadership_guard=self.leadership_guard,
            job_repository=self.job_repository,
            model_repository=self.model_repository,
            data_preparation_service=self.data_preparation_service,
            experiment_planner=self.experiment_planner,
            training_orchestrator=self.training_orchestrator,
            validation_coordinator=self.validation_coordinator,
            model_selector=self.model_selector,
            model_manifest_builder=self.model_manifest_builder,
        )

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

        job_id = self._generate_job_id()

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

        try:
            created_job_id = self.training_job_service.start_training_job(training_request)
        except Exception as exc:
            return rf_pb2.SubmitTrainingResponse(
                job_id="",
                status=rf_pb2.FAILED,
                message=str(exc),
            )

        return rf_pb2.SubmitTrainingResponse(
            job_id=created_job_id,
            status=rf_pb2.PENDING,
            message="Training started",
        )

    # --------------------------------------------------------
    # RPC: training status
    # --------------------------------------------------------

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
            workers=[worker.worker_id for worker in self.registry.alive_workers()],
        )

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

    # --------------------------------------------------------
    # Helpers
    # --------------------------------------------------------

    def _job_status_to_proto(self, status) -> int:
        status_str = getattr(status, "value", status)
        mapping = {
            "PENDING": rf_pb2.PENDING,
            "RUNNING": rf_pb2.RUNNING,
            "COMPLETED": rf_pb2.COMPLETED,
            "FAILED": rf_pb2.FAILED,
        }
        return mapping.get(status_str, rf_pb2.FAILED)

    def _generate_job_id(self) -> str:
        from common.ids import generate_job_id
        return generate_job_id()


# ============================================================
# Server bootstrap
# ============================================================

def serve(
    host: str = "0.0.0.0",
    port: int = 50051,
    artifact_root: str = "/shared/artifacts",
):
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