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
from masterPackage.retry_policy import RetryPolicy
from masterPackage.shard_planner import ShardPlanner
from masterPackage.task_lease_manager import TaskLeaseManager
from masterPackage.training_job_service import TrainingJobService
from masterPackage.training_orchestrator import TrainingOrchestrator
from masterPackage.validation_coordinator import ValidationCoordinator
from masterPackage.worker_client import WorkerClient
from masterPackage.worker_heartbeat_monitor import WorkerHeartbeatMonitor
from masterPackage.recovery_planner import RecoveryPlanner

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
GRPC_MAX_MESSAGE_LENGTH = 64 * 1024 * 1024  # 64 MB

GRPC_OPTIONS = [
    ("grpc.max_send_message_length", GRPC_MAX_MESSAGE_LENGTH),
    ("grpc.max_receive_message_length", GRPC_MAX_MESSAGE_LENGTH),
]

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
        self.active_task_ids: list[str] = []

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

    def heartbeat(
        self,
        worker_id: str,
        running_tasks: int,
        active_task_ids: Optional[list[str]] = None,
    ) -> bool:
        with self._lock:
            worker = self._workers.get(worker_id)
            if worker is None:
                return False

            worker.last_heartbeat = now_ts()
            worker.running_tasks = running_tasks
            worker.active_task_ids = list(active_task_ids or [])
            return True

    def alive_workers(self) -> list[WorkerInfo]:
        cutoff = now_ts() - HEARTBEAT_TIMEOUT_SECONDS
        with self._lock:
            return [
                worker
                for worker in self._workers.values()
                if worker.last_heartbeat >= cutoff
            ]

    def get_retry_candidate(
        self,
        exclude_worker_id: str | None = None,
    ) -> Optional[WorkerInfo]:
        candidates = [
            worker
            for worker in self.alive_workers()
            if exclude_worker_id is None or worker.worker_id != exclude_worker_id
        ]
        if not candidates:
            return None
        return sorted(candidates, key=lambda worker: worker.running_tasks)[0]

    def list_workers(self) -> list[WorkerInfo]:
        with self._lock:
            return list(self._workers.values())

# ============================================================
# Master coordinator
# ============================================================

class MasterCoordinator(rf_pb2_grpc.CoordinatorServiceServicer):
    """
    Facciata RPC del master.

    Stato coerente col proto rf_v2.proto:
    - RegisterWorker
    - Heartbeat
    - SubmitTraining

    Responsabilità:
    - ricevere le RPC esposte dal CoordinatorService
    - validare input minimi a livello RPC
    - applicare leader-only execution dove richiesto
    - tradurre protobuf -> contratti di dominio
    - delegare ai servizi applicativi del control plane

    Non deve:
    - orchestrare direttamente training o validation
    - contenere logica di inferenza distribuita
    - sostituire TrainingJobService / TrainingOrchestrator
    """

    def __init__(self, artifact_root: str = "/shared/artifacts") -> None:
        self.artifact_root = Path(artifact_root)
        self.artifact_root.mkdir(parents=True, exist_ok=True)

        # stato condiviso del master
        self.registry = WorkerRegistry()

        self.store = SharedArtifactStore(str(self.artifact_root))
        self.layout = StorageLayout(str(self.artifact_root))

        self.job_repository = JobRepository(self.store)
        self.model_repository = ModelRepository(self.store)
        self.task_ledger = TaskLedger(self.store)

        # leadership / consenso
        self.consensus = InMemoryLeaderConsensusService(
            node_id="master-1",
            start_as_leader=True,
        )
        self.leadership_guard = LeadershipGuard(self.consensus)

        # monitor heartbeat: va creato PRIMA di orchestrator/recovery
        self.worker_heartbeat_monitor = WorkerHeartbeatMonitor(
            worker_registry=self.registry,
            heartbeat_timeout_seconds=HEARTBEAT_TIMEOUT_SECONDS,
        )

        # data prep
        self.data_preparation_service = DataPreparationService(
            dataset_loader=DatasetLoader(),
            dataset_validator=DatasetValidator(),
            split_manager=SplitManager(),
            artifact_store=self.store,
        )

        self.experiment_planner = ExperimentPlanner()
        self.model_selector = ModelSelector(selection_metric="auto")
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
            retry_policy=RetryPolicy(
                max_attempts_per_task=2,
                base_backoff_seconds=0.5,
                retry_on_timeout=True,
                retry_on_worker_failure=True,
                retry_on_unknown_error=False,
            ),
            task_lease_manager=TaskLeaseManager(
                task_ledger=self.task_ledger,
                lease_timeout_seconds=600.0,
            ),
            worker_heartbeat_monitor=self.worker_heartbeat_monitor,
        )

        self.validation_coordinator = ValidationCoordinator(
            leadership_guard=self.leadership_guard,
            worker_registry=self.registry,
            worker_client=self.worker_client,
        )

        self.recovery_planner = RecoveryPlanner(
            task_ledger=self.task_ledger,
            shard_planner=self.shard_planner,
            worker_heartbeat_monitor=self.worker_heartbeat_monitor,
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
    # --------------------------------------------------------
    # RPC: worker lifecycle
    # --------------------------------------------------------

    def RegisterWorker(self, request, context):
        if not request.worker_id.strip():
            return rf_pb2.RegisterWorkerResponse(
                accepted=False,
                message="worker_id must be non-empty",
            )

        if not request.host.strip():
            return rf_pb2.RegisterWorkerResponse(
                accepted=False,
                message="host must be non-empty",
            )

        if request.port <= 0:
            return rf_pb2.RegisterWorkerResponse(
                accepted=False,
                message="port must be > 0",
            )

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
            active_task_ids=list(request.active_task_ids),
        )
        return rf_pb2.HeartbeatResponse(ok=ok)

    # --------------------------------------------------------
    # RPC: submit training
    # --------------------------------------------------------

    def SubmitTraining(self, request, context):
        try:
            self.leadership_guard.require_leader()
        except Exception as exc:
            return self._failed_submit_training_response(f"Not leader: {exc}")

        validation_error = self._validate_submit_training_request(request)
        if validation_error is not None:
            return self._failed_submit_training_response(validation_error)

        alive_workers = self.registry.alive_workers()
        if not alive_workers:
            return self._failed_submit_training_response("No alive workers available")

        training_request = self._build_training_request(request)

        try:
            created_job_id = self.training_job_service.start_training_job(training_request)
        except Exception as exc:
            return self._failed_submit_training_response(str(exc))

        return rf_pb2.SubmitTrainingResponse(
            job_id=created_job_id,
            status=rf_pb2.PENDING,
            message="Training started",
        )

    # --------------------------------------------------------
    # Helpers
    # --------------------------------------------------------

    def _failed_submit_training_response(self, message: str) -> rf_pb2.SubmitTrainingResponse:
        return rf_pb2.SubmitTrainingResponse(
            job_id="",
            status=rf_pb2.FAILED,
            message=message,
        )

    def _validate_submit_training_request(self, request) -> Optional[str]:
        task_type = request.task_type.strip().lower()
        if task_type not in {"classification", "regression"}:
            return "task_type must be 'classification' or 'regression'"

        if request.n_estimators_total <= 0:
            return "n_estimators_total must be > 0"

        if not request.dataset_url.strip():
            return "dataset_url must be non-empty"

        if not request.target_column.strip():
            return "target_column must be non-empty"

        if request.validation_ratio < 0.0 or request.test_ratio < 0.0:
            return "validation_ratio and test_ratio must be >= 0"

        if request.validation_ratio + request.test_ratio >= 1.0:
            return "validation_ratio + test_ratio must be < 1.0"

        return None

    def _build_training_request(self, request) -> TrainingRequest:
        task_type = request.task_type.strip().lower()
        job_id = self._generate_job_id()

        max_depth_candidates = [
            value if value > 0 else None
            for value in request.max_depth_candidates
        ]
        if not max_depth_candidates:
            max_depth_candidates = [None]

        max_features_candidates = [
            self._parse_max_features_candidate(value)
            for value in request.max_features_candidates
        ]
        if not max_features_candidates:
            max_features_candidates = ["sqrt" if task_type == "classification" else 1.0]

        min_samples_split_candidates = list(request.min_samples_split_candidates)
        if not min_samples_split_candidates:
            min_samples_split_candidates = [2]

        min_samples_leaf_candidates = list(request.min_samples_leaf_candidates)
        if not min_samples_leaf_candidates:
            min_samples_leaf_candidates = [1]

        criterion_candidates = list(request.criterion_candidates)
        if not criterion_candidates:
            criterion_candidates = ["gini"] if task_type == "classification" else ["squared_error"]

        return TrainingRequest(
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

    def _parse_max_features_candidate(self, raw_value: str):
        value = raw_value.strip()
        if not value:
            return None

        lowered = value.lower()
        if lowered in {"none", "null"}:
            return None

        try:
            return float(value)
        except ValueError:
            return value

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
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=32),
        options=GRPC_OPTIONS,
    )
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