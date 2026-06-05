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


from masterPackage.test_evaluator import TestEvaluator
from masterPackage.data.data_preparation_service import DataPreparationService
from masterPackage.data.dataset_loader import DatasetLoader
from masterPackage.data.dataset_validator import DatasetValidator
from masterPackage.data.split_manager import SplitManager
from masterPackage.experiment_planner import ExperimentPlanner
from masterPackage.fault_tolerance import (
    InMemoryLeaderConsensusService,
    LeadershipGuard,
    RaftConsensusService,
    build_raft_node_config_from_env,
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
from common.enums import WorkerLivenessStatus

#Valore precedente ai test: HEARTBEAT_TIMEOUT_SECONDS = 15.0
HEARTBEAT_TIMEOUT_SECONDS = 6.0
#Valore precedente ai test: DEFAULT_RPC_TIMEOUT_SECONDS = 600.0
DEFAULT_RPC_TIMEOUT_SECONDS = 60.0
GRPC_MAX_MESSAGE_LENGTH = 64 * 1024 * 1024  # 64 MB

GRPC_OPTIONS = [
    ("grpc.max_send_message_length", GRPC_MAX_MESSAGE_LENGTH),
    ("grpc.max_receive_message_length", GRPC_MAX_MESSAGE_LENGTH),
]

SUPPORTED_DATASET_SCENARIOS = {
    "baseline_original",
    "baseline_no_leakage",
    "no_diagnostic_features",
    "no_diagnostic_extended",
    "clinical_only",
    "glucose_only",
}
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

        # nuovo: timestamp ultimo progresso per task
        self.active_task_last_progress_ts: dict[str, float] = {}

        # nuovo: stato di liveness del worker
        self.liveness_status = WorkerLivenessStatus.ALIVE
        self.quarantined_at: float | None = None
        self.quarantine_reason: str | None = None

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
        active_task_last_progress_ts: Optional[dict[str, float]] = None,
    ) -> bool:
        with self._lock:
            worker = self._workers.get(worker_id)
            if worker is None:
                return False

            # se il worker è già quarantinato, non lo riattiviamo
            if worker.liveness_status == WorkerLivenessStatus.DEAD:
                return False

            worker.last_heartbeat = now_ts()
            worker.running_tasks = running_tasks
            worker.active_task_ids = list(active_task_ids or [])
            worker.active_task_last_progress_ts = dict(active_task_last_progress_ts or {})
            return True

    def quarantine(self, worker_id: str, reason: str) -> bool:
        with self._lock:
            worker = self._workers.get(worker_id)
            if worker is None:
                return False

            worker.liveness_status = WorkerLivenessStatus.DEAD
            worker.quarantined_at = now_ts()
            worker.quarantine_reason = reason

            print(
                "[WorkerRegistry] quarantined "
                f"worker={worker_id} "
                f"reason={reason}",
                flush=True,
            )

            return True

    def quarantined_workers(self) -> list[WorkerInfo]:
        with self._lock:
            return [
                worker
                for worker in self._workers.values()
                if worker.liveness_status == WorkerLivenessStatus.DEAD
            ]

    def quarantined_worker_ids(self) -> list[str]:
        return [worker.worker_id for worker in self.quarantined_workers()]

    def alive_workers(self) -> list[WorkerInfo]:
        cutoff = now_ts() - HEARTBEAT_TIMEOUT_SECONDS
        with self._lock:
            return [
                worker
                for worker in self._workers.values()
                if worker.last_heartbeat >= cutoff
                   and worker.liveness_status == WorkerLivenessStatus.ALIVE
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

    def __init__(self, artifact_root: str = "/mnt/efs/gp_artifacts") -> None:
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
        self.consensus = self._build_consensus_service()
        self.consensus.start()
        self.leadership_guard = LeadershipGuard(self.consensus)

        # monitor heartbeat: va creato PRIMA di orchestrator/recovery
        self.worker_heartbeat_monitor = WorkerHeartbeatMonitor(
            worker_registry=self.registry,
            heartbeat_timeout_seconds=HEARTBEAT_TIMEOUT_SECONDS,

            ###########################################################################
            #introdotto temporaneamente per rendere il timeout per gli zombie più breve
            ###########################################################################
            task_progress_timeout_seconds=8,
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
        self.shard_planner = ShardPlanner(
            self.layout,
            max_running_tasks_per_worker=2,
        )
        
        self.worker_client = WorkerClient(
            timeout_train_seconds=DEFAULT_RPC_TIMEOUT_SECONDS,
            timeout_predict_seconds=DEFAULT_RPC_TIMEOUT_SECONDS,
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
        self.test_evaluator = TestEvaluator(
            leadership_guard=self.leadership_guard,
            worker_registry=self.registry,
            worker_client=self.worker_client,
        )
        self.recovery_planner = RecoveryPlanner(
            task_ledger=self.task_ledger,
            shard_planner=self.shard_planner,
            worker_heartbeat_monitor=self.worker_heartbeat_monitor,
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
                #lease_timeout_seconds=600.0,
                lease_timeout_seconds=20.0,
            ),
            worker_heartbeat_monitor=self.worker_heartbeat_monitor,
        recovery_planner=self.recovery_planner)
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
            test_evaluator=self.test_evaluator,
        )
        self._start_recovery_on_startup_if_enabled()
    # --------------------------------------------------------
    # RPC: worker lifecycle
    # --------------------------------------------------------

    def RegisterWorker(self, request, context):
        if not self.consensus.is_leader():
            return rf_pb2.RegisterWorkerResponse(
                accepted=False,
                message=(
                    f"Not leader. role={self.consensus.current_role()} "
                    f"term={self.consensus.current_term()}"
                ),
            )
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
        if not self.consensus.is_leader():
            return rf_pb2.HeartbeatResponse(ok=False)


        active_task_progress = {
        item.task_id: float(item.last_progress_ts)
        for item in getattr(request, "active_tasks", [])
    }

        ok = self.registry.heartbeat(
            worker_id=request.worker_id,
            running_tasks=request.running_tasks,
            active_task_ids=list(request.active_task_ids),
            active_task_last_progress_ts=active_task_progress,
        )

        if ok:
            snapshot = next(
                (
                    item
                    for item in self.worker_heartbeat_monitor.snapshot()
                    if item.worker_id == request.worker_id
                ),
                None,
            )

            if snapshot is not None and snapshot.is_zombie:
                print(
                    "[MasterCoordinator] zombie worker detected: "
                    f"worker_id={snapshot.worker_id} "
                    f"running_tasks={snapshot.running_tasks} "
                    f"last_progress_age_seconds={snapshot.last_progress_age_seconds:.1f} "
                    f"age_seconds={snapshot.age_seconds:.1f}",
                    flush=True,
                )
                self.registry.quarantine(
                    request.worker_id,
                    reason=(
                        f"zombie worker: "
                        f"last_progress_age_seconds={snapshot.last_progress_age_seconds:.1f}"
                    ),
                )
        return rf_pb2.HeartbeatResponse(ok=ok)

# --------------------------------------------------------------
#               consenso
# ----------------------------------------------------------
    def _build_consensus_service(self):
        backend = os.getenv("CONSENSUS_BACKEND", "memory").strip().lower()

        if backend == "raft":
            config = build_raft_node_config_from_env(
                artifact_root=str(self.artifact_root),
            )
            return RaftConsensusService(config)

        node_id = os.getenv("MASTER_NODE_ID", "master-1").strip()

        return InMemoryLeaderConsensusService(
            node_id=node_id,
            start_as_leader=True,
        )

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
    def SubmitInference(self, request, context):
        try:
            self.leadership_guard.require_leader()
        except Exception as exc:
            return self._failed_submit_inference_response(f"Not leader: {exc}")

        model_id = request.model_id.strip()
        if not model_id:
            return self._failed_submit_inference_response("model_id must be non-empty")

        try:
            features = matrix_from_proto(request.features)
        except Exception as exc:
            return self._failed_submit_inference_response(f"Invalid features matrix: {exc}")

        alive_workers = self.registry.alive_workers()
        if not alive_workers:
            return self._failed_submit_inference_response("No alive workers available")

        try:
            result = self.inference_coordinator.run_inference(
                model_id=model_id,
                features=features,
            )

            return rf_pb2.SubmitInferenceResponse(
                success=True,
                error="",
                task_type=result.task_type,
                predicted_labels=result.predicted_labels or [],
                predicted_values=result.predicted_values or [],
            )

        except Exception as exc:
            return self._failed_submit_inference_response(str(exc))
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

        dataset_scenario = self._extract_dataset_scenario(request)
        if dataset_scenario not in SUPPORTED_DATASET_SCENARIOS:
            return (
                f"dataset_scenario must be one of "
                f"{sorted(SUPPORTED_DATASET_SCENARIOS)}"
            )

        leakage_columns = self._extract_leakage_columns(request)
        if leakage_columns is not None:
            target_column = request.target_column.strip()
            if target_column in leakage_columns:
                return "leakage_columns cannot contain the target_column"

        return None

    def _failed_submit_inference_response(self, message: str) -> rf_pb2.SubmitInferenceResponse:
        return rf_pb2.SubmitInferenceResponse(
            success=False,
            error=message,
            task_type="",
            predicted_labels=[],
            predicted_values=[],
        )

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

        dataset_scenario = self._extract_dataset_scenario(request)
        leakage_columns = self._extract_leakage_columns(request)

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
            dataset_scenario=dataset_scenario,
            leakage_columns=leakage_columns,
        )

    def _extract_dataset_scenario(self, request) -> str:
        """
        Estrae lo scenario dataset dalla request gRPC, se presente.

        Compatibilità:
        - se il proto è stato esteso con dataset_scenario, usa quello;
        - altrimenti usa la variabile d'ambiente DATASET_SCENARIO;
        - se nessuno dei due è presente, usa baseline_original.
        """
        raw_value = getattr(request, "dataset_scenario", "")

        if raw_value is None or not str(raw_value).strip():
            raw_value = os.getenv("DATASET_SCENARIO", "baseline_original")

        scenario = str(raw_value).strip().lower()

        if not scenario:
            return "baseline_original"

        return scenario

    def _extract_leakage_columns(self, request) -> list[str] | None:
        """
        Estrae le colonne sospette di leakage.

        Compatibilità:
        - se il proto ha repeated string leakage_columns, usa quello;
        - se il proto non lo ha, usa la variabile d'ambiente LEAKAGE_COLUMNS;
        - LEAKAGE_COLUMNS può essere una stringa comma-separated:
          LEAKAGE_COLUMNS=diabetes_stage,altra_colonna
        """
        raw_columns = getattr(request, "leakage_columns", None)

        columns: list[str] = []

        if raw_columns is not None:
            if isinstance(raw_columns, str):
                columns = raw_columns.split(",")
            else:
                columns = list(raw_columns)

        if not columns:
            env_value = os.getenv("LEAKAGE_COLUMNS", "").strip()
            if env_value:
                columns = env_value.split(",")

        normalized_columns: list[str] = []
        for column in columns:
            normalized = str(column).strip()
            if normalized and normalized not in normalized_columns:
                normalized_columns.append(normalized)

        return normalized_columns or None

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
    # --------------------------------------------------------
    # Startup recovery
    # --------------------------------------------------------

    def _start_recovery_on_startup_if_enabled(self) -> None:
        enabled = os.getenv("RECOVER_INCOMPLETE_JOBS_ON_STARTUP", "true").lower()

        if enabled not in {"1", "true", "yes", "y"}:
            print("[MasterCoordinator] Startup recovery disabled")
            return

        thread = threading.Thread(
            target=self._recover_incomplete_jobs_on_startup,
            daemon=True,
        )
        thread.start()

    def _recover_incomplete_jobs_on_startup(self) -> None:
        startup_delay_seconds = float(
            os.getenv("RECOVERY_STARTUP_DELAY_SECONDS", "10")
        )
        poll_seconds = float(
            os.getenv("RECOVERY_LEADER_POLL_SECONDS", "2")
        )

        print(
            "[MasterCoordinator] Startup recovery watcher scheduled "
            f"in {startup_delay_seconds} seconds"
        )

        time.sleep(startup_delay_seconds)

        recovered_terms: set[int] = set()

        while True:
            try:
                if not self.consensus.is_leader():
                    time.sleep(poll_seconds)
                    continue

                term = self.consensus.current_term()

                if term in recovered_terms:
                    time.sleep(poll_seconds)
                    continue

                recovered_terms.add(term)

                print(
                    "[MasterCoordinator] This node is leader, "
                    f"starting recovery check for term {term}"
                )

                self._wait_for_at_least_one_worker()

                recoverable_jobs = self._list_recoverable_jobs()

                if not recoverable_jobs:
                    print("[MasterCoordinator] No recoverable jobs found")
                    time.sleep(poll_seconds)
                    continue

                print(
                    "[MasterCoordinator] Recoverable jobs found: "
                    f"{[job.job_id for job in recoverable_jobs]}"
                )

                for job_record in recoverable_jobs:
                    try:
                        print(
                            "[MasterCoordinator] Resuming job "
                            f"{job_record.job_id}"
                        )

                        self.training_job_service.resume_training_job(
                            job_id=job_record.job_id,
                            async_run=False,
                        )

                    except Exception as exc:
                        print(
                            "[MasterCoordinator] Failed to resume job "
                            f"{job_record.job_id}: {exc}"
                        )

            except Exception as exc:
                print(
                    "[MasterCoordinator] Recovery watcher error: "
                    f"{exc}"
                )

            time.sleep(poll_seconds)
    def _wait_for_at_least_one_worker(self) -> None:
        timeout_seconds = float(
            os.getenv("RECOVERY_WAIT_WORKERS_TIMEOUT_SECONDS", "60")
        )
        poll_interval_seconds = float(
            os.getenv("RECOVERY_WAIT_WORKERS_POLL_SECONDS", "2")
        )

        deadline = time.time() + timeout_seconds

        while time.time() < deadline:
            try:
                workers = self.registry.alive_workers()

                if workers:
                    print(
                        "[MasterCoordinator] Workers available for recovery: "
                        f"{[worker.worker_id for worker in workers]}",
                        flush=True,
                    )
                    return

            except Exception as exc:
                print(
                    "[MasterCoordinator] Worker availability check failed: "
                    f"{exc}",
                    flush=True,
                )

            time.sleep(poll_interval_seconds)

        print(
            "[MasterCoordinator] No workers available before recovery timeout. "
            "Recovery will still be attempted.",
            flush=True,
        )
    def _list_recoverable_jobs(self):
        jobs = self._list_all_jobs_from_repository()

        recover_failed_jobs = os.getenv(
            "RECOVER_FAILED_JOBS_ON_STARTUP",
            "false",
        ).lower() in {"1", "true", "yes", "y"}

        recoverable = []

        for job in jobs:
            status = self._status_value(job.status)

            if status in {"PENDING", "RUNNING", "VALIDATING"}:
                recoverable.append(job)
                continue

            if recover_failed_jobs and status == "FAILED":
                recoverable.append(job)

        return recoverable

    def _list_all_jobs_from_repository(self):
        if hasattr(self.job_repository, "list_jobs"):
            return self.job_repository.list_jobs()

        if hasattr(self.job_repository, "list_all"):
            return self.job_repository.list_all()

        if hasattr(self.job_repository, "all"):
            return self.job_repository.all()

        if hasattr(self.job_repository, "load_all"):
            return self.job_repository.load_all()

        raise AttributeError(
            "JobRepository must expose one of: "
            "list_jobs, list_all, all, load_all"
        )

    def _status_value(self, status) -> str:
        if hasattr(status, "value"):
            return str(status.value)
        return str(status)

# ============================================================
# Server bootstrap
# ============================================================

def serve(
    host: str = "0.0.0.0",
    port: int = 50051,
    artifact_root: str = "/mnt/efs/gp_artifacts",
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
        #modificato questo codice da os.getenv("ARTIFACT_ROOT", "/mnt/efs/gp_artifacts") per salvare direttamente nel path condiviso di efs
        artifact_root=os.getenv("ARTIFACT_ROOT","/mnt/efs/gp_artifacts")
    )