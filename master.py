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

import rf_v2_pb2 as rf_pb2
import rf_v2_pb2_grpc as rf_pb2_grpc

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
from master.fault_tolerance import (
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
    Versione ponte del master:
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

        self.store = SharedArtifactStore(str(self.artifact_root))
        self.layout = StorageLayout(str(self.artifact_root))
        self.job_repository = JobRepository(self.store)
        self.model_repository = ModelRepository(self.store)
        self.task_ledger = TaskLedger(self.store)

        # Consensus/leader guard: placeholder leader-only service.
        self.consensus = InMemoryLeaderConsensusService(is_leader=True)
        self.leadership_guard = LeadershipGuard(self.consensus)

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

        if request.n_trees <= 0:
            return rf_pb2.SubmitTrainingResponse(
                job_id="",
                status=rf_pb2.FAILED,
                message="n_trees must be > 0",
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

        # Per ora trattiamo una sola configurazione “resolved”.
        # In seguito HyperparameterSpace + ExperimentPlanner genereranno più esperimenti.
        experiment_id = generate_experiment_id(job_id, 0)

        training_request = TrainingRequest(
            job_id=job_id,
            dataset_uri=request.dataset_url,
            target_column=request.target_column,
            task_type=request.model_type.strip().lower(),
            hyperparameter_space=HyperparameterSpace(
                n_estimators_candidates=[request.n_trees],
                max_depth_candidates=[request.max_depth if request.max_depth > 0 else None],
                max_features_candidates=[request.max_features],
                min_samples_split_candidates=[request.min_samples_split],
                min_samples_leaf_candidates=[1],
                criterion_candidates=["gini"] if request.model_type.strip().lower() == "classification" else ["squared_error"],
                bootstrap=request.bootstrap,
                global_random_seed=request.random_seed,
            ),
            n_estimators_total=request.n_trees,
            validation_ratio=0.0,  # fase successiva
            test_ratio=0.0,        # fase successiva
            global_random_seed=request.random_seed,
            bootstrap=request.bootstrap,
        )

        job_record = TrainingJobRecord(
            job_id=job_id,
            status="PENDING",
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

            record.status = "RUNNING"
            record.message = "Training in progress"
            record.updated_at = now_ts()
            self.job_repository.save(record)

            req = record.training_request
            model_type = req.task_type
            if model_type not in {"classification", "regression"}:
                raise ValueError("task_type/model_type must be 'classification' or 'regression'")

            # TODO fase 3: questo blocco sarà rimpiazzato da DataPreparationService
            df = read_csv_dataset(req.dataset_uri)
            if req.target_column not in df.columns:
                raise ValueError(f"Target column '{req.target_column}' not found in dataset")

            class_labels: list[str] = []
            if model_type == "classification":
                class_labels = sorted(df[req.target_column].astype(str).unique().tolist())

            forest_config = ForestConfiguration(
                experiment_id=experiment_id,
                task_type=model_type,
                n_estimators=req.n_estimators_total,
                max_depth=req.hyperparameter_space.max_depth_candidates[0],
                max_features=req.hyperparameter_space.max_features_candidates[0],
                min_samples_split=req.hyperparameter_space.min_samples_split_candidates[0],
                min_samples_leaf=req.hyperparameter_space.min_samples_leaf_candidates[0],
                criterion=req.hyperparameter_space.criterion_candidates[0],
                bootstrap=req.bootstrap,
                global_random_seed=req.global_random_seed,
            )

            # Persistiamo un record di esperimento base.
            experiment = ExperimentRecord(
                experiment_id=experiment_id,
                forest_config=forest_config,
                status="RUNNING",
                assigned_workers=[],
                expected_tree_count=forest_config.n_estimators,
                completed_tree_count=0,
                validation_metrics=None,
            )
            self.job_repository.save_experiment(job_id, experiment)

            alive_workers = self.registry.alive_workers()
            if not alive_workers:
                raise RuntimeError("No alive workers available during scheduling")

            shard_specs = self._split_trees(forest_config.n_estimators, alive_workers)
            collected_artifacts: list[TreeArtifactMetadata] = []

            with ThreadPoolExecutor(max_workers=len(shard_specs)) as pool:
                future_map = {}

                for worker, start_idx, tree_count in shard_specs:
                    task_id = generate_task_id(job_id, experiment_id, start_idx, tree_count)
                    attempt_id = 1
                    seed_base = req.global_random_seed

                    shard = TrainingShard(
                        task_id=task_id,
                        job_id=job_id,
                        experiment_id=experiment_id,
                        worker_id=worker.worker_id,
                        tree_start_index=start_idx,
                        tree_count=tree_count,
                        forest_config=forest_config,
                        train_features_uri=req.dataset_uri,  # TODO fase 3: train split URI reale
                        train_labels_uri=req.target_column,  # TODO fase 3: labels URI reale
                        artifact_output_dir=self.layout.experiment_dir(job_id, experiment_id),
                        seed_base=seed_base,
                    )

                    task_record = TaskRecord(
                        task_id=task_id,
                        job_id=job_id,
                        experiment_id=experiment_id,
                        worker_id=worker.worker_id,
                        attempt_id=attempt_id,
                        status="PENDING",
                        lease_expiration_ts=None,
                        tree_ids=[
                            generate_tree_id(experiment_id, i)
                            for i in range(start_idx, start_idx + tree_count)
                        ],
                        completed_tree_ids=[],
                        updated_at=now_ts(),
                    )
                    self.task_ledger.save(task_record)

                    grpc_request = self._build_train_shard_request(model_id, req, shard, attempt_id)
                    fut = pool.submit(self._call_train_shard, worker, grpc_request)
                    future_map[fut] = (worker, shard, attempt_id, req)

                for fut in as_completed(future_map):
                    worker, shard, attempt_id, req = future_map[fut]

                    response = fut.result()
                    if not response.success:
                        retry_worker = self.registry.get_retry_candidate(exclude_worker_id=worker.worker_id)
                        if retry_worker is None:
                            raise RuntimeError(
                                f"Training shard failed on worker {worker.worker_id} and no retry worker available: {response.error}"
                            )

                        retry_attempt = attempt_id + 1
                        retry_request = self._build_train_shard_request(model_id, req, shard, retry_attempt)
                        response = self._call_train_shard(retry_worker, retry_request)

                    if not response.success:
                        self.task_ledger.mark_failed(shard.task_id, response.error)
                        raise RuntimeError(
                            f"Training shard failed permanently for task {shard.task_id}: {response.error}"
                        )

                    shard_artifacts = []
                    for a in response.artifacts:
                        meta = TreeArtifactMetadata(
                            tree_id=a.tree_id,
                            job_id=job_id,
                            experiment_id=experiment_id,
                            task_id=shard.task_id,
                            tree_index=a.tree_index,
                            worker_id=a.worker_id,
                            seed=a.seed,
                            artifact_uri=a.artifact_uri,
                            status="COMPLETED",
                            training_time_seconds=a.training_time_seconds,
                        )
                        shard_artifacts.append(meta)

                    collected_artifacts.extend(shard_artifacts)
                    self.task_ledger.mark_completed(
                        task_id=shard.task_id,
                        completed_tree_ids=[a.tree_id for a in shard_artifacts],
                    )

                    completed_trees = self.task_ledger.count_completed_trees(job_id)
                    record.message = f"Completed {completed_trees}/{forest_config.n_estimators} trees"
                    record.updated_at = now_ts()
                    self.job_repository.save(record)

            collected_artifacts = sorted(collected_artifacts, key=lambda x: x.tree_index)
            if len(collected_artifacts) != forest_config.n_estimators:
                raise RuntimeError(
                    f"Expected {forest_config.n_estimators} trees, got {len(collected_artifacts)}"
                )

            # Validation placeholder: fase 6
            validation_metrics = ValidationMetrics(
                experiment_id=experiment_id,
                accuracy=0.0,
                classification_report={},
                confusion_matrix=[],
                feature_importances=[],
                evaluated_at=now_ts(),
            )

            experiment.status = "COMPLETED"
            experiment.completed_tree_count = len(collected_artifacts)
            experiment.validation_metrics = validation_metrics
            self.job_repository.save_experiment(job_id, experiment)

            manifest = ModelManifest(
                model_id=model_id,
                job_id=job_id,
                experiment_id=experiment_id,
                model_type=model_type,
                forest_config=forest_config,
                class_labels=class_labels,
                feature_names=[c for c in df.columns if c != req.target_column],
                target_column=req.target_column,
                train_split_uri=req.dataset_uri,
                validation_split_uri="",
                test_split_uri="",
                tree_artifacts=collected_artifacts,
                validation_metrics=validation_metrics,
                test_metrics=None,
                created_at=now_ts(),
                status="READY",
            )

            self.model_repository.save(manifest)

            record.status = "COMPLETED"
            record.message = "Training completed successfully"
            record.selected_experiment_id = experiment_id
            record.updated_at = now_ts()
            self.job_repository.save(record)

        except Exception as exc:
            record = self.job_repository.load(job_id)
            if record is not None:
                record.status = "FAILED"
                record.message = str(exc)
                record.updated_at = now_ts()
                self.job_repository.save(record)

    def _split_trees(self, n_trees: int, workers: list[WorkerInfo]) -> list[tuple[WorkerInfo, int, int]]:
        chunks: list[tuple[WorkerInfo, int, int]] = []
        base = n_trees // len(workers)
        rem = n_trees % len(workers)

        start = 0
        for i, worker in enumerate(workers):
            count = base + (1 if i < rem else 0)
            if count == 0:
                continue
            chunks.append((worker, start, count))
            start += count
        return chunks

    def _build_train_shard_request(
        self,
        model_id: str,
        req: TrainingRequest,
        shard: TrainingShard,
        attempt_id: int,
    ) -> rf_pb2.TrainShardRequest:
        """
        Mapping master-side TrainingShard -> protobuf request.
        Adatta i nomi qui se il proto definitivo usa campi diversi.
        """
        fc = shard.forest_config
        return rf_pb2.TrainShardRequest(
            task_id=shard.task_id,
            attempt_id=attempt_id,
            job_id=shard.job_id,
            experiment_id=shard.experiment_id,
            model_id=model_id,
            worker_id=shard.worker_id,
            dataset_url=req.dataset_uri,
            target_column=req.target_column,
            model_type=fc.task_type,
            start_tree_index=shard.tree_start_index,
            tree_count=shard.tree_count,
            max_depth=fc.max_depth or 0,
            min_samples_split=fc.min_samples_split,
            min_samples_leaf=fc.min_samples_leaf,
            max_features=str(fc.max_features),
            bootstrap=fc.bootstrap,
            seed_base=shard.seed_base,
            artifact_output_dir=shard.artifact_output_dir,
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
            self.leadership_guard.require_leader()
        except Exception as exc:
            return rf_pb2.SubmitInferenceResponse(
                status=rf_pb2.FAILED,
                message=f"Not leader: {exc}",
            )

        manifest = self.model_repository.load(request.model_id)
        if manifest is None:
            return rf_pb2.SubmitInferenceResponse(
                status=rf_pb2.FAILED,
                message="Model not found",
            )

        alive_workers = self.registry.alive_workers()
        if not alive_workers:
            return rf_pb2.SubmitInferenceResponse(
                status=rf_pb2.FAILED,
                message="No alive workers available",
            )

        try:
            X = matrix_from_proto(request.features)
            if X.size == 0:
                raise ValueError("Empty inference batch")

            tree_uris = [t.artifact_uri for t in manifest.tree_artifacts]
            shards = self._split_tree_uris(tree_uris, len(alive_workers))
            features_msg = matrix_to_proto(X)

            responses = []
            with ThreadPoolExecutor(max_workers=len(shards)) as pool:
                future_map = {}
                used_workers = alive_workers[: len(shards)]

                for worker, uri_shard in zip(used_workers, shards):
                    shard_req = rf_pb2.PredictShardRequest(
                        model_id=manifest.model_id,
                        model_type=manifest.model_type,
                        features=features_msg,
                        tree_artifact_uris=uri_shard,
                        class_labels=manifest.class_labels,
                    )
                    fut = pool.submit(self._call_predict_shard, worker, shard_req)
                    future_map[fut] = (worker, shard_req)

                for fut in as_completed(future_map):
                    worker, shard_req = future_map[fut]
                    response = fut.result()

                    if not response.success:
                        retry_worker = self.registry.get_retry_candidate(worker.worker_id)
                        if retry_worker is not None:
                            response = self._call_predict_shard(retry_worker, shard_req)

                    if not response.success:
                        raise RuntimeError(
                            f"Inference shard failed on worker {worker.worker_id}: {response.error}"
                        )

                    responses.append(response)

            if manifest.model_type == "classification":
                votes = np.zeros((X.shape[0], len(manifest.class_labels)), dtype=float)
                for resp in responses:
                    votes += np.asarray(resp.values, dtype=float).reshape(resp.n_rows, resp.n_cols)

                pred_idx = np.argmax(votes, axis=1)
                predicted_labels = [manifest.class_labels[i] for i in pred_idx]

                return rf_pb2.SubmitInferenceResponse(
                    status=rf_pb2.COMPLETED,
                    predicted_labels=predicted_labels,
                    message="Inference completed",
                )

            else:
                sums = np.zeros((X.shape[0], 1), dtype=float)
                for resp in responses:
                    sums += np.asarray(resp.values, dtype=float).reshape(resp.n_rows, resp.n_cols)

                preds = (sums[:, 0] / len(manifest.tree_artifacts)).tolist()

                return rf_pb2.SubmitInferenceResponse(
                    status=rf_pb2.COMPLETED,
                    predicted_values=preds,
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