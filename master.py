from __future__ import annotations

import json
import threading
import time
import uuid
from concurrent import futures
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional

import grpc
import numpy as np
import pandas as pd

import rf_pb2
import rf_pb2_grpc


HEARTBEAT_TIMEOUT_SECONDS = 15.0


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
        return dataset_url.replace("file:///shared/datasets/diabetes_dataset.csv", "", 1)
    return dataset_url


def read_csv_dataset(dataset_url: str) -> pd.DataFrame:
    # Prima versione: path locale, file://..., oppure URL CSV leggibile da pandas
    path_or_url = parse_dataset_url(dataset_url)
    return pd.read_csv(path_or_url)


@dataclass
class WorkerInfo:
    worker_id: str
    host: str
    port: int
    last_heartbeat: float = field(default_factory=time.time)
    running_tasks: int = 0

    @property
    def address(self) -> str:
        return f"{self.host}:{self.port}"


@dataclass
class TrainingJob:
    job_id: str
    model_id: str
    status: int
    total_trees: int
    completed_trees: int
    message: str
    workers: list[str]


@dataclass
class TreeArtifact:
    tree_index: int
    tree_path: str
    worker_id: str


@dataclass
class ModelManifest:
    model_id: str
    model_type: str
    class_labels: list[str]
    tree_paths: list[str]


class WorkerRegistry:
    def __init__(self) -> None:
        self._workers: dict[str, WorkerInfo] = {}
        self._lock = threading.Lock()

    def register(self, worker_id: str, host: str, port: int) -> None:
        with self._lock:
            self._workers[worker_id] = WorkerInfo(
                worker_id=worker_id,
                host=host,
                port=port,
                last_heartbeat=time.time(),
                running_tasks=0,
            )

    def heartbeat(self, worker_id: str, running_tasks: int) -> bool:
        with self._lock:
            worker = self._workers.get(worker_id)
            if worker is None:
                return False
            worker.last_heartbeat = time.time()
            worker.running_tasks = running_tasks
            return True

    def alive_workers(self) -> list[WorkerInfo]:
        now = time.time()
        with self._lock:
            return [
                w
                for w in self._workers.values()
                if now - w.last_heartbeat <= HEARTBEAT_TIMEOUT_SECONDS
            ]

    def get_retry_candidate(self, exclude_worker_id: str) -> Optional[WorkerInfo]:
        candidates = [w for w in self.alive_workers() if w.worker_id != exclude_worker_id]
        if not candidates:
            return None
        return sorted(candidates, key=lambda w: w.running_tasks)[0]


class MasterCoordinator(rf_pb2_grpc.CoordinatorServiceServicer):
    def __init__(self, artifact_root: str = "/shared/artifacts") -> None:
        self.registry = WorkerRegistry()
        self.artifact_root = Path(artifact_root)
        self.artifact_root.mkdir(parents=True, exist_ok=True)

        self.jobs: dict[str, TrainingJob] = {}
        self.models: dict[str, ModelManifest] = {}

        self._lock = threading.Lock()

    # ---------------------------
    # RPC: worker lifecycle
    # ---------------------------

    def RegisterWorker(self, request, context):
        self.registry.register(request.worker_id, request.host, request.port)
        return rf_pb2.RegisterWorkerResponse(
            accepted=True,
            message=f"Worker {request.worker_id} registered",
        )

    def Heartbeat(self, request, context):
        ok = self.registry.heartbeat(request.worker_id, request.running_tasks)
        return rf_pb2.HeartbeatResponse(ok=ok)

    # ---------------------------
    # RPC: training
    # ---------------------------

    def SubmitTraining(self, request, context):
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

        job_id = str(uuid.uuid4())
        model_id = str(uuid.uuid4())

        with self._lock:
            self.jobs[job_id] = TrainingJob(
                job_id=job_id,
                model_id=model_id,
                status=rf_pb2.PENDING,
                total_trees=request.n_trees,
                completed_trees=0,
                message="Training queued",
                workers=[w.worker_id for w in alive_workers],
            )

        request_data = {
            "dataset_url": request.dataset_url,
            "target_column": request.target_column,
            "model_type": request.model_type,
            "n_trees": request.n_trees,
            "max_depth": request.max_depth,
            "min_samples_split": request.min_samples_split,
            "max_features": request.max_features,
            "bootstrap": request.bootstrap,
            "random_seed": request.random_seed,
        }

        threading.Thread(
            target=self._run_training_job,
            args=(job_id, model_id, request_data),
            daemon=True,
        ).start()

        return rf_pb2.SubmitTrainingResponse(
            job_id=job_id,
            status=rf_pb2.PENDING,
            message="Training started",
        )

    def GetTrainingStatus(self, request, context):
        with self._lock:
            job = self.jobs.get(request.job_id)

        if job is None:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details("Job not found")
            return rf_pb2.GetTrainingStatusResponse()

        return rf_pb2.GetTrainingStatusResponse(
            job_id=job.job_id,
            model_id=job.model_id,
            status=job.status,
            total_trees=job.total_trees,
            completed_trees=job.completed_trees,
            message=job.message,
            workers=job.workers,
        )

    def _run_training_job(self, job_id: str, model_id: str, cfg: dict) -> None:
        try:
            with self._lock:
                self.jobs[job_id].status = rf_pb2.RUNNING
                self.jobs[job_id].message = "Training in progress"

            model_type = cfg["model_type"].strip().lower()
            if model_type not in {"classification", "regression"}:
                raise ValueError("model_type must be 'classification' or 'regression'")

            class_labels: list[str] = []
            if model_type == "classification":
                df = read_csv_dataset(cfg["dataset_url"])
                if cfg["target_column"] not in df.columns:
                    raise ValueError(f"Target column '{cfg['target_column']}' not found")
                class_labels = sorted(df[cfg["target_column"]].astype(str).unique().tolist())

            alive_workers = self.registry.alive_workers()
            if not alive_workers:
                raise RuntimeError("No alive workers available during scheduling")

            chunks = self._split_trees(cfg["n_trees"], alive_workers)
            artifact_dir = str(self.artifact_root.resolve())

            collected: list[TreeArtifact] = []

            with ThreadPoolExecutor(max_workers=len(chunks)) as pool:
                future_map = {}
                for worker, start_idx, n_trees in chunks:
                    shard_req = rf_pb2.TrainShardRequest(
                        job_id=job_id,
                        model_id=model_id,
                        dataset_url=cfg["dataset_url"],
                        target_column=cfg["target_column"],
                        model_type=model_type,
                        start_tree_index=start_idx,
                        n_trees=n_trees,
                        max_depth=cfg["max_depth"],
                        min_samples_split=cfg["min_samples_split"],
                        max_features=cfg["max_features"],
                        bootstrap=cfg["bootstrap"],
                        random_seed=cfg["random_seed"],
                        artifact_dir=artifact_dir,
                    )
                    fut = pool.submit(self._call_train_shard, worker, shard_req)
                    future_map[fut] = (worker, shard_req)

                for fut in as_completed(future_map):
                    worker, shard_req = future_map[fut]
                    response = fut.result()

                    if not response.success:
                        retry_worker = self.registry.get_retry_candidate(worker.worker_id)
                        if retry_worker is not None:
                            response = self._call_train_shard(retry_worker, shard_req)

                    if not response.success:
                        raise RuntimeError(
                            f"Training shard failed on worker {worker.worker_id}: {response.error}"
                        )

                    shard_artifacts = [
                        TreeArtifact(
                            tree_index=a.tree_index,
                            tree_path=a.tree_path,
                            worker_id=a.worker_id,
                        )
                        for a in response.artifacts
                    ]
                    collected.extend(shard_artifacts)

                    with self._lock:
                        self.jobs[job_id].completed_trees += len(shard_artifacts)
                        self.jobs[job_id].message = (
                            f"Completed {self.jobs[job_id].completed_trees}/"
                            f"{self.jobs[job_id].total_trees} trees"
                        )

            collected = sorted(collected, key=lambda x: x.tree_index)
            tree_paths = [a.tree_path for a in collected]

            if len(tree_paths) != cfg["n_trees"]:
                raise RuntimeError(
                    f"Expected {cfg['n_trees']} trees, got {len(tree_paths)}"
                )

            manifest = ModelManifest(
                model_id=model_id,
                model_type=model_type,
                class_labels=class_labels,
                tree_paths=tree_paths,
            )

            self.models[model_id] = manifest
            self._save_manifest(manifest)

            with self._lock:
                self.jobs[job_id].status = rf_pb2.COMPLETED
                self.jobs[job_id].message = "Training completed successfully"

        except Exception as exc:
            with self._lock:
                self.jobs[job_id].status = rf_pb2.FAILED
                self.jobs[job_id].message = str(exc)

    def _split_trees(
        self, n_trees: int, workers: list[WorkerInfo]
    ) -> list[tuple[WorkerInfo, int, int]]:
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

    def _call_train_shard(
        self, worker: WorkerInfo, request: rf_pb2.TrainShardRequest
    ) -> rf_pb2.TrainShardResponse:
        with grpc.insecure_channel(worker.address) as channel:
            stub = rf_pb2_grpc.WorkerServiceStub(channel)
            return stub.TrainShard(request, timeout=600)

    def _save_manifest(self, manifest: ModelManifest) -> None:
        out_dir = self.artifact_root / manifest.model_id
        out_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = out_dir / "manifest.json"

        payload = {
            "model_id": manifest.model_id,
            "model_type": manifest.model_type,
            "class_labels": manifest.class_labels,
            "tree_paths": manifest.tree_paths,
        }

        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    # ---------------------------
    # RPC: inference
    # ---------------------------

    def SubmitInference(self, request, context):
        manifest = self.models.get(request.model_id)
        if manifest is None:
            manifest = self._load_manifest(request.model_id)

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

            shards = self._split_tree_paths(manifest.tree_paths, len(alive_workers))
            features_msg = matrix_to_proto(X)

            responses = []
            with ThreadPoolExecutor(max_workers=len(shards)) as pool:
                future_map = {}
                used_workers = alive_workers[: len(shards)]

                for worker, tree_paths in zip(used_workers, shards):
                    shard_req = rf_pb2.PredictShardRequest(
                        model_id=manifest.model_id,
                        model_type=manifest.model_type,
                        features=features_msg,
                        tree_paths=tree_paths,
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

                preds = (sums[:, 0] / len(manifest.tree_paths)).tolist()

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
        self, worker: WorkerInfo, request: rf_pb2.PredictShardRequest
    ) -> rf_pb2.PredictShardResponse:
        with grpc.insecure_channel(worker.address) as channel:
            stub = rf_pb2_grpc.WorkerServiceStub(channel)
            return stub.PredictShard(request, timeout=600)

    def _split_tree_paths(self, tree_paths: list[str], n_parts: int) -> list[list[str]]:
        if n_parts <= 0:
            raise ValueError("n_parts must be > 0")

        shards = [[] for _ in range(min(n_parts, len(tree_paths)))]
        for i, path in enumerate(tree_paths):
            shards[i % len(shards)].append(path)
        return [s for s in shards if s]

    def _load_manifest(self, model_id: str) -> Optional[ModelManifest]:
        manifest_path = self.artifact_root / model_id / "manifest.json"
        if not manifest_path.exists():
            return None

        with open(manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        manifest = ModelManifest(
            model_id=data["model_id"],
            model_type=data["model_type"],
            class_labels=data.get("class_labels", []),
            tree_paths=data["tree_paths"],
        )
        self.models[model_id] = manifest
        return manifest


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