from __future__ import annotations

import json
import os
import socket
import threading
import time
import uuid
from concurrent import futures
from dataclasses import asdict, dataclass
from typing import Optional

import grpc
import joblib
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

import rf_v2_pb2 as rf_pb2
import rf_v2_pb2_grpc as rf_pb2_grpc

from common.contracts import TreeArtifactMetadata, WorkerProgressSnapshot
from common.ids import generate_tree_id, tree_seed
import paths


# ============================================================
# Artifact Store Abstraction
# ============================================================

class ArtifactStore:
    def exists(self, key: str) -> bool:
        raise NotImplementedError

    def save_json(self, key: str, data: dict) -> None:
        raise NotImplementedError

    def load_json(self, key: str) -> dict:
        raise NotImplementedError

    def save_joblib(self, key: str, obj) -> None:
        raise NotImplementedError

    def load_joblib(self, key: str):
        raise NotImplementedError


class FilesystemArtifactStore(ArtifactStore):
    def __init__(self, root: str):
        self.root = root
        os.makedirs(self.root, exist_ok=True)

    def _full_path(self, key: str) -> str:
        return os.path.join(self.root, key)

    def exists(self, key: str) -> bool:
        return os.path.exists(self._full_path(key))

    def save_json(self, key: str, data: dict) -> None:
        path = self._full_path(key)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp_path = path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp_path, path)

    def load_json(self, key: str) -> dict:
        path = self._full_path(key)
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def save_joblib(self, key: str, obj) -> None:
        path = self._full_path(key)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp_path = path + ".tmp"
        joblib.dump(obj, tmp_path)
        os.replace(tmp_path, path)

    def load_joblib(self, key: str):
        path = self._full_path(key)
        return joblib.load(path)


# ============================================================
# Utility
# ============================================================

def now_ts() -> float:
    return time.time()


def generate_worker_id() -> str:
    explicit = os.getenv("WORKER_ID")
    if explicit:
        return explicit
    return f"worker-{socket.gethostname()}-{uuid.uuid4().hex[:8]}"


def detect_advertise_host(master_host: str, master_port: int) -> str:
    explicit = os.getenv("WORKER_ADVERTISE_HOST")
    if explicit:
        return explicit

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect((master_host, master_port))
        return sock.getsockname()[0]
    except Exception:
        try:
            return socket.gethostbyname(socket.gethostname())
        except Exception:
            return "127.0.0.1"
    finally:
        sock.close()


def matrix_from_proto(msg: rf_pb2.DenseMatrix) -> np.ndarray:
    arr = np.asarray(msg.values, dtype=float)
    if msg.n_rows * msg.n_cols != arr.size:
        raise ValueError("DenseMatrix shape mismatch")
    return arr.reshape(msg.n_rows, msg.n_cols)


def parse_dataset_url(dataset_url: str) -> str:
    if dataset_url.startswith("file://"):
        return dataset_url.replace("file://", "", 1)
    return dataset_url


def read_csv_dataset(dataset_url: str) -> pd.DataFrame:
    path_or_url = parse_dataset_url(dataset_url)
    return pd.read_csv(path_or_url)


def parse_max_features(value: str):
    value = (value or "").strip().lower()
    if value in {"", "none"}:
        return None
    if value in {"sqrt", "log2"}:
        return value
    try:
        return float(value)
    except ValueError:
        return value


# ============================================================
# Worker config/state
# ============================================================

@dataclass
class WorkerConfig:
    worker_id: str
    bind_host: str
    port: int
    master_host: str
    master_port: int
    artifact_root: str = "/shared/artifacts"
    advertise_host: Optional[str] = None


class WorkerState:
    def __init__(self) -> None:
        self.running_tasks = 0
        self._lock = threading.Lock()

    def inc(self) -> None:
        with self._lock:
            self.running_tasks += 1

    def dec(self) -> None:
        with self._lock:
            self.running_tasks = max(0, self.running_tasks - 1)

    def get(self) -> int:
        with self._lock:
            return self.running_tasks


# ============================================================
# Worker service
# ============================================================

class WorkerService(rf_pb2_grpc.WorkerServiceServicer):
    """
    Versione ponte del worker:
    - supporta task_id / attempt_id / experiment_id
    - usa tree_id deterministici
    - salva gli alberi in modo atomico
    - aggiorna progress snapshot su storage condiviso
    - è retry-safe: se un tree artifact esiste già, lo salta
    """

    def __init__(self, config: WorkerConfig, state: WorkerState):
        self.config = config
        self.state = state
        self.store: ArtifactStore = FilesystemArtifactStore(config.artifact_root)

    def TrainShard(self, request, context):
        self.state.inc()
        completed_tree_ids: list[str] = []
        failed_tree_ids: list[str] = []
        trained_artifacts: list[rf_pb2.TrainedTreeArtifact] = []
        running_tree_ids: list[str] = []

        try:
            model_type = request.model_type.strip().lower()
            if model_type not in {"classification", "regression"}:
                raise ValueError("model_type must be 'classification' or 'regression'")

            df = read_csv_dataset(request.dataset_url)
            if request.target_column not in df.columns:
                raise ValueError(f"Target column '{request.target_column}' not found")

            y = df[request.target_column].to_numpy()
            X = df.drop(columns=[request.target_column]).to_numpy(dtype=float)

            max_depth = None if request.max_depth <= 0 else request.max_depth
            min_samples_split = max(2, request.min_samples_split)
            min_samples_leaf = max(1, request.min_samples_leaf)
            max_features = parse_max_features(request.max_features)

            # initial snapshot
            self._write_progress_snapshot(
                job_id=request.job_id,
                experiment_id=request.experiment_id,
                task_id=request.task_id,
                completed_tree_ids=[],
                running_tree_ids=[],
                failed_tree_ids=[],
            )

            for offset in range(request.tree_count):
                tree_index = request.start_tree_index + offset
                tree_id = generate_tree_id(request.experiment_id, tree_index)
                seed = tree_seed(request.seed_base, tree_index)

                artifact_key = paths.tree_artifact_path(request.job_id, request.experiment_id, tree_index)
                metadata_key = artifact_key + ".meta.json"

                running_tree_ids = [tree_id]

                self._write_progress_snapshot(
                    job_id=request.job_id,
                    experiment_id=request.experiment_id,
                    task_id=request.task_id,
                    completed_tree_ids=completed_tree_ids,
                    running_tree_ids=running_tree_ids,
                    failed_tree_ids=failed_tree_ids,
                )

                # retry-safe: skip if exists
                if self.store.exists(artifact_key):
                    if self.store.exists(metadata_key):
                        meta_dict = self.store.load_json(metadata_key)
                        meta = TreeArtifactMetadata(**meta_dict)
                    else:
                        meta = TreeArtifactMetadata(
                            tree_id=tree_id,
                            job_id=request.job_id,
                            experiment_id=request.experiment_id,
                            task_id=request.task_id,
                            tree_index=tree_index,
                            worker_id=self.config.worker_id,
                            seed=seed,
                            artifact_uri=artifact_key,
                            status="COMPLETED",
                            training_time_seconds=0.0,
                        )
                        self.store.save_json(metadata_key, asdict(meta))

                    trained_artifacts.append(self._to_proto_tree_artifact(meta))

                    if tree_id not in completed_tree_ids:
                        completed_tree_ids.append(tree_id)

                    running_tree_ids = []

                    self._write_progress_snapshot(
                        job_id=request.job_id,
                        experiment_id=request.experiment_id,
                        task_id=request.task_id,
                        completed_tree_ids=completed_tree_ids,
                        running_tree_ids=running_tree_ids,
                        failed_tree_ids=failed_tree_ids,
                    )

                    continue

                t0 = now_ts()

                X_fit = X
                y_fit = y
                if request.bootstrap:
                    rng = np.random.default_rng(seed)
                    idx = rng.integers(0, X.shape[0], size=X.shape[0])
                    X_fit = X[idx]
                    y_fit = y[idx]

                if model_type == "classification":
                    model = DecisionTreeClassifier(
                        max_depth=max_depth,
                        min_samples_split=min_samples_split,
                        min_samples_leaf=min_samples_leaf,
                        max_features=max_features,
                        random_state=seed,
                    )
                else:
                    model = DecisionTreeRegressor(
                        max_depth=max_depth,
                        min_samples_split=min_samples_split,
                        min_samples_leaf=min_samples_leaf,
                        max_features=max_features,
                        random_state=seed,
                    )

                model.fit(X_fit, y_fit)

                self.store.save_joblib(artifact_key, model)

                training_time = now_ts() - t0

                meta = TreeArtifactMetadata(
                    tree_id=tree_id,
                    job_id=request.job_id,
                    experiment_id=request.experiment_id,
                    task_id=request.task_id,
                    tree_index=tree_index,
                    worker_id=self.config.worker_id,
                    seed=seed,
                    artifact_uri=artifact_key,
                    status="COMPLETED",
                    training_time_seconds=training_time,
                )

                self.store.save_json(metadata_key, asdict(meta))

                trained_artifacts.append(self._to_proto_tree_artifact(meta))
                completed_tree_ids.append(tree_id)
                running_tree_ids = []

                self._write_progress_snapshot(
                    job_id=request.job_id,
                    experiment_id=request.experiment_id,
                    task_id=request.task_id,
                    completed_tree_ids=completed_tree_ids,
                    running_tree_ids=running_tree_ids,
                    failed_tree_ids=failed_tree_ids,
                )

            return rf_pb2.TrainShardResponse(
                worker_id=self.config.worker_id,
                task_id=request.task_id,
                attempt_id=request.attempt_id,
                success=True,
                error="",
                artifacts=trained_artifacts,
            )

        except Exception as exc:
            if running_tree_ids:
                for tree_id in running_tree_ids:
                    if tree_id not in failed_tree_ids:
                        failed_tree_ids.append(tree_id)

            try:
                self._write_progress_snapshot(
                    job_id=request.job_id,
                    experiment_id=request.experiment_id,
                    task_id=request.task_id,
                    completed_tree_ids=completed_tree_ids,
                    running_tree_ids=[],
                    failed_tree_ids=failed_tree_ids,
                )
            except Exception:
                pass

            return rf_pb2.TrainShardResponse(
                worker_id=self.config.worker_id,
                task_id=request.task_id,
                attempt_id=request.attempt_id,
                success=False,
                error=str(exc),
                artifacts=trained_artifacts,
            )
        finally:
            self.state.dec()

    def PredictShard(self, request, context):
        self.state.inc()
        try:
            X = matrix_from_proto(request.features)

            if not request.tree_artifact_uris:
                raise ValueError("No tree artifact URIs provided")

            model_type = request.model_type.strip().lower()

            if model_type == "classification":
                class_labels = list(request.class_labels)
                if not class_labels:
                    raise ValueError("class_labels required for classification")

                class_to_idx = {label: i for i, label in enumerate(class_labels)}
                votes = np.zeros((X.shape[0], len(class_labels)), dtype=float)

                for artifact_key in request.tree_artifact_uris:
                    model = self.store.load_joblib(artifact_key)
                    pred = model.predict(X)
                    for row_idx, label in enumerate(pred):
                        votes[row_idx, class_to_idx[str(label)]] += 1.0

                return rf_pb2.PredictShardResponse(
                    worker_id=self.config.worker_id,
                    success=True,
                    error="",
                    values=votes.ravel().tolist(),
                    n_rows=votes.shape[0],
                    n_cols=votes.shape[1],
                )

            elif model_type == "regression":
                sums = np.zeros((X.shape[0], 1), dtype=float)

                for artifact_key in request.tree_artifact_uris:
                    model = self.store.load_joblib(artifact_key)
                    pred = model.predict(X)
                    sums[:, 0] += pred

                return rf_pb2.PredictShardResponse(
                    worker_id=self.config.worker_id,
                    success=True,
                    error="",
                    values=sums.ravel().tolist(),
                    n_rows=sums.shape[0],
                    n_cols=sums.shape[1],
                )

            else:
                raise ValueError("Unsupported model_type")

        except Exception as exc:
            return rf_pb2.PredictShardResponse(
                worker_id=self.config.worker_id,
                success=False,
                error=str(exc),
                values=[],
                n_rows=0,
                n_cols=0,
            )
        finally:
            self.state.dec()

    def _write_progress_snapshot(
        self,
        job_id: str,
        experiment_id: str,
        task_id: str,
        completed_tree_ids: list[str],
        running_tree_ids: list[str],
        failed_tree_ids: list[str],
    ) -> None:
        snapshot = WorkerProgressSnapshot(
            worker_id=self.config.worker_id,
            task_id=task_id,
            experiment_id=experiment_id,
            completed_tree_ids=list(completed_tree_ids),
            running_tree_ids=list(running_tree_ids),
            failed_tree_ids=list(failed_tree_ids),
            last_update_ts=now_ts(),
        )

        snapshot_key = paths.worker_snapshot_path(
            job_id,
            experiment_id,
            self.config.worker_id,
            task_id,
        )

        self.store.save_json(snapshot_key, asdict(snapshot))

    def _to_proto_tree_artifact(self, meta: TreeArtifactMetadata) -> rf_pb2.TrainedTreeArtifact:
        return rf_pb2.TrainedTreeArtifact(
            tree_id=meta.tree_id,
            tree_index=meta.tree_index,
            artifact_uri=meta.artifact_uri,
            worker_id=meta.worker_id,
            seed=meta.seed,
            training_time_seconds=meta.training_time_seconds,
        )


# ============================================================
# Worker node lifecycle
# ============================================================

class WorkerNode:
    def __init__(self, config: WorkerConfig):
        self.config = config
        self.state = WorkerState()

    @property
    def master_address(self) -> str:
        return f"{self.config.master_host}:{self.config.master_port}"

    def register_to_master(self) -> None:
        advertise_host = self.config.advertise_host or detect_advertise_host(
            self.config.master_host,
            self.config.master_port,
        )

        with grpc.insecure_channel(self.master_address) as channel:
            stub = rf_pb2_grpc.CoordinatorServiceStub(channel)
            stub.RegisterWorker(
                rf_pb2.RegisterWorkerRequest(
                    worker_id=self.config.worker_id,
                    host=advertise_host,
                    port=self.config.port,
                ),
                timeout=10,
            )

    def start_heartbeat_loop(self, interval_seconds: float = 5.0) -> None:
        def _loop():
            while True:
                try:
                    with grpc.insecure_channel(self.master_address) as channel:
                        stub = rf_pb2_grpc.CoordinatorServiceStub(channel)
                        stub.Heartbeat(
                            rf_pb2.HeartbeatRequest(
                                worker_id=self.config.worker_id,
                                running_tasks=self.state.get(),
                            ),
                            timeout=5,
                        )
                except grpc.RpcError:
                    pass

                time.sleep(interval_seconds)

        threading.Thread(target=_loop, daemon=True).start()

    def serve(self) -> None:
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=16))
        rf_pb2_grpc.add_WorkerServiceServicer_to_server(
            WorkerService(self.config, self.state),
            server,
        )
        server.add_insecure_port(f"{self.config.bind_host}:{self.config.port}")
        server.start()

        self.register_to_master()
        self.start_heartbeat_loop()

        advertised = self.config.advertise_host or self.config.bind_host
        print(f"[WORKER {self.config.worker_id}] listening on {advertised}:{self.config.port}")
        server.wait_for_termination()


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    master_host = os.getenv("MASTER_HOST", "master")
    master_port = int(os.getenv("MASTER_PORT", "50051"))

    config = WorkerConfig(
        worker_id=generate_worker_id(),
        bind_host=os.getenv("WORKER_BIND_HOST", "0.0.0.0"),
        port=int(os.getenv("WORKER_PORT", "50061")),
        master_host=master_host,
        master_port=master_port,
        artifact_root=os.getenv("ARTIFACT_ROOT", "/shared/artifacts"),
        advertise_host=os.getenv("WORKER_ADVERTISE_HOST"),
    )

    WorkerNode(config).serve()
