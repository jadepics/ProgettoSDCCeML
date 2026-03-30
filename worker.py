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
    path_or_url =   parse_dataset_url(dataset_url)
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

    def __init__(self, config, state, shard_trainer, shard_predictor):
        self.config = config
        self.state = state
        self.trainer = shard_trainer
        self.predictor = shard_predictor

    def TrainShard(self, request, context):
        self.state.inc()
        try:
            metas = self.trainer.train(request)

            artifacts = [
                self._to_proto_tree_artifact(m)
                for m in metas
            ]

            return rf_pb2.TrainShardResponse(
                worker_id=self.config.worker_id,
                task_id=request.task_id,
                attempt_id=request.attempt_id,
                success=True,
                artifacts=artifacts,
            )

        except Exception as e:
            return rf_pb2.TrainShardResponse(
                worker_id=self.config.worker_id,
                task_id=request.task_id,
                attempt_id=request.attempt_id,
                success=False,
                error=str(e),
                artifacts=[],
            )
        finally:
            self.state.dec()

    def PredictShard(self, request, context):
        self.state.inc()
        try:
            result = self.predictor.predict(request)

            return rf_pb2.PredictShardResponse(
                worker_id=self.config.worker_id,
                success=True,
                error="",
                values=result.ravel().tolist(),
                n_rows=result.shape[0],
                n_cols=result.shape[1],
            )

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
from storage.filesystem_store import FilesystemArtifactStore
from storage import paths
from worker.training.shard_trainer import ShardTrainer
from worker.prediction.shard_predictor import ShardPredictor
from worker.training.tree_artifact_writer import TreeArtifactWriter
from worker.progress.worker_progress_store import WorkerProgressStore

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
        store = FilesystemArtifactStore(self.config.artifact_root)
        progress_store = WorkerProgressStore(store, paths)

        writer = TreeArtifactWriter(
            store=store,
            paths=paths,
            worker_id=self.config.worker_id,
        )

        trainer = ShardTrainer(
            config=self.config,
            state=self.state,
            store=store,
            paths=paths,
            artifact_writer=writer,
            progress_store=progress_store,
        )

        predictor = ShardPredictor(store=store)

        rf_pb2_grpc.add_WorkerServiceServicer_to_server(
            WorkerService(self.config, self.state, trainer, predictor),
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
    master_host = os.getenv("MASTER_HOST", "masterPackage")
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
