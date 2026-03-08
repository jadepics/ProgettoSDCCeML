from __future__ import annotations

import os
import socket
import uuid
import threading
import time
import uuid
from concurrent import futures
from dataclasses import dataclass
from pathlib import Path

import grpc
import joblib
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

import rf_pb2
import rf_pb2_grpc

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
        # Non invia traffico reale, serve solo per capire quale IP locale usare
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


class WorkerService(rf_pb2_grpc.WorkerServiceServicer):
    def __init__(self, config: WorkerConfig, state: WorkerState):
        self.config = config
        self.state = state
        Path(self.config.artifact_root).mkdir(parents=True, exist_ok=True)

    def TrainShard(self, request, context):
        self.state.inc()
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
            max_features = parse_max_features(request.max_features)

            model_dir = Path(request.artifact_dir) / request.model_id
            model_dir.mkdir(parents=True, exist_ok=True)

            artifacts = []

            for offset in range(request.n_trees):
                tree_index = request.start_tree_index + offset
                tree_seed = request.random_seed + tree_index

                X_fit = X
                y_fit = y
                if request.bootstrap:
                    rng = np.random.default_rng(tree_seed)
                    idx = rng.integers(0, X.shape[0], size=X.shape[0])
                    X_fit = X[idx]
                    y_fit = y[idx]

                if model_type == "classification":
                    model = DecisionTreeClassifier(
                        max_depth=max_depth,
                        min_samples_split=min_samples_split,
                        max_features=max_features,
                        random_state=tree_seed,
                    )
                else:
                    model = DecisionTreeRegressor(
                        max_depth=max_depth,
                        min_samples_split=min_samples_split,
                        max_features=max_features,
                        random_state=tree_seed,
                    )

                model.fit(X_fit, y_fit)

                tree_path = model_dir / f"tree_{tree_index}.joblib"
                joblib.dump(model, tree_path)

                artifacts.append(
                    rf_pb2.TrainedTreeArtifact(
                        tree_index=tree_index,
                        tree_path=str(tree_path),
                        worker_id=self.config.worker_id,
                    )
                )

            return rf_pb2.TrainShardResponse(
                worker_id=self.config.worker_id,
                success=True,
                error="",
                artifacts=artifacts,
            )

        except Exception as exc:
            return rf_pb2.TrainShardResponse(
                worker_id=self.config.worker_id,
                success=False,
                error=str(exc),
                artifacts=[],
            )
        finally:
            self.state.dec()

    def PredictShard(self, request, context):
        self.state.inc()
        try:
            X = matrix_from_proto(request.features)
            if X.size == 0:
                raise ValueError("Empty input batch")
            if not request.tree_paths:
                raise ValueError("No tree paths provided")

            model_type = request.model_type.strip().lower()

            if model_type == "classification":
                class_labels = list(request.class_labels)
                if not class_labels:
                    raise ValueError("class_labels required for classification")

                class_to_idx = {label: i for i, label in enumerate(class_labels)}
                votes = np.zeros((X.shape[0], len(class_labels)), dtype=float)

                for tree_path in request.tree_paths:
                    model = joblib.load(tree_path)
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

                for tree_path in request.tree_paths:
                    model = joblib.load(tree_path)
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

        thread = threading.Thread(target=_loop, daemon=True)
        thread.start()

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

        print(
            f"[WORKER {self.config.worker_id}] listening on "
            f"{self.config.host}:{self.config.port}"
        )
        server.wait_for_termination()


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