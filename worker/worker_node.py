from __future__ import annotations

import grpc
from concurrent import futures
import time

import rf_v2_pb2_grpc as rf_pb2_grpc
from worker.training.bootstrap_sampler import BootstrapSampler
from worker.training.tree_artifact_writer import TreeArtifactWriter
from worker.utils.io_utils import DataLoader
from worker.progress.worker_progress_store import WorkerProgressStore

from worker.worker_service import WorkerService
from worker.worker_config import WorkerConfig
from worker.worker_state import WorkerState

from worker.training.shard_trainer import ShardTrainer
from worker.training.decision_tree_factory import DecisionTreeFactory
from worker.prediction.shard_predictor import ShardPredictor

from worker.master_client.master_client import MasterClient
from worker.runtime.heartbeat_loop import HeartbeatLoop

from worker.storage.filesystem_store import FilesystemArtifactStore


class WorkerNode:
    """
    Entry point del worker.

    Responsabilità:
    - inizializzare configurazione
    - costruire tutte le dipendenze
    - avviare server gRPC
    """

    def __init__(self, config: WorkerConfig):
        self.config = config

        # --------------------------------------------------
        # gRPC Server
        # --------------------------------------------------
        self.server = grpc.server(
            futures.ThreadPoolExecutor(max_workers=config.max_workers)
        )

        # --------------------------------------------------
        # State
        # --------------------------------------------------
        self.state = WorkerState()

        # --------------------------------------------------
        # Storage
        # --------------------------------------------------
        self.artifact_store = FilesystemArtifactStore(
            root_dir=config.artifact_root
        )

        # --------------------------------------------------
        # Progress Store (MISSING PRIMA)
        # --------------------------------------------------
        self.progress_store = WorkerProgressStore(
            artifact_store=self.artifact_store,
            worker_id=config.worker_id
        )

        # --------------------------------------------------
        # Training components (MISSING PRIMA)
        # --------------------------------------------------
        self.bootstrap_sampler = BootstrapSampler()

        self.artifact_writer = TreeArtifactWriter(
            artifact_store=self.artifact_store,
            worker_id=config.worker_id
        )

        # --------------------------------------------------
        # Prediction
        # --------------------------------------------------
        self.shard_predictor = ShardPredictor(
            artifact_store=self.artifact_store
        )

        # --------------------------------------------------
        # Data Loader (NUOVO)
        # --------------------------------------------------
        self.data_loader = DataLoader(
            artifact_store=self.artifact_store
        )

        self.tree_factory=DecisionTreeFactory()

        self.shard_trainer = ShardTrainer(
            bootstrap_sampler=self.bootstrap_sampler,
            tree_factory=self.tree_factory,
            artifact_writer = self.artifact_writer,
            progress_store = self.progress_store,
            data_loader = self.data_loader
        )

        # --------------------------------------------------
        # Service
        # --------------------------------------------------
        self.service = WorkerService(
            config=config,
            state=self.state,
            shard_trainer=self.shard_trainer,
            shard_predictor=self.shard_predictor
        )

        # --------------------------------------------------
        # gRPC registration
        # --------------------------------------------------
        rf_pb2_grpc.add_WorkerServiceServicer_to_server(
            self.service,
            self.server
        )

        # --------------------------------------------------
        # Master client
        # --------------------------------------------------
        self.master_client = MasterClient(
            host=config.master_host,
            port=config.master_port
        )

        # --------------------------------------------------
        # Heartbeat loop
        # --------------------------------------------------
        self.heartbeat_loop = HeartbeatLoop(
            master_client=self.master_client,
            worker_state=self.state,
            worker_id=config.worker_id,
            interval_sec=5
        )

        self.server.add_insecure_port(f"[::]:{config.port}")
    # --------------------------------------------------
    # Lifecycle
    # --------------------------------------------------

    def start(self):
        # ----------------------------------------
        # 1. START gRPC SERVER (PRIMA!)
        # ----------------------------------------
        self.server.start()
        print(f"[WorkerNode] gRPC server started on port {self.config.port}")

        # ----------------------------------------
        # 2. REGISTER (ora il worker è raggiungibile)
        # ----------------------------------------
        advertise_host=self._resolve_advertise_host()

        self.master_client.register_worker(
            worker_id=self.config.worker_id,
            host=advertise_host,
            port=self.config.port
        )

        print(f"[WorkerNode] Registered as {advertise_host}:{self.config.port}")

        # ----------------------------------------
        # 3. START HEARTBEAT
        # ----------------------------------------
        self.heartbeat_loop.start()

        # ----------------------------------------
        # 4. LOOP
        # ----------------------------------------
        try:
            while True:
                time.sleep(86400)
        except KeyboardInterrupt:
            self.stop()


    def _resolve_advertise_host(self) -> str:
        if self.config.advertise_host:
            return self.config.advertise_host

        # fallback intelligente per sviluppo locale
        return "localhost"

    def stop(self):
        print("[WorkerNode] Shutting down...")
        self.heartbeat_loop.stop()
        self.server.stop(0)