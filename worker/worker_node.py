from __future__ import annotations

import grpc
from concurrent import futures
import time

import rf_v2_pb2_grpc as rf_pb2_grpc

from worker.worker_service import WorkerService
from worker.worker_config import WorkerConfig
from worker.worker_state import WorkerState

from worker.training.shard_trainer import ShardTrainer
from worker.prediction.shard_predictor import ShardPredictor

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
        # Infrastructure
        # --------------------------------------------------
        self.server = grpc.server(
            futures.ThreadPoolExecutor(max_workers=config.max_workers)
        )

        # --------------------------------------------------
        # State
        # --------------------------------------------------
        self.state = WorkerState(worker_id=config.worker_id)

        # --------------------------------------------------
        # Storage
        # --------------------------------------------------
        self.artifact_store = FilesystemArtifactStore(
            base_dir=config.artifact_base_dir
        )

        # --------------------------------------------------
        # Training / Prediction components
        # --------------------------------------------------
        self.shard_trainer = ShardTrainer(
            config=config,
            artifact_store=self.artifact_store
        )

        self.shard_predictor = ShardPredictor(
            config=config,
            artifact_store=self.artifact_store
        )

        # --------------------------------------------------
        # Service (RPC layer)
        # --------------------------------------------------
        self.service = WorkerService(
            config=config,
            state=self.state,
            shard_trainer=self.shard_trainer,
            shard_predictor=self.shard_predictor
        )

        # Register gRPC service
        rf_pb2_grpc.add_WorkerServiceServicer_to_server(
            self.service,
            self.server
        )

        self.server.add_insecure_port(f"[::]:{config.port}")

    # --------------------------------------------------
    # Lifecycle
    # --------------------------------------------------

    def start(self):
        self.server.start()
        print(f"[WorkerNode] Started on port {self.config.port}")

        try:
            while True:
                time.sleep(86400)
        except KeyboardInterrupt:
            self.stop()

    def stop(self):
        print("[WorkerNode] Shutting down...")
        self.server.stop(0)