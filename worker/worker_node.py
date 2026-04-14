from __future__ import annotations

import grpc
from concurrent import futures
import time

import rf_v2_pb2_grpc as rf_pb2_grpc
from worker.training.bootstrap_sampler import BootstrapSampler
from worker.training.decision_tree_factory import DecisionTreeFactory
from worker.training.tree_artifact_writer import TreeArtifactWriter
from worker.utils.io_utils import DataLoader
from worker.progress.worker_progress_store import WorkerProgressStore

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
        
        self.tree_factory = DecisionTreeFactory()

        self.artifact_writer = TreeArtifactWriter(
            artifact_store=self.artifact_store,
            worker_id=config.worker_id
        )

        self.shard_trainer = ShardTrainer(
            bootstrap_sampler=self.bootstrap_sampler,
            tree_factory=self.tree_factory,
            artifact_writer=self.artifact_writer,
            progress_store=self.progress_store,
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

        # --------------------------------------------------
        # Service
        # --------------------------------------------------
        self.service = WorkerService(
            config=config,
            state=self.state,
            shard_trainer=self.shard_trainer,
            shard_predictor=self.shard_predictor,
            progress_store=self.progress_store,
            artifact_store=self.artifact_store,
            data_loader=self.data_loader,
        )

        # --------------------------------------------------
        # gRPC registration
        # --------------------------------------------------
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