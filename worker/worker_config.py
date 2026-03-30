from dataclasses import dataclass
from typing import Optional
import os

from worker.utils.id_utils import generate_worker_id


@dataclass
class WorkerConfig:
    worker_id: str
    bind_host: str
    port: int
    master_host: str
    master_port: int
    artifact_root: str
    advertise_host: Optional[str] = None
    max_workers: int = 16  # aggiunto (lo stai usando nel server)

    @staticmethod
    def from_env() -> "WorkerConfig":
        return WorkerConfig(
            worker_id=generate_worker_id(),  # ✅ QUI
            bind_host=os.getenv("WORKER_BIND_HOST", "0.0.0.0"),
            port=int(os.getenv("WORKER_PORT", "50061")),
            master_host=os.getenv("MASTER_HOST", "localhost"),
            master_port=int(os.getenv("MASTER_PORT", "50051")),
            artifact_root=os.getenv("ARTIFACT_ROOT", "./artifacts"),
            advertise_host=os.getenv("WORKER_ADVERTISE_HOST"),
            max_workers=int(os.getenv("WORKER_MAX_WORKERS", "16")),
        )