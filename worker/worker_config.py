from dataclasses import dataclass
from typing import Optional


@dataclass
class WorkerConfig:
    worker_id: str
    bind_host: str
    port: int
    master_host: str
    master_port: int
    artifact_root: str
    advertise_host: Optional[str] = None