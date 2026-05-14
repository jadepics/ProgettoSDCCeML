from dataclasses import dataclass
from typing import Optional
import os
from urllib.request import Request, urlopen

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
        worker_id = os.getenv("WORKER_ID") or generate_worker_id()

        bind_host = os.getenv("WORKER_BIND_HOST", "0.0.0.0")
        port = int(os.getenv("WORKER_PORT", "50061"))

        master_host = os.getenv("MASTER_HOST")
        if not master_host:
            raise ValueError("MASTER_HOST is required")

        master_port = int(os.getenv("MASTER_PORT", "50051"))

        artifact_root = os.getenv("ARTIFACT_ROOT", "/mnt/efs/gp_artifacts")

        advertise_host = os.getenv("WORKER_ADVERTISE_HOST")
        if not advertise_host:
            advertise_host = WorkerConfig._resolve_private_ip()

        max_workers = int(os.getenv("WORKER_MAX_WORKERS", "16"))

        return WorkerConfig(
            worker_id=worker_id,
            bind_host=bind_host,
            port=port,
            master_host=master_host,
            master_port=master_port,
            artifact_root=artifact_root,
            advertise_host=advertise_host,
            max_workers=max_workers,
        )

    @staticmethod
    def _resolve_private_ip() -> str:
        """
        Prova a recuperare l'IP privato dell'istanza EC2 tramite IMDS.
        Utile se non vuoi scrivere WORKER_ADVERTISE_HOST a mano.
        """

        # IMDSv2 token
        token_req = Request(
            "http://169.254.169.254/latest/api/token",
            method="PUT",
            headers={"X-aws-ec2-metadata-token-ttl-seconds": "21600"},
        )

        try:
            with urlopen(token_req, timeout=2) as resp:
                token = resp.read().decode("utf-8")

            meta_req = Request(
                "http://169.254.169.254/latest/meta-data/local-ipv4",
                headers={"X-aws-ec2-metadata-token": token},
            )

            with urlopen(meta_req, timeout=2) as resp:
                return resp.read().decode("utf-8")

        except Exception:
            # fallback solo per sviluppo locale/non-EC2
            return "127.0.0.1"