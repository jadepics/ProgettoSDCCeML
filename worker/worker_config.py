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

            worker_id=os.getenv(
                "WORKER_ID",
                generate_worker_id()
            ),

            # corretto mantenere "0.0.0.0"
            bind_host=os.getenv(
                "WORKER_BIND_HOST",
                "0.0.0.0"
            ),

            #porta di ascolto, unica per istanza di EC2 ma ripetibile per diverse istanze
            port=int(
                os.getenv("WORKER_PORT", "50061")
            ),

            #host privato del master
            master_host=os.getenv(
                "MASTER_HOST",
                "172.31.37.47"
            ),

            #porta del master che manteniamo sempre 50051
            master_port=int(
                os.getenv("MASTER_PORT", "50051")
            ),

            #root per la scrittura dei files all'interno del sistema
            artifact_root=os.getenv(
                "ARTIFACT_ROOT",
                "/mnt/efs/gp_artifacts"
            ),

            #Ip privato del worker. Per adesso lo hardcodiamo, poi sarà da modificare in modo che se creiamo diverse istanze ec2, questo venga preso in automatico
            advertise_host=os.getenv(
                "WORKER_ADVERTISE_HOST",
                "172.31.39.5"
            ),

            max_workers=int(
                os.getenv("WORKER_MAX_WORKERS", "16")
            ),
        )