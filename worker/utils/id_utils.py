import os
import socket
import uuid


def generate_worker_id() -> str:
    """
  Genera un identificatore univoco per il worker.

Priorità:
Se la variabile d'ambiente WORKER_ID è impostata → utilizzala
Altrimenti → genera un ID basato sul nome host + suffisso casuale

    """

    # consente override esplciito
    explicit = os.getenv("WORKER_ID")
    if explicit:
        return explicit

    # Fallback: hostname + random UUID suffix
    hostname = socket.gethostname()
    random_suffix = uuid.uuid4().hex[:8]

    return f"worker-{hostname}-{random_suffix}"