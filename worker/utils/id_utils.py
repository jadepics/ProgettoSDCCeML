import os
import socket
import uuid


def generate_worker_id() -> str:
    """
    Generates a unique worker identifier.

    Priority:
    1. If WORKER_ID env variable is set → use it (useful for deterministic deployment)
    2. Otherwise → generate an ID based on hostname + random suffix

    Returns:
        str: unique worker identifier
    """

    # 1. Allow explicit override (useful in Docker / Kubernetes)
    explicit = os.getenv("WORKER_ID")
    if explicit:
        return explicit

    # 2. Fallback: hostname + random UUID suffix
    hostname = socket.gethostname()
    random_suffix = uuid.uuid4().hex[:8]

    return f"worker-{hostname}-{random_suffix}"