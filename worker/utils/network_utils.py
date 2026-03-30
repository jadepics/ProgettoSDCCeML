import os
import socket


def detect_advertise_host(master_host: str, master_port: int) -> str:
    """
    Detects the IP address that the worker should advertise to the master.

    Priority:
    1. Environment variable WORKER_ADVERTISE_HOST
    2. OS routing decision via UDP socket
    3. Hostname resolution
    4. Fallback to localhost
    """

    # 1. Explicit override via environment variable
    explicit = os.getenv("WORKER_ADVERTISE_HOST")
    if explicit:
        return explicit

    # 2. Try to infer IP by opening a UDP connection
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # No real connection is made; this just helps determine the outbound IP
        sock.connect((master_host, master_port))
        return sock.getsockname()[0]

    except Exception:
        pass

    finally:
        sock.close()

    # 3. Fallback to hostname resolution
    try:
        return socket.gethostbyname(socket.gethostname())
    except Exception:
        pass

    # 4. Final fallback
    return "127.0.0.1"


def get_hostname() -> str:
    """
    Returns the current machine hostname.
    """
    return socket.gethostname()


def get_local_ip() -> str:
    """
    Attempts to retrieve the local IP address of the machine.
    Useful for debugging and diagnostics.
    """
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.connect(("8.8.8.8", 80))
        ip = sock.getsockname()[0]
        sock.close()
        return ip
    except Exception:
        return "127.0.0.1"