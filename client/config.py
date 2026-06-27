from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


CLIENT_DIR = Path(__file__).resolve().parent
DEFAULT_ENV_FILE = CLIENT_DIR / ".env.client"

DEFAULT_MASTER_HOST = "127.0.0.1"
DEFAULT_MASTER_PORT = "50051"
DEFAULT_GRPC_MAX_MESSAGE_LENGTH_MB = 256
DEFAULT_CLIENT_GRPC_TIMEOUT_SECONDS = 120.0


@dataclass(frozen=True)
class ClientConfig:
    master_addresses: list[str]
    grpc_max_message_length_mb: int
    grpc_timeout_seconds: float

    @property
    def grpc_max_message_length_bytes(self) -> int:
        return self.grpc_max_message_length_mb * 1024 * 1024


def load_env_file(path: Path = DEFAULT_ENV_FILE) -> None:
    if not path.exists():
        return

    with path.open("r", encoding="utf-8") as file:
        for raw_line in file:
            line = raw_line.strip()

            if not line or line.startswith("#"):
                continue

            if "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")

            if key:
                os.environ.setdefault(key, value)


def load_client_config(env_file: Path | None = None) -> ClientConfig:
    load_env_file(env_file or DEFAULT_ENV_FILE)

    return ClientConfig(
        master_addresses=load_master_addresses(),
        grpc_max_message_length_mb=_read_positive_int_env(
            "GRPC_MAX_MESSAGE_LENGTH_MB",
            DEFAULT_GRPC_MAX_MESSAGE_LENGTH_MB,
        ),
        grpc_timeout_seconds=_read_positive_float_env(
            "CLIENT_GRPC_TIMEOUT_SECONDS",
            DEFAULT_CLIENT_GRPC_TIMEOUT_SECONDS,
        ),
    )


def load_master_addresses() -> list[str]:
    raw_seeds = os.getenv("MASTER_SEEDS", "").strip()

    if raw_seeds:
        addresses = [
            _normalize_master_address(item)
            for item in raw_seeds.split(",")
            if item.strip()
        ]

        addresses = _deduplicate(addresses)
        if addresses:
            return addresses

    master_host = os.getenv("MASTER_HOST", DEFAULT_MASTER_HOST).strip()
    master_port = os.getenv("MASTER_PORT", DEFAULT_MASTER_PORT).strip()

    return [_normalize_master_address(f"{master_host}:{master_port}")]


def build_grpc_options(config: ClientConfig) -> list[tuple[str, int]]:
    max_message_length = config.grpc_max_message_length_bytes

    return [
        ("grpc.max_send_message_length", max_message_length),
        ("grpc.max_receive_message_length", max_message_length),
    ]


def _normalize_master_address(value: str) -> str:
    address = value.strip()

    if not address:
        raise ValueError("Master address cannot be empty")

    if address.startswith("http://"):
        address = address.removeprefix("http://")

    if address.startswith("https://"):
        address = address.removeprefix("https://")

    if ":" not in address:
        address = f"{address}:{DEFAULT_MASTER_PORT}"

    host, port = address.rsplit(":", 1)
    host = host.strip()
    port = port.strip()

    if not host:
        raise ValueError(f"Invalid master address '{value}': missing host")

    if not port.isdigit():
        raise ValueError(f"Invalid master address '{value}': port must be numeric")

    port_number = int(port)
    if port_number <= 0 or port_number > 65535:
        raise ValueError(f"Invalid master address '{value}': port out of range")

    return f"{host}:{port_number}"


def _deduplicate(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()

    for value in values:
        if value in seen:
            continue

        result.append(value)
        seen.add(value)

    return result


def _read_positive_int_env(name: str, default: int) -> int:
    raw_value = os.getenv(name, str(default)).strip()

    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc

    if value <= 0:
        raise ValueError(f"{name} must be > 0")

    return value


def _read_positive_float_env(name: str, default: float) -> float:
    raw_value = os.getenv(name, str(default)).strip()

    try:
        value = float(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number") from exc

    if value <= 0:
        raise ValueError(f"{name} must be > 0")

    return value