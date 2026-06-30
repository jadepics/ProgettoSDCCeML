from __future__ import annotations

import os

# configurazioni grpc
def _message_limit_bytes() -> int:
    raw_value = os.getenv("GRPC_MAX_MESSAGE_LENGTH_MB", "256").strip()

    try:
        value_mb = int(raw_value)
    except ValueError as exc:
        raise ValueError(
            "GRPC_MAX_MESSAGE_LENGTH_MB must be an integer number of MiB"
        ) from exc

    if value_mb <= 0:
        raise ValueError("GRPC_MAX_MESSAGE_LENGTH_MB must be > 0")

    return value_mb * 1024 * 1024


GRPC_MAX_MESSAGE_LENGTH = _message_limit_bytes()

GRPC_OPTIONS = [
    ("grpc.max_send_message_length", GRPC_MAX_MESSAGE_LENGTH),
    ("grpc.max_receive_message_length", GRPC_MAX_MESSAGE_LENGTH),
]