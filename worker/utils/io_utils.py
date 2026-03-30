import json
import os
import tempfile
from pathlib import Path


def atomic_json_write(path: Path, payload: dict) -> None:
    """
    Writes a JSON file atomically to avoid corruption in case of failures.

    Strategy:
    - Write to a temporary file
    - Flush + fsync to disk
    - Atomically replace the target file
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        mode="w",
        delete=False,
        dir=str(path.parent),
        encoding="utf-8"
    ) as tmp:
        json.dump(payload, tmp, indent=2)
        tmp.flush()
        os.fsync(tmp.fileno())
        temp_path = Path(tmp.name)

    # Atomic replace
    temp_path.replace(path)