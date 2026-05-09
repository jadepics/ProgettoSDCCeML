import json
import os
import tempfile
from pathlib import Path

import pandas as pd


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

import numpy as np
from urllib.parse import urlparse

class DataLoader:
    def __init__(self, artifact_store):
        self.store = artifact_store

    def load_numpy(self, uri: str) -> np.ndarray:
        path = self._resolve_uri(uri)

        df = pd.read_parquet(path)

        return df.to_numpy()


    def _resolve_uri(self, uri) -> Path:
        if uri.startswith("file://"):
            parsed = urlparse(uri)
            return Path(parsed.path)

        return Path(uri)