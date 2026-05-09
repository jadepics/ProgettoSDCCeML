import json
import os
import tempfile
from pathlib import Path
from urllib.parse import urlparse, unquote


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

    def _resolve_uri(self, uri: str) -> Path:
        parsed = urlparse(uri)

        # Caso 1: path locale normale, es:
        # C:\Users\micci\...
        # oppure Dataset\file.parquet
        if parsed.scheme == "":
            return Path(uri)

        # Caso 2: URI file://...
        if parsed.scheme == "file":
            path = unquote(parsed.path)

            # Caso Windows:
            # file:///C:/Users/... viene parsato come /C:/Users/...
            # dobbiamo togliere lo slash iniziale
            if os.name == "nt" and len(path) >= 3 and path[0] == "/" and path[2] == ":":
                path = path[1:]

            # Caso file://server/share/file.parquet
            if parsed.netloc:
                path = f"//{parsed.netloc}{path}"

            return Path(path)

        raise ValueError(f"Unsupported URI scheme '{parsed.scheme}' for URI '{uri}'")