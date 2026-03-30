import json
import os
import joblib
from typing import Any

from artifact_store import ArtifactStore


class FilesystemArtifactStore(ArtifactStore):

    def __init__(self, root_dir: str):
        self.root_dir = root_dir

    def _full_path(self, path: str) -> str:
        return os.path.join(self.root_dir, path)

    def save_tree_artifact(self, path: str, model: Any) -> str:
        full_path = self._full_path(path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        joblib.dump(model, full_path)
        return full_path

    def load_tree_artifact(self, path: str) -> Any:
        full_path = self._full_path(path)
        return joblib.load(full_path)

    def tree_artifact_exists(self, path: str) -> bool:
        return os.path.exists(self._full_path(path))

    def save_json(self, path: str, data: dict) -> None:
        full_path = self._full_path(path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, "w") as f:
            json.dump(data, f)

    def load_json(self, path: str) -> dict:
        full_path = self._full_path(path)
        with open(full_path, "r") as f:
            return json.load(f)

    def save_bytes(self, path: str, data: bytes) -> None:
        full_path = self._full_path(path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, "wb") as f:
            f.write(data)

    def load_bytes(self, path: str) -> bytes:
        full_path = self._full_path(path)
        with open(full_path, "rb") as f:
            return f.read()

    def exists(self, path: str) -> bool:
        return os.path.exists(self._full_path(path))

    def delete(self, path: str) -> None:
        full_path = self._full_path(path)
        if os.path.exists(full_path):
            os.remove(full_path)