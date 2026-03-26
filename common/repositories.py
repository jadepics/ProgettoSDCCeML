from __future__ import annotations

import json
import threading
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

from .contracts import (
    ModelManifest,
    TaskRecord,
    TrainingJobRecord,
    WorkerProgressSnapshot,
)
from .storage_layout import StorageLayout


class JsonFileStore:
    """Small atomic JSON store suitable for the first project milestone.

    The purpose is not performance; it is deterministic persistence and recoverability.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()

    def write_json(self, path: str | Path, payload: dict[str, Any]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_suffix(path.suffix + ".tmp")
        with self._lock:
            with temp_path.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, sort_keys=True)
            temp_path.replace(path)

    def read_json(self, path: str | Path) -> dict[str, Any]:
        path = Path(path)
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def exists(self, path: str | Path) -> bool:
        return Path(path).exists()


class SharedArtifactStore:
    def __init__(self, root: str | Path, json_store: Optional[JsonFileStore] = None):
        self.layout = StorageLayout(root)
        self.json_store = json_store or JsonFileStore()
        Path(root).mkdir(parents=True, exist_ok=True)

    def write_json(self, path: str | Path, payload: dict[str, Any]) -> None:
        self.json_store.write_json(path, payload)

    def read_json(self, path: str | Path) -> dict[str, Any]:
        return self.json_store.read_json(path)

    def exists(self, path: str | Path) -> bool:
        return self.json_store.exists(path)


class JobRepository:
    def __init__(self, artifact_store: SharedArtifactStore):
        self.artifact_store = artifact_store

    def save(self, record: TrainingJobRecord) -> None:
        path = self.artifact_store.layout.job_record_path(record.job_id)
        self.artifact_store.write_json(path, record.to_dict())

    def load_raw(self, job_id: str) -> dict[str, Any]:
        path = self.artifact_store.layout.job_record_path(job_id)
        return self.artifact_store.read_json(path)


class ModelRepository:
    def __init__(self, artifact_store: SharedArtifactStore):
        self.artifact_store = artifact_store

    def save_manifest(self, manifest: ModelManifest) -> None:
        path = self.artifact_store.layout.model_manifest_path(manifest.model_id)
        self.artifact_store.write_json(path, manifest.to_dict())

    def load_manifest_raw(self, model_id: str) -> dict[str, Any]:
        path = self.artifact_store.layout.model_manifest_path(model_id)
        return self.artifact_store.read_json(path)


class TaskLedger:
    def __init__(self, artifact_store: SharedArtifactStore):
        self.artifact_store = artifact_store
        self._lock = threading.Lock()

    def _read_all(self, job_id: str) -> dict[str, Any]:
        path = self.artifact_store.layout.task_ledger_path(job_id)
        if not self.artifact_store.exists(path):
            return {"tasks": {}}
        return self.artifact_store.read_json(path)

    def _write_all(self, job_id: str, payload: dict[str, Any]) -> None:
        path = self.artifact_store.layout.task_ledger_path(job_id)
        self.artifact_store.write_json(path, payload)

    def upsert(self, record: TaskRecord) -> None:
        with self._lock:
            payload = self._read_all(record.job_id)
            payload.setdefault("tasks", {})[record.task_id] = record.to_dict()
            self._write_all(record.job_id, payload)

    def get_raw(self, job_id: str, task_id: str) -> Optional[dict[str, Any]]:
        payload = self._read_all(job_id)
        return payload.get("tasks", {}).get(task_id)

    def list_raw(self, job_id: str) -> list[dict[str, Any]]:
        payload = self._read_all(job_id)
        return list(payload.get("tasks", {}).values())


class WorkerProgressStore:
    def __init__(self, artifact_store: SharedArtifactStore):
        self.artifact_store = artifact_store

    def save_snapshot(self, job_id: str, snapshot: WorkerProgressSnapshot) -> None:
        path = self.artifact_store.layout.worker_progress_path(job_id, snapshot.worker_id)
        self.artifact_store.write_json(path, snapshot.to_dict())

    def load_snapshot_raw(self, job_id: str, worker_id: str) -> Optional[dict[str, Any]]:
        path = self.artifact_store.layout.worker_progress_path(job_id, worker_id)
        if not self.artifact_store.exists(path):
            return None
        return self.artifact_store.read_json(path)
