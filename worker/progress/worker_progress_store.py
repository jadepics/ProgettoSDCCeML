from pathlib import Path
from typing import Dict, Any, Optional
from worker.utils.io_utils import atomic_json_write
from worker.utils.time_utils import current_time_seconds
import json


class WorkerProgressStore:
    """
    Persistent store for tracking worker progress and enabling fault tolerance.
    """

    def __init__(self, storage_path: Path):
        self.storage_path = Path(storage_path)
        self.state: Dict[str, Any] = {
            "tasks": {}
        }

        self._load()

    # ------------------------
    # Internal persistence
    # ------------------------

    def _load(self) -> None:
        if self.storage_path.exists():
            with open(self.storage_path, "r", encoding="utf-8") as f:
                self.state = json.load(f)

    def _persist(self) -> None:
        atomic_json_write(self.storage_path, self.state)

    # ------------------------
    # Task management
    # ------------------------

    def start_task(self, task_id: str, metadata: Dict[str, Any]) -> None:
        self.state["tasks"][task_id] = {
            "status": "RUNNING",
            "metadata": metadata,
            "progress": 0.0,
            "shards_completed": [],
            "last_update": current_time_seconds()
        }
        self._persist()

    def update_progress(self, task_id: str, shard_id: int, progress: float) -> None:
        task = self._get_task(task_id)

        if shard_id not in task["shards_completed"]:
            task["shards_completed"].append(shard_id)

        task["progress"] = progress
        task["last_update"] = current_time_seconds()

        self._persist()

    def complete_task(self, task_id: str) -> None:
        task = self._get_task(task_id)
        task["status"] = "COMPLETED"
        task["progress"] = 1.0
        task["last_update"] = current_time_seconds()
        self._persist()

    def fail_task(self, task_id: str, error: str) -> None:
        task = self._get_task(task_id)
        task["status"] = "FAILED"
        task["error"] = error
        task["last_update"] = current_time_seconds()
        self._persist()

    def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        return self.state["tasks"].get(task_id)

    def _get_task(self, task_id: str) -> Dict[str, Any]:
        if task_id not in self.state["tasks"]:
            raise KeyError(f"Task {task_id} not found")
        return self.state["tasks"][task_id]

    # ------------------------
    # Recovery helpers
    # ------------------------

    def get_running_tasks(self):
        return {
            tid: t for tid, t in self.state["tasks"].items()
            if t["status"] == "RUNNING"
        }

    def get_completed_shards(self, task_id: str):
        task = self._get_task(task_id)
        return set(task.get("shards_completed", []))