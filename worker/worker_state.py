import threading
from typing import Dict


class WorkerState:
    """
    Runtime state del worker (NON persistente).

    Responsabilità:
    - tracciare task attivi
    - tracciare stato task (debug / monitoring)
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()

        # task attivi
        self.running_tasks: int = 0

        # opzionale ma molto utile
        self.task_status: Dict[str, str] = {}

    # --------------------------------------------------
    # Lifecycle hooks (usati da WorkerService)
    # --------------------------------------------------

    def on_task_start(self, task_id: str) -> None:
        with self._lock:
            self.running_tasks += 1
            self.task_status[task_id] = "RUNNING"

    def on_task_success(self, task_id: str) -> None:
        with self._lock:
            self.task_status[task_id] = "SUCCESS"

    def on_task_failure(self, task_id: str, error: str) -> None:
        with self._lock:
            self.task_status[task_id] = f"FAILED: {error}"

    def on_task_end(self, task_id: str) -> None:
        with self._lock:
            self.running_tasks = max(0, self.running_tasks - 1)

    # --------------------------------------------------
    # Query methods
    # --------------------------------------------------

    def get_running_tasks(self) -> int:
        with self._lock:
            return self.running_tasks

    def get_task_status(self, task_id: str) -> str:
        with self._lock:
            return self.task_status.get(task_id, "UNKNOWN")