import threading
import time
from typing import Dict, List, Set


class WorkerState:
    """
    Runtime state del worker (NON persistente).

    Responsabilità:
    - tracciare task attivi (source of truth)
    - tracciare stato task (debug / monitoring)
    - fornire dati coerenti per heartbeat verso master
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()

        # 🔴 SOURCE OF TRUTH
        self._active_tasks: Set[str] = set()

        # opzionale ma utile per debug/monitoring
        self._task_status: Dict[str, str] = {}

        # ultimo progresso osservato per task
        self._task_last_progress_ts: Dict[str, float] = {}


    # --------------------------------------------------
    # Lifecycle hooks (usati da WorkerService)
    # --------------------------------------------------

    def on_task_start(self, task_id: str) -> None:
        now = time.time()

        with self._lock:
            self._active_tasks.add(task_id)
            self._task_status[task_id] = "RUNNING"
            self._task_last_progress_ts[task_id] = now

    def on_task_progress(self, task_id: str) -> None:
        now = time.time()
        with self._lock:
            if task_id in self._active_tasks:
                self._task_last_progress_ts[task_id] = now

    def on_task_success(self, task_id: str) -> None:
        now = time.time()
        with self._lock:
            # non rimuoviamo ancora: lo farà on_task_end
            self._task_status[task_id] = "SUCCESS"
            self._task_last_progress_ts[task_id] = now


    def on_task_failure(self, task_id: str, error: str) -> None:
        now = time.time()
        with self._lock:
            self._task_status[task_id] = f"FAILED: {error}"
            self._task_last_progress_ts[task_id] = now

    def on_task_end(self, task_id: str) -> None:
        with self._lock:
            self._active_tasks.discard(task_id)

    # --------------------------------------------------
    # Query methods (usati da heartbeat)
    # --------------------------------------------------

    def running_tasks_count(self) -> int:
        with self._lock:
            return len(self._active_tasks)

    def active_task_ids(self) -> List[str]:
        with self._lock:
            return list(self._active_tasks)

    def active_tasks_snapshot(self) -> List[dict]:
        """
        Snapshot usato per heartbeat verso master.

        Output:
        [
            {
                "task_id": "...",
                "last_progress_ts": 1234567890.0
            }
        ]
        """

        with self._lock:
            snapshot = []

            for task_id in self._active_tasks:
                snapshot.append(
                    {
                        "task_id": task_id,
                        "last_progress_ts": self._task_last_progress_ts.get(
                            task_id,
                            0.0,
                        ),
                    }
                )

            return snapshot

    # --------------------------------------------------
    # Debug / monitoring
    # --------------------------------------------------

    def get_task_status(self, task_id: str) -> str:
        with self._lock:
            return self._task_status.get(task_id, "UNKNOWN")