import threading


class WorkerState:
    def __init__(self) -> None:
        self.running_tasks = 0
        self._lock = threading.Lock()

    def inc(self) -> None:
        with self._lock:
            self.running_tasks += 1

    def dec(self) -> None:
        with self._lock:
            self.running_tasks = max(0, self.running_tasks - 1)

    def get(self) -> int:
        with self._lock:
            return self.running_tasks