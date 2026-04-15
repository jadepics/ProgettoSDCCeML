import threading
import time


class HeartbeatLoop:

    def __init__(self, master_client, worker_state, worker_id, interval_sec=5):
        self.master_client = master_client
        self.worker_state = worker_state
        self.worker_id = worker_id
        self.interval_sec = interval_sec
        self._stop = False

    def start(self):
        thread = threading.Thread(target=self._run, daemon=True)
        thread.start()

    def stop(self):
        self._stop = True

    def _run(self):
        while not self._stop:
            try:
                running_tasks = self.worker_state.running_tasks_count()
                active_task_ids = self.worker_state.active_task_ids()

                self.master_client.send_heartbeat(
                    worker_id=self.worker_id,
                    running_tasks=running_tasks,
                    active_task_ids=active_task_ids
                )

            except Exception as e:
                print(f"[HeartbeatLoop] Failed: {e}")

            time.sleep(self.interval_sec)