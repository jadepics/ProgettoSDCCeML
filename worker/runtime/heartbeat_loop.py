import threading
import time

from common.chaos import maybe_fail


class HeartbeatLoop:
    # Loop in background che invia periodicamente al master l'heartbeat del worker,
    # includendo numero e stato dei task attivi, così il master può monitorarne
    # liveness e progresso; se il master non riconosce più il worker, tenta la ri-registrazione.
    def __init__(
        self,
        master_client,
        worker_state,
        worker_id,
        interval_sec=5,
    ):
        self.master_client = master_client
        self.worker_state = worker_state
        self.worker_id = worker_id
        self.interval_sec = interval_sec
        self._stop = False

    def start(self):
        thread = threading.Thread(
            target=self._run,
            daemon=True,
        )
        thread.start()

    def stop(self):
        self._stop = True

    def _run(self):
        while not self._stop:
            try:
                running_tasks = self.worker_state.running_tasks_count()
                active_task_ids = self.worker_state.active_task_ids()
                active_tasks = self.worker_state.active_tasks_snapshot()
                maybe_fail("worker.heartbeat.before_send")

                response = self.master_client.send_heartbeat(
                    worker_id=self.worker_id,
                    running_tasks=running_tasks,
                    active_task_ids=active_task_ids,
                    active_tasks = active_tasks,
                )

                if not response.ok:
                    print(
                        "[HeartbeatLoop] Master does not know this worker. "
                        "Re-registering..."
                    )
                    self.master_client.reregister_worker()

            except Exception as e:
                print(f"[HeartbeatLoop] Failed: {e}")

            time.sleep(self.interval_sec)