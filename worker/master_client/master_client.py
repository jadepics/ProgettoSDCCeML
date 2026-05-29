import grpc
import time

import rf_v2_pb2 as rf_pb2
import rf_v2_pb2_grpc as rf_pb2_grpc

GRPC_MAX_MESSAGE_LENGTH = 64 * 1024 * 1024

GRPC_OPTIONS = [
    ("grpc.max_send_message_length", GRPC_MAX_MESSAGE_LENGTH),
    ("grpc.max_receive_message_length", GRPC_MAX_MESSAGE_LENGTH),
]


class MasterClient:

    def __init__(self, host: str, port: int):
        self.address = f"{host}:{port}"
        self.channel = grpc.insecure_channel(self.address, options=GRPC_OPTIONS)
        self.stub = rf_pb2_grpc.CoordinatorServiceStub(self.channel)

        self._registered_worker_id = None
        self._registered_host = None
        self._registered_port = None

    # --------------------------------------------------
    # REGISTER
    # --------------------------------------------------

    def register_worker(
        self,
        worker_id: str,
        host: str,
        port: int,
        retry: bool = True,
    ):
        self._registered_worker_id = worker_id
        self._registered_host = host
        self._registered_port = port

        request = rf_pb2.RegisterWorkerRequest(
            worker_id=worker_id,
            host=host,
            port=port,
        )

        while True:
            try:
                response = self.stub.RegisterWorker(
                    request,
                    timeout=10,
                )

                if response.accepted:
                    print(
                        f"[MasterClient] Registered worker "
                        f"{worker_id} as {host}:{port}"
                    )
                    return response

                raise RuntimeError(
                    f"Registration rejected: {response.message}"
                )

            except Exception as e:
                print(f"[MasterClient] Register failed: {e}")

                if not retry:
                    raise

                time.sleep(2)

    def reregister_worker(self):
        if (
            self._registered_worker_id is None
            or self._registered_host is None
            or self._registered_port is None
        ):
            raise RuntimeError(
                "Cannot re-register worker: registration data missing"
            )

        return self.register_worker(
            worker_id=self._registered_worker_id,
            host=self._registered_host,
            port=self._registered_port,
            retry=True,
        )

    # --------------------------------------------------
    # HEARTBEAT
    # --------------------------------------------------
    def send_heartbeat(self, worker_id: str, running_tasks: int, active_task_ids, active_tasks):
        request = rf_pb2.HeartbeatRequest(
            worker_id=worker_id,
            running_tasks=running_tasks,
            active_task_ids=active_task_ids,
            active_tasks = active_tasks
        )

        return self.stub.Heartbeat(
            request,
            timeout=10,
        )