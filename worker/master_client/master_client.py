import grpc
import time

import rf_v2_pb2 as rf_pb2
import rf_v2_pb2_grpc as rf_pb2_grpc

GRPC_MAX_MESSAGE_LENGTH = 64 * 1024 * 1024  # 64 MB

GRPC_OPTIONS = [
    ("grpc.max_send_message_length", GRPC_MAX_MESSAGE_LENGTH),
    ("grpc.max_receive_message_length", GRPC_MAX_MESSAGE_LENGTH),
]
class MasterClient:

    def __init__(self, host: str, port: int):
        self.address = f"{host}:{port}"
        self.channel = grpc.insecure_channel(self.address, options=GRPC_OPTIONS)
        self.stub = rf_pb2_grpc.CoordinatorServiceStub(self.channel)

    # --------------------------------------------------
    # REGISTER
    # --------------------------------------------------
    def register_worker(self, worker_id: str, host: str, port: int, retry=True):
        request = rf_pb2.RegisterWorkerRequest(
            worker_id=worker_id,
            host=host,
            port=port
        )

        while True:
            try:
                response = self.stub.RegisterWorker(request)

                if response.accepted:
                    print(f"[MasterClient] Registered worker {worker_id}")
                    return

                raise RuntimeError(f"Registration rejected: {response.message}")

            except Exception as e:
                print(f"[MasterClient] Register failed: {e}")

                if not retry:
                    raise

                time.sleep(2)

    # --------------------------------------------------
    # HEARTBEAT
    # --------------------------------------------------
    def send_heartbeat(self, worker_id: str, running_tasks: int, active_task_ids):
        request = rf_pb2.HeartbeatRequest(
            worker_id=worker_id,
            running_tasks=running_tasks,
            active_task_ids=active_task_ids
        )

        return self.stub.Heartbeat(request)