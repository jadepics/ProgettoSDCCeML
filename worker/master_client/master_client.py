import os
import grpc
import time
from typing import Optional

import rf_v2_pb2 as rf_pb2
import rf_v2_pb2_grpc as rf_pb2_grpc

from common.grpc_config import GRPC_OPTIONS


class MasterClient:

    def __init__(self, host: str, port: int):
        self.default_address = f"{host}:{port}"
        self.master_addresses = self._load_master_addresses(
            fallback_address=self.default_address,
        )

        self.address: Optional[str] = None
        self.channel = None
        self.stub = None

        self._registered_worker_id = None
        self._registered_host = None
        self._registered_port = None

        self._connect_to(self.master_addresses[0])

    # --------------------------------------------------
    # MASTER DISCOVERY / CONNECTION
    # --------------------------------------------------

    def _load_master_addresses(self, fallback_address: str) -> list[str]:
        raw_seeds = os.getenv("MASTER_SEEDS", "").strip()

        if not raw_seeds:
            return [fallback_address]

        addresses: list[str] = []

        for item in raw_seeds.split(","):
            address = item.strip()
            if not address:
                continue

            if ":" not in address:
                raise ValueError(
                    f"Invalid MASTER_SEEDS item '{address}'. "
                    "Expected host:port"
                )

            if address not in addresses:
                addresses.append(address)

        return addresses or [fallback_address]

    def _connect_to(self, address: str) -> None:
        if self.address == address and self.stub is not None:
            return

        self.address = address
        self.channel = grpc.insecure_channel(
            self.address,
            options=GRPC_OPTIONS,
        )
        self.stub = rf_pb2_grpc.CoordinatorServiceStub(self.channel)

        print(
            f"[MasterClient] Using master candidate {self.address}",
            flush=True,
        )

    def _candidate_addresses(self) -> list[str]:
        if self.address is None:
            return list(self.master_addresses)

        ordered = [self.address]

        for address in self.master_addresses:
            if address not in ordered:
                ordered.append(address)

        return ordered

    def _is_not_leader_message(self, message: str) -> bool:
        normalized = str(message or "").lower()
        return "not leader" in normalized or "operation allowed only" in normalized

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
            last_error = None

            for address in self._candidate_addresses():
                try:
                    self._connect_to(address)

                    response = self.stub.RegisterWorker(
                        request,
                        timeout=10,
                    )

                    if response.accepted:
                        print(
                            f"[MasterClient] Registered worker {worker_id} "
                            f"as {host}:{port} on master {self.address}",
                            flush=True,
                        )
                        return response

                    message = response.message or ""

                    if self._is_not_leader_message(message):
                        print(
                            f"[MasterClient] Master {self.address} rejected "
                            f"registration because it is not leader",
                            flush=True,
                        )
                        last_error = RuntimeError(message)
                        continue

                    raise RuntimeError(
                        f"Registration rejected by {self.address}: {message}"
                    )

                except Exception as exc:
                    last_error = exc
                    print(
                        f"[MasterClient] Register failed on {address}: {exc}",
                        flush=True,
                    )
                    continue

            if not retry:
                raise RuntimeError(
                    f"Unable to register worker on any master: {last_error}"
                )

            print(
                "[MasterClient] No leader accepted registration. Retrying...",
                flush=True,
            )
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

    def send_heartbeat(
            self,
            worker_id: str,
            running_tasks: int,
            active_task_ids=None,
            active_tasks=None,
    ):
        active_task_ids = list(active_task_ids or [])
        active_tasks = list(active_tasks or [])

        request = rf_pb2.HeartbeatRequest(
            worker_id=worker_id,
            running_tasks=running_tasks,
            active_task_ids=active_task_ids,
            active_tasks=[
                rf_pb2.ActiveTaskHeartbeat(
                    task_id=str(item["task_id"]),
                    last_progress_ts=float(item["last_progress_ts"]),
                )
                for item in active_tasks
            ],
        )

        last_error = None
        last_response = None

        for address in self._candidate_addresses():
            try:
                self._connect_to(address)

                response = self.stub.Heartbeat(
                    request,
                    timeout=10,
                )

                if response.ok:
                    return response

                last_response = response

                print(
                    f"[MasterClient] Heartbeat rejected by {self.address}. "
                    "Trying next master candidate...",
                    flush=True,
                )

                continue

            except Exception as exc:
                last_error = exc
                print(
                    f"[MasterClient] Heartbeat failed on {address}: {exc}",
                    flush=True,
                )
                continue

        if last_response is not None:
            return last_response

        raise RuntimeError(
            f"Heartbeat failed on all master candidates: {last_error}"
        )