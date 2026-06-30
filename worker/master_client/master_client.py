"""
Client gRPC usato dal worker per comunicare con il cluster dei master.

Il client gestisce la connessione ai master candidati, la registrazione
del worker, il reinvio della registrazione e l'invio degli heartbeat.
In caso di errore o di risposta da un master non leader, prova gli altri
indirizzi disponibili.
"""

import os
import grpc
import time
from typing import Optional

import rf_v2_pb2 as rf_pb2
import rf_v2_pb2_grpc as rf_pb2_grpc

from common.grpc_config import GRPC_OPTIONS


class MasterClient:
    """
    Incapsula la comunicazione dal worker verso il master.

    La classe mantiene una lista di master candidati e conserva l'ultimo
    indirizzo utilizzato. Le chiamate gRPC vengono tentate prima sul master
    corrente e poi sugli altri candidati, così da supportare scenari con
    più master e cambio di leader.
    """

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
    # Scoperta e connessione ai master
    # --------------------------------------------------

    def _load_master_addresses(self, fallback_address: str) -> list[str]:
        """
        Legge la lista dei master candidati dalla variabile d'ambiente MASTER_SEEDS.

        MASTER_SEEDS deve contenere una lista di indirizzi nel formato host:port,
        separati da virgola. Se la variabile non è presente o non contiene valori
        validi, viene usato l'indirizzo di fallback.
        """

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
        """
        Apre un canale gRPC verso il master indicato e aggiorna lo stub usato
        per le chiamate successive.

        Se il client è già connesso allo stesso indirizzo, non ricrea il canale.
        """

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
        """
        Restituisce gli indirizzi dei master nell'ordine in cui devono essere provati.

        Il master attualmente selezionato viene provato per primo, seguito dagli
        altri candidati configurati in MASTER_SEEDS.
        """

        if self.address is None:
            return list(self.master_addresses)

        ordered = [self.address]

        for address in self.master_addresses:
            if address not in ordered:
                ordered.append(address)

        return ordered

    def _is_not_leader_message(self, message: str) -> bool:
        """
        Verifica se la risposta del master indica che la richiesta è stata inviata
        a un nodo che non è leader.

        In questo caso il client non considera la chiamata definitivamente fallita,
        ma prova il master candidato successivo.
        """

        normalized = str(message or "").lower()
        return "not leader" in normalized or "operation allowed only" in normalized

    # --------------------------------------------------
    # Registrazione del worker
    # --------------------------------------------------

    def register_worker(
        self,
        worker_id: str,
        host: str,
        port: int,
        retry: bool = True,
    ):
        """
        Registra il worker presso un master disponibile.

        La richiesta viene tentata sui master candidati finché un leader non la
        accetta. Se un master risponde che non è leader, il client passa al
        candidato successivo.
        """

        self._registered_worker_id = worker_id
        self._registered_host = host
        self._registered_port = port

        request = rf_pb2.RegisterWorkerRequest(
            worker_id=worker_id,
            host=host,
            port=port,
        )

        # Il ciclo permette di riprovare la registrazione finché non viene
        # trovato un master leader disponibile.
        while True:
            last_error = None

            # Ogni tentativo prova prima il master corrente e poi gli altri candidati.
            for address in self._candidate_addresses():
                try:
                    self._connect_to(address)

                    response = self.stub.RegisterWorker(
                        request,
                        timeout=10,
                    )

                    # La registrazione termina solo quando un master accetta
                    # esplicitamente il worker.
                    if response.accepted:
                        print(
                            f"[MasterClient] Registered worker {worker_id} "
                            f"as {host}:{port} on master {self.address}",
                            flush=True,
                        )
                        return response

                    message = response.message or ""

                    # Se il master raggiunto non è leader, non è un errore definitivo:
                    # il client prova il prossimo master candidato.
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
        """
        Ripete la registrazione usando i dati salvati durante la prima registrazione.

        È usato quando il worker deve ristabilire la propria presenza presso il
        master senza ricostruire manualmente la richiesta.
        """

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
    # Invio degli heartbeat
    # --------------------------------------------------

    def send_heartbeat(
            self,
            worker_id: str,
            running_tasks: int,
            active_task_ids=None,
            active_tasks=None,
    ):
        """
        Invia al master lo stato corrente del worker.

        L'heartbeat comunica quanti task sono in esecuzione e, quando disponibili,
        quali task sono attivi e il timestamp dell'ultimo progresso registrato.
        Se il master corrente non risponde o rifiuta la richiesta, vengono provati
        gli altri master candidati.
        """

        active_task_ids = list(active_task_ids or [])
        active_tasks = list(active_tasks or [])

        # La richiesta include sia il conteggio dei task attivi sia il dettaglio
        # dei task, usato dal master per monitorare avanzamento e possibili stalli.
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

        # L'heartbeat viene inviato al master corrente e, in caso di fallimento,
        # agli altri master candidati.
        for address in self._candidate_addresses():
            try:
                self._connect_to(address)

                response = self.stub.Heartbeat(
                    request,
                    timeout=10,
                )

                # Una risposta positiva conclude l'heartbeat.
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

        # Se almeno un master ha risposto, anche negativamente, restituiamo
        # l'ultima risposta ricevuta invece di trasformarla in errore di rete.
        if last_response is not None:
            return last_response

        raise RuntimeError(
            f"Heartbeat failed on all master candidates: {last_error}"
        )