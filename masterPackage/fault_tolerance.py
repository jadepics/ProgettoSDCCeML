from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
import json
import random
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from common.contracts import MasterCommand
from common.enums import MasterRole


@dataclass(slots=True)
class RaftPeer:
    """
    Descrive un nodo remoto del cluster Raft.

    Attributes:
        node_id: identificatore logico univoco del peer nel cluster.
        host: indirizzo di rete usato per la comunicazione HTTP interna Raft.
        port: porta TCP del server Raft esposto dal peer.
    """
    node_id: str
    host: str
    port: int


@dataclass(slots=True)
class RaftNodeConfig:
    """
    Configurazione locale del nodo Raft.

    Attributes:
        node_id: identificatore del nodo corrente.
        host: indirizzo su cui il server HTTP Raft locale va in ascolto.
        port: porta su cui il server HTTP Raft locale va in ascolto.
        peer_nodes: elenco degli altri nodi del cluster, escluso il nodo corrente.
        log_dir: directory persistente usata per salvare lo stato locale minimo di Raft.
        election_timeout_ms: timeout base per l'avvio di una nuova elezione.
        heartbeat_interval_ms: intervallo con cui il leader invia heartbeat ai follower.
    """
    node_id: str
    host: str
    port: int
    peer_nodes: list[RaftPeer]
    log_dir: str
    election_timeout_ms: int = 3000
    heartbeat_interval_ms: int = 500


class ConsensusService(ABC):
    """
    Interfaccia astratta del livello di consenso usato dal master.

    Questa interfaccia astrae il meccanismo che decide:
    - quale nodo è leader;
    - quale termine è corrente;
    - se un'operazione può essere eseguita solo dal leader.

    Nota importante:
    in questa fase del progetto il consenso viene usato soprattutto
    per garantire l'esecuzione leader-only delle operazioni del master.
    La replica completa dello stato applicativo non è ancora gestita qui,
    ma viene supportata a livello pratico da storage condiviso EFS e recovery.
    """

    @abstractmethod
    def start(self) -> None:
        """Avvia il servizio di consenso e le sue risorse interne."""
        raise NotImplementedError

    @abstractmethod
    def stop(self) -> None:
        """Arresta il servizio di consenso e rilascia le risorse aperte."""
        raise NotImplementedError

    @abstractmethod
    def current_role(self) -> MasterRole:
        """Restituisce il ruolo corrente del nodo nel cluster."""
        raise NotImplementedError

    @abstractmethod
    def is_leader(self) -> bool:
        """Indica se il nodo corrente è il leader attivo."""
        raise NotImplementedError

    @abstractmethod
    def current_term(self) -> int:
        """Restituisce il termine Raft locale corrente."""
        raise NotImplementedError

    @abstractmethod
    def append_command(self, command: MasterCommand) -> int:
        """
        Registra logicamente un comando leader-only.

        Nella versione completa di Raft questo metodo sarebbe il punto naturale
        per inserire la log replication e il commit dei comandi.
        Nella versione attuale viene usato come controllo di coerenza:
        solo il leader può accettare il comando.
        """
        raise NotImplementedError


class InMemoryLeaderConsensusService(ConsensusService):
    """
    Implementazione minimale del consenso per esecuzione single-leader.

    Questa classe non implementa una vera elezione distribuita:
    serve come fallback semplice o come stub per ambienti locali/testing.
    Il nodo viene creato come leader oppure follower e mantiene lo stato
    solo in memoria.

    Uso previsto:
    - test rapidi;
    - esecuzione semplificata senza cluster Raft reale;
    - applicazione della regola architetturale "solo il leader orchestra".
    """

    def __init__(self, node_id: str, start_as_leader: bool = True) -> None:
        self.node_id = node_id
        self._role = MasterRole.LEADER if start_as_leader else MasterRole.FOLLOWER
        self._term = 1
        self._log_index = 0

    def start(self) -> None:
        """Nessuna inizializzazione runtime necessaria nella versione in-memory."""
        return None

    def stop(self) -> None:
        """Nessuna risorsa esterna da chiudere nella versione in-memory."""
        return None

    def current_role(self) -> MasterRole:
        return self._role

    def is_leader(self) -> bool:
        return self._role == MasterRole.LEADER

    def current_term(self) -> int:
        return self._term

    def append_command(self, command: MasterCommand) -> int:
        """
        Accetta il comando solo se il nodo è leader.

        Qui non viene replicato nessun log: l'indice restituito è solo
        un contatore locale utile a mantenere una semantica coerente
        con l'interfaccia del servizio di consenso.
        """
        if not self.is_leader():
            raise PermissionError("Only the leader can append commands")
        self._log_index += 1
        return self._log_index


class LeadershipGuard:
    """
    Guard applicativo che protegge le operazioni leader-only del master.

    Questa classe centralizza il controllo di leadership ed evita che
    componenti applicativi diversi replichino la stessa logica di validazione.
    È utile soprattutto nei punti in cui il master:
    - assegna shard;
    - aggiorna stato di job/esperimenti;
    - pubblica manifest o decisioni finali.
    """

    def __init__(self, consensus_service: ConsensusService):
        self.consensus_service = consensus_service

    def require_leader(self) -> None:
        """
        Solleva eccezione se il nodo corrente non è leader.

        Da usare prima di operazioni che devono essere eseguite da un solo master.
        """
        if not self.consensus_service.is_leader():
            raise PermissionError("Operation allowed only on the current leader master")

    def assert_leader_for(self, job_id: str) -> None:
        """
        Variante contestualizzata per job specifici.

        Migliora i messaggi di errore quando un'operazione leader-only
        viene rifiutata durante la gestione di uno specifico job.
        """
        try:
            self.require_leader()
        except PermissionError as exc:
            raise PermissionError(
                f"Leader-only operation rejected for job {job_id}"
            ) from exc


def build_raft_node_config_from_env(artifact_root: str) -> RaftNodeConfig:
    """
    Costruisce la configurazione Raft leggendo le variabili d'ambiente.

    Formato atteso per RAFT_PEERS:
        node_id:host:port,node_id:host:port,...

    Il nodo corrente viene escluso automaticamente dall'elenco dei peer.
    La directory di stato locale viene derivata da artifact_root se non
    è esplicitamente configurata.
    """
    import os

    node_id = os.getenv("MASTER_NODE_ID", "master-1").strip()
    raft_host = os.getenv("RAFT_HOST", "0.0.0.0").strip()
    raft_port = int(os.getenv("RAFT_PORT", "50151"))

    raw_peers = os.getenv("RAFT_PEERS", "").strip()
    peers: list[RaftPeer] = []

    if raw_peers:
        for item in raw_peers.split(","):
            item = item.strip()
            if not item:
                continue

            parts = item.split(":")
            if len(parts) != 3:
                raise ValueError(
                    "Invalid RAFT_PEERS item. Expected node_id:host:port, "
                    f"got: {item}"
                )

            peer_node_id, peer_host, peer_port = parts
            if peer_node_id == node_id:
                continue

            peers.append(
                RaftPeer(
                    node_id=peer_node_id,
                    host=peer_host,
                    port=int(peer_port),
                )
            )

    log_dir = os.getenv(
        "RAFT_LOG_DIR",
        str(Path(artifact_root) / "raft" / node_id),
    )

    return RaftNodeConfig(
        node_id=node_id,
        host=raft_host,
        port=raft_port,
        peer_nodes=peers,
        log_dir=log_dir,
        election_timeout_ms=int(os.getenv("RAFT_ELECTION_TIMEOUT_MS", "3000")),
        heartbeat_interval_ms=int(os.getenv("RAFT_HEARTBEAT_INTERVAL_MS", "500")),
    )


class RaftConsensusService(ConsensusService):
    """
    Implementazione minimale di Raft per la fault tolerance lato master.

    Responsabilità implementate:
    - elezione di un leader unico;
    - gestione dei ruoli follower/candidate/leader;
    - RequestVote tra master;
    - heartbeat periodici tramite AppendEntries vuoti;
    - persistenza locale dello stato minimo Raft
      (term, voted_for, leader_id, log_index).

    Scopo architetturale:
    questa classe rende possibile l'esecuzione leader-only del control plane
    del master cluster. In caso di crash del leader, un altro master può essere
    eletto e il sistema può recuperare lo stato applicativo leggendo EFS.

    Limiti intenzionali di questa versione:
    - non replica il log applicativo completo;
    - non implementa commit index e replicated state machine;
    - non garantisce quindi una replica formale dello stato master,
      ma una leader election reale con recovery applicativo basato su storage condiviso.
    """

    def __init__(self, config: RaftNodeConfig) -> None:
        self.config = config
        self.node_id = config.node_id

        # Stato volatile del nodo Raft.
        self._role = MasterRole.FOLLOWER
        self._term = 0
        self._voted_for: Optional[str] = None
        self._leader_id: Optional[str] = None
        self._log_index = 0

        # Lock re-entrant per proteggere accessi concorrenti da thread di election,
        # heartbeat e server RPC.
        self._lock = threading.RLock()
        self._stop_event = threading.Event()

        # Risorse runtime.
        self._server: Optional[ThreadingHTTPServer] = None
        self._server_thread: Optional[threading.Thread] = None
        self._election_thread: Optional[threading.Thread] = None
        self._heartbeat_thread: Optional[threading.Thread] = None

        # Deadline oltre la quale un follower/candidate avvia una nuova elezione.
        self._election_deadline_ts = 0.0

        # File locale che persiste lo stato minimo di Raft.
        self._state_path = Path(config.log_dir) / "raft_state.json"

    def start(self) -> None:
        """
        Avvia il nodo Raft.

        Sequenza:
        1. crea la directory di stato;
        2. ricarica lo stato persistito locale;
        3. inizializza la prossima election deadline;
        4. avvia il server RPC HTTP;
        5. avvia i thread di election e heartbeat.
        """
        Path(self.config.log_dir).mkdir(parents=True, exist_ok=True)
        self._load_state()
        self._reset_election_deadline()

        self._start_rpc_server()

        self._election_thread = threading.Thread(
            target=self._election_loop,
            daemon=True,
        )
        self._election_thread.start()

        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop,
            daemon=True,
        )
        self._heartbeat_thread.start()

        print(
            "[RaftConsensusService] started:",
            f"node_id={self.node_id}",
            f"raft={self.config.host}:{self.config.port}",
            f"peers={[peer.node_id for peer in self.config.peer_nodes]}",
            flush=True,
        )

    def stop(self) -> None:
        """
        Arresta il servizio e ferma il server HTTP.

        I thread in background terminano leggendo _stop_event.
        """
        self._stop_event.set()

        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()

    def current_role(self) -> MasterRole:
        with self._lock:
            return self._role

    def is_leader(self) -> bool:
        with self._lock:
            return self._role == MasterRole.LEADER

    def current_term(self) -> int:
        with self._lock:
            return self._term

    def append_command(self, command: MasterCommand) -> int:
        """
        Punto di estensione per il futuro log Raft applicativo.

        Nella versione corrente:
        - verifica che il nodo sia leader;
        - incrementa un indice locale;
        - restituisce quell'indice.

        In una versione successiva questo metodo dovrebbe:
        - appendere il comando al log;
        - replicarlo ai follower;
        - attendere commit di maggioranza;
        - applicarlo alla state machine.
        """
        with self._lock:
            if self._role != MasterRole.LEADER:
                raise PermissionError("Only the leader can append commands")

            self._log_index += 1
            return self._log_index

    def _candidate_role(self) -> MasterRole:
        """
        Restituisce il ruolo candidate, se definito nell'enum.

        Fallback: follower.
        Questo mantiene compatibilità anche se MasterRole non espone esplicitamente CANDIDATE.
        """
        return getattr(MasterRole, "CANDIDATE", MasterRole.FOLLOWER)

    def _cluster_size(self) -> int:
        """Numero totale di nodi del cluster, incluso il nodo corrente."""
        return len(self.config.peer_nodes) + 1

    def _majority(self) -> int:
        """Quorum minimo richiesto per eleggere un leader."""
        return self._cluster_size() // 2 + 1

    def _reset_election_deadline(self) -> None:
        """
        Aggiorna la scadenza per la prossima elezione.

        Il timeout è randomizzato in un intervallo [T, 2T] per ridurre
        il rischio di elezioni simultanee tra follower.
        """
        base_seconds = self.config.election_timeout_ms / 1000.0
        timeout = random.uniform(base_seconds, base_seconds * 2.0)
        self._election_deadline_ts = time.time() + timeout

    def _load_state(self) -> None:
        """
        Carica dallo storage locale lo stato minimo persistito di Raft.

        Se il file non esiste o è corrotto, il nodo parte con lo stato di default.
        """
        if not self._state_path.exists():
            return

        try:
            payload = json.loads(self._state_path.read_text())
        except Exception:
            return

        with self._lock:
            self._term = int(payload.get("term", 0))
            self._voted_for = payload.get("voted_for")
            self._leader_id = payload.get("leader_id")
            self._log_index = int(payload.get("log_index", 0))

    def _persist_state(self) -> None:
        """
        Salva lo stato locale di Raft in modo atomico.

        La scrittura avviene su file temporaneo seguito da rename,
        così da ridurre il rischio di file parziali in caso di crash.
        """
        payload = {
            "node_id": self.node_id,
            "term": self._term,
            "voted_for": self._voted_for,
            "leader_id": self._leader_id,
            "log_index": self._log_index,
            "updated_at": time.time(),
        }

        self._state_path.parent.mkdir(parents=True, exist_ok=True)

        tmp_path = self._state_path.with_name(
            f".{self._state_path.name}.tmp.{time.time_ns()}.{threading.get_ident()}"
        )

        try:
            tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            tmp_path.replace(self._state_path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    def _start_rpc_server(self) -> None:
        """
        Avvia il piccolo server HTTP interno usato per le RPC Raft minime.

        Endpoint supportati:
        - /request_vote
        - /append_entries
        - /status
        """
        service = self

        class Handler(BaseHTTPRequestHandler):
            """Handler HTTP minimale che inoltra le richieste al servizio Raft."""

            def do_POST(self):
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    raw_body = self.rfile.read(length)
                    payload = json.loads(raw_body.decode("utf-8") or "{}")

                    if self.path == "/request_vote":
                        response = service._handle_request_vote(payload)
                    elif self.path == "/append_entries":
                        response = service._handle_append_entries(payload)
                    elif self.path == "/status":
                        response = service._handle_status()
                    else:
                        self.send_response(404)
                        self.end_headers()
                        return

                    encoded = json.dumps(response).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(encoded)))
                    self.end_headers()
                    self.wfile.write(encoded)

                except Exception as exc:
                    encoded = json.dumps(
                        {"ok": False, "error": str(exc)}
                    ).encode("utf-8")

                    self.send_response(500)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(encoded)))
                    self.end_headers()
                    self.wfile.write(encoded)

            def log_message(self, format, *args):
                # Sopprime il logging HTTP standard per evitare rumore nei log del nodo.
                return

        self._server = ThreadingHTTPServer(
            (self.config.host, self.config.port),
            Handler,
        )

        self._server_thread = threading.Thread(
            target=self._server.serve_forever,
            daemon=True,
        )
        self._server_thread.start()

    def _election_loop(self) -> None:
        """
        Loop periodico che monitora l'election timeout.

        Un follower o candidate che non riceve heartbeat entro la deadline
        prova ad avviare una nuova elezione.
        """
        while not self._stop_event.is_set():
            time.sleep(0.05)

            with self._lock:
                if self._role == MasterRole.LEADER:
                    continue

                if time.time() < self._election_deadline_ts:
                    continue

            self._start_election()

    def _start_election(self) -> None:
        """
        Avvia una nuova elezione locale.

        Il nodo:
        - incrementa il term;
        - vota per sé stesso;
        - invia RequestVote a tutti i peer;
        - diventa leader se ottiene la maggioranza.
        """
        with self._lock:
            self._term += 1
            term = self._term
            self._role = self._candidate_role()
            self._voted_for = self.node_id
            self._leader_id = None
            self._reset_election_deadline()
            self._persist_state()

        votes = 1

        print(
            "[RaftConsensusService] election started:",
            f"node_id={self.node_id}",
            f"term={term}",
            flush=True,
        )

        for peer in self.config.peer_nodes:
            response = self._post_json(
                peer=peer,
                path="/request_vote",
                payload={"term": term, "candidate_id": self.node_id},
            )

            if response is None:
                continue

            peer_term = int(response.get("term", 0))
            if peer_term > term:
                self._become_follower(
                    term=peer_term,
                    leader_id=None,
                    voted_for=None,
                )
                return

            if response.get("vote_granted") is True:
                votes += 1

        with self._lock:
            if self._term != term:
                return

            if self._role != self._candidate_role():
                return

            if votes >= self._majority():
                self._role = MasterRole.LEADER
                self._leader_id = self.node_id
                self._persist_state()

                print(
                    "[RaftConsensusService] leader elected:",
                    f"node_id={self.node_id}",
                    f"term={self._term}",
                    f"votes={votes}/{self._cluster_size()}",
                    flush=True,
                )
            else:
                print(
                    "[RaftConsensusService] election lost:",
                    f"node_id={self.node_id}",
                    f"term={term}",
                    f"votes={votes}/{self._cluster_size()}",
                    flush=True,
                )

    def _heartbeat_loop(self) -> None:
        """
        Loop periodico del leader.

        Il leader invia AppendEntries vuoti ai follower per:
        - mantenere la leadership;
        - aggiornare i follower sul term corrente;
        - impedire l'avvio di nuove elezioni.
        """
        interval_seconds = self.config.heartbeat_interval_ms / 1000.0

        while not self._stop_event.is_set():
            time.sleep(interval_seconds)

            with self._lock:
                if self._role != MasterRole.LEADER:
                    continue

                term = self._term

            for peer in self.config.peer_nodes:
                response = self._post_json(
                    peer=peer,
                    path="/append_entries",
                    payload={"term": term, "leader_id": self.node_id},
                )

                if response is None:
                    continue

                peer_term = int(response.get("term", 0))
                if peer_term > term:
                    self._become_follower(
                        term=peer_term,
                        leader_id=None,
                        voted_for=None,
                    )
                    break

    def _handle_request_vote(self, payload: dict[str, Any]) -> dict[str, Any]:
        """
        Gestisce una richiesta di voto ricevuta da un candidato.

        Regole applicate:
        - rifiuta i term più vecchi;
        - aggiorna il term locale se il candidato è più aggiornato;
        - concede il voto se non ha ancora votato nel term corrente
          o se aveva già votato per lo stesso candidato.
        """
        candidate_term = int(payload.get("term", 0))
        candidate_id = str(payload.get("candidate_id", "")).strip()

        with self._lock:
            if candidate_term < self._term:
                return {"term": self._term, "vote_granted": False}

            if candidate_term > self._term:
                self._role = MasterRole.FOLLOWER
                self._term = candidate_term
                self._voted_for = None
                self._leader_id = None

            can_vote = self._voted_for is None or self._voted_for == candidate_id

            if can_vote:
                self._voted_for = candidate_id
                self._role = MasterRole.FOLLOWER
                self._leader_id = None
                self._reset_election_deadline()
                self._persist_state()

                return {"term": self._term, "vote_granted": True}

            return {"term": self._term, "vote_granted": False}

    def _handle_append_entries(self, payload: dict[str, Any]) -> dict[str, Any]:
        """
        Gestisce un heartbeat ricevuto dal leader.
        In questa implementazione il messaggio viene usato solo per aggiornare il term, registrare il leader corrente
        e resettare la election deadline.
        """
        leader_term = int(payload.get("term", 0))
        leader_id = str(payload.get("leader_id", "")).strip()

        with self._lock:
            if leader_term < self._term:
                return {"term": self._term, "success": False}

            if leader_term > self._term:
                self._term = leader_term
                self._voted_for = None

            if leader_id != self.node_id:
                self._role = MasterRole.FOLLOWER
                self._leader_id = leader_id

            self._reset_election_deadline()
            self._persist_state()

            return {"term": self._term, "success": True}

    def _become_follower(
        self,
        term: int,
        leader_id: Optional[str],
        voted_for: Optional[str],
    ) -> None:
        """
        Forza la transizione del nodo a follower.
        È usato quando il nodo scopre un term più alto oppure perde la leadership.
        """
        with self._lock:
            self._term = term
            self._role = MasterRole.FOLLOWER
            self._leader_id = leader_id
            self._voted_for = voted_for
            self._reset_election_deadline()
            self._persist_state()

        print(
            "[RaftConsensusService] became follower:",
            f"node_id={self.node_id}",
            f"term={term}",
            f"leader_id={leader_id}",
            flush=True,
        )

    def _post_json(
        self,
        peer: RaftPeer,
        path: str,
        payload: dict[str, Any],
    ) -> Optional[dict[str, Any]]:
        """
        Invia una richiesta HTTP JSON a un peer Raft.
        Ritorna:
        - il payload JSON decodificato in caso di successo;
        - None se il peer non è raggiungibile o non risponde entro il timeout.
        """
        url = f"http://{peer.host}:{peer.port}{path}"
        data = json.dumps(payload).encode("utf-8")

        request = urllib.request.Request(
            url=url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=1.0) as response:
                raw = response.read().decode("utf-8")
                return json.loads(raw or "{}")
        except (urllib.error.URLError, TimeoutError, OSError):
            return None

    def _handle_status(self) -> dict[str, Any]:
        """
        Restituisce lo stato locale del nodo.
        Utile per debug, health check e strumenti operativi.
        """
        with self._lock:
            return {
                "ok": True,
                "node_id": self.node_id,
                "role": str(self._role.value if hasattr(self._role, "value") else self._role),
                "term": self._term,
                "leader_id": self._leader_id,
            }