from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
import json
import random
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from common.contracts import MasterCommand
from common.enums import MasterRole


@dataclass(slots=True)
class RaftPeer:
    node_id: str
    host: str
    port: int


@dataclass(slots=True)
class RaftNodeConfig:
    node_id: str
    host: str
    port: int
    peer_nodes: list[RaftPeer]
    log_dir: str
    election_timeout_ms: int = 3000
    heartbeat_interval_ms: int = 500


class ConsensusService(ABC):
    """Interface intentionally kept small for the first milestone.

    Start with a single-leader in-memory implementation; replace it later with a
    real Raft-backed implementation without forcing changes into orchestration code.
    """

    @abstractmethod
    def start(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def stop(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def current_role(self) -> MasterRole:
        raise NotImplementedError

    @abstractmethod
    def is_leader(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def current_term(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def append_command(self, command: MasterCommand) -> int:
        raise NotImplementedError


class InMemoryLeaderConsensusService(ConsensusService):
    """Single-process development stub.

    It already enforces the architectural rule 'leader-only execution', while the
    real Raft implementation can be introduced in a later milestone.
    """

    def __init__(self, node_id: str, start_as_leader: bool = True) -> None:
        self.node_id = node_id
        self._role = MasterRole.LEADER if start_as_leader else MasterRole.FOLLOWER
        self._term = 1
        self._log_index = 0

    def start(self) -> None:
        return None

    def stop(self) -> None:
        return None

    def current_role(self) -> MasterRole:
        return self._role

    def is_leader(self) -> bool:
        return self._role == MasterRole.LEADER

    def current_term(self) -> int:
        return self._term

    def append_command(self, command: MasterCommand) -> int:
        if not self.is_leader():
            raise PermissionError("Only the leader can append commands")
        self._log_index += 1
        return self._log_index


class LeadershipGuard:
    def __init__(self, consensus_service: ConsensusService):
        self.consensus_service = consensus_service

    def require_leader(self) -> None:
        if not self.consensus_service.is_leader():
            raise PermissionError("Operation allowed only on the current leader masterPackage")

    def assert_leader_for(self, job_id: str) -> None:
        try:
            self.require_leader()
        except PermissionError as exc:
            raise PermissionError(f"Leader-only operation rejected for job {job_id}") from exc

def build_raft_node_config_from_env(
    artifact_root: str,
) -> RaftNodeConfig:
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
    Raft minimale per BLOCCO C.

    Implementa:
    - leader election;
    - request vote;
    - heartbeat AppendEntries vuoto;
    - current_term persistito;
    - voted_for persistito;
    - is_leader() reale.

    Non implementa ancora:
    - log replication piena;
    - commit index;
    - ReplicatedStateMachine.

    Quella parte resta BLOCCO D.
    """

    def __init__(self, config: RaftNodeConfig) -> None:
        self.config = config
        self.node_id = config.node_id

        self._role = MasterRole.FOLLOWER
        self._term = 0
        self._voted_for: Optional[str] = None
        self._leader_id: Optional[str] = None
        self._log_index = 0

        self._lock = threading.RLock()
        self._stop_event = threading.Event()

        self._server: Optional[ThreadingHTTPServer] = None
        self._server_thread: Optional[threading.Thread] = None
        self._election_thread: Optional[threading.Thread] = None
        self._heartbeat_thread: Optional[threading.Thread] = None

        self._election_deadline_ts = 0.0

        self._state_path = Path(config.log_dir) / "raft_state.json"

    def start(self) -> None:
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
        with self._lock:
            if self._role != MasterRole.LEADER:
                raise PermissionError("Only the leader can append commands")

            self._log_index += 1
            return self._log_index

    def _candidate_role(self) -> MasterRole:
        return getattr(MasterRole, "CANDIDATE", MasterRole.FOLLOWER)

    def _cluster_size(self) -> int:
        return len(self.config.peer_nodes) + 1

    def _majority(self) -> int:
        return self._cluster_size() // 2 + 1

    def _reset_election_deadline(self) -> None:
        base_seconds = self.config.election_timeout_ms / 1000.0
        timeout = random.uniform(base_seconds, base_seconds * 2.0)
        self._election_deadline_ts = time.time() + timeout

    def _load_state(self) -> None:
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
        payload = {
            "node_id": self.node_id,
            "term": self._term,
            "voted_for": self._voted_for,
            "leader_id": self._leader_id,
            "log_index": self._log_index,
            "updated_at": time.time(),
        }

        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self._state_path.with_suffix(".json.tmp")
        tmp_path.write_text(json.dumps(payload, indent=2))
        tmp_path.replace(self._state_path)

    def _start_rpc_server(self) -> None:
        service = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    raw_body = self.rfile.read(length)
                    payload = json.loads(raw_body.decode("utf-8") or "{}")

                    if self.path == "/request_vote":
                        response = service._handle_request_vote(payload)
                    elif self.path == "/append_entries":
                        response = service._handle_append_entries(payload)
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
                        {
                            "ok": False,
                            "error": str(exc),
                        }
                    ).encode("utf-8")

                    self.send_response(500)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(encoded)))
                    self.end_headers()
                    self.wfile.write(encoded)

            def log_message(self, format, *args):
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
        while not self._stop_event.is_set():
            time.sleep(0.05)

            with self._lock:
                if self._role == MasterRole.LEADER:
                    continue

                if time.time() < self._election_deadline_ts:
                    continue

            self._start_election()

    def _start_election(self) -> None:
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
                payload={
                    "term": term,
                    "candidate_id": self.node_id,
                },
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
                    payload={
                        "term": term,
                        "leader_id": self.node_id,
                    },
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
        candidate_term = int(payload.get("term", 0))
        candidate_id = str(payload.get("candidate_id", "")).strip()

        with self._lock:
            if candidate_term < self._term:
                return {
                    "term": self._term,
                    "vote_granted": False,
                }

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

                return {
                    "term": self._term,
                    "vote_granted": True,
                }

            return {
                "term": self._term,
                "vote_granted": False,
            }

    def _handle_append_entries(self, payload: dict[str, Any]) -> dict[str, Any]:
        leader_term = int(payload.get("term", 0))
        leader_id = str(payload.get("leader_id", "")).strip()

        with self._lock:
            if leader_term < self._term:
                return {
                    "term": self._term,
                    "success": False,
                }

            if leader_term > self._term:
                self._term = leader_term
                self._voted_for = None

            if leader_id != self.node_id:
                self._role = MasterRole.FOLLOWER
                self._leader_id = leader_id

            self._reset_election_deadline()
            self._persist_state()

            return {
                "term": self._term,
                "success": True,
            }

    def _become_follower(
        self,
        term: int,
        leader_id: Optional[str],
        voted_for: Optional[str],
    ) -> None:
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