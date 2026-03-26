from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

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
            raise PermissionError("Operation allowed only on the current leader master")

    def assert_leader_for(self, job_id: str) -> None:
        try:
            self.require_leader()
        except PermissionError as exc:
            raise PermissionError(f"Leader-only operation rejected for job {job_id}") from exc
