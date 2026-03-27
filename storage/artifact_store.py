from abc import ABC, abstractmethod
from typing import Any, Optional


class ArtifactStore(ABC):

    # --- Tree artifacts ---
    @abstractmethod
    def save_tree_artifact(self, path: str, model: Any) -> str:
        pass

    @abstractmethod
    def load_tree_artifact(self, path: str) -> Any:
        pass

    @abstractmethod
    def tree_artifact_exists(self, path: str) -> bool:
        pass

    # --- Generic JSON ---
    @abstractmethod
    def save_json(self, path: str, data: dict) -> None:
        pass

    @abstractmethod
    def load_json(self, path: str) -> dict:
        pass

    # --- Raw file ---
    @abstractmethod
    def save_bytes(self, path: str, data: bytes) -> None:
        pass

    @abstractmethod
    def load_bytes(self, path: str) -> bytes:
        pass

    # --- Utilities ---
    @abstractmethod
    def exists(self, path: str) -> bool:
        pass

    @abstractmethod
    def delete(self, path: str) -> None:
        pass