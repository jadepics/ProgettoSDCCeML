from abc import ABC, abstractmethod
from typing import Any, Dict


class ArtifactStore(ABC):

    # --- Tree artifacts ---
    @abstractmethod
    def save_tree_artifact(self, path: str, model: Any) -> str:
        pass

    @abstractmethod
    def save_tree_artifact_if_not_exists(self, path: str, model: Any) -> bool:
        """
        Returns:
            True -> artifact was created
            False -> artifact already existed
        """
        pass

    @abstractmethod
    def save_json_atomic(self, key: str, data: Dict[str, Any]) -> None:
        """
        Salva JSON in modo atomico:
        - scrive su file temporaneo
        - rename atomico su key finale
        """
        pass

    @abstractmethod
    def rename(self, src: str, dst: str) -> None:
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