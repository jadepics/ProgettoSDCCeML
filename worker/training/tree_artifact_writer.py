from __future__ import annotations

from dataclasses import asdict

from common.ids import generate_tree_id
from common.contracts import TreeArtifactMetadata
from common.enums import TreeStatus
from worker.storage.artifact_store import ArtifactStore
from worker.storage.paths import (
    tree_artifact_path,
    tree_metadata_path,
)


class TreeArtifactWriter:
    """
    Responsabile del salvataggio degli alberi (artifact) e dei relativi metadata.

    Garantisce:
    - path deterministici
    - idempotenza storage
    - consistenza tra artifact e metadata
    """

    def __init__(self, artifact_store: ArtifactStore, worker_id: str):
        self.store: ArtifactStore = artifact_store
        self.worker_id: str = worker_id

    def write_tree(
        self,
        model: object,
        job_id: str,
        experiment_id: str,
        task_id: str,
        tree_index: int,
        seed: int,
        training_time_seconds: float,
    ) -> TreeArtifactMetadata:
        """
        model: object
        job_id: str
        experiment_id: str
        task_id: str
        tree_index: int
        seed: int
        training_time_seconds: float

        return: TreeArtifactMetadata

        Idempotente:
        - metadata esiste → ritorna
        - artifact esiste ma metadata no → ricostruisce
        - niente esiste → scrive tutto
        """

        # --------------------------------------------------
        # 1. Path deterministici
        # --------------------------------------------------
        artifact_key: str = tree_artifact_path(
            job_id=job_id,
            experiment_id=experiment_id,
            tree_index=tree_index,
        )

        metadata_key: str = tree_metadata_path(
            job_id=job_id,
            experiment_id=experiment_id,
            tree_index=tree_index,
        )

        tree_id = generate_tree_id(experiment_id, tree_index)

        # --------------------------------------------------
        # 2. FAST PATH → metadata già esiste
        # (più veloce + evita load modello inutile)
        # --------------------------------------------------
        if self.store.exists(metadata_key):
            data = self.store.load_json(metadata_key)
            return TreeArtifactMetadata(**data)

        # --------------------------------------------------
        # 3. Scrittura artifact (idempotente)
        # --------------------------------------------------
        created: bool = self.store.save_tree_artifact_if_not_exists(
            artifact_key,
            model
        )

        # --------------------------------------------------
        # 4. COSTRUZIONE METADATA (sempre)
        # --------------------------------------------------
        metadata = TreeArtifactMetadata(
            tree_id=tree_id,
            job_id=job_id,
            experiment_id=experiment_id,
            task_id=task_id,
            tree_index=tree_index,
            worker_id=self.worker_id,
            seed=seed,
            artifact_uri=artifact_key,
            status=TreeStatus.COMPLETED,
            training_time_seconds=training_time_seconds,
        )

        # --------------------------------------------------
        # 5. Scrittura metadata ATOMICA (sempre)
        # --------------------------------------------------
        # ⚠️ FIX CRITICO:
        # prima usavi save_json nel caso crash → NON atomico
        # ora sempre atomico
        self.store.save_json_atomic(
            metadata_key,
            asdict(metadata)
        )

        return metadata