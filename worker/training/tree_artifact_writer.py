from __future__ import annotations

from dataclasses import asdict

from common.contracts import TreeArtifactMetadata
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
    - separazione logica/physical storage
    - consistenza tra artifact e metadata
    """

    def __init__(self, artifact_store: ArtifactStore, worker_id: str):
        self.store = artifact_store
        self.worker_id = worker_id

    def write_tree(
        self,
        model,
        job_id: str,
        experiment_id: str,
        task_id: str,
        tree_index: int,
        seed: int,
        training_time_seconds: float,
    ) -> TreeArtifactMetadata:

        # --------------------------------------------------
        # 1. Chiave logica artifact
        # --------------------------------------------------
        artifact_key = tree_artifact_path(
            job_id=job_id,
            experiment_id=experiment_id,
            tree_index=tree_index,
        )

        # --------------------------------------------------
        # 2. Salvataggio modello (delegato allo store)
        # --------------------------------------------------
        self.store.save_tree_artifact(artifact_key, model)

        # --------------------------------------------------
        # 3. Costruzione metadata
        # --------------------------------------------------
        tree_id = f"{experiment_id}_tree_{tree_index}"

        metadata = TreeArtifactMetadata(
            tree_id=tree_id,
            job_id=job_id,
            experiment_id=experiment_id,
            task_id=task_id,
            tree_index=tree_index,
            worker_id=self.worker_id,
            seed=seed,
            artifact_uri=artifact_key,  # 👈 chiave logica, NON path fisico
            status="COMPLETED",
            training_time_seconds=training_time_seconds,
        )

        # --------------------------------------------------
        # 4. Chiave logica metadata
        # --------------------------------------------------
        metadata_key = tree_metadata_path(
            job_id=job_id,
            experiment_id=experiment_id,
            tree_index=tree_index,
        )

        # --------------------------------------------------
        # 5. Salvataggio metadata (via store)
        # --------------------------------------------------
        self.store.save_json(metadata_key, asdict(metadata))

        # --------------------------------------------------
        # 6. Return
        # --------------------------------------------------
        return metadata