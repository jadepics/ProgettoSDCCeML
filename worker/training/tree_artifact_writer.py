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
        """
        Restituisce i metadati dell'albero al completamento dell'operazione.
        Se esisteva già l'alberi, recupera i dati in memoria
        :param model:
        :param job_id:
        :param experiment_id:
        :param task_id:
        :param tree_index:
        :param seed:
        :param training_time_seconds:
        :return:
        """
        # --------------------------------------------------
        # 1. Path logici
        # --------------------------------------------------
        artifact_key = tree_artifact_path(
            job_id=job_id,
            experiment_id=experiment_id,
            tree_index=tree_index,
        )

        metadata_key = tree_metadata_path(
            job_id=job_id,
            experiment_id=experiment_id,
            tree_index=tree_index,
        )

        # --------------------------------------------------
        # 2. Prova scrittura idempotente
        # --------------------------------------------------
        created = self.store.save_tree_artifact_if_not_exists(
            artifact_key,
            model
        )

        # --------------------------------------------------
        # 3. Caso: artifact GIÀ esistente
        # --------------------------------------------------
        if not created:
            # 👉 Qui è la parte che mancava
            if self.store.exists(metadata_key):
                data = self.store.load_json(metadata_key)
                return TreeArtifactMetadata(**data)

            # ⚠️ Caso raro ma possibile:
            # artifact esiste ma metadata no (crash intermedio)
            # → ricostruiamo metadata
            tree_id = f"{experiment_id}_tree_{tree_index}"

            metadata = TreeArtifactMetadata(
                tree_id=tree_id,
                job_id=job_id,
                experiment_id=experiment_id,
                task_id=task_id,
                tree_index=tree_index,
                worker_id=self.worker_id,
                seed=seed,
                artifact_uri=artifact_key,
                status="COMPLETED",
                training_time_seconds=training_time_seconds,
            )

            self.store.save_json(metadata_key, asdict(metadata))
            return metadata

        # --------------------------------------------------
        # 4. Caso: NUOVO artifact
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
            artifact_uri=artifact_key,
            status="COMPLETED",
            training_time_seconds=training_time_seconds,
        )

        self.store.save_json(metadata_key, asdict(metadata))

        return metadata