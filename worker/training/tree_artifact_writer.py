from __future__ import annotations

import time
from dataclasses import asdict

from common.contracts import TreeArtifactMetadata
from common.ids import generate_tree_id


def now_ts():
    return time.time()


class TreeArtifactWriter:

    def __init__(self, store, paths, worker_id: str):
        self.store = store
        self.paths = paths
        self.worker_id = worker_id

    def write_tree(
        self,
        *,
        job_id: str,
        experiment_id: str,
        task_id: str,
        tree_index: int,
        seed: int,
        model,
        training_time: float,
    ) -> TreeArtifactMetadata:

        tree_id = generate_tree_id(experiment_id, tree_index)

        artifact_key = self.paths.tree_artifact_path(
            experiment_id, tree_id
        )

        metadata_key = self.paths.tree_metadata_path(
            experiment_id, tree_id
        )

        # ✅ idempotenza: se esiste già → non riscrivere
        if self.store.exists(artifact_key):
            meta = self._load_metadata_if_exists(metadata_key)
            if meta:
                return meta

        # salva modello
        self.store.save_joblib(artifact_key, model)

        meta = TreeArtifactMetadata(
            tree_id=tree_id,
            job_id=job_id,
            experiment_id=experiment_id,
            task_id=task_id,
            tree_index=tree_index,
            worker_id=self.worker_id,
            seed=seed,
            artifact_uri=artifact_key,
            status="COMPLETED",
            training_time_seconds=training_time,
        )

        self.store.save_json(metadata_key, asdict(meta))

        return meta

    def _load_metadata_if_exists(self, key):
        if not self.store.exists(key):
            return None
        data = self.store.load_json(key)
        return TreeArtifactMetadata(**data)