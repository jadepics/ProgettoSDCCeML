from __future__ import annotations

import numpy as np
import pandas as pd
import joblib
import time
from dataclasses import asdict
from typing import List

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from common.contracts import TreeArtifactMetadata, WorkerProgressSnapshot
from common.ids import generate_tree_id, tree_seed


def now_ts() -> float:
    return time.time()


class ShardTrainer:

    def __init__(self, config, state, store, paths, artifact_writer, progress_store):
        self.config = config
        self.state = state
        self.store = store
        self.paths = paths
        self.writer = artifact_writer
        self.progress_store = progress_store

    def train(self, request) -> List[TreeArtifactMetadata]:
        completed_tree_ids = []
        failed_tree_ids = []
        trained = []
        running_tree_ids = []

        df = pd.read_csv(request.dataset_url)
        y = df[request.target_column].to_numpy()
        X = df.drop(columns=[request.target_column]).to_numpy(dtype=float)

        # snapshot iniziale
        self._snapshot(request, completed_tree_ids, [], failed_tree_ids)

        for offset in range(request.tree_count):
            tree_index = request.start_tree_index + offset
            tree_id = generate_tree_id(request.experiment_id, tree_index)
            seed = tree_seed(request.seed_base, tree_index)

            artifact_key = self.paths.tree_artifact_path(
                request.experiment_id, tree_id
            )
            metadata_key = self.paths.tree_metadata_path(
                request.experiment_id, tree_id
            )

            running_tree_ids = [tree_id]
            self._snapshot(request, completed_tree_ids, running_tree_ids, failed_tree_ids)

            # idempotenza
            if self.store.exists(artifact_key):
                meta = self._load_metadata_if_exists(metadata_key)
                if meta is None:
                    meta = TreeArtifactMetadata(
                        tree_id=tree_id,
                        job_id=request.job_id,
                        experiment_id=request.experiment_id,
                        task_id=request.task_id,
                        tree_index=tree_index,
                        worker_id=self.config.worker_id,
                        seed=seed,
                        artifact_uri=artifact_key,
                        status="COMPLETED",
                        training_time_seconds=0.0,
                    )
                    self.store.save_json(metadata_key, asdict(meta))

                trained.append(meta)
                completed_tree_ids.append(tree_id)
                self._snapshot(request, completed_tree_ids, [], failed_tree_ids)
                continue

            t0 = now_ts()

            # bootstrap
            X_fit, y_fit = X, y
            if request.bootstrap:
                rng = np.random.default_rng(seed)
                idx = rng.integers(0, X.shape[0], size=X.shape[0])
                X_fit = X[idx]
                y_fit = y[idx]

            # model
            if request.model_type == "classification":
                model = DecisionTreeClassifier(random_state=seed)
            else:
                model = DecisionTreeRegressor(random_state=seed)

            model.fit(X_fit, y_fit)
            self.store.save_joblib(artifact_key, model)

            training_time = now_ts() - t0

            meta = self.writer.write_tree(
                job_id=request.job_id,
                experiment_id=request.experiment_id,
                task_id=request.task_id,
                tree_index=tree_index,
                seed=seed,
                model=model,
                training_time=training_time,
            )

            self.store.save_json(metadata_key, asdict(meta))

            trained.append(meta)
            completed_tree_ids.append(tree_id)

            self._snapshot(request, completed_tree_ids, [], failed_tree_ids)

        return trained

    def _snapshot(self, request, completed, running, failed):
        snapshot = WorkerProgressSnapshot(
            worker_id=self.config.worker_id,
            task_id=request.task_id,
            experiment_id=request.experiment_id,
            completed_tree_ids=list(completed),
            running_tree_ids=list(running),
            failed_tree_ids=list(failed),
            last_update_ts=now_ts(),
        )

        self.progress_store.save_snapshot(snapshot)

    def _load_metadata_if_exists(self, key):
        if not self.store.exists(key):
            return None
        return TreeArtifactMetadata(**self.store.load_json(key))