from __future__ import annotations

import time
from typing import List

import numpy as np

from common.contracts import (
    TrainingShard,
    ShardTrainingResult,
    TreeArtifactMetadata,
)
from worker.storage.paths import tree_artifact_path

from worker.training.bootstrap_sampler import BootstrapSampler
from worker.training.decision_tree_factory import DecisionTreeFactory
from worker.training.tree_artifact_writer import TreeArtifactWriter
from worker.progress.worker_progress_store import WorkerProgressStore


class ShardTrainer:
    """
    Core del training lato worker.

    Responsabilità:
    - eseguire training shard
    - delegare stato e idempotenza al WorkerProgressStore
    """

    def __init__(
        self,
        bootstrap_sampler: BootstrapSampler,
        tree_factory: DecisionTreeFactory,
        artifact_writer: TreeArtifactWriter,
        progress_store: WorkerProgressStore,
    ):
        self.bootstrap_sampler = bootstrap_sampler
        self.tree_factory = tree_factory
        self.artifact_writer = artifact_writer
        self.progress_store = progress_store

    def train(
            self,
            shard: TrainingShard,
            X: np.ndarray,
            y: np.ndarray,
    ) -> ShardTrainingResult:

        job_id = shard.job_id
        experiment_id = shard.experiment_id
        task_id = shard.task_id

        # --------------------------------------------------
        # Stato persistente
        # --------------------------------------------------
        task = self.progress_store.get_task(job_id, experiment_id, task_id)

        completed_tree_ids: List[str] = []
        failed_tree_ids: List[str] = []

        if task:
            completed_tree_ids = list(task.get("completed_tree_ids", []))
            failed_tree_ids = list(task.get("failed_tree_ids", []))

        tree_artifacts: List[TreeArtifactMetadata] = []

        # --------------------------------------------------
        # Loop alberi
        # --------------------------------------------------
        for offset in range(shard.tree_count):
            tree_index = shard.tree_start_index + offset
            seed = shard.seed_base + tree_index

            tree_id = f"{experiment_id}_tree_{tree_index}"

            artifact_key = tree_artifact_path(
                job_id=job_id,
                experiment_id=experiment_id,
                tree_index=tree_index,
            )

            # --------------------------------------------------
            # ✅ Idempotenza 1: snapshot (progress store)
            # --------------------------------------------------
            if tree_id in completed_tree_ids:
                continue

            # --------------------------------------------------
            # ✅ Idempotenza 2: storage (source of truth)
            # --------------------------------------------------
            if self.artifact_writer.store.tree_artifact_exists(artifact_key):
                # Artifact già presente → consideriamo il tree completato

                completed_tree_ids.append(tree_id)

                # Aggiorna progress snapshot (recupero implicito)
                self.progress_store.update_progress(
                    job_id=job_id,
                    experiment_id=experiment_id,
                    task_id=task_id,
                    shard_id=tree_index,
                    progress=len(completed_tree_ids) / shard.tree_count,
                )

                continue

            # --------------------------------------------------
            # ✅ Idempotenza 3: artifact già esistente
            # --------------------------------------------------
            artifact_key = tree_artifact_path(
                job_id=job_id,
                experiment_id=experiment_id,
                tree_index=tree_index,
            )

            if self.artifact_writer.store.exists(artifact_key):
                completed_tree_ids.append(tree_id)

                # opzionale: aggiornare progress
                self.progress_store.update_progress(
                    job_id=job_id,
                    experiment_id=experiment_id,
                    task_id=task_id,
                    shard_id=tree_index,
                    progress=len(completed_tree_ids) / shard.tree_count,
                )

                continue

            # --------------------------------------------------
            # TRAINING
            # --------------------------------------------------
            try:
                t0 = time.time()

                # 1. bootstrap
                indices = self.bootstrap_sampler.sample_indices(len(X), seed)
                X_fit = X[indices]
                y_fit = y[indices]

                # 2. modello
                model = self.tree_factory.create(
                    max_depth=shard.forest_config.max_depth,
                    min_samples_split=shard.forest_config.min_samples_split,
                    min_samples_leaf=shard.forest_config.min_samples_leaf,
                    max_features=shard.forest_config.max_features,
                    seed=seed,
                )

                # 3. training
                model.fit(X_fit, y_fit)

                training_time = time.time() - t0

                # 4. persistenza
                meta = self.artifact_writer.write_tree(
                    model=model,
                    job_id=job_id,
                    experiment_id=experiment_id,
                    task_id=task_id,
                    tree_index=tree_index,
                    seed=seed,
                    training_time_seconds=training_time,
                )

                tree_artifacts.append(meta)
                completed_tree_ids.append(meta.tree_id)

                # --------------------------------------------------
                # Progress persistente
                # --------------------------------------------------
                self.progress_store.update_progress(
                    job_id=job_id,
                    experiment_id=experiment_id,
                    task_id=task_id,
                    shard_id=tree_index,
                    progress=len(completed_tree_ids) / shard.tree_count,
                )

            except Exception:
                failed_tree_ids.append(tree_id)

                # ⚠️ NON blocchiamo tutto il task subito
                # (puoi migliorarlo in step successivi)
                self.progress_store.fail_task(
                    job_id=job_id,
                    experiment_id=experiment_id,
                    task_id=task_id,
                    error=f"Tree {tree_id} failed",
                )

        # --------------------------------------------------
        # Finalizzazione
        # --------------------------------------------------
        success = len(failed_tree_ids) == 0

        if success:
            self.progress_store.complete_task(
                job_id=job_id,
                experiment_id=experiment_id,
                task_id=task_id,
            )

        return ShardTrainingResult(
            task_id=task_id,
            attempt_id=shard.attempt_id,
            worker_id=shard.assigned_worker_id,
            success=success,
            tree_artifacts=tree_artifacts,
            completed_tree_ids=completed_tree_ids,
            failed_tree_ids=failed_tree_ids,
            completed_tree_count=len(completed_tree_ids),
            failed_tree_count=len(failed_tree_ids),
            error_message=None if success else "Some trees failed",
            elapsed_time_seconds=0.0,
        )