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
from worker.utils.io_utils import DataLoader


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
        data_loader: DataLoader,
    ):
        self.bootstrap_sampler = bootstrap_sampler
        self.tree_factory = tree_factory
        self.artifact_writer = artifact_writer
        self.progress_store = progress_store
        self.data_loader = data_loader

    # --------------------------------------------------
    # MAIN
    # --------------------------------------------------
    def train(self, shard: TrainingShard) -> ShardTrainingResult:
        """
        shard: TrainingShard
        return: ShardTrainingResult
        """

        import time

        start_time: float = time.time()

        # ----------------------------------------
        # Accumulatori risultato
        # ----------------------------------------
        tree_artifacts: list[TreeArtifactMetadata] = []
        completed_tree_ids: set[str] = set()
        failed_tree_ids: set[str] = set()

        # ----------------------------------------
        # START TASK (idempotente) (start_task precedentemente era in trainshard)
        # ----------------------------------------
        status: str = self.progress_store.start_task(
            job_id=shard.job_id,
            experiment_id=shard.experiment_id,
            task_id=shard.task_id,
            metadata={
                "attempt_id": shard.attempt_id,
                "worker_id": shard.assigned_worker_id,
            },
        )

        # ----------------------------------------
        # Caso: già completato (idempotenza forte, ovvero idempotenza per task)
        # ----------------------------------------
        if status == "ALREADY_COMPLETED":
            existing = self.progress_store.get_task(
                job_id=shard.job_id,
                experiment_id=shard.experiment_id,
                task_id=shard.task_id,
            )

            return ShardTrainingResult(
                task_id=shard.task_id,
                attempt_id=shard.attempt_id,
                worker_id=shard.assigned_worker_id,
                success=True,
                tree_artifacts=[],
                completed_tree_ids=existing.completed_tree_ids,
                failed_tree_ids=existing.failed_tree_ids,
                completed_tree_count=len(existing.completed_tree_ids),
                failed_tree_count=len(existing.failed_tree_ids),
                error_message=None,
                elapsed_time_seconds=0.0,
            )

        # ----------------------------------------
        # Carico snapshot esistente (retry-safe)
        # ----------------------------------------
        existing = self.progress_store.get_task(
            job_id=shard.job_id,
            experiment_id=shard.experiment_id,
            task_id=shard.task_id,
        )

        if existing is not None:
            completed_tree_ids = set(existing.get("completed_tree_ids", []))
            failed_tree_ids = set(existing.get("failed_tree_ids", []))

        # ----------------------------------------
        # Load dataset (URI → numpy)
        # ----------------------------------------
        X: "np.ndarray" = self.data_loader.load_numpy(shard.train_features_uri)
        y: "np.ndarray" = self.data_loader.load_numpy(shard.train_labels_uri)

        # ----------------------------------------
        # Da qui partirà:
        # try:
        #     for each tree ...
        # ----------------------------------------
        # LOOP sugli alberi (setup + skip)
        # ----------------------------------------
        for i in range(shard.tree_count):
            """
            i: int
            """

            # indice globale dell'albero nella foresta
            tree_index: int = shard.tree_start_index + i

            # id logico dell'albero (deterministico)
            tree_id: str = f"{shard.task_id}_tree_{tree_index}"

            # ----------------------------------------
            # IDPOTENZA (CRITICO)
            # ----------------------------------------
            if self.artifact_writer.exists(tree_id):
                completed_tree_ids.append(tree_id)
                continue

            # ----------------------------------------
            # SKIP se già completato (retry-safe)
            # ----------------------------------------
            if tree_id in completed_tree_ids:
                continue

            # ----------------------------------------
            # SEED deterministico per l'albero
            # ----------------------------------------
            seed: int = shard.seed_base + tree_index

            # ----------------------------------------
            # BOOTSTRAP (firma corretta)
            # sample_indices(n_samples: int, seed: int, bootstrap: bool)
            # ----------------------------------------
            indices = self.bootstrap_sampler.sample_indices(
                n_samples=len(X),
                seed=seed,
                bootstrap=shard.forest_config.bootstrap,
            )

            X_sample = X[indices]
            y_sample = y[indices]

            # ----------------------------------------
            # CREAZIONE MODELLO (firma corretta)
            # create(max_depth, min_samples_split, min_samples_leaf, max_features, seed)
            # ----------------------------------------
            fc = shard.forest_config  # alias per leggibilità

            tree = self.tree_factory.create(
                max_depth=fc.max_depth,
                min_samples_split=fc.min_samples_split,
                min_samples_leaf=fc.min_samples_leaf,
                max_features=fc.max_features,
                seed=seed,
            )

            # ----------------------------------------
            # Da qui partirà:
            # try:
            #     tree.fit(...)
            # ----------------------------------------
            try:
                """
                Training + persistenza atomica albero
                """

                # ----------------------------------------
                # TRAIN
                # ----------------------------------------
                t0: float = time.time()

                tree.fit(X_sample, y_sample)

                training_time: float = time.time() - t0

                # ----------------------------------------
                # WRITE ARTIFACT (idempotente lato storage)
                # ----------------------------------------
                metadata: TreeArtifactMetadata = self.artifact_writer.write_tree(
                    model=tree,
                    job_id=shard.job_id,
                    experiment_id=shard.experiment_id,
                    task_id=shard.task_id,
                    tree_index=tree_index,
                    seed=seed,
                    training_time_seconds=training_time,
                )

                # ----------------------------------------
                # UPDATE SUCCESS
                # ----------------------------------------
                tree_artifacts.append(metadata)
                completed_tree_ids.add(tree_id)

                # aggiornamento snapshot consistente
                self.progress_store.update_task(
                    job_id=shard.job_id,
                    experiment_id=shard.experiment_id,
                    task_id=shard.task_id,
                    completed_tree_ids=list(completed_tree_ids),
                    failed_tree_ids=list(failed_tree_ids),
                )


            except Exception as exc:
                """
                Gestione errore per singolo albero
                """

                failed_tree_ids.add(tree_id)

                # aggiorniamo comunque lo snapshot
                self.progress_store.update_task(
                    job_id=shard.job_id,
                    experiment_id=shard.experiment_id,
                    task_id=shard.task_id,
                    completed_tree_ids=list(completed_tree_ids),
                    failed_tree_ids=list(failed_tree_ids),
                )
        # ----------------------------------------
        # FINALIZZAZIONE
        # ----------------------------------------
        # ----------------------------------------
        # FINALIZZAZIONE TASK
        # ----------------------------------------

        # successo globale
        success: bool = len(failed_tree_ids) == 0

        # ----------------------------------------
        # Aggiornamento finale snapshot
        # ----------------------------------------
        self.progress_store.update_task(
            job_id=shard.job_id,
            experiment_id=shard.experiment_id,
            task_id=shard.task_id,
            completed_tree_ids=list(completed_tree_ids),
            failed_tree_ids=list(failed_tree_ids),
        )

        # ----------------------------------------
        # Stato finale
        # ----------------------------------------
        if success:
            self.progress_store.complete_task(
                job_id=shard.job_id,
                experiment_id=shard.experiment_id,
                task_id=shard.task_id,
            )
        else:
            self.progress_store.fail_task(
                job_id=shard.job_id,
                experiment_id=shard.experiment_id,
                task_id=shard.task_id,
                error="Some trees failed",
            )

        # ----------------------------------------
        # RETURN coerente
        # ----------------------------------------
        return ShardTrainingResult(
            task_id=shard.task_id,
            attempt_id=shard.attempt_id,
            worker_id=shard.assigned_worker_id,
            success=success,
            tree_artifacts=tree_artifacts,
            completed_tree_ids=list(completed_tree_ids),
            failed_tree_ids=list(failed_tree_ids),
            completed_tree_count=len(completed_tree_ids),
            failed_tree_count=len(failed_tree_ids),
            error_message=None if success else "Some trees failed",
            elapsed_time_seconds=time.time() - start_time,
        )