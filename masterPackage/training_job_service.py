from __future__ import annotations

import threading
import time
import uuid

from common.contracts import (
    HyperparameterSpace,
    TrainingJobRecord,
    TrainingRequest,
)
from common.enums import ExperimentStatus, JobStatus, ModelStatus


class TrainingJobService:
    """
    Servizio applicativo del master per il ciclo completo di un training job.

    Responsabilità:
    - creare job + esperimento iniziale
    - lanciare il workflow di training in background
    - orchestrare data preparation, training, validation, selection e manifest finale

    Nota:
    questa classe NON parla protobuf e NON espone RPC.
    """

    def __init__(
        self,
        leadership_guard,
        job_repository,
        model_repository,
        data_preparation_service,
        experiment_planner,
        training_orchestrator,
        validation_coordinator,
        model_selector,
        model_manifest_builder,
    ) -> None:
        self.leadership_guard = leadership_guard
        self.job_repository = job_repository
        self.model_repository = model_repository
        self.data_preparation_service = data_preparation_service
        self.experiment_planner = experiment_planner
        self.training_orchestrator = training_orchestrator
        self.validation_coordinator = validation_coordinator
        self.model_selector = model_selector
        self.model_manifest_builder = model_manifest_builder

    def start_training_job(self, training_request: TrainingRequest) -> str:
        self.leadership_guard.require_leader()

        model_id = str(uuid.uuid4())
        initial_experiment = self.experiment_planner.select_initial_experiment(training_request)

        job_record = TrainingJobRecord(
            job_id=training_request.job_id,
            status=JobStatus.PENDING,
            training_request=training_request,
            prepared_dataset=None,
            experiment_ids=[initial_experiment.experiment_id],
            selected_experiment_id=None,
            model_id=model_id,
            message="Training queued",
            created_at=self._now_ts(),
            updated_at=self._now_ts(),
        )

        self.job_repository.save(job_record)
        self.job_repository.save_experiment(training_request.job_id, initial_experiment)

        threading.Thread(
            target=self._run_training_job,
            args=(training_request.job_id, model_id, initial_experiment.experiment_id),
            daemon=True,
        ).start()

        return training_request.job_id

    def _run_training_job(self, job_id: str, model_id: str, experiment_id: str) -> None:
        record = self.job_repository.load(job_id)
        if record is None:
            return

        try:
            self.leadership_guard.require_leader()

            record.status = JobStatus.RUNNING
            record.message = "Training in progress"
            record.updated_at = self._now_ts()
            self.job_repository.save(record)

            req = record.training_request
            model_type = req.task_type.strip().lower()
            if model_type not in {"classification", "regression"}:
                raise ValueError("task_type must be 'classification' or 'regression'")

            prepared_dataset = self.data_preparation_service.prepare(
                job_id=req.job_id,
                dataset_uri=req.dataset_uri,
                target_column=req.target_column,
                task_type=req.task_type,
                validation_ratio=req.validation_ratio,
                test_ratio=req.test_ratio,
                random_seed=req.global_random_seed,
            )

            record.prepared_dataset = prepared_dataset
            record.updated_at = self._now_ts()
            self.job_repository.save(record)

            experiment = self.job_repository.load_experiment(job_id, experiment_id)
            if experiment is None:
                raise RuntimeError(f"Experiment '{experiment_id}' not found")

            forest_config = experiment.forest_config

            collected_artifacts = self.training_orchestrator.run_experiment(
                job_id=job_id,
                experiment_id=experiment_id,
                forest_config=forest_config,
            )

            validation_result = self.validation_coordinator.validate_experiment(
                experiment_id=experiment_id,
                task_type=model_type,
                validation_features_uri=prepared_dataset.validation_features_uri,
                validation_labels_uri=prepared_dataset.validation_labels_uri,
                tree_artifacts=collected_artifacts,
                class_labels=prepared_dataset.class_labels or [],
            )

            experiment = self.job_repository.load_experiment(job_id, experiment_id)
            if experiment is None:
                raise RuntimeError(
                    f"Experiment '{experiment_id}' not found after training orchestration"
                )

            experiment.status = ExperimentStatus.COMPLETED
            experiment.completed_tree_count = len(collected_artifacts)
            experiment.validation_metrics = validation_result.metrics
            self.job_repository.save_experiment(job_id, experiment)

            selected_experiment = self.model_selector.select_best_for_job(
                self.job_repository,
                job_id,
            )

            if selected_experiment.validation_metrics is None:
                raise RuntimeError(
                    f"Selected experiment '{selected_experiment.experiment_id}' "
                    f"has no validation metrics"
                )

            manifest = self.model_manifest_builder.build(
                model_id=model_id,
                job_id=job_id,
                experiment_id=selected_experiment.experiment_id,
                model_type=model_type,
                forest_config=selected_experiment.forest_config,
                prepared_dataset=prepared_dataset,
                tree_artifacts=collected_artifacts,
                validation_metrics=selected_experiment.validation_metrics,
                test_metrics=None,
                status=ModelStatus.READY,
            )

            self.model_repository.save(manifest)

            record = self.job_repository.load(job_id)
            if record is None:
                raise RuntimeError(f"Job '{job_id}' disappeared before completion update")

            record.status = JobStatus.COMPLETED
            record.message = "Training completed successfully"
            record.selected_experiment_id = selected_experiment.experiment_id
            record.updated_at = self._now_ts()
            self.job_repository.save(record)

        except Exception as exc:
            record = self.job_repository.load(job_id)
            if record is not None:
                record.status = JobStatus.FAILED
                record.message = str(exc)
                record.updated_at = self._now_ts()
                self.job_repository.save(record)

    def _now_ts(self) -> float:
        return time.time()