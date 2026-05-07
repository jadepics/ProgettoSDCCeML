from __future__ import annotations

import threading
import time
from typing import Optional

from common.contracts import (
    ExperimentRecord,
    ForestConfiguration,
    TrainingJobRecord,
    TrainingRequest,
    TreeArtifactMetadata,
)
from common.enums import ExperimentStatus, JobStatus


class TrainingJobService:
    """
    Responsabilità:
    - avviare un training job lato master
    - persistire lo stato del job tramite JobRepository
    - orchestrare data preparation, planning, training, validation,
      model selection e manifest build
    - NON fare lavoro RPC diretto
    - NON sostituire TrainingOrchestrator / ValidationCoordinator
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

    # --------------------------------------------------------
    # public API
    # --------------------------------------------------------

    def start_training_job(self, training_request: TrainingRequest) -> str:
        self.leadership_guard.require_leader()

        job_id = training_request.job_id

        initial_record = TrainingJobRecord(
            job_id=job_id,
            status=JobStatus.PENDING,
            training_request=training_request,
            prepared_dataset=None,
            experiment_ids=[],
            selected_experiment_id=None,
            model_id=None,
            message="Training job created",
            created_at=time.time(),
            updated_at=time.time(),
        )
        self.job_repository.save(initial_record)

        worker = threading.Thread(
            target=self._run_training_job,
            args=(job_id,),
            daemon=True,
        )
        worker.start()

        return job_id

    # --------------------------------------------------------
    # internal workflow
    # --------------------------------------------------------

    def _run_training_job(self, job_id: str) -> None:
        current_experiment_id: str | None = None

        try:
            self.leadership_guard.require_leader()

            job_record = self._load_job_or_raise(job_id)
            training_request = job_record.training_request

            self.job_repository.mark_running(
                job_id=job_id,
                message="Preparing dataset",
            )

            prepared_dataset = self.data_preparation_service.prepare(training_request)
            self.job_repository.attach_prepared_dataset(job_id, prepared_dataset)

            experiments = self._plan_experiments(training_request)
            if not experiments:
                raise RuntimeError("ExperimentPlanner produced no experiments")

            artifacts_by_experiment_id: dict[str, list[TreeArtifactMetadata]] = {}

            for experiment in experiments:
                self.job_repository.save_experiment(job_id, experiment)

            for experiment in experiments:
                current_experiment_id = experiment.experiment_id

                self.job_repository.update_experiment_status(
                    job_id=job_id,
                    experiment_id=experiment.experiment_id,
                    status=ExperimentStatus.RUNNING,
                )
                self.job_repository.mark_running(
                    job_id=job_id,
                    message=f"Training experiment {experiment.experiment_id}",
                )

                tree_artifacts = self.training_orchestrator.run_experiment(
                    job_id=job_id,
                    experiment_id=experiment.experiment_id,
                    forest_config=experiment.forest_config,
                )
                artifacts_by_experiment_id[experiment.experiment_id] = list(tree_artifacts)

                validation_result = self._validate_experiment(
                    job_id=job_id,
                    experiment=experiment,
                    tree_artifacts=tree_artifacts,
                )

                self.job_repository.update_experiment_status(
                    job_id=job_id,
                    experiment_id=experiment.experiment_id,
                    status=ExperimentStatus.COMPLETED,
                    completed_tree_count=len(tree_artifacts),
                    validation_metrics=validation_result.metrics,
                )

            completed_experiments = self.job_repository.list_experiments(job_id)
            if not completed_experiments:
                raise RuntimeError("No persisted experiments found after training")

            winning_experiment = self._select_best_experiment(completed_experiments)
            if winning_experiment is None:
                raise RuntimeError("ModelSelector did not return a winning experiment")

            self.job_repository.set_selected_experiment(
                job_id=job_id,
                experiment_id=winning_experiment.experiment_id,
            )

            selected_tree_artifacts = artifacts_by_experiment_id.get(
                winning_experiment.experiment_id,
                [],
            )
            if not selected_tree_artifacts:
                raise RuntimeError(
                    f"No tree artifacts found for selected experiment "
                    f"{winning_experiment.experiment_id}"
                )

            current_job = self._load_job_or_raise(job_id)
            if current_job.prepared_dataset is None:
                raise RuntimeError(f"Job '{job_id}' has no prepared dataset")

            manifest = self._build_model_manifest(
                job_record=current_job,
                experiment_record=winning_experiment,
                tree_artifacts=selected_tree_artifacts,
            )

            self.model_repository.save(manifest)
            if hasattr(self.model_repository, "mark_ready"):
                self.model_repository.mark_ready(manifest.model_id)

            self.job_repository.mark_completed(
                job_id=job_id,
                selected_experiment_id=winning_experiment.experiment_id,
                model_id=manifest.model_id,
                message=(
                    f"Training completed. Selected experiment "
                    f"{winning_experiment.experiment_id}"
                ),
            )

        except Exception as exc:
            self._mark_job_failed(
                job_id=job_id,
                error_message=str(exc),
                experiment_id=current_experiment_id,
            )

    # --------------------------------------------------------
    # planning helpers
    # --------------------------------------------------------

    def _plan_experiments(
        self,
        training_request: TrainingRequest,
    ) -> list[ExperimentRecord]:
        """
        Normalizza l'output del planner in una lista di ExperimentRecord.
        Supporta diversi nomi di metodo del planner per restare compatibile
        con versioni leggermente diverse del progetto.
        """
        planned = None

        if hasattr(self.experiment_planner, "plan"):
            planned = self.experiment_planner.plan(training_request)
        elif hasattr(self.experiment_planner, "plan_experiments"):
            planned = self.experiment_planner.plan_experiments(training_request)
        elif hasattr(self.experiment_planner, "select_initial_experiment"):
            planned = self.experiment_planner.select_initial_experiment(training_request)
        else:
            raise AttributeError(
                "ExperimentPlanner must expose one of: "
                "plan, plan_experiments, select_initial_experiment"
            )

        if planned is None:
            return []

        if isinstance(planned, ExperimentRecord):
            return [planned]

        if isinstance(planned, ForestConfiguration):
            return [self._experiment_from_forest_config(planned)]

        if isinstance(planned, list):
            normalized: list[ExperimentRecord] = []
            for item in planned:
                if isinstance(item, ExperimentRecord):
                    normalized.append(item)
                elif isinstance(item, ForestConfiguration):
                    normalized.append(self._experiment_from_forest_config(item))
                else:
                    raise TypeError(
                        f"Unsupported planner output item type: {type(item)!r}"
                    )
            return normalized

        raise TypeError(f"Unsupported planner output type: {type(planned)!r}")

    def _experiment_from_forest_config(
        self,
        forest_config: ForestConfiguration,
    ) -> ExperimentRecord:
        experiment_id = getattr(forest_config, "experiment_id", None)
        if not experiment_id:
            raise ValueError(
                "ForestConfiguration must carry experiment_id to build ExperimentRecord"
            )

        return ExperimentRecord(
            experiment_id=experiment_id,
            forest_config=forest_config,
            status=ExperimentStatus.PENDING,
            assigned_workers=[],
            expected_tree_count=forest_config.n_estimators,
            completed_tree_count=0,
            validation_metrics=None,
        )

    # --------------------------------------------------------
    # validation / selection helpers
    # --------------------------------------------------------

    def _validate_experiment(
        self,
        job_id: str,
        experiment: ExperimentRecord,
        tree_artifacts: list[TreeArtifactMetadata],
    ):
        job_record = self._load_job_or_raise(job_id)
        prepared_dataset = job_record.prepared_dataset
        if prepared_dataset is None:
            raise ValueError(f"Job '{job_id}' has no prepared dataset")

        return self.validation_coordinator.validate_experiment(
            experiment_id=experiment.experiment_id,
            task_type=job_record.training_request.task_type,
            validation_features_uri=prepared_dataset.validation_features_uri,
            validation_labels_uri=prepared_dataset.validation_labels_uri,
            tree_artifacts=tree_artifacts,
            class_labels=prepared_dataset.class_labels,
        )

    def _select_best_experiment(
        self,
        experiments: list[ExperimentRecord],
    ) -> Optional[ExperimentRecord]:
        return self.model_selector.select_best(experiments)

    # --------------------------------------------------------
    # manifest helper
    # --------------------------------------------------------

    def _build_model_manifest(
        self,
        job_record: TrainingJobRecord,
        experiment_record: ExperimentRecord,
        tree_artifacts: list[TreeArtifactMetadata],
    ):
        from common.enums import ModelStatus
        from common.ids import generate_model_id

        prepared_dataset = job_record.prepared_dataset
        if prepared_dataset is None:
            raise ValueError(f"Job '{job_record.job_id}' has no prepared dataset")

        if experiment_record.validation_metrics is None:
            raise ValueError(
                f"Experiment '{experiment_record.experiment_id}' has no validation metrics"
            )

        model_id = generate_model_id(job_record.job_id)

        return self.model_manifest_builder.build(
            model_id=model_id,
            job_id=job_record.job_id,
            experiment_id=experiment_record.experiment_id,
            model_type=job_record.training_request.task_type,
            forest_config=experiment_record.forest_config,
            prepared_dataset=prepared_dataset,
            tree_artifacts=tree_artifacts,
            validation_metrics=experiment_record.validation_metrics,
            test_metrics=None,
            status=ModelStatus.READY,
        )

    # --------------------------------------------------------
    # failure handling
    # --------------------------------------------------------

    def _mark_job_failed(
        self,
        job_id: str,
        error_message: str,
        experiment_id: str | None = None,
    ) -> None:
        try:
            if experiment_id is not None:
                self.job_repository.update_experiment_status(
                    job_id=job_id,
                    experiment_id=experiment_id,
                    status=ExperimentStatus.FAILED,
                )
        finally:
            try:
                self.job_repository.mark_failed(
                    job_id=job_id,
                    message=error_message,
                )
            except Exception:
                pass

    def _load_job_or_raise(self, job_id: str) -> TrainingJobRecord:
        record = self.job_repository.load(job_id)
        if record is None:
            raise ValueError(f"Job '{job_id}' not found")
        return record