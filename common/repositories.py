from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any, Optional

from .contracts import (
    DatasetSchema,
    ExperimentRecord,
    ForestConfiguration,
    HyperparameterSpace,
    ModelManifest,
    PreparedDataset,
    TaskRecord,
    TrainingJobRecord,
    TrainingRequest,
    ValidationMetrics,
    WorkerProgressSnapshot, TreeArtifactMetadata,
)
from .enums import ExperimentStatus, JobStatus, ModelStatus,TaskStatus, TreeStatus
from .storage_layout import StorageLayout
from typing import Any, Optional
from common.contracts import ModelManifest


class JsonFileStore:
    """Small atomic JSON store suitable for the first project milestone.

    The purpose is not performance; it is deterministic persistence and recoverability.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()

    def write_json(self, path: str | Path, payload: dict[str, Any]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_suffix(path.suffix + ".tmp")
        with self._lock:
            with temp_path.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, sort_keys=True)
            temp_path.replace(path)

    def read_json(self, path: str | Path) -> dict[str, Any]:
        path = Path(path)
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def exists(self, path: str | Path) -> bool:
        return Path(path).exists()


class SharedArtifactStore:
    def __init__(self, root: str | Path, json_store: Optional[JsonFileStore] = None):
        self.layout = StorageLayout(root)
        self.json_store = json_store or JsonFileStore()
        Path(root).mkdir(parents=True, exist_ok=True)

    def write_json(self, path: str | Path, payload: dict[str, Any]) -> None:
        self.json_store.write_json(path, payload)

    def read_json(self, path: str | Path) -> dict[str, Any]:
        return self.json_store.read_json(path)

    def exists(self, path: str | Path) -> bool:
        return self.json_store.exists(path)


def _job_status_from_raw(value: str | JobStatus) -> JobStatus:
    if isinstance(value, JobStatus):
        return value
    return JobStatus(value)


def _experiment_status_from_raw(value: str | ExperimentStatus) -> ExperimentStatus:
    if isinstance(value, ExperimentStatus):
        return value
    return ExperimentStatus(value)


def _hyperparameter_space_from_dict(payload: dict[str, Any]) -> HyperparameterSpace:
    return HyperparameterSpace(
        n_estimators_candidates=payload["n_estimators_candidates"],
        max_depth_candidates=payload["max_depth_candidates"],
        max_features_candidates=payload["max_features_candidates"],
        min_samples_split_candidates=payload["min_samples_split_candidates"],
        min_samples_leaf_candidates=payload["min_samples_leaf_candidates"],
        criterion_candidates=payload["criterion_candidates"],
        bootstrap=payload["bootstrap"],
        global_random_seed=payload["global_random_seed"],
    )


def _training_request_from_dict(payload: dict[str, Any]) -> TrainingRequest:
    return TrainingRequest(
        job_id=payload["job_id"],
        dataset_uri=payload["dataset_uri"],
        target_column=payload["target_column"],
        task_type=payload["task_type"],
        hyperparameter_space=_hyperparameter_space_from_dict(payload["hyperparameter_space"]),
        n_estimators_total=payload["n_estimators_total"],
        validation_ratio=payload["validation_ratio"],
        test_ratio=payload["test_ratio"],
        global_random_seed=payload["global_random_seed"],
        bootstrap=payload["bootstrap"],
        created_at=payload.get("created_at", 0.0),
    )


def _dataset_schema_from_dict(payload: dict[str, Any]) -> DatasetSchema:
    return DatasetSchema(
        dataset_uri=payload["dataset_uri"],
        target_column=payload["target_column"],
        feature_names=payload["feature_names"],
        task_type=payload["task_type"],
        label_mapping=payload.get("label_mapping"),
        preprocessing_uri=payload.get("preprocessing_uri"),
    )


def _prepared_dataset_from_dict(payload: dict[str, Any]) -> PreparedDataset:
    return PreparedDataset(
        dataset_id=payload["dataset_id"],
        schema=_dataset_schema_from_dict(payload["schema"]),
        train_features_uri=payload["train_features_uri"],
        train_labels_uri=payload["train_labels_uri"],
        validation_features_uri=payload["validation_features_uri"],
        validation_labels_uri=payload["validation_labels_uri"],
        test_features_uri=payload["test_features_uri"],
        test_labels_uri=payload["test_labels_uri"],
        class_labels=payload.get("class_labels"),
        n_features=payload["n_features"],
        n_train=payload["n_train"],
        n_validation=payload["n_validation"],
        n_test=payload["n_test"],
    )


def _forest_configuration_from_dict(payload: dict[str, Any]) -> ForestConfiguration:
    return ForestConfiguration(
        experiment_id=payload["experiment_id"],
        task_type=payload["task_type"],
        n_estimators=payload["n_estimators"],
        max_depth=payload.get("max_depth"),
        max_features=payload.get("max_features"),
        min_samples_split=payload["min_samples_split"],
        min_samples_leaf=payload["min_samples_leaf"],
        criterion=payload["criterion"],
        bootstrap=payload["bootstrap"],
        global_random_seed=payload["global_random_seed"],
    )


def _validation_metrics_from_dict(payload: dict[str, Any]) -> ValidationMetrics:
    return ValidationMetrics(
        experiment_id=payload["experiment_id"],
        accuracy=payload["accuracy"],
        classification_report=payload["classification_report"],
        confusion_matrix=payload["confusion_matrix"],
        feature_importances=payload["feature_importances"],
        evaluated_at=payload.get("evaluated_at", 0.0),
    )


def _experiment_record_from_dict(payload: dict[str, Any]) -> ExperimentRecord:
    validation_metrics_payload = payload.get("validation_metrics")
    validation_metrics = None
    if validation_metrics_payload is not None:
        validation_metrics = _validation_metrics_from_dict(validation_metrics_payload)

    return ExperimentRecord(
        experiment_id=payload["experiment_id"],
        forest_config=_forest_configuration_from_dict(payload["forest_config"]),
        status=_experiment_status_from_raw(payload["status"]),
        assigned_workers=payload["assigned_workers"],
        expected_tree_count=payload["expected_tree_count"],
        completed_tree_count=payload["completed_tree_count"],
        validation_metrics=validation_metrics,
    )


def _training_job_record_from_dict(payload: dict[str, Any]) -> TrainingJobRecord:
    prepared_dataset_payload = payload.get("prepared_dataset")
    prepared_dataset = None
    if prepared_dataset_payload is not None:
        prepared_dataset = _prepared_dataset_from_dict(prepared_dataset_payload)

    return TrainingJobRecord(
        job_id=payload["job_id"],
        status=_job_status_from_raw(payload["status"]),
        training_request=_training_request_from_dict(payload["training_request"]),
        prepared_dataset=prepared_dataset,
        experiment_ids=payload.get("experiment_ids", []),
        selected_experiment_id=payload.get("selected_experiment_id"),
        model_id=payload.get("model_id"),
        message=payload.get("message", ""),
        created_at=payload.get("created_at", 0.0),
        updated_at=payload.get("updated_at", 0.0),
    )


class JobRepository:
    def __init__(self, artifact_store: SharedArtifactStore):
        self.artifact_store = artifact_store

    # --------------------------------------------------------
    # basic persistence
    # --------------------------------------------------------

    def save(self, record: TrainingJobRecord) -> None:
        record.updated_at = time.time()
        path = self.artifact_store.layout.job_record_path(record.job_id)
        self.artifact_store.write_json(path, record.to_dict())

    def exists(self, job_id: str) -> bool:
        path = self.artifact_store.layout.job_record_path(job_id)
        return self.artifact_store.exists(path)

    def load_raw(self, job_id: str) -> Optional[dict[str, Any]]:
        path = self.artifact_store.layout.job_record_path(job_id)
        if not self.artifact_store.exists(path):
            return None
        return self.artifact_store.read_json(path)

    def load(self, job_id: str) -> Optional[TrainingJobRecord]:
        payload = self.load_raw(job_id)
        if payload is None:
            return None
        return _training_job_record_from_dict(payload)

    def list_jobs(self) -> list[TrainingJobRecord]:
        jobs_root = self.artifact_store.layout.root / "jobs"
        if not jobs_root.exists():
            return []

        records: list[TrainingJobRecord] = []

        for job_dir in jobs_root.iterdir():
            if not job_dir.is_dir():
                continue

            job_id = job_dir.name
            record = self.load(job_id)
            if record is not None:
                records.append(record)

        records.sort(key=lambda item: item.created_at)
        return records

    # --------------------------------------------------------
    # internal update helper
    # --------------------------------------------------------

    def _update_job(
        self,
        job_id: str,
        *,
        status: Optional[JobStatus] = None,
        message: Optional[str] = None,
        prepared_dataset: Optional[PreparedDataset] = None,
        selected_experiment_id: Optional[str] = None,
        model_id: Optional[str] = None,
        append_experiment_id: Optional[str] = None,
    ) -> TrainingJobRecord:
        record = self.load(job_id)
        if record is None:
            raise ValueError(f"Job '{job_id}' not found")

        if status is not None:
            record.status = status

        if message is not None:
            record.message = message

        if prepared_dataset is not None:
            record.prepared_dataset = prepared_dataset

        if selected_experiment_id is not None:
            record.selected_experiment_id = selected_experiment_id

        if model_id is not None:
            record.model_id = model_id

        if append_experiment_id is not None and append_experiment_id not in record.experiment_ids:
            record.experiment_ids.append(append_experiment_id)

        self.save(record)
        return record

    # --------------------------------------------------------
    # experiment persistence
    # --------------------------------------------------------

    def save_experiment(self, job_id: str, record: ExperimentRecord) -> None:
        path = self.artifact_store.layout.experiment_record_path(job_id, record.experiment_id)
        self.artifact_store.write_json(path, record.to_dict())

        self._update_job(
            job_id,
            append_experiment_id=record.experiment_id,
        )

    def load_experiment_raw(self, job_id: str, experiment_id: str) -> Optional[dict[str, Any]]:
        path = self.artifact_store.layout.experiment_record_path(job_id, experiment_id)
        if not self.artifact_store.exists(path):
            return None
        return self.artifact_store.read_json(path)

    def load_experiment(self, job_id: str, experiment_id: str) -> Optional[ExperimentRecord]:
        payload = self.load_experiment_raw(job_id, experiment_id)
        if payload is None:
            return None
        return _experiment_record_from_dict(payload)

    def list_experiments(self, job_id: str) -> list[ExperimentRecord]:
        job_record = self.load(job_id)
        if job_record is None:
            return []

        experiments: list[ExperimentRecord] = []
        for experiment_id in job_record.experiment_ids:
            experiment = self.load_experiment(job_id, experiment_id)
            if experiment is not None:
                experiments.append(experiment)

        return experiments

    def update_experiment_status(
        self,
        job_id: str,
        experiment_id: str,
        status: ExperimentStatus,
        *,
        completed_tree_count: Optional[int] = None,
        assigned_workers: Optional[list[str]] = None,
        validation_metrics: Optional[ValidationMetrics] = None,
    ) -> ExperimentRecord:
        record = self.load_experiment(job_id, experiment_id)
        if record is None:
            raise ValueError(
                f"Experiment '{experiment_id}' not found for job '{job_id}'"
            )

        record.status = status

        if completed_tree_count is not None:
            record.completed_tree_count = completed_tree_count

        if assigned_workers is not None:
            record.assigned_workers = list(assigned_workers)

        if validation_metrics is not None:
            record.validation_metrics = validation_metrics

        self.save_experiment(job_id, record)
        return record

    # --------------------------------------------------------
    # explicit job state helpers
    # --------------------------------------------------------

    def update_job_status(
        self,
        job_id: str,
        status: JobStatus,
        message: Optional[str] = None,
        selected_experiment_id: Optional[str] = None,
        model_id: Optional[str] = None,
    ) -> None:
        self._update_job(
            job_id,
            status=status,
            message=message,
            selected_experiment_id=selected_experiment_id,
            model_id=model_id,
        )

    def mark_pending(self, job_id: str, message: Optional[str] = None) -> TrainingJobRecord:
        return self._update_job(
            job_id,
            status=JobStatus.PENDING,
            message=message,
        )

    def mark_running(self, job_id: str, message: Optional[str] = None) -> TrainingJobRecord:
        return self._update_job(
            job_id,
            status=JobStatus.RUNNING,
            message=message,
        )

    def mark_completed(
        self,
        job_id: str,
        *,
        selected_experiment_id: Optional[str] = None,
        model_id: Optional[str] = None,
        message: Optional[str] = None,
    ) -> TrainingJobRecord:
        return self._update_job(
            job_id,
            status=JobStatus.COMPLETED,
            message=message,
            selected_experiment_id=selected_experiment_id,
            model_id=model_id,
        )

    def mark_failed(self, job_id: str, message: str) -> TrainingJobRecord:
        return self._update_job(
            job_id,
            status=JobStatus.FAILED,
            message=message,
        )

    # --------------------------------------------------------
    # explicit field helpers
    # --------------------------------------------------------

    def attach_prepared_dataset(self, job_id: str, prepared_dataset: PreparedDataset) -> TrainingJobRecord:
        return self._update_job(
            job_id,
            prepared_dataset=prepared_dataset,
        )

    def add_experiment_id(self, job_id: str, experiment_id: str) -> TrainingJobRecord:
        return self._update_job(
            job_id,
            append_experiment_id=experiment_id,
        )

    def set_selected_experiment(self, job_id: str, experiment_id: str) -> TrainingJobRecord:
        return self._update_job(
            job_id,
            selected_experiment_id=experiment_id,
        )

    def set_model_id(self, job_id: str, model_id: str) -> TrainingJobRecord:
        return self._update_job(
            job_id,
            model_id=model_id,
        )

def _model_status_from_raw(value: str | ModelStatus) -> ModelStatus:
    if isinstance(value, ModelStatus):
        return value
    return ModelStatus(value)


def _tree_status_from_raw(value: str | TreeStatus) -> TreeStatus:
    if isinstance(value, TreeStatus):
        return value
    return TreeStatus(value)


def _tree_artifact_metadata_from_dict(payload: dict[str, Any]) -> TreeArtifactMetadata:
    return TreeArtifactMetadata(
        tree_id=payload["tree_id"],
        job_id=payload["job_id"],
        experiment_id=payload["experiment_id"],
        task_id=payload["task_id"],
        tree_index=payload["tree_index"],
        worker_id=payload["worker_id"],
        seed=payload["seed"],
        artifact_uri=payload["artifact_uri"],
        status=_tree_status_from_raw(payload["status"]),
        training_time_seconds=payload.get("training_time_seconds", 0.0),
    )


def _model_manifest_from_dict(payload: dict[str, Any]) -> ModelManifest:
    return ModelManifest(
        model_id=payload["model_id"],
        job_id=payload["job_id"],
        experiment_id=payload["experiment_id"],
        model_type=payload["model_type"],
        forest_config=_forest_configuration_from_dict(payload["forest_config"]),
        class_labels=payload.get("class_labels", []),
        feature_names=payload.get("feature_names", []),
        target_column=payload["target_column"],
        train_features_uri=payload["train_features_uri"],
        train_labels_uri=payload["train_labels_uri"],
        validation_features_uri=payload["validation_features_uri"],
        validation_labels_uri=payload["validation_labels_uri"],
        test_features_uri=payload["test_features_uri"],
        test_labels_uri=payload["test_labels_uri"],
        tree_artifacts=[
            _tree_artifact_metadata_from_dict(item)
            for item in payload.get("tree_artifacts", [])
        ],
        validation_metrics=_validation_metrics_from_dict(payload["validation_metrics"]),
        test_metrics=payload.get("test_metrics"),
        created_at=payload.get("created_at", 0.0),
        status=_model_status_from_raw(payload["status"]),
    )

class ModelRepository:
    def __init__(self, artifact_store: SharedArtifactStore):
        self.artifact_store = artifact_store

    # --------------------------------------------------------
    # basic persistence
    # --------------------------------------------------------

    def save(self, manifest: ModelManifest) -> None:
        path = self.artifact_store.layout.model_manifest_path(manifest.model_id)
        self.artifact_store.write_json(path, manifest.to_dict())

    def exists(self, model_id: str) -> bool:
        path = self.artifact_store.layout.model_manifest_path(model_id)
        return self.artifact_store.exists(path)

    def load_raw(self, model_id: str) -> Optional[dict[str, Any]]:
        path = self.artifact_store.layout.model_manifest_path(model_id)
        if not self.artifact_store.exists(path):
            return None
        return self.artifact_store.read_json(path)

    def load(self, model_id: str) -> Optional[ModelManifest]:
        payload = self.load_raw(model_id)
        if payload is None:
            return None
        return _model_manifest_from_dict(payload)

    def list_models(self) -> list[ModelManifest]:
        models_root = self.artifact_store.layout.root / "models"
        if not models_root.exists():
            return []

        manifests: list[ModelManifest] = []

        for model_dir in models_root.iterdir():
            if not model_dir.is_dir():
                continue

            model_id = model_dir.name
            manifest = self.load(model_id)
            if manifest is not None:
                manifests.append(manifest)

        manifests.sort(key=lambda item: item.model_id)
        return manifests

    # --------------------------------------------------------
    # status updates
    # --------------------------------------------------------

    def update_status(self, model_id: str, status: ModelStatus) -> ModelManifest:
        manifest = self.load(model_id)
        if manifest is None:
            raise ValueError(f"Model '{model_id}' not found")

        manifest.status = status
        self.save(manifest)
        return manifest

    def mark_ready(self, model_id: str) -> ModelManifest:
        return self.update_status(model_id, ModelStatus.READY)

    def mark_failed(self, model_id: str) -> ModelManifest:
        return self.update_status(model_id, ModelStatus.FAILED)

    def mark_training(self, model_id: str) -> ModelManifest:
        return self.update_status(model_id, ModelStatus.TRAINING)

def _task_status_from_raw(value: str | TaskStatus) -> TaskStatus:
    if isinstance(value, TaskStatus):
        return value
    return TaskStatus(value)


def _task_record_from_dict(payload: dict[str, Any]) -> TaskRecord:
    return TaskRecord(
        task_id=payload["task_id"],
        attempt_id=payload["attempt_id"],
        job_id=payload["job_id"],
        experiment_id=payload["experiment_id"],
        worker_id=payload["worker_id"],
        status=_task_status_from_raw(payload["status"]),
        tree_ids=payload.get("tree_ids", []),
        completed_tree_ids=payload.get("completed_tree_ids", []),
        failed_tree_ids=payload.get("failed_tree_ids", []),
        lease_expires_at_ts=payload["lease_expires_at_ts"],
        updated_at=payload.get("updated_at", 0.0),
        error_message=payload.get("error_message"),
    )


class TaskLedger:
    """
    Persistent task ledger grouped by logical task_id and versioned by attempt_id.

    JSON shape per job:
    {
      "tasks": {
        "<task_id>": {
          "task_id": "<task_id>",
          "job_id": "<job_id>",
          "experiment_id": "<experiment_id>",
          "latest_attempt_id": 2,
          "updated_at": 1712345678.12,
          "attempts": {
            "1": { ... TaskRecord.to_dict() ... },
            "2": { ... TaskRecord.to_dict() ... }
          }
        }
      }
    }

    Compatibility notes:
    - save(record) / upsert(record) keep the current external API.
    - load(job_id, task_id) returns the latest attempt by default.
    - mark_running / mark_completed / mark_failed keep accepting only task_id,
      so the current orchestrator can continue to work.
    - If an old flat ledger exists, it is transparently normalized in memory
      the first time it is read.
    """

    def __init__(self, artifact_store: SharedArtifactStore):
        self.artifact_store = artifact_store
        self._lock = threading.Lock()

    # --------------------------------------------------------
    # internal persistence
    # --------------------------------------------------------

    def _empty_payload(self) -> dict[str, Any]:
        return {"tasks": {}}

    def _read_all(self, job_id: str) -> dict[str, Any]:
        path = self.artifact_store.layout.task_ledger_path(job_id)
        if not self.artifact_store.exists(path):
            return self._empty_payload()

        raw_payload = self.artifact_store.read_json(path)
        return self._normalize_payload(raw_payload)

    def mark_completed(
            self,
            task_id: str,
            completed_tree_ids: list[str],
            attempt_id: Optional[int] = None,
            job_id: Optional[str] = None,
    ) -> None:
        def _updater(raw: dict[str, Any]) -> None:
            existing = raw.get("completed_tree_ids", [])
            merged = list(dict.fromkeys(existing + list(completed_tree_ids or [])))
            raw["completed_tree_ids"] = merged
            raw["failed_tree_ids"] = []
            raw["status"] = TaskStatus.COMPLETED.value
            raw["error_message"] = None

        self._update_attempt_raw(
            task_id=task_id,
            updater=_updater,
            attempt_id=attempt_id,
            job_id=job_id,
        )

    def _write_all(self, job_id: str, payload: dict[str, Any]) -> None:
        path = self.artifact_store.layout.task_ledger_path(job_id)
        self.artifact_store.write_json(path, payload)

    def _normalize_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        """
        Normalizes the on-disk JSON so both old and new formats are accepted.

        Old format:
        {
          "tasks": {
            "task-1": { ... TaskRecord ... }
          }
        }

        New format:
        {
          "tasks": {
            "task-1": {
              "task_id": "task-1",
              "job_id": "...",
              "experiment_id": "...",
              "latest_attempt_id": 2,
              "updated_at": ...,
              "attempts": {
                "1": { ... TaskRecord ... },
                "2": { ... TaskRecord ... }
              }
            }
          }
        }
        """
        tasks_payload = payload.get("tasks")
        if not isinstance(tasks_payload, dict):
            return self._empty_payload()

        normalized_tasks: dict[str, Any] = {}

        for task_id, raw_value in tasks_payload.items():
            if not isinstance(raw_value, dict):
                continue

            # New format already grouped by attempts
            if "attempts" in raw_value:
                attempts_payload = raw_value.get("attempts", {})
                normalized_attempts: dict[str, Any] = {}

                for attempt_key, attempt_raw in attempts_payload.items():
                    if not isinstance(attempt_raw, dict):
                        continue

                    attempt_id = attempt_raw.get("attempt_id")
                    if attempt_id is None:
                        try:
                            attempt_id = int(attempt_key)
                        except Exception:
                            continue

                    normalized_attempts[str(int(attempt_id))] = attempt_raw

                if normalized_attempts:
                    latest_attempt_id = raw_value.get("latest_attempt_id")
                    if latest_attempt_id is None:
                        latest_attempt_id = max(int(k) for k in normalized_attempts.keys())

                    first_attempt = normalized_attempts[str(min(int(k) for k in normalized_attempts.keys()))]

                    normalized_tasks[task_id] = {
                        "task_id": raw_value.get("task_id", task_id),
                        "job_id": raw_value.get("job_id", first_attempt.get("job_id")),
                        "experiment_id": raw_value.get("experiment_id", first_attempt.get("experiment_id")),
                        "latest_attempt_id": int(latest_attempt_id),
                        "updated_at": raw_value.get("updated_at", time.time()),
                        "attempts": normalized_attempts,
                    }
                continue

            # Old flat format: single TaskRecord directly under task_id
            attempt_id = int(raw_value.get("attempt_id", 1))
            normalized_tasks[task_id] = {
                "task_id": raw_value.get("task_id", task_id),
                "job_id": raw_value.get("job_id"),
                "experiment_id": raw_value.get("experiment_id"),
                "latest_attempt_id": attempt_id,
                "updated_at": raw_value.get("updated_at", time.time()),
                "attempts": {
                    str(attempt_id): raw_value,
                },
            }

        return {"tasks": normalized_tasks}

    # --------------------------------------------------------
    # internal task/attempt helpers
    # --------------------------------------------------------

    def _ensure_task_bucket(self, payload: dict[str, Any], record: TaskRecord) -> dict[str, Any]:
        tasks = payload.setdefault("tasks", {})
        bucket = tasks.get(record.task_id)

        if bucket is None:
            bucket = {
                "task_id": record.task_id,
                "job_id": record.job_id,
                "experiment_id": record.experiment_id,
                "latest_attempt_id": record.attempt_id,
                "updated_at": time.time(),
                "attempts": {},
            }
            tasks[record.task_id] = bucket
            return bucket

        if bucket.get("job_id") not in {None, record.job_id}:
            raise ValueError(
                f"Task '{record.task_id}' already exists for another job: "
                f"{bucket.get('job_id')} != {record.job_id}"
            )

        if bucket.get("experiment_id") not in {None, record.experiment_id}:
            raise ValueError(
                f"Task '{record.task_id}' already exists for another experiment: "
                f"{bucket.get('experiment_id')} != {record.experiment_id}"
            )

        bucket["job_id"] = record.job_id
        bucket["experiment_id"] = record.experiment_id
        bucket.setdefault("attempts", {})
        return bucket

    def _resolve_attempt_id(
        self,
        bucket: dict[str, Any],
        attempt_id: Optional[int] = None,
    ) -> Optional[int]:
        attempts = bucket.get("attempts", {})
        if not attempts:
            return None

        if attempt_id is not None:
            if str(attempt_id) not in attempts:
                return None
            return int(attempt_id)

        latest_attempt_id = bucket.get("latest_attempt_id")
        if latest_attempt_id is not None and str(latest_attempt_id) in attempts:
            return int(latest_attempt_id)

        return max(int(key) for key in attempts.keys())

    def _find_task_bucket(
        self,
        task_id: str,
        attempt_id: Optional[int] = None,
        job_id: Optional[str] = None,
    ) -> tuple[Optional[str], Optional[dict[str, Any]], Optional[int]]:
        if job_id is not None:
            payload = self._read_all(job_id)
            bucket = payload.get("tasks", {}).get(task_id)
            if bucket is None:
                return None, None, None
            resolved_attempt_id = self._resolve_attempt_id(bucket, attempt_id)
            return job_id, bucket, resolved_attempt_id

        jobs_root = self.artifact_store.layout.root / "jobs"
        if not jobs_root.exists():
            return None, None, None

        for job_dir in jobs_root.iterdir():
            if not job_dir.is_dir():
                continue

            current_job_id = job_dir.name
            payload = self._read_all(current_job_id)
            bucket = payload.get("tasks", {}).get(task_id)
            if bucket is None:
                continue

            resolved_attempt_id = self._resolve_attempt_id(bucket, attempt_id)
            return current_job_id, bucket, resolved_attempt_id

        return None, None, None

    def _update_attempt_raw(
        self,
        task_id: str,
        updater,
        *,
        attempt_id: Optional[int] = None,
        job_id: Optional[str] = None,
    ) -> None:
        with self._lock:
            resolved_job_id, bucket, resolved_attempt_id = self._find_task_bucket(
                task_id=task_id,
                attempt_id=attempt_id,
                job_id=job_id,
            )
            if resolved_job_id is None or bucket is None or resolved_attempt_id is None:
                raise ValueError(
                    f"Task '{task_id}'"
                    + (f" attempt '{attempt_id}'" if attempt_id is not None else "")
                    + " not found"
                )

            payload = self._read_all(resolved_job_id)
            task_bucket = payload.setdefault("tasks", {}).get(task_id)
            if task_bucket is None:
                raise ValueError(f"Task '{task_id}' not found in job '{resolved_job_id}'")

            attempts = task_bucket.setdefault("attempts", {})
            attempt_key = str(resolved_attempt_id)
            raw = attempts.get(attempt_key)
            if raw is None:
                raise ValueError(f"Task '{task_id}' attempt '{resolved_attempt_id}' not found")

            updater(raw)

            raw["updated_at"] = time.time()
            attempts[attempt_key] = raw
            task_bucket["updated_at"] = raw["updated_at"]

            latest_attempt_id = task_bucket.get("latest_attempt_id")
            if latest_attempt_id is None or int(resolved_attempt_id) >= int(latest_attempt_id):
                task_bucket["latest_attempt_id"] = int(resolved_attempt_id)

            payload["tasks"][task_id] = task_bucket
            self._write_all(resolved_job_id, payload)

    # --------------------------------------------------------
    # public CRUD
    # --------------------------------------------------------

    def save(self, record: TaskRecord) -> None:
        with self._lock:
            payload = self._read_all(record.job_id)
            bucket = self._ensure_task_bucket(payload, record)

            attempts = bucket.setdefault("attempts", {})
            attempts[str(record.attempt_id)] = record.to_dict()

            latest_attempt_id = bucket.get("latest_attempt_id")
            if latest_attempt_id is None or int(record.attempt_id) >= int(latest_attempt_id):
                bucket["latest_attempt_id"] = int(record.attempt_id)

            bucket["updated_at"] = time.time()
            payload.setdefault("tasks", {})[record.task_id] = bucket
            self._write_all(record.job_id, payload)

    def upsert(self, record: TaskRecord) -> None:
        self.save(record)

    def load_raw(
        self,
        job_id: str,
        task_id: str,
        attempt_id: Optional[int] = None,
    ) -> Optional[dict[str, Any]]:
        payload = self._read_all(job_id)
        bucket = payload.get("tasks", {}).get(task_id)
        if bucket is None:
            return None

        resolved_attempt_id = self._resolve_attempt_id(bucket, attempt_id)
        if resolved_attempt_id is None:
            return None

        return bucket.get("attempts", {}).get(str(resolved_attempt_id))

    def load(
        self,
        job_id: str,
        task_id: str,
        attempt_id: Optional[int] = None,
    ) -> Optional[TaskRecord]:
        raw = self.load_raw(job_id, task_id, attempt_id)
        if raw is None:
            return None
        return _task_record_from_dict(raw)

    def load_latest_attempt(self, job_id: str, task_id: str) -> Optional[TaskRecord]:
        return self.load(job_id, task_id, attempt_id=None)

    def load_attempt(self, job_id: str, task_id: str, attempt_id: int) -> Optional[TaskRecord]:
        return self.load(job_id, task_id, attempt_id=attempt_id)

    def list_raw(self, job_id: str) -> list[dict[str, Any]]:
        """
        Returns the latest attempt for each logical task.
        This keeps behavior intuitive for existing callers.
        """
        payload = self._read_all(job_id)
        result: list[dict[str, Any]] = []

        for task_id, bucket in payload.get("tasks", {}).items():
            resolved_attempt_id = self._resolve_attempt_id(bucket)
            if resolved_attempt_id is None:
                continue

            raw = bucket.get("attempts", {}).get(str(resolved_attempt_id))
            if raw is not None:
                result.append(raw)

        return result

    def list(self, job_id: str) -> list[TaskRecord]:
        return [_task_record_from_dict(item) for item in self.list_raw(job_id)]

    def list_attempts_raw(self, job_id: str, task_id: str) -> list[dict[str, Any]]:
        payload = self._read_all(job_id)
        bucket = payload.get("tasks", {}).get(task_id)
        if bucket is None:
            return []

        attempts = bucket.get("attempts", {})
        ordered_keys = sorted((int(key) for key in attempts.keys()))
        return [attempts[str(key)] for key in ordered_keys]

    def list_attempts(self, job_id: str, task_id: str) -> list[TaskRecord]:
        return [_task_record_from_dict(item) for item in self.list_attempts_raw(job_id, task_id)]

    def list_all_attempts(self, job_id: str) -> list[TaskRecord]:
        payload = self._read_all(job_id)
        records: list[TaskRecord] = []

        for task_id in payload.get("tasks", {}).keys():
            records.extend(self.list_attempts(job_id, task_id))

        return records

    def list_by_experiment(self, job_id: str, experiment_id: str) -> list[TaskRecord]:
        """
        Returns the latest attempt per task filtered by experiment.
        """
        return [
            record
            for record in self.list(job_id)
            if record.experiment_id == experiment_id
        ]

    def list_attempts_by_experiment(self, job_id: str, experiment_id: str) -> list[TaskRecord]:
        return [
            record
            for record in self.list_all_attempts(job_id)
            if record.experiment_id == experiment_id
        ]

    # --------------------------------------------------------
    # public state transitions
    # --------------------------------------------------------

    def mark_running(
        self,
        task_id: str,
        attempt_id: Optional[int] = None,
        job_id: Optional[str] = None,
    ) -> None:
        def _updater(raw: dict[str, Any]) -> None:
            raw["status"] = TaskStatus.RUNNING.value
            raw["error_message"] = None

        self._update_attempt_raw(
            task_id=task_id,
            updater=_updater,
            attempt_id=attempt_id,
            job_id=job_id,
        )

    def mark_completed(
        self,
        task_id: str,
        completed_tree_ids: list[str],
        attempt_id: Optional[int] = None,
        job_id: Optional[str] = None,
    ) -> None:
        def _updater(raw: dict[str, Any]) -> None:
            existing = raw.get("completed_tree_ids", [])
            merged = list(dict.fromkeys(existing + completed_tree_ids))
            raw["completed_tree_ids"] = merged
            raw["status"] = TaskStatus.COMPLETED.value
            raw["error_message"] = None

        self._update_attempt_raw(
            task_id=task_id,
            updater=_updater,
            attempt_id=attempt_id,
            job_id=job_id,
        )

    def mark_failed(
        self,
        task_id: str,
        error_message: str,
        attempt_id: Optional[int] = None,
        job_id: Optional[str] = None,
    ) -> None:
        def _updater(raw: dict[str, Any]) -> None:
            raw["status"] = TaskStatus.FAILED.value
            raw["error_message"] = error_message

        self._update_attempt_raw(
            task_id=task_id,
            updater=_updater,
            attempt_id=attempt_id,
            job_id=job_id,
        )

    # --------------------------------------------------------
    # public queries / aggregation
    # --------------------------------------------------------

    def latest_attempt_id(self, job_id: str, task_id: str) -> Optional[int]:
        payload = self._read_all(job_id)
        bucket = payload.get("tasks", {}).get(task_id)
        if bucket is None:
            return None
        resolved = self._resolve_attempt_id(bucket)
        return int(resolved) if resolved is not None else None

    def count_completed_trees(
        self,
        job_id: str,
        experiment_id: Optional[str] = None,
    ) -> int:
        return len(self.completed_tree_ids(job_id, experiment_id))

    def is_task_completed(self, job_id: str, task_id: str) -> bool:
        """
        A logical task is considered completed if at least one of its attempts completed.
        """
        for record in self.list_attempts(job_id, task_id):
            if record.status == TaskStatus.COMPLETED:
                return True
        return False

    def completed_tree_ids(
        self,
        job_id: str,
        experiment_id: Optional[str] = None,
    ) -> list[str]:
        """
        Returns unique completed tree IDs across all attempts.
        This avoids double-counting the same tree on retries.
        """
        all_ids: list[str] = []

        for record in self.list_all_attempts(job_id):
            if experiment_id is not None and record.experiment_id != experiment_id:
                continue
            all_ids.extend(record.completed_tree_ids)

        return list(dict.fromkeys(all_ids))

    def failed_tree_ids(
        self,
        job_id: str,
        experiment_id: Optional[str] = None,
    ) -> list[str]:
        all_ids: list[str] = []

        for record in self.list_all_attempts(job_id):
            if experiment_id is not None and record.experiment_id != experiment_id:
                continue
            all_ids.extend(record.failed_tree_ids)

        return list(dict.fromkeys(all_ids))

    def missing_tree_ids(
        self,
        job_id: str,
        experiment_id: str,
        expected_tree_ids: list[str],
    ) -> list[str]:
        completed = set(self.completed_tree_ids(job_id, experiment_id))
        return [tree_id for tree_id in expected_tree_ids if tree_id not in completed]

class WorkerProgressStore:
    def __init__(self, artifact_store: SharedArtifactStore):
        self.artifact_store = artifact_store

    def save_snapshot(self, job_id: str, snapshot: WorkerProgressSnapshot) -> None:
        path = self.artifact_store.layout.worker_progress_path(job_id, snapshot.worker_id)
        self.artifact_store.write_json(path, snapshot.to_dict())

    def load_snapshot_raw(self, job_id: str, worker_id: str) -> Optional[dict[str, Any]]:
        path = self.artifact_store.layout.worker_progress_path(job_id, worker_id)
        if not self.artifact_store.exists(path):
            return None
        return self.artifact_store.read_json(path)
