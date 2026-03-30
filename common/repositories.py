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

    def save(self, record: TrainingJobRecord) -> None:
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

    def save_experiment(self, job_id: str, record: ExperimentRecord) -> None:
        path = self.artifact_store.layout.experiment_record_path(job_id, record.experiment_id)
        self.artifact_store.write_json(path, record.to_dict())

        job_record = self.load(job_id)
        if job_record is None:
            raise ValueError(f"Cannot save experiment for missing job '{job_id}'")

        if record.experiment_id not in job_record.experiment_ids:
            job_record.experiment_ids.append(record.experiment_id)
            job_record.updated_at = time.time()
            self.save(job_record)

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

    def update_job_status(
        self,
        job_id: str,
        status: JobStatus,
        message: Optional[str] = None,
        selected_experiment_id: Optional[str] = None,
        model_id: Optional[str] = None,
    ) -> None:
        record = self.load(job_id)
        if record is None:
            raise ValueError(f"Job '{job_id}' not found")

        record.status = status
        if message is not None:
            record.message = message
        if selected_experiment_id is not None:
            record.selected_experiment_id = selected_experiment_id
        if model_id is not None:
            record.model_id = model_id
        record.updated_at = time.time()

        self.save(record)

    def attach_prepared_dataset(self, job_id: str, prepared_dataset: PreparedDataset) -> None:
        record = self.load(job_id)
        if record is None:
            raise ValueError(f"Job '{job_id}' not found")

        record.prepared_dataset = prepared_dataset
        record.updated_at = time.time()
        self.save(record)

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

    def save(self, manifest: ModelManifest) -> None:
        path = self.artifact_store.layout.model_manifest_path(manifest.model_id)
        self.artifact_store.write_json(path, manifest.to_dict())

    def save_manifest(self, manifest: ModelManifest) -> None:
        self.save(manifest)

    def exists(self, model_id: str) -> bool:
        path = self.artifact_store.layout.model_manifest_path(model_id)
        return self.artifact_store.exists(path)

    def load_raw(self, model_id: str) -> Optional[dict[str, Any]]:
        path = self.artifact_store.layout.model_manifest_path(model_id)
        if not self.artifact_store.exists(path):
            return None
        return self.artifact_store.read_json(path)

    def load_manifest_raw(self, model_id: str) -> Optional[dict[str, Any]]:
        return self.load_raw(model_id)

    def load(self, model_id: str) -> Optional[ModelManifest]:
        payload = self.load_raw(model_id)
        if payload is None:
            return None
        return _model_manifest_from_dict(payload)

    def load_manifest(self, model_id: str) -> Optional[ModelManifest]:
        return self.load(model_id)

    def mark_ready(self, model_id: str) -> None:
        manifest = self.load(model_id)
        if manifest is None:
            raise ValueError(f"Model '{model_id}' not found")

        manifest.status = ModelStatus.READY
        self.save(manifest)


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
    def __init__(self, artifact_store: SharedArtifactStore):
        self.artifact_store = artifact_store
        self._lock = threading.Lock()

    def _read_all(self, job_id: str) -> dict[str, Any]:
        path = self.artifact_store.layout.task_ledger_path(job_id)
        if not self.artifact_store.exists(path):
            return {"tasks": {}}
        return self.artifact_store.read_json(path)

    def _write_all(self, job_id: str, payload: dict[str, Any]) -> None:
        path = self.artifact_store.layout.task_ledger_path(job_id)
        self.artifact_store.write_json(path, payload)

    def save(self, record: TaskRecord) -> None:
        with self._lock:
            payload = self._read_all(record.job_id)
            payload.setdefault("tasks", {})[record.task_id] = record.to_dict()
            self._write_all(record.job_id, payload)

    def upsert(self, record: TaskRecord) -> None:
        self.save(record)

    def load_raw(self, job_id: str, task_id: str) -> Optional[dict[str, Any]]:
        payload = self._read_all(job_id)
        return payload.get("tasks", {}).get(task_id)

    def load(self, job_id: str, task_id: str) -> Optional[TaskRecord]:
        raw = self.load_raw(job_id, task_id)
        if raw is None:
            return None
        return _task_record_from_dict(raw)

    def list_raw(self, job_id: str) -> list[dict[str, Any]]:
        payload = self._read_all(job_id)
        return list(payload.get("tasks", {}).values())

    def list(self, job_id: str) -> list[TaskRecord]:
        return [_task_record_from_dict(item) for item in self.list_raw(job_id)]

    def list_by_experiment(self, job_id: str, experiment_id: str) -> list[TaskRecord]:
        return [
            record
            for record in self.list(job_id)
            if record.experiment_id == experiment_id
        ]

    def mark_completed(self, task_id: str, completed_tree_ids: list[str]) -> None:
        with self._lock:
            job_id, raw = self._find_task_raw(task_id)
            if raw is None or job_id is None:
                raise ValueError(f"Task '{task_id}' not found")

            completed_set = list(dict.fromkeys(raw.get("completed_tree_ids", []) + completed_tree_ids))
            raw["completed_tree_ids"] = completed_set
            raw["status"] = TaskStatus.COMPLETED.value
            raw["updated_at"] = time.time()
            raw["error_message"] = None

            payload = self._read_all(job_id)
            payload.setdefault("tasks", {})[task_id] = raw
            self._write_all(job_id, payload)

    def mark_failed(self, task_id: str, error_message: str) -> None:
        with self._lock:
            job_id, raw = self._find_task_raw(task_id)
            if raw is None or job_id is None:
                raise ValueError(f"Task '{task_id}' not found")

            raw["status"] = TaskStatus.FAILED.value
            raw["error_message"] = error_message
            raw["updated_at"] = time.time()

            payload = self._read_all(job_id)
            payload.setdefault("tasks", {})[task_id] = raw
            self._write_all(job_id, payload)

    def mark_running(self, task_id: str) -> None:
        with self._lock:
            job_id, raw = self._find_task_raw(task_id)
            if raw is None or job_id is None:
                raise ValueError(f"Task '{task_id}' not found")

            raw["status"] = TaskStatus.RUNNING.value
            raw["updated_at"] = time.time()

            payload = self._read_all(job_id)
            payload.setdefault("tasks", {})[task_id] = raw
            self._write_all(job_id, payload)

    def count_completed_trees(self, job_id: str) -> int:
        count = 0
        for record in self.list(job_id):
            count += len(record.completed_tree_ids)
        return count

    def is_task_completed(self, job_id: str, task_id: str) -> bool:
        record = self.load(job_id, task_id)
        if record is None:
            return False
        return record.status == TaskStatus.COMPLETED

    def completed_tree_ids(self, job_id: str, experiment_id: Optional[str] = None) -> list[str]:
        records = self.list(job_id)
        if experiment_id is not None:
            records = [record for record in records if record.experiment_id == experiment_id]

        all_ids: list[str] = []
        for record in records:
            all_ids.extend(record.completed_tree_ids)

        return list(dict.fromkeys(all_ids))

    def _find_task_raw(self, task_id: str) -> tuple[Optional[str], Optional[dict[str, Any]]]:
        jobs_root = self.artifact_store.layout.root / "jobs"
        if not jobs_root.exists():
            return None, None

        for job_dir in jobs_root.iterdir():
            if not job_dir.is_dir():
                continue

            job_id = job_dir.name
            payload = self._read_all(job_id)
            raw = payload.get("tasks", {}).get(task_id)
            if raw is not None:
                return job_id, raw

        return None, None

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
