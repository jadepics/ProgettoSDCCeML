from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

from .enums import (
    CommandType,
    ExperimentStatus,
    JobStatus,
    ModelStatus,
    TaskStatus,
    TreeStatus,
)


@dataclass(slots=True)
class HyperparameterSpace:
    n_estimators_candidates: list[int]
    max_depth_candidates: list[Optional[int]]
    max_features_candidates: list[str | float | None]
    min_samples_split_candidates: list[int]
    min_samples_leaf_candidates: list[int]
    criterion_candidates: list[str]
    bootstrap: bool = True
    global_random_seed: int = 42

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ForestConfiguration:
    experiment_id: str
    task_type: str
    n_estimators: int
    max_depth: Optional[int]
    max_features: str | float | None
    min_samples_split: int
    min_samples_leaf: int
    criterion: str
    bootstrap: bool
    global_random_seed: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class TrainingRequest:
    job_id: str
    dataset_uri: str
    target_column: str
    task_type: str
    hyperparameter_space: HyperparameterSpace
    n_estimators_total: int
    validation_ratio: float
    test_ratio: float
    global_random_seed: int
    bootstrap: bool
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class DatasetSchema:
    dataset_uri: str
    target_column: str
    feature_names: list[str]
    task_type: str
    label_mapping: Optional[dict[str, int]] = None
    preprocessing_uri: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class PreparedDataset:
    dataset_id: str
    schema: DatasetSchema
    train_features_uri: str
    train_labels_uri: str
    validation_features_uri: str
    validation_labels_uri: str
    test_features_uri: str
    test_labels_uri: str
    class_labels: Optional[list[str]]
    n_features: int
    n_train: int
    n_validation: int
    n_test: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class TrainingShard:
    task_id: str
    attempt_id: int
    job_id: str
    experiment_id: str
    assigned_worker_id: str
    tree_start_index: int
    tree_count: int
    forest_config: ForestConfiguration
    train_features_uri: str
    train_labels_uri: str
    artifact_output_dir: str
    seed_base: int
    lease_expires_at_ts: float

    @property
    def tree_end_index(self) -> int:
        return self.tree_start_index + self.tree_count - 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class TreeArtifactMetadata:
    tree_id: str
    job_id: str
    experiment_id: str
    task_id: str
    tree_index: int
    worker_id: str
    seed: int
    artifact_uri: str
    status: TreeStatus
    training_time_seconds: float

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["status"] = self.status.value
        return payload


@dataclass(slots=True)
class ShardTrainingResult:
    task_id: str
    attempt_id: int
    worker_id: str
    success: bool
    tree_artifacts: list[TreeArtifactMetadata]
    completed_tree_ids: list[str]
    failed_tree_ids: list[str]
    completed_tree_count: int
    failed_tree_count: int
    error_message: Optional[str]
    elapsed_time_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "attempt_id": self.attempt_id,
            "worker_id": self.worker_id,
            "success": self.success,
            "tree_artifacts": [item.to_dict() for item in self.tree_artifacts],
            "completed_tree_ids": self.completed_tree_ids,
            "failed_tree_ids": self.failed_tree_ids,
            "completed_tree_count": self.completed_tree_count,
            "failed_tree_count": self.failed_tree_count,
            "error_message": self.error_message,
            "elapsed_time_seconds": self.elapsed_time_seconds,
        }


@dataclass(slots=True)
class ValidationMetrics:
    experiment_id: str
    accuracy: float
    classification_report: dict[str, Any]
    confusion_matrix: list[list[int]]
    feature_importances: list[float]
    evaluated_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ExperimentRecord:
    experiment_id: str
    forest_config: ForestConfiguration
    status: ExperimentStatus
    assigned_workers: list[str]
    expected_tree_count: int
    completed_tree_count: int
    validation_metrics: Optional[ValidationMetrics] = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["status"] = self.status.value
        if self.validation_metrics is not None:
            payload["validation_metrics"] = self.validation_metrics.to_dict()
        return payload


@dataclass(slots=True)
class TrainingJobRecord:
    job_id: str
    status: JobStatus
    training_request: TrainingRequest
    prepared_dataset: Optional[PreparedDataset]
    experiment_ids: list[str]
    selected_experiment_id: Optional[str]
    model_id: Optional[str]
    message: str
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["status"] = self.status.value
        return payload


@dataclass(slots=True)
class ModelManifest:
    model_id: str
    job_id: str
    experiment_id: str
    model_type: str
    forest_config: ForestConfiguration
    class_labels: list[str]
    feature_names: list[str]
    target_column: str
    train_features_uri: str
    train_labels_uri: str
    validation_features_uri: str
    validation_labels_uri: str
    test_features_uri: str
    test_labels_uri: str
    tree_artifacts: list[TreeArtifactMetadata]
    validation_metrics: ValidationMetrics
    test_metrics: Optional[dict[str, Any]]
    created_at: float = field(default_factory=time.time)
    status: ModelStatus = ModelStatus.TRAINING

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["status"] = self.status.value
        payload["tree_artifacts"] = [item.to_dict() for item in self.tree_artifacts]
        payload["validation_metrics"] = self.validation_metrics.to_dict()
        return payload


@dataclass(slots=True)
class MasterCommand:
    command_id: str
    job_id: str
    command_type: CommandType
    payload: dict[str, Any]
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["command_type"] = self.command_type.value
        return payload


@dataclass(slots=True)
class WorkerProgressSnapshot:
    worker_id: str
    task_id: str
    experiment_id: str
    completed_tree_ids: list[str]
    running_tree_ids: list[str]
    failed_tree_ids: list[str]
    last_update_ts: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class TaskRecord:
    task_id: str
    attempt_id: int
    job_id: str
    experiment_id: str
    worker_id: str
    status: TaskStatus
    tree_ids: list[str]
    completed_tree_ids: list[str]
    failed_tree_ids: list[str]
    lease_expires_at_ts: float
    updated_at: float = field(default_factory=time.time)
    error_message: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["status"] = self.status.value
        return payload
