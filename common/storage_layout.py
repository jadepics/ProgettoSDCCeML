from __future__ import annotations

from pathlib import Path


class StorageLayout:
    """Deterministic storage layout shared by masterPackage and workers.

    This is the first class both developers should use, because it freezes the
    naming convention of all artifacts and prevents path drift between masterPackage- and
    worker-side implementations.
    """

    def __init__(self, root: str | Path):
        self.root = Path(root)

    def jobs_dir(self) -> Path:
        return self.root / "jobs"

    def job_dir(self, job_id: str) -> Path:
        return self.jobs_dir() / job_id

    def job_record_path(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "job_record.json"

    def task_ledger_path(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "task_ledger.json"

    def worker_progress_dir(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "worker_progress"

    def worker_progress_path(self, job_id: str, worker_id: str) -> Path:
        return self.worker_progress_dir(job_id) / f"{worker_id}.json"

    def prepared_dataset_dir(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "prepared_dataset"

    def dataset_schema_path(self, job_id: str) -> Path:
        return self.prepared_dataset_dir(job_id) / "schema.json"

    def train_features_path(self, job_id: str) -> Path:
        return self.prepared_dataset_dir(job_id) / "train_features.parquet"

    def train_labels_path(self, job_id: str) -> Path:
        return self.prepared_dataset_dir(job_id) / "train_labels.parquet"

    def validation_features_path(self, job_id: str) -> Path:
        return self.prepared_dataset_dir(job_id) / "validation_features.parquet"

    def validation_labels_path(self, job_id: str) -> Path:
        return self.prepared_dataset_dir(job_id) / "validation_labels.parquet"

    def test_features_path(self, job_id: str) -> Path:
        return self.prepared_dataset_dir(job_id) / "test_features.parquet"

    def test_labels_path(self, job_id: str) -> Path:
        return self.prepared_dataset_dir(job_id) / "test_labels.parquet"

    def experiments_dir(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "experiments"

    def experiment_dir(self, job_id: str, experiment_id: str) -> Path:
        return self.experiments_dir(job_id) / experiment_id

    def experiment_record_path(self, job_id: str, experiment_id: str) -> Path:
        return self.experiment_dir(job_id, experiment_id) / "experiment_record.json"

    def trees_dir(self, job_id: str, experiment_id: str) -> Path:
        return self.experiment_dir(job_id, experiment_id) / "trees"

    def tree_artifact_path(self, job_id: str, experiment_id: str, tree_id: str) -> Path:
        return self.trees_dir(job_id, experiment_id) / f"{tree_id}.joblib"

    def metrics_dir(self, job_id: str, experiment_id: str) -> Path:
        return self.experiment_dir(job_id, experiment_id) / "metrics"

    def validation_metrics_path(self, job_id: str, experiment_id: str) -> Path:
        return self.metrics_dir(job_id, experiment_id) / "validation_metrics.json"

    def models_dir(self) -> Path:
        return self.root / "models"

    def model_dir(self, model_id: str) -> Path:
        return self.models_dir() / model_id

    def model_manifest_path(self, model_id: str) -> Path:
        return self.model_dir(model_id) / "manifest.json"
