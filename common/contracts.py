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
"""
Contratti di dominio condivisi tra master, worker e layer di persistenza.

Il modulo definisce le strutture dati serializzabili usate per training,
inference, monitoring, recovery e pubblicazione del modello.
"""

@dataclass(slots=True)
class HyperparameterSpace:
    # spazio degli hyperparametri esplorabile durante la pianificazione degli exp di train
    n_estimators_candidates: list[int]
    max_depth_candidates: list[Optional[int]]
    max_features_candidates: list[str | int | float | None]
    min_samples_split_candidates: list[int]
    min_samples_leaf_candidates: list[int]
    criterion_candidates: list[str]
    bootstrap: bool = True
    global_random_seed: int = 42

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ForestConfiguration:
    """
       Configurazione concreta di una foresta per un singolo esperimento.

       A differenza di HyperparameterSpace, qui ogni parametro ha già
       un valore fissato e pronto per essere usato nel training.
       """
    experiment_id: str
    task_type: str
    n_estimators: int
    max_depth: Optional[int]
    max_features: str | int | float | None
    min_samples_split: int
    min_samples_leaf: int
    criterion: str
    bootstrap: bool
    global_random_seed: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class TrainingRequest:
    """
    Richiesta logica di training inviata dal client al master.

    Contiene il riferimento al dataset, il target, i parametri globali
    del job e lo spazio di ricerca degli iperparametri.
    """
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
    dataset_scenario: str = "baseline_original"
    leakage_columns: Optional[list[str]] = None
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

#introduco per baseline no leakage, estendo per dati da tenere keep e rimuovere drop
@dataclass(slots=True)
class DatasetPreparationMetadata:
    """
    Metadata del preprocessing e della preparazione del dataset.

    Serve a tracciare in modo esplicito:
    - scenario applicato;
    - colonne richieste in keep/drop/leakage;
    - colonne effettivamente mantenute o rimosse;
    - eventuali richieste non soddisfatte;
    - dimensioni iniziali e finali del dataset.
    """
    dataset_scenario: str = "baseline_original"
    scenario_type: str = "none"

    requested_drop_columns: Optional[list[str]] = None
    dropped_columns: list[str] = field(default_factory=list)
    missing_requested_drop_columns: list[str] = field(default_factory=list)

    requested_keep_columns: Optional[list[str]] = None
    kept_columns: list[str] = field(default_factory=list)
    missing_requested_keep_columns: list[str] = field(default_factory=list)

    requested_leakage_columns: Optional[list[str]] = None
    missing_requested_leakage_columns: list[str] = field(default_factory=list)

    original_column_count: int = 0
    final_column_count: int = 0
    original_row_count: int = 0
    final_row_count: int = 0

    scenario_report_uri: Optional[str] = None
    scenario_parameters: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

@dataclass(slots=True)
class DatasetSchema:
    """
    Schema logico del dataset preparato.

    Descrive il target, le feature finali e gli eventuali artefatti
    di preprocessing necessari per training e inferenza.
    """
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
    """
    Dataset già validato, trasformato e suddiviso nei tre split standard:
    train, validation e test.

    Questo è il formato che il master passa alla pipeline di training.
    """
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
    preparation_metadata: DatasetPreparationMetadata = field(
        default_factory=DatasetPreparationMetadata
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

@dataclass(slots=True)
class TrainingShard:
    """
    Unità di lavoro assegnata a un worker durante il training distribuito.

    Ogni shard identifica:
    - il task e la sua attempt;
    - il worker destinatario;
    - il sottoinsieme di alberi da addestrare;
    - i riferimenti ai dati di training;
    - i parametri della foresta da usare.
    """
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
    """
    Metadata persistiti per un singolo albero addestrato.

    Non contiene l'albero, ma le informazioni necessarie
    per identificarlo, localizzarlo e verificarne lo stato.
    """
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
    feature_importances: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["status"] = self.status.value
        return payload

    @staticmethod
    def from_dict(data: dict) -> "TreeArtifactMetadata":
        #riscostruisce i metadata da una rappresentazione serializzata
        return TreeArtifactMetadata(
            tree_id=data["tree_id"],
            job_id=data["job_id"],
            experiment_id=data["experiment_id"],
            task_id=data["task_id"],
            tree_index=data["tree_index"],
            worker_id=data["worker_id"],
            seed=data["seed"],
            artifact_uri=data["artifact_uri"],
            status=TreeStatus(data["status"]),
            training_time_seconds=data["training_time_seconds"],
            feature_importances=data.get("feature_importances", []),
        )

@dataclass(slots=True)
class ShardTrainingResult:
    """
    Risultato restituito da un worker dopo l'esecuzione di uno shard di training.

    Riporta sia gli artifact prodotti sia il dettaglio degli alberi completati
    o falliti, così il master può aggiornare ledger e recovery state.
    """
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
    """
    Contenitore uniforme delle metriche di valutazione.

    Include sia metriche di classificazione sia metriche di regressione,
    così lo stesso contratto può essere riusato in task_type diversi.
    """
    # Classification metrics
    accuracy: Optional[float] = None
    balanced_accuracy: Optional[float] = None

    macro_precision: Optional[float] = None
    macro_recall: Optional[float] = None
    macro_f1: Optional[float] = None

    weighted_precision: Optional[float] = None
    weighted_recall: Optional[float] = None
    weighted_f1: Optional[float] = None

    classification_report: Optional[dict[str, Any]] = None
    confusion_matrix: Optional[list[list[int]]] = None

    # Regression metrics
    mae: Optional[float] = None
    mse: Optional[float] = None
    rmse: Optional[float] = None
    r2: Optional[float] = None

    # Shared metrics / diagnostics
    feature_importances: list[float] = field(default_factory=list)
    feature_importances_by_name: dict[str, float] = field(default_factory=dict)

    evaluated_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ExperimentRecord:
    """
    Stato persistito di un singolo esperimento.

    Un job può contenere più esperimenti, ciascuno con una diversa
    configurazione della foresta.
    """
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
    """
    Stato persistito dell'intero training job.

    Tiene insieme richiesta iniziale, dataset preparato, esperimenti creati,
    modello selezionato e messaggi di stato leggibili dal client.
    """
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
    """
    Descrizione completa del modello pubblicato.

    Il manifest è la vista logica globale del modello:
    dice quali artifact lo compongono, con quali feature è stato costruito,
    su quali split è stato valutato e con quali metriche.
    """
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
    preparation_metadata: DatasetPreparationMetadata = field(
        default_factory=DatasetPreparationMetadata
    )
    created_at: float = field(default_factory=time.time)
    status: ModelStatus = ModelStatus.TRAINING

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["status"] = self.status.value
        payload["tree_artifacts"] = [item.to_dict() for item in self.tree_artifacts]
        payload["validation_metrics"] = self.validation_metrics.to_dict()
        payload["preparation_metadata"] = self.preparation_metadata.to_dict()
        return payload


@dataclass(slots=True)
class MasterCommand:
    """
     Comando logico del control plane del master.

     È il formato adatto a rappresentare decisioni critiche da tracciare
     o, in futuro, da replicare formalmente tramite consenso.
     """
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
    """
    Snapshot del progresso locale osservato su un worker.

    Serve al master per monitorare avanzamento, stalli e possibili
    recovery dei task distribuiti.
    """
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
    """
     Stato persistito di un task/shard assegnato a un worker.

     Include attempt-aware bookkeeping, lease e dettaglio degli alberi
     completati o falliti in quello specifico task.
     """
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
