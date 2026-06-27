from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


TaskType = Literal["classification", "regression"]


EFS_DATASETS_ROOT = "/mnt/efs/gp_artifacts/datasets"

DEFAULT_CLASSIFICATION_CRITERION = "gini"
DEFAULT_REGRESSION_CRITERION = "squared_error"


@dataclass(frozen=True)
class TrainingPreset:
    key: str
    label: str
    dataset_path: str
    task_type: TaskType
    target_column: str
    dataset_scenario: str
    leakage_columns: tuple[str, ...]
    criterion: str

    def to_dict(self) -> dict[str, object]:
        return {
            "key": self.key,
            "label": self.label,
            "dataset_path": self.dataset_path,
            "task_type": self.task_type,
            "target_column": self.target_column,
            "dataset_scenario": self.dataset_scenario,
            "leakage_columns": list(self.leakage_columns),
            "criterion": self.criterion,
        }


def dataset_path(filename: str) -> str:
    return f"{EFS_DATASETS_ROOT}/{filename}"


TRAINING_PRESETS: dict[str, TrainingPreset] = {
    "real_classification": TrainingPreset(
        key="real_classification",
        label="real diabetes classification",
        dataset_path=dataset_path("diabetes_dataset.csv"),
        task_type="classification",
        target_column="diagnosed_diabetes",
        dataset_scenario="baseline_no_leakage",
        leakage_columns=(
            "diabetes_stage",
            "diabetes_risk_score",
        ),
        criterion=DEFAULT_CLASSIFICATION_CRITERION,
    ),
    "real_classification_no_diagnostic_features": TrainingPreset(
        key="real_classification_no_diagnostic_features",
        label="real diabetes classification - no diagnostic features",
        dataset_path=dataset_path("diabetes_dataset.csv"),
        task_type="classification",
        target_column="diagnosed_diabetes",
        dataset_scenario="no_diagnostic_features",
        leakage_columns=(
            "diabetes_stage",
            "diabetes_risk_score",
        ),
        criterion=DEFAULT_CLASSIFICATION_CRITERION,
    ),
    "real_classification_no_diagnostic_extended": TrainingPreset(
        key="real_classification_no_diagnostic_extended",
        label="real diabetes classification - no diagnostic extended",
        dataset_path=dataset_path("diabetes_dataset.csv"),
        task_type="classification",
        target_column="diagnosed_diabetes",
        dataset_scenario="no_diagnostic_extended",
        leakage_columns=(
            "diabetes_stage",
            "diabetes_risk_score",
        ),
        criterion=DEFAULT_CLASSIFICATION_CRITERION,
    ),
    "real_classification_clinical_only": TrainingPreset(
        key="real_classification_clinical_only",
        label="real diabetes classification - clinical only",
        dataset_path=dataset_path("diabetes_dataset.csv"),
        task_type="classification",
        target_column="diagnosed_diabetes",
        dataset_scenario="clinical_only",
        leakage_columns=(
            "diabetes_stage",
            "diabetes_risk_score",
        ),
        criterion=DEFAULT_CLASSIFICATION_CRITERION,
    ),
    "real_classification_glucose_only": TrainingPreset(
        key="real_classification_glucose_only",
        label="real diabetes classification - glucose only",
        dataset_path=dataset_path("diabetes_dataset.csv"),
        task_type="classification",
        target_column="diagnosed_diabetes",
        dataset_scenario="glucose_only",
        leakage_columns=(
            "diabetes_stage",
            "diabetes_risk_score",
        ),
        criterion=DEFAULT_CLASSIFICATION_CRITERION,
    ),
    "real_classification_noise_10": TrainingPreset(
        key="real_classification_noise_10",
        label="real diabetes classification - diagnostic noise 10%",
        dataset_path=dataset_path("diabetes_dataset.csv"),
        task_type="classification",
        target_column="diagnosed_diabetes",
        dataset_scenario="diagnostic_noise_10pct",
        leakage_columns=(
            "diabetes_stage",
            "diabetes_risk_score",
        ),
        criterion=DEFAULT_CLASSIFICATION_CRITERION,
    ),
    "real_classification_noise_25": TrainingPreset(
        key="real_classification_noise_25",
        label="real diabetes classification - diagnostic noise 25%",
        dataset_path=dataset_path("diabetes_dataset.csv"),
        task_type="classification",
        target_column="diagnosed_diabetes",
        dataset_scenario="diagnostic_noise_25pct",
        leakage_columns=(
            "diabetes_stage",
            "diabetes_risk_score",
        ),
        criterion=DEFAULT_CLASSIFICATION_CRITERION,
    ),
    "real_classification_noise_50": TrainingPreset(
        key="real_classification_noise_50",
        label="real diabetes classification - diagnostic noise 50%",
        dataset_path=dataset_path("diabetes_dataset.csv"),
        task_type="classification",
        target_column="diagnosed_diabetes",
        dataset_scenario="diagnostic_noise_50pct",
        leakage_columns=(
            "diabetes_stage",
            "diabetes_risk_score",
        ),
        criterion=DEFAULT_CLASSIFICATION_CRITERION,
    ),
    "real_classification_imbalance_positive_80": TrainingPreset(
        key="real_classification_imbalance_positive_80",
        label="real diabetes classification - 80% positives",
        dataset_path=dataset_path("diabetes_dataset.csv"),
        task_type="classification",
        target_column="diagnosed_diabetes",
        dataset_scenario="imbalance_positive_80",
        leakage_columns=(
            "diabetes_stage",
            "diabetes_risk_score",
        ),
        criterion=DEFAULT_CLASSIFICATION_CRITERION,
    ),
    "real_classification_imbalance_positive_90": TrainingPreset(
        key="real_classification_imbalance_positive_90",
        label="real diabetes classification - 90% positives",
        dataset_path=dataset_path("diabetes_dataset.csv"),
        task_type="classification",
        target_column="diagnosed_diabetes",
        dataset_scenario="imbalance_positive_90",
        leakage_columns=(
            "diabetes_stage",
            "diabetes_risk_score",
        ),
        criterion=DEFAULT_CLASSIFICATION_CRITERION,
    ),
    "real_classification_imbalance_negative_80": TrainingPreset(
        key="real_classification_imbalance_negative_80",
        label="real diabetes classification - 80% negatives",
        dataset_path=dataset_path("diabetes_dataset.csv"),
        task_type="classification",
        target_column="diagnosed_diabetes",
        dataset_scenario="imbalance_negative_80",
        leakage_columns=(
            "diabetes_stage",
            "diabetes_risk_score",
        ),
        criterion=DEFAULT_CLASSIFICATION_CRITERION,
    ),
    "real_stage_multiclass": TrainingPreset(
        key="real_stage_multiclass",
        label="real diabetes stage multi-class classification",
        dataset_path=dataset_path("diabetes_dataset.csv"),
        task_type="classification",
        target_column="diabetes_stage",
        dataset_scenario="stage_multiclass_no_leakage",
        leakage_columns=(
            "diagnosed_diabetes",
            "diabetes_risk_score",
        ),
        criterion=DEFAULT_CLASSIFICATION_CRITERION,
    ),
    "real_regression": TrainingPreset(
        key="real_regression",
        label="real diabetes regression",
        dataset_path=dataset_path("diabetes_dataset.csv"),
        task_type="regression",
        target_column="diabetes_risk_score",
        dataset_scenario="baseline_no_leakage",
        leakage_columns=(
            "diagnosed_diabetes",
            "diabetes_stage",
        ),
        criterion=DEFAULT_REGRESSION_CRITERION,
    ),
    "synthetic_classification_100k": TrainingPreset(
        key="synthetic_classification_100k",
        label="synthetic classification 100000x40",
        dataset_path=dataset_path(
            "synthetic_classification_100000_samples_40_features.csv"
        ),
        task_type="classification",
        target_column="target",
        dataset_scenario="baseline_original",
        leakage_columns=(),
        criterion=DEFAULT_CLASSIFICATION_CRITERION,
    ),
    "synthetic_regression_100k": TrainingPreset(
        key="synthetic_regression_100k",
        label="synthetic regression 100000x40",
        dataset_path=dataset_path(
            "synthetic_regression_100000_samples_40_features.csv"
        ),
        task_type="regression",
        target_column="target",
        dataset_scenario="baseline_original",
        leakage_columns=(),
        criterion=DEFAULT_REGRESSION_CRITERION,
    ),
    "synthetic_classification_2m": TrainingPreset(
        key="synthetic_classification_2m",
        label="synthetic classification 2000000x47",
        dataset_path=dataset_path(
            "synthetic_classification_2000000_samples_47_features.csv"
        ),
        task_type="classification",
        target_column="target",
        dataset_scenario="baseline_original",
        leakage_columns=(),
        criterion=DEFAULT_CLASSIFICATION_CRITERION,
    ),
    "synthetic_regression_2m": TrainingPreset(
        key="synthetic_regression_2m",
        label="synthetic regression 2000000x42",
        dataset_path=dataset_path(
            "synthetic_regression_2000000_samples_42_features.csv"
        ),
        task_type="regression",
        target_column="target",
        dataset_scenario="baseline_original",
        leakage_columns=(),
        criterion=DEFAULT_REGRESSION_CRITERION,
    ),
}


def get_training_preset(key: str) -> TrainingPreset:
    try:
        return TRAINING_PRESETS[key]
    except KeyError as exc:
        available = ", ".join(TRAINING_PRESETS.keys())
        raise ValueError(
            f"Unknown training preset '{key}'. Available presets: {available}"
        ) from exc


def get_training_presets_by_task(task_type: TaskType) -> list[TrainingPreset]:
    return [
        preset
        for preset in TRAINING_PRESETS.values()
        if preset.task_type == task_type
    ]


def list_training_preset_keys(task_type: TaskType | None = None) -> list[str]:
    if task_type is None:
        return list(TRAINING_PRESETS.keys())

    return [
        key
        for key, preset in TRAINING_PRESETS.items()
        if preset.task_type == task_type
    ]


def build_custom_training_preset(
    *,
    dataset_path_value: str,
    task_type: TaskType,
    target_column: str,
    dataset_scenario: str = "baseline_original",
    leakage_columns: list[str] | tuple[str, ...] | None = None,
    criterion: str | None = None,
) -> TrainingPreset:
    if not dataset_path_value.strip():
        raise ValueError("dataset_path_value cannot be empty")

    if not target_column.strip():
        raise ValueError("target_column cannot be empty")

    if criterion is None:
        if task_type == "classification":
            criterion = DEFAULT_CLASSIFICATION_CRITERION
        else:
            criterion = DEFAULT_REGRESSION_CRITERION

    return TrainingPreset(
        key="custom",
        label="custom",
        dataset_path=dataset_path_value.strip(),
        task_type=task_type,
        target_column=target_column.strip(),
        dataset_scenario=dataset_scenario.strip() or "baseline_original",
        leakage_columns=tuple(leakage_columns or ()),
        criterion=criterion,
    )