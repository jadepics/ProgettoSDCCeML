from __future__ import annotations

import sys
from pathlib import Path

# Permette di lanciare questo file sia con:
# python -m local_baseline.local_baseline_cli
# sia direttamente da IDE / path assoluto.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from local_baseline.local_baseline_runner import (
    LocalBaselineConfig,
    run_local_baseline,
)


DEFAULT_DATASET_PATH = "Dataset/diabetes_dataset.csv"
DEFAULT_OUTPUT_DIR = "local_baseline/results"

DEFAULT_N_ESTIMATORS = 240
DEFAULT_MAX_DEPTH = 5
DEFAULT_MAX_FEATURES = "sqrt"
DEFAULT_MIN_SAMPLES_SPLIT = 2
DEFAULT_MIN_SAMPLES_LEAF = 1
DEFAULT_BOOTSTRAP = True
DEFAULT_VALIDATION_RATIO = 0.2
DEFAULT_TEST_RATIO = 0.2
DEFAULT_RANDOM_SEED = 42
DEFAULT_LOCAL_N_JOBS = 1

CLASSIFICATION_SCENARIO_CHOICES = [
    ("baseline_original", "BASELINE ORIGINAL"),
    ("baseline_no_leakage", "BASELINE NO LEAKAGE"),
    ("diagnostic_noise_10pct", "DIAGNOSTIC NOISE 10%"),
    ("diagnostic_noise_25pct", "DIAGNOSTIC NOISE 25%"),
    ("diagnostic_noise_50pct", "DIAGNOSTIC NOISE 50%"),
    ("imbalance_positive_80", "IMBALANCE POSITIVE 80%"),
    ("imbalance_positive_90", "IMBALANCE POSITIVE 90%"),
    ("imbalance_negative_80", "IMBALANCE NEGATIVE 80%"),
    ("stage_multiclass_no_leakage", "STAGE MULTICLASS NO LEAKAGE"),
    ("no_diagnostic_features", "NO DIAGNOSTIC FEATURES"),
    ("no_diagnostic_extended", "NO DIAGNOSTIC EXTENDED"),
    ("clinical_only", "CLINICAL ONLY"),
    ("glucose_only", "GLUCOSE ONLY"),
]


def default_classification_target_column(dataset_scenario: str) -> str:
    if dataset_scenario == "stage_multiclass_no_leakage":
        return "diabetes_stage"

    return "diagnosed_diabetes"


def default_classification_drop_columns(
    dataset_scenario: str,
    target_column: str,
) -> list[str]:
    """
    Drop automatici coerenti con gli scenari distribuiti.

    Nota:
    il target viene già rimosso automaticamente dal runner.
    Queste colonne servono solo come protezione ulteriore da leakage,
    soprattutto se il runner locale non applica già internamente
    tutte le regole dello scenario.
    """
    if target_column == "diabetes_stage":
        return [
            "diagnosed_diabetes",
            "diabetes_risk_score",
        ]

    if target_column == "diagnosed_diabetes":
        return [
            "diabetes_stage",
            "diabetes_risk_score",
        ]

    return []

def ask(label: str, default: str | None = None) -> str:
    if default is None:
        value = input(f"{label}: ").strip()
    else:
        value = input(f"{label} [{default}]: ").strip()

    if not value and default is not None:
        return default

    return value


def ask_int(label: str, default: int) -> int:
    while True:
        raw = ask(label, str(default))

        try:
            return int(raw)
        except ValueError:
            print("[ERROR] Insert an integer")


def ask_choice(title: str, choices: list[tuple[str, str]]) -> str:
    print()
    print(title)
    print("=" * len(title))

    for index, (_, label) in enumerate(choices, start=1):
        print(f"{index} -> {label}")

    while True:
        raw = input("\nSelect option: ").strip()

        try:
            selected_index = int(raw)
        except ValueError:
            print("[ERROR] Insert a valid number")
            continue

        if 1 <= selected_index <= len(choices):
            return choices[selected_index - 1][0]

        print("[ERROR] Invalid option")


def build_default_output_path(
    task_type: str,
    dataset_scenario: str,
    target_column: str,
    n_estimators: int,
) -> str:
    safe_target = (
        target_column
        .replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
    )

    filename = (
        f"local_{task_type}_{dataset_scenario}_"
        f"{safe_target}_{n_estimators}_trees.json"
    )

    return str(Path(DEFAULT_OUTPUT_DIR) / filename)


def default_regression_drop_columns(target_column: str) -> list[str]:
    """
    Drop automatici ragionevoli per evitare leakage nei task di regressione.

    Il target viene già rimosso automaticamente dal runner.
    """
    if target_column == "diabetes_risk_score":
        return [
            "diagnosed_diabetes",
            "diabetes_stage",
        ]

    if target_column == "hba1c":
        return [
            "diagnosed_diabetes",
            "diabetes_stage",
            "diabetes_risk_score",
        ]

    if target_column in {"glucose_fasting", "glucose_postprandial"}:
        return [
            "diagnosed_diabetes",
            "diabetes_stage",
            "diabetes_risk_score",
            "hba1c",
        ]

    return [
        "diagnosed_diabetes",
        "diabetes_stage",
        "diabetes_risk_score",
    ]


def print_quick_configuration(config: LocalBaselineConfig) -> None:
    print()
    print("[INFO] Local baseline configuration")
    print("-" * 80)
    print(f"dataset_url:              {config.dataset_url}")
    print(f"task_type:                {config.task_type}")
    print(f"target_column:            {config.target_column}")
    print(f"dataset_scenario:         {config.dataset_scenario}")
    print(f"extra_drop_columns:       {config.extra_drop_columns or []}")
    print(f"n_estimators:             {config.n_estimators}")
    print(f"max_depth:                {config.max_depth}")
    print(f"max_features:             {config.max_features}")
    print(f"min_samples_split:        {config.min_samples_split}")
    print(f"min_samples_leaf:         {config.min_samples_leaf}")

    if config.task_type == "classification":
        print(f"classification_criterion: {config.classification_criterion}")
        print(f"class_weight:             {config.class_weight}")
    else:
        print(f"regression_criterion:     {config.regression_criterion}")

    print(f"bootstrap:                {config.bootstrap}")
    print(f"validation_ratio:         {config.validation_ratio}")
    print(f"test_ratio:               {config.test_ratio}")
    print(f"random_seed:              {config.random_seed}")
    print(f"local n_jobs:             {config.n_jobs}")
    print(f"output_json:              {config.output_json}")
    print()


def print_result_summary(result: dict) -> None:
    config = result["config"]
    task_type = config["task_type"]

    print()
    print("LOCAL BASELINE RESULT")
    print("=" * 80)
    print(f"task_type:                   {task_type}")
    print(f"dataset_scenario:            {config['dataset_scenario']}")
    print(f"target_column:               {config['target_column']}")
    print(f"n_estimators:                {config['n_estimators']}")
    print(f"n_samples_total:             {result['n_samples_total']}")
    print(f"n_features_input:            {result['n_features_input']}")

    if "n_features_after_preprocessing" in result:
        print(
            "n_features_after_preproc:   "
            f"{result['n_features_after_preprocessing']}"
        )

    print(f"n_train:                     {result['n_train']}")
    print(f"n_validation:                {result['n_validation']}")
    print(f"n_test:                      {result['n_test']}")
    print(f"training_time_seconds:       {result['training_time_seconds']:.4f}")
    print(
        "validation_inference:       "
        f"{result['validation_inference_time_seconds']:.4f}"
    )
    print(
        "test_inference:             "
        f"{result['test_inference_time_seconds']:.4f}"
    )

    if task_type == "classification":
        print(
            "validation_accuracy:        "
            f"{result['validation_metrics']['accuracy']:.6f}"
        )
        print(
            "test_accuracy:              "
            f"{result['test_metrics']['accuracy']:.6f}"
        )
    else:
        validation_metrics = result["validation_metrics"]
        test_metrics = result["test_metrics"]

        print(f"validation_r2:              {validation_metrics['r2']:.6f}")
        print(f"validation_mae:             {validation_metrics['mae']:.6f}")
        print(f"validation_rmse:            {validation_metrics['rmse']:.6f}")
        print(f"test_r2:                    {test_metrics['r2']:.6f}")
        print(f"test_mae:                   {test_metrics['mae']:.6f}")
        print(f"test_rmse:                  {test_metrics['rmse']:.6f}")

    output_json = config.get("output_json")

    if output_json:
        print()
        print(f"Saved JSON result to: {output_json}")

    print()


def run_quick_classification_flow() -> None:
    dataset_scenario = ask_choice(
        "CHOOSE CLASSIFICATION DATASET SCENARIO",
        CLASSIFICATION_SCENARIO_CHOICES,
    )

    dataset_url = ask("Dataset path", DEFAULT_DATASET_PATH)

    default_target = default_classification_target_column(dataset_scenario)

    target_column = ask("Target column", default_target)

    n_estimators = ask_int("Number of trees", DEFAULT_N_ESTIMATORS)

    output_default = build_default_output_path(
        task_type="classification",
        dataset_scenario=dataset_scenario,
        target_column=target_column,
        n_estimators=n_estimators,
    )

    output_json = ask("Output JSON path", output_default)

    extra_drop_columns = default_classification_drop_columns(
        dataset_scenario=dataset_scenario,
        target_column=target_column,
    )

    config = LocalBaselineConfig(
        dataset_url=dataset_url,
        target_column=target_column,
        task_type="classification",
        dataset_scenario=dataset_scenario,
        extra_drop_columns=extra_drop_columns,

        n_estimators=n_estimators,
        max_depth=DEFAULT_MAX_DEPTH,
        max_features=DEFAULT_MAX_FEATURES,
        min_samples_split=DEFAULT_MIN_SAMPLES_SPLIT,
        min_samples_leaf=DEFAULT_MIN_SAMPLES_LEAF,

        classification_criterion="gini",
        regression_criterion="squared_error",

        bootstrap=DEFAULT_BOOTSTRAP,

        # Importante:
        # class_weight=None per avvicinarci al training distribuito,
        # dove non abbiamo class_weight="balanced" negli artifact.
        class_weight=None,

        validation_ratio=DEFAULT_VALIDATION_RATIO,
        test_ratio=DEFAULT_TEST_RATIO,
        random_seed=DEFAULT_RANDOM_SEED,

        # Baseline non distribuita: singolo processo locale.
        n_jobs=DEFAULT_LOCAL_N_JOBS,

        output_json=output_json,
    )

    print()
    print("[INFO] Classification extra drop columns used to reduce leakage:")
    print(extra_drop_columns)

    print_quick_configuration(config)

    result = run_local_baseline(config)
    print_result_summary(result)

def run_quick_regression_flow() -> None:
    dataset_url = ask("Dataset path", DEFAULT_DATASET_PATH)

    target_column = ask_choice(
        "CHOOSE REGRESSION TARGET",
        [
            ("diabetes_risk_score", "diabetes_risk_score"),
            ("hba1c", "hba1c"),
            ("glucose_fasting", "glucose_fasting"),
            ("bmi", "bmi"),
            ("cholesterol_total", "cholesterol_total"),
            ("custom", "custom target"),
        ],
    )

    if target_column == "custom":
        target_column = ask("Custom target column")

    dataset_scenario = ask_choice(
        "CHOOSE REGRESSION FEATURE SCENARIO",
        [
            ("baseline_original", "BASELINE ORIGINAL"),
            ("baseline_no_leakage", "BASELINE NO LEAKAGE"),
            ("no_diagnostic_features", "NO DIAGNOSTIC FEATURES"),
            ("no_diagnostic_extended", "NO DIAGNOSTIC EXTENDED"),
            ("clinical_only", "CLINICAL ONLY"),
            ("glucose_only", "GLUCOSE ONLY"),
        ],
    )

    n_estimators = ask_int("Number of trees", DEFAULT_N_ESTIMATORS)

    output_default = build_default_output_path(
        task_type="regression",
        dataset_scenario=dataset_scenario,
        target_column=target_column,
        n_estimators=n_estimators,
    )

    output_json = ask("Output JSON path", output_default)

    extra_drop_columns = default_regression_drop_columns(target_column)

    config = LocalBaselineConfig(
        dataset_url=dataset_url,
        target_column=target_column,
        task_type="regression",
        dataset_scenario=dataset_scenario,
        extra_drop_columns=extra_drop_columns,

        n_estimators=n_estimators,
        max_depth=DEFAULT_MAX_DEPTH,
        max_features=DEFAULT_MAX_FEATURES,
        min_samples_split=DEFAULT_MIN_SAMPLES_SPLIT,
        min_samples_leaf=DEFAULT_MIN_SAMPLES_LEAF,

        classification_criterion="gini",
        regression_criterion="squared_error",

        bootstrap=DEFAULT_BOOTSTRAP,
        class_weight=None,

        validation_ratio=DEFAULT_VALIDATION_RATIO,
        test_ratio=DEFAULT_TEST_RATIO,
        random_seed=DEFAULT_RANDOM_SEED,

        # Baseline non distribuita: singolo processo locale.
        n_jobs=DEFAULT_LOCAL_N_JOBS,

        output_json=output_json,
    )

    print()
    print("[INFO] Regression extra drop columns used to reduce leakage:")
    print(extra_drop_columns)

    print_quick_configuration(config)

    result = run_local_baseline(config)
    print_result_summary(result)


def main() -> None:
    while True:
        print()
        print("===================================")
        print("LOCAL BASELINE CLI")
        print("===================================")
        print("1 -> Quick classification baseline")
        print("2 -> Quick regression baseline")
        print("0 -> Exit")

        choice = input("\nSelect option: ").strip()

        if choice == "1":
            run_quick_classification_flow()

        elif choice == "2":
            run_quick_regression_flow()

        elif choice == "0":
            print("Bye.")
            return

        else:
            print("[ERROR] Invalid option")


if __name__ == "__main__":
    main()