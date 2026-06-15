from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from synthetic_data.synthetic_dataset_generator import (
    SyntheticClassificationConfig,
    SyntheticRegressionConfig,
    generate_synthetic_classification_dataset,
    generate_synthetic_regression_dataset,
)


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


def ask_float(label: str, default: float) -> float:
    while True:
        raw = ask(label, str(default))

        try:
            return float(raw)
        except ValueError:
            print("[ERROR] Insert a number")


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


def default_classification_output_path(
    n_samples: int,
    n_features: int,
) -> str:
    return (
        "Dataset/"
        f"synthetic_classification_{n_samples}_samples_"
        f"{n_features}_features.csv"
    )


def default_regression_output_path(
    n_samples: int,
    n_features: int,
) -> str:
    return (
        "Dataset/"
        f"synthetic_regression_{n_samples}_samples_"
        f"{n_features}_features.csv"
    )


def print_metadata_summary(metadata: dict) -> None:
    print()
    print("SYNTHETIC DATASET GENERATED")
    print("=" * 80)
    print(f"dataset_type:     {metadata['dataset_type']}")
    print(f"generator:        {metadata['generator']}")
    print(f"output_csv:       {metadata['output_csv_resolved']}")
    print(f"n_rows:           {metadata['n_rows']}")
    print(f"n_features:       {metadata['n_features']}")
    print(f"target_column:    {metadata['target_column']}")

    if metadata["dataset_type"] == "synthetic_classification":
        print(f"class_distribution: {metadata['class_distribution']}")
    else:
        print(f"target_mean:      {metadata['target_mean']:.6f}")
        print(f"target_std:       {metadata['target_std']:.6f}")
        print(f"target_min:       {metadata['target_min']:.6f}")
        print(f"target_max:       {metadata['target_max']:.6f}")

    print()


def run_classification_generation_flow() -> None:
    print()
    print("QUICK SYNTHETIC CLASSIFICATION DATASET")
    print("=" * 80)

    n_samples = ask_int("Number of samples", 100_000)
    n_features = ask_int("Number of features", 40)
    n_informative = ask_int("Number of informative features", 20)
    n_redundant = ask_int("Number of redundant features", 10)

    default_output = default_classification_output_path(
        n_samples=n_samples,
        n_features=n_features,
    )

    output_csv = ask("Output CSV path", default_output)

    config = SyntheticClassificationConfig(
        output_csv=output_csv,
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=n_redundant,
        n_classes=2,
        class_sep=1.0,
        flip_y=0.01,
        random_seed=42,
        target_column="target",
    )

    metadata = generate_synthetic_classification_dataset(config)
    print_metadata_summary(metadata)


def run_regression_generation_flow() -> None:
    print()
    print("QUICK SYNTHETIC REGRESSION DATASET")
    print("=" * 80)

    n_samples = ask_int("Number of samples", 100_000)
    n_features = ask_int("Number of features", 40)
    n_informative = ask_int("Number of informative features", 20)
    noise = ask_float("Noise", 10.0)

    default_output = default_regression_output_path(
        n_samples=n_samples,
        n_features=n_features,
    )

    output_csv = ask("Output CSV path", default_output)

    config = SyntheticRegressionConfig(
        output_csv=output_csv,
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        noise=noise,
        random_seed=42,
        target_column="target",
    )

    metadata = generate_synthetic_regression_dataset(config)
    print_metadata_summary(metadata)


def main() -> None:
    while True:
        print()
        print("===================================")
        print("SYNTHETIC DATASET CLI")
        print("===================================")
        print("1 -> Generate synthetic classification dataset")
        print("2 -> Generate synthetic regression dataset")
        print("0 -> Exit")

        choice = input("\nSelect option: ").strip()

        if choice == "1":
            run_classification_generation_flow()

        elif choice == "2":
            run_regression_generation_flow()

        elif choice == "0":
            print("Bye.")
            return

        else:
            print("[ERROR] Invalid option")


if __name__ == "__main__":
    main()