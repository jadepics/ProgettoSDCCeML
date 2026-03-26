from __future__ import annotations

import time
import uuid


def generate_job_id() -> str:
    return f"job_{uuid.uuid4().hex}"


def generate_model_id() -> str:
    return f"model_{uuid.uuid4().hex}"


def generate_dataset_id() -> str:
    return f"dataset_{uuid.uuid4().hex}"


def generate_command_id() -> str:
    return f"cmd_{uuid.uuid4().hex}"


def generate_experiment_id(job_id: str, ordinal: int) -> str:
    return f"{job_id}_exp_{ordinal:03d}"


def generate_task_id(experiment_id: str, tree_start_index: int, tree_count: int) -> str:
    end_index = tree_start_index + tree_count - 1
    return f"{experiment_id}_trees_{tree_start_index:06d}_{end_index:06d}"


def generate_tree_id(experiment_id: str, tree_index: int) -> str:
    return f"{experiment_id}_tree_{tree_index:06d}"


def generate_attempt_id(previous_attempts: int) -> int:
    return previous_attempts + 1


def tree_seed(global_seed: int, tree_index: int) -> int:
    return global_seed + tree_index


def now_ts() -> float:
    return time.time()
