from pathlib import Path
from typing import List, Optional, Union

import grpc

import rf_v2_pb2 as pb
import rf_v2_pb2_grpc as pbgrpc

from common.grpc_config import GRPC_OPTIONS

def main(
    PrivateIp_Port: str,
    dataset_path: Union[str, Path],
    n_estimators_total: int,
    dataset_scenario: str = "baseline_no_leakage",
    leakage_columns: Optional[List[str]] = None,
    target_column: str = "diabetes_risk_score",
    criterion: str = "squared_error",
    max_depth_candidates: Optional[List[int]] = None,
    max_features_candidates: Optional[List[str]] = None,
    min_samples_split_candidates: Optional[List[int]] = None,
    min_samples_leaf_candidates: Optional[List[int]] = None,
    criterion_candidates: Optional[List[str]] = None,
    bootstrap: bool = True,
    global_random_seed: int = 42,
):
    if leakage_columns is None:
        leakage_columns = []

    if max_depth_candidates is None:
        max_depth_candidates = [5]

    if max_features_candidates is None:
        max_features_candidates = ["sqrt"]

    if min_samples_split_candidates is None:
        min_samples_split_candidates = [2]

    if min_samples_leaf_candidates is None:
        min_samples_leaf_candidates = [1]

    if criterion_candidates is None:
        criterion_candidates = [criterion]

    channel = grpc.insecure_channel(PrivateIp_Port, options=GRPC_OPTIONS)
    stub = pbgrpc.CoordinatorServiceStub(channel)

    request = pb.SubmitTrainingRequest(
        dataset_url=str(dataset_path),
        target_column=target_column,
        task_type="regression",
        dataset_scenario=dataset_scenario,
        validation_ratio=0.2,
        test_ratio=0.2,
        bootstrap=bootstrap,
        global_random_seed=global_random_seed,

        max_depth_candidates=max_depth_candidates,
        n_estimators_total=n_estimators_total,
        max_features_candidates=max_features_candidates,
        min_samples_split_candidates=min_samples_split_candidates,
        min_samples_leaf_candidates=min_samples_leaf_candidates,
        criterion_candidates=criterion_candidates,
    )

    request.leakage_columns.extend(leakage_columns)

    print()
    print("Submitting regression training")
    print("dataset_url:", str(dataset_path))
    print("target_column:", target_column)
    print("task_type: regression")
    print("dataset_scenario:", dataset_scenario)
    print("leakage_columns:", leakage_columns)
    print("n_estimators_total:", n_estimators_total)
    print("max_depth_candidates:", max_depth_candidates)
    print("max_features_candidates:", max_features_candidates)
    print("min_samples_split_candidates:", min_samples_split_candidates)
    print("min_samples_leaf_candidates:", min_samples_leaf_candidates)
    print("criterion_candidates:", criterion_candidates)
    print("bootstrap:", bootstrap)
    print("global_random_seed:", global_random_seed)
    print()

    response = stub.SubmitTraining(request, timeout=30)

    print("job_id:", response.job_id)
    print("status:", response.status)
    print("message:", response.message)

    return response