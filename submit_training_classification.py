from pathlib import Path
from typing import Optional

import grpc

import rf_v2_pb2 as pb
import rf_v2_pb2_grpc as pbgrpc


def main(
    PrivateIp_Port: str,
    dataset_path: str | Path,
    dataset_scenario: str = "baseline_original",
    leakage_columns: Optional[list[str]] = None,
):
    if leakage_columns is None:
        leakage_columns = []

    channel = grpc.insecure_channel(PrivateIp_Port)
    stub = pbgrpc.CoordinatorServiceStub(channel)

    request = pb.SubmitTrainingRequest(
        dataset_url=str(dataset_path),
        target_column="diagnosed_diabetes",
        task_type="classification",
        n_estimators_total=4,
        validation_ratio=0.2,
        test_ratio=0.2,
        bootstrap=True,
        global_random_seed=42,
        max_depth_candidates=[5],
        max_features_candidates=["sqrt"],
        min_samples_split_candidates=[2],
        min_samples_leaf_candidates=[1],
        criterion_candidates=["gini"],
        dataset_scenario=dataset_scenario,
    )

    request.leakage_columns.extend(leakage_columns)

    print()
    print("Submitting classification training")
    print("dataset_url:", str(dataset_path))
    print("dataset_scenario:", dataset_scenario)
    print("leakage_columns:", leakage_columns)
    print()

    response = stub.SubmitTraining(request, timeout=30)

    print("job_id:", response.job_id)
    print("status:", response.status)
    print("message:", response.message)