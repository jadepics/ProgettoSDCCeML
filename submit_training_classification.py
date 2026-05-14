from pathlib import Path
import grpc

import rf_v2_pb2 as pb
import rf_v2_pb2_grpc as pbgrpc

def main(PrivateIp_Port : str, dataset_path : str):

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
    )

    response = stub.SubmitTraining(request, timeout=30)

    print("job_id:", response.job_id)
    print("status:", response.status)
    print("message:", response.message)
