from pathlib import Path
import grpc

import rf_v2_pb2 as pb
import rf_v2_pb2_grpc as pbgrpc

def main(PrivateIp_Port : str, dataset_path :Path, n_estimators_total : int):

    channel = grpc.insecure_channel(PrivateIp_Port)
    stub = pbgrpc.CoordinatorServiceStub(channel)

    request = pb.SubmitTrainingRequest(
        dataset_url=str(dataset_path),
        target_column="hba1c",
        task_type="regression",
        dataset_scenario="baseline_no_leakage",
        leakage_columns=[
            "diagnosed_diabetes",
            "diabetes_stage",
            "diabetes_risk_score",
        ],
        validation_ratio=0.2,
        test_ratio=0.2,
        bootstrap=True,
        global_random_seed=42,
        #HP
        #profondità degli alberi
        max_depth_candidates=[5],
        #numero di alberi
        n_estimators_total=n_estimators_total,
        #Numero di feature considerate a ogni split.
        max_features_candidates=["sqrt"],
        #Numero minimo di campioni per fare uno split.
        min_samples_split_candidates=[2],
        #Numero minimo di esempi in una foglia.
        min_samples_leaf_candidates=[1],
        #Funzione usata per valutare gli split.
        criterion_candidates=["squared_error"],
    )

    response = stub.SubmitTraining(request, timeout=30)

    print("job_id:", response.job_id)
    print("status:", response.status)
    print("message:", response.message)
