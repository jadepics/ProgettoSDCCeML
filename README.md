HOW TO RUN:
Setup iniziale

Apri PowerShell nella root del progetto:

cd C:\Users\micci\PycharmProjects\PythonProject1
.\.venv\Scripts\Activate.ps1

Poi fai un controllo veloce:

python -m compileall -q master.py masterPackage common worker

Se questo passa senza output, prosegui.

Pulisci gli artifact vecchi:

Remove-Item -Recurse -Force .\shared_artifacts -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force .\shared_artifacts | Out-Null
Terminale 1 — Avvia il master

Apri un primo terminale PowerShell:

cd C:\Users\micci\PycharmProjects\PythonProject1
.\.venv\Scripts\Activate.ps1

$env:MASTER_HOST="0.0.0.0"
$env:MASTER_PORT="50051"
$env:ARTIFACT_ROOT=(Join-Path (Get-Location) "shared_artifacts")

python master.py

Devi vedere qualcosa tipo:

[MASTER] listening on 0.0.0.0:50051

Lascia questo terminale aperto.

Terminale 2 — Avvia worker 1

Apri un secondo terminale PowerShell:

cd C:\Users\micci\PycharmProjects\PythonProject1
.\.venv\Scripts\Activate.ps1

$env:WORKER_ID="worker-1"
$env:WORKER_BIND_HOST="0.0.0.0"
$env:WORKER_PORT="50061"
$env:WORKER_ADVERTISE_HOST="127.0.0.1"
$env:MASTER_HOST="127.0.0.1"
$env:MASTER_PORT="50051"
$env:ARTIFACT_ROOT=(Join-Path (Get-Location) "shared_artifacts")

python -c "from worker.worker_config import WorkerConfig; from worker.worker_node import WorkerNode; WorkerNode(WorkerConfig.from_env()).start()"

Dovresti vedere:

[WorkerNode] gRPC server started on port 50061
[MasterClient] Registered worker worker-1
[WorkerNode] Registered as 127.0.0.1:50061

Lascia aperto anche questo terminale.

Terminale 3 — Avvia worker 2

Apri un terzo terminale PowerShell:

cd C:\Users\micci\PycharmProjects\PythonProject1
.\.venv\Scripts\Activate.ps1

$env:WORKER_ID="worker-2"
$env:WORKER_BIND_HOST="0.0.0.0"
$env:WORKER_PORT="50062"
$env:WORKER_ADVERTISE_HOST="127.0.0.1"
$env:MASTER_HOST="127.0.0.1"
$env:MASTER_PORT="50051"
$env:ARTIFACT_ROOT=(Join-Path (Get-Location) "shared_artifacts")

python -c "from worker.worker_config import WorkerConfig; from worker.worker_node import WorkerNode; WorkerNode(WorkerConfig.from_env()).start()"

Per una prima run puoi anche usare un solo worker, ma con due inizi già a verificare la distribuzione.

Terminale 4 — Invia il training job

Apri un quarto terminale PowerShell:

cd C:\Users\micci\PycharmProjects\PythonProject1
.\.venv\Scripts\Activate.ps1

Crea un piccolo client temporaneo:

@'
from pathlib import Path
import grpc

import rf_v2_pb2 as pb
import rf_v2_pb2_grpc as pbgrpc

dataset_path = Path("Dataset/diabetes_dataset.csv").resolve()

channel = grpc.insecure_channel("127.0.0.1:50051")
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
'@ | Set-Content .\submit_training.py

Poi lancia:

python .\submit_training.py

Dovresti ottenere una risposta tipo:

job_id: job_...
status: 1
message: Training started

Lo status 1 corrisponde a PENDING, perché il training parte in thread separato lato master.