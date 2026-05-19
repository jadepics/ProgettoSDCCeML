
# 1. Setup iniziale

Aprire PowerShell nella root del progetto:

```powershell
cd C:\Users\micci\PycharmProjects\PythonProject1
.\.venv\Scripts\Activate.ps1
```

Controllare che il codice compili:

```powershell
python -m compileall -q master.py masterPackage common worker
```

Se il comando non stampa nulla, il controllo è passato.

Pulire gli artifact vecchi prima di una nuova run:

```powershell
Remove-Item -Recurse -Force .\shared_artifacts -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force .\shared_artifacts | Out-Null
```

---

# 2. Terminale 1 — Avviare il master

Aprire un primo terminale PowerShell:

```powershell
cd C:\Users\micci\PycharmProjects\PythonProject1
.\.venv\Scripts\Activate.ps1

$env:MASTER_HOST="0.0.0.0"
$env:MASTER_PORT="50051"
$env:ARTIFACT_ROOT=(Join-Path (Get-Location) "shared_artifacts")

python master.py
```

Output atteso:

```text
[MASTER] listening on 0.0.0.0:50051
```

Lasciare aperto questo terminale.

---

# 3. Terminale 2 — Avviare worker 1

Aprire un secondo terminale PowerShell:

```powershell
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
```

Output atteso:

```text
[WorkerNode] gRPC server started on port 50061
[MasterClient] Registered worker worker-1
[WorkerNode] Registered as 127.0.0.1:50061
```

Lasciare aperto questo terminale.

---

# 4. Terminale 3 — Avviare worker 2

Aprire un terzo terminale PowerShell:

```powershell
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
```

Output atteso:

```text
[WorkerNode] gRPC server started on port 50062
[MasterClient] Registered worker worker-2
[WorkerNode] Registered as 127.0.0.1:50062
```

Lasciare aperto questo terminale.

---

# 5. Opzionale — Avviare worker 3

Per testare meglio la scalabilità locale, aprire un quarto terminale worker:

```powershell
cd C:\Users\micci\PycharmProjects\PythonProject1
.\.venv\Scripts\Activate.ps1

$env:WORKER_ID="worker-3"
$env:WORKER_BIND_HOST="0.0.0.0"
$env:WORKER_PORT="50063"
$env:WORKER_ADVERTISE_HOST="127.0.0.1"
$env:MASTER_HOST="127.0.0.1"
$env:MASTER_PORT="50051"
$env:ARTIFACT_ROOT=(Join-Path (Get-Location) "shared_artifacts")

python -c "from worker.worker_config import WorkerConfig; from worker.worker_node import WorkerNode; WorkerNode(WorkerConfig.from_env()).start()"
```

---

# 6. Terminale client — Creare client classification

Aprire un nuovo terminale PowerShell:

```powershell
cd C:\Users\micci\PycharmProjects\PythonProject1
.\.venv\Scripts\Activate.ps1
```

Creare `submit_training_classification.py`:

```powershell
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
'@ | Set-Content .\submit_training_classification.py
```

Lanciare la run classification:

```powershell
python .\submit_training_classification.py
```

Output atteso:

```text
job_id: job_...
status: 1
message: Training started
```

`status: 1` indica che il job è stato accettato. Il training prosegue in background lato master.

---

# 7. Terminale client — Creare client regression

Creare `submit_training_regression.py`:

```powershell
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
    target_column="hba1c",
    task_type="regression",
    n_estimators_total=4,
    validation_ratio=0.2,
    test_ratio=0.2,
    bootstrap=True,
    global_random_seed=42,
    max_depth_candidates=[5],
    max_features_candidates=["sqrt"],
    min_samples_split_candidates=[2],
    min_samples_leaf_candidates=[1],
    criterion_candidates=["squared_error"],
)

response = stub.SubmitTraining(request, timeout=30)

print("job_id:", response.job_id)
print("status:", response.status)
print("message:", response.message)
'@ | Set-Content .\submit_training_regression.py
```

Lanciare la run regression:

```powershell
python .\submit_training_regression.py
```

Output atteso:

```text
job_id: job_...
status: 1
message: Training started
```

---

# 8. Recuperare l'ultimo job creato

Se non vuoi copiare manualmente il `job_id`, puoi recuperare l'ultimo job creato:

```powershell
$job = (Get-ChildItem ".\shared_artifacts\jobs" -Directory |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1).Name

$job
```

Da qui in poi puoi usare `$job` nei comandi successivi.

---

# 9. Controllare lo stato del job

Impostare il job manualmente:

```powershell
$job="job_QUI_METTI_ID"
```

Leggere `job_record.json`:

```powershell
$j = Get-Content ".\shared_artifacts\jobs\$job\job_record.json" -Raw | ConvertFrom-Json

$j.status
$j.message
$j.selected_experiment_id
$j.model_id
```

Risultato atteso a fine run:

```text
COMPLETED
Training completed. Selected experiment ...
job_..._exp_000
model_...
```

Se il job è fallito:

```powershell
$j.status
$j.message
```

Il campo `$j.message` contiene l'errore principale.

---

# 10. Controllare gli esperimenti del job

```powershell
$job="job_QUI_METTI_ID"

Get-ChildItem ".\shared_artifacts\jobs\$job\experiments" -Directory | ForEach-Object {
    Get-Content "$($_.FullName)\experiment_record.json" -Raw | ConvertFrom-Json |
    Select-Object experiment_id,status,expected_tree_count,completed_tree_count,assigned_workers
}
```

Risultato atteso:

```text
experiment_id        : job_..._exp_000
status               : COMPLETED
expected_tree_count  : 4
completed_tree_count : 4
assigned_workers     : {worker-1, worker-2}
```

Se `assigned_workers` contiene solo `worker-1`, la run è completata ma il training non è stato realmente distribuito su più worker.

---

# 11. Contare gli alberi salvati

```powershell


Get-ChildItem ".\shared_artifacts\jobs\$job\experiments" -Directory | ForEach-Object {
    Write-Host "Experiment:" $_.Name

    Write-Host "Tree model files:"
    Get-ChildItem "$($_.FullName)\trees" -Filter "*.joblib" -ErrorAction SilentlyContinue | Measure-Object

    Write-Host "Tree metadata files:"
    Get-ChildItem "$($_.FullName)\trees" -Filter "*.json" -ErrorAction SilentlyContinue | Measure-Object
}
```

Con:

```python
n_estimators_total=4
```

risultato atteso:

```text
.joblib Count = 4
.json   Count = 4
```

---

# 12. Verificare che il training sia davvero distribuito

Questo è il controllo principale per verificare la distribuzione su più nodi worker.

```powershell

$exp="${job}_exp_000"

Get-ChildItem ".\shared_artifacts\jobs\$job\experiments\$exp\trees" -Filter "*.json" | ForEach-Object {
    Get-Content $_.FullName -Raw | ConvertFrom-Json |
    Select-Object tree_id, tree_index, worker_id, task_id
}
```

Risultato atteso con 2 worker e 4 alberi:

```text
tree_0    0    worker-1    ...
tree_1    1    worker-1    ...
tree_2    2    worker-2    ...
tree_3    3    worker-2    ...
```

Una distribuzione equivalente va bene. L'importante è che compaiano entrambi i worker.

Vista sintetica per contare quanti alberi ha prodotto ogni worker:

```powershell
$exp="${job}_exp_000"

Get-ChildItem ".\shared_artifacts\jobs\$job\experiments\$exp\trees" -Filter "*.json" | ForEach-Object {
    (Get-Content $_.FullName -Raw | ConvertFrom-Json).worker_id
} | Group-Object
```

Risultato atteso con 2 worker e 4 alberi:

```text
Count Name
2     worker-1
2     worker-2
```

Risultato atteso con 2 worker e 20 alberi:

```text
Count Name
10    worker-1
10    worker-2
```

Risultato atteso con 3 worker e 21 alberi:

```text
Count Name
7     worker-1
7     worker-2
7     worker-3
```

---

# 13. Leggere le metriche di validation

Le metriche vengono lette da `experiment_record.json`.

```powershell

$exp="${job}_exp_000"

$er = Get-Content ".\shared_artifacts\jobs\$job\experiments\$exp\experiment_record.json" -Raw | ConvertFrom-Json

$er.status
$er.expected_tree_count
$er.completed_tree_count
$er.assigned_workers
$er.validation_metrics
```

---

# 14. Metriche classification

Per una run classification:

```powershell
$job="job_QUI_METTI_ID"
$exp="${job}_exp_000"

$er = Get-Content ".\shared_artifacts\jobs\$job\experiments\$exp\experiment_record.json" -Raw | ConvertFrom-Json

$er.validation_metrics.accuracy
$er.validation_metrics.confusion_matrix
$er.validation_metrics.classification_report | ConvertTo-Json -Depth 10
```

Campi importanti:

```text
accuracy
confusion_matrix
precision
recall
f1-score
support
macro avg
weighted avg
```

Esempio di lettura singola:

```powershell
$er.validation_metrics.accuracy
```

---

# 15. Metriche regression

Per una run regression, le metriche sono salvate nel campo `classification_report`, anche se il nome del campo è improprio.

```powershell
$exp="${job}_exp_000"

$er = Get-Content ".\shared_artifacts\jobs\$job\experiments\$exp\experiment_record.json" -Raw | ConvertFrom-Json

$er.validation_metrics.classification_report | ConvertTo-Json -Depth 10
```

Campi attesi:

```text
mse
rmse
r2
```

Lettura diretta:

```powershell
$er.validation_metrics.classification_report.mse
$er.validation_metrics.classification_report.rmse
$er.validation_metrics.classification_report.r2
```

---

# 16. Leggere il manifest finale del modello

Dopo una run completata:

```powershell

$j = Get-Content ".\shared_artifacts\jobs\$job\job_record.json" -Raw | ConvertFrom-Json
$model = $j.model_id

$mf = Get-Content ".\shared_artifacts\models\$model\manifest.json" -Raw | ConvertFrom-Json

$mf.status
$mf.model_id
$mf.job_id
$mf.experiment_id
$mf.model_type
$mf.tree_artifacts.Count
$mf.validation_metrics.accuracy
```

Risultato atteso:

```text
tree_artifacts.Count = n_estimators_total
```

Per classification:

```powershell
$mf.validation_metrics.accuracy
$mf.validation_metrics.confusion_matrix
$mf.validation_metrics.classification_report | ConvertTo-Json -Depth 10
```

Per regression:

```powershell
$mf.validation_metrics.classification_report | ConvertTo-Json -Depth 10
```

---

# 17. Controllare il TaskLedger

Il `TaskLedger` permette di verificare quali shard sono stati creati, su quali worker sono stati assegnati e quali alberi risultano completati.

```powershell


Get-Content ".\shared_artifacts\jobs\$job\task_ledger.json" -Raw |
ConvertFrom-Json |
ConvertTo-Json -Depth 20
```

Con 2 worker e 4 alberi, una run distribuita dovrebbe avere più task, ad esempio:

```text
trees_000000_000001 -> worker-1
trees_000002_000003 -> worker-2
```

Se invece compare un solo task:

```text
trees_000000_000003 -> worker-1
```

allora il job è completato ma il training non è stato realmente distribuito.

---

# 18. Controllare velocemente tutto dopo una run

Impostare il job:

```powershell
$job="job_QUI_METTI_ID"
$exp="${job}_exp_000"
```

Job:

```powershell
$j = Get-Content ".\shared_artifacts\jobs\$job\job_record.json" -Raw | ConvertFrom-Json
$j.status
$j.message
$j.selected_experiment_id
$j.model_id
```

Experiment:

```powershell
$er = Get-Content ".\shared_artifacts\jobs\$job\experiments\$exp\experiment_record.json" -Raw | ConvertFrom-Json
$er.status
$er.expected_tree_count
$er.completed_tree_count
$er.assigned_workers
```

Alberi:

```powershell
Get-ChildItem ".\shared_artifacts\jobs\$job\experiments\$exp\trees" -Filter "*.joblib" | Measure-Object
Get-ChildItem ".\shared_artifacts\jobs\$job\experiments\$exp\trees" -Filter "*.json" | Measure-Object
```

Distribuzione worker:

```powershell
Get-ChildItem ".\shared_artifacts\jobs\$job\experiments\$exp\trees" -Filter "*.json" | ForEach-Object {
    (Get-Content $_.FullName -Raw | ConvertFrom-Json).worker_id
} | Group-Object
```

Metriche:

```powershell
$er.validation_metrics
$er.validation_metrics.classification_report | ConvertTo-Json -Depth 10
```

Manifest:

```powershell
$model = $j.model_id
$mf = Get-Content ".\shared_artifacts\models\$model\manifest.json" -Raw | ConvertFrom-Json

$mf.status
$mf.model_id
$mf.tree_artifacts.Count
$mf.validation_metrics
```

---

# 19. Test consigliati

## Test A — Classification minima distribuita

Configurazione nello script:

```python
target_column="diagnosed_diabetes"
task_type="classification"
n_estimators_total=4
criterion_candidates=["gini"]
```

Controlli attesi:

```text
job status = COMPLETED
completed_tree_count = 4
.joblib Count = 4
.json Count = 4
assigned_workers contiene worker-1 e worker-2
classification metrics presenti
manifest presente
```

---

## Test B — Regression minima distribuita

Configurazione nello script:

```python
target_column="hba1c"
task_type="regression"
n_estimators_total=4
criterion_candidates=["squared_error"]
```

Controlli attesi:

```text
job status = COMPLETED
completed_tree_count = 4
.joblib Count = 4
.json Count = 4
assigned_workers contiene worker-1 e worker-2
mse/rmse/r2 presenti
manifest presente
```

---

## Test C — Scalabilità locale con 2 worker

Modificare lo script:

```python
n_estimators_total=20
```

Controllo distribuzione:

```powershell
$job="job_QUI_METTI_ID"
$exp="${job}_exp_000"

Get-ChildItem ".\shared_artifacts\jobs\$job\experiments\$exp\trees" -Filter "*.json" | ForEach-Object {
    (Get-Content $_.FullName -Raw | ConvertFrom-Json).worker_id
} | Group-Object
```

Risultato atteso:

```text
10 worker-1
10 worker-2
```

---

## Test D — Scalabilità locale con 3 worker

Avviare anche `worker-3`.

Modificare lo script:

```python
n_estimators_total=21
```

Controllo distribuzione:

```powershell
$job="job_QUI_METTI_ID"
$exp="${job}_exp_000"

Get-ChildItem ".\shared_artifacts\jobs\$job\experiments\$exp\trees" -Filter "*.json" | ForEach-Object {
    (Get-Content $_.FullName -Raw | ConvertFrom-Json).worker_id
} | Group-Object
```

Risultato atteso:

```text
7 worker-1
7 worker-2
7 worker-3
```

---

# 20. Quando una run fallisce

Impostare il job:

```powershell
$job="job_QUI_METTI_ID"
```

Leggere errore principale:

```powershell
$j = Get-Content ".\shared_artifacts\jobs\$job\job_record.json" -Raw | ConvertFrom-Json

$j.status
$j.message
```

Leggere task ledger:

```powershell
Get-Content ".\shared_artifacts\jobs\$job\task_ledger.json" -Raw |
ConvertFrom-Json |
ConvertTo-Json -Depth 20
```

Controllare se esistono alberi parziali:

```powershell
Get-ChildItem ".\shared_artifacts\jobs\$job\experiments" -Directory | ForEach-Object {
    Write-Host "Experiment:" $_.Name
    Get-ChildItem "$($_.FullName)\trees" -Filter "*.joblib" -ErrorAction SilentlyContinue | Measure-Object
    Get-ChildItem "$($_.FullName)\trees" -Filter "*.json" -ErrorAction SilentlyContinue | Measure-Object
}
```

---

# 21. Stop e nuova run pulita

Per fare una nuova run da zero:

1. Fermare master e worker con `CTRL+C`.
2. Pulire `shared_artifacts`.
3. Riavviare master.
4. Riavviare worker.
5. Lanciare client classification o regression.

Comando di pulizia:

```powershell
Remove-Item -Recurse -Force .\shared_artifacts -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force .\shared_artifacts | Out-Null
```

Poi ripetere:

```text
Terminale 1 -> master
Terminale 2 -> worker-1
Terminale 3 -> worker-2
Terminale client -> submit training
```

---

# 22. Note per esecuzione distribuita reale su AWS

In locale viene usato:

```powershell
$env:WORKER_ADVERTISE_HOST="127.0.0.1"
```

Questo funziona solo perché master e worker girano sulla stessa macchina.

Su AWS o su macchine diverse, ogni worker deve usare un indirizzo raggiungibile dal master:

```powershell
$env:WORKER_ADVERTISE_HOST="<IP_PRIVATO_O_DNS_DEL_WORKER>"
```

Anche lo storage deve essere condiviso tra master e worker. In locale:

```powershell
$env:ARTIFACT_ROOT=(Join-Path (Get-Location) "shared_artifacts")
```

Su AWS, ad esempio con EFS montato:

```powershell
$env:ARTIFACT_ROOT="/mnt/efs/shared_artifacts"
```

Tutti i nodi devono vedere gli stessi artifact allo stesso path logico o tramite una stessa astrazione di storage.
