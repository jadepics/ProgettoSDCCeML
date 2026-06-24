# Distributed Random Forest — Master/Worker con gRPC, EFS e Fault Tolerance

## Descrizione del progetto

Questo progetto implementa un sistema di **Random Forest distribuito** per task di **classificazione** (binaria e multiclasse) e **regressione**. L'obiettivo è distribuire il training e l'inferenza di una foresta di decision tree su più worker, mantenendo un master responsabile del coordinamento, della preparazione dei dati, della validazione, della selezione del modello e della pubblicazione del manifest finale.

La distribuzione del sistema è realizzata tramite un'architettura **Master/Worker** basata su **Python**, **gRPC** e storage condiviso Amazon EFS, n ambiente AWS, montato sui nodi master, worker e client nel path:

```text
/mnt/efs/gp_artifacts
```

Il progetto include inoltre una fault tolerance pratica basata su:

- cluster di master con leader election tramite Raft minimale;
- esecuzione delle operazioni critiche solo sul master leader;
- worker con registrazione e heartbeat;
- persistenza di job, task ledger, split, alberi e manifest su EFS;
- recovery degli alberi mancanti dopo crash worker o crash del master leader.

> Nota importante: Raft è usato per la **leader election**. Lo stato applicativo del training viene recuperato da EFS tramite `JobRepository`, `TaskLedger`, artifact e manifest persistenti. Il progetto non replica ancora tutte le transizioni applicative tramite una state machine Raft completa.

---

## Architettura del sistema

Il sistema è composto da quattro componenti principali.

### Client

Il client invia richieste al master leader tramite gRPC. Può:

- lanciare job di training;
- interrogare lo stato dei job;
- eseguire inferenza distribuita;
- scaricare il modello addestrato;
- confrontare i risultati distribuiti con baseline locali.

La CLI principale è:

```bash
python training_debug_cli.py
```

### Master cluster

Il master rappresenta il **control plane** del sistema. In configurazione distribuita sono previsti tre master:

```text
master1 -> gRPC 50051, Raft 50151
master2 -> gRPC 50052, Raft 50152
master3 -> gRPC 50053, Raft 50153
```

Un solo nodo è leader alla volta. Il leader:

- riceve `SubmitTraining`, `SubmitInference`, `ResumeTraining` e `DownloadModel`;
- prepara il dataset;
- crea gli esperimenti;
- divide la foresta in shard;
- assegna shard ai worker;
- coordina validation e test;
- seleziona il miglior modello;
- costruisce e salva il `ModelManifest` finale.

### Worker

Il worker rappresenta l'**execution plane**. Ogni worker:

- riceve shard di training dal master leader;
- legge gli split del dataset da EFS;
- addestra un sottoinsieme di alberi;
- salva ogni albero e i relativi metadata su EFS;
- produce predizioni parziali per validation, testing e inferenza.

I worker espongono un servizio gRPC e possono essere avviati in più container sulla stessa istanza EC2.

### Storage condiviso EFS

EFS è lo strato di persistenza degli artifact. Contiene:

- dataset sorgenti;
- split train/validation/test;
- alberi `.joblib`;
- metadata JSON degli alberi;
- `job_record.json`;
- `task_ledger.json`;
- metriche di validation e test;
- prediction artifact;
- manifest dei modelli finali;
- log Raft dei master.

---

## Flusso di training distribuito

```text
Client
  |
  | SubmitTraining
  v
Master leader
  |
  | load dataset
  | validate dataset
  | apply scenario / preprocessing
  | split train / validation / test
  | persist prepared dataset on EFS
  v
ExperimentPlanner
  |
  | generate ForestConfiguration(s)
  v
ShardPlanner
  |
  | split n_estimators into deterministic tree shards
  v
TrainingOrchestrator
  |
  | assign shards to alive workers
  v
Workers
  |
  | train DecisionTreeClassifier / DecisionTreeRegressor
  | save tree artifact and metadata on EFS
  v
Master leader
  |
  | collect completed artifacts
  | run distributed validation
  | select best experiment
  | run distributed test evaluation
  | build and save model manifest
  v
EFS /models/<model_id>/manifest.json
```

Ogni albero ha un identificativo deterministico:

```text
<experiment_id>_tree_<tree_index>
```

Questa scelta supporta idempotenza e recovery: se un worker fallisce dopo aver completato alcuni alberi, il sistema può recuperare solo quelli mancanti.

---

## Flusso di inferenza distribuita

L'inferenza usa il manifest del modello finale.

```text
Client
  |
  | SubmitInference(model_id, features_uri)
  v
Master leader
  |
  | load manifest
  | split tree_artifact_uris among workers
  v
Workers
  |
  | load assigned trees from EFS
  | load input features from EFS
  | compute partial predictions
  | save partial predictions on EFS
  v
Master leader
  |
  | aggregate partial predictions
  | classification -> majority vote
  | regression -> average prediction
  | save predictions.npy
  v
Client receives prediction_uri
```

Nel percorso scalabile non vengono inviate matrici dense via gRPC: master e worker scambiano principalmente URI verso file salvati su EFS.

---

## Prerequisiti

### Software

- Python 3.11 o superiore;
- `pip`;
- `venv`;
- Docker;
- gRPC / Protocol Buffers;
- storage condiviso montato in `/mnt/efs/gp_artifacts`;
- su AWS: EC2 + EFS.

Dipendenze Python presenti in `requirements.txt`:

```text
grpcio
grpcio-tools
numpy
pandas
scikit-learn
joblib
pyarrow
matplotlib
```

### Porte principali

| Componente | Porte | Uso |
|---|---:|---|
| Master gRPC | `50051`, `50052`, `50053` | API master verso client e worker |
| Master Raft | `50151`, `50152`, `50153` | leader election tra master |
| Worker gRPC | `50061+` | training e inferenza shard |
| EFS / NFS | `2049` | mount storage condiviso |

Il deployment Docker usa `--network host`, quindi le porte dei container coincidono con quelle dell'host.

---

## Installazione

### 1. Clonare il repository

```bash
git clone https://github.com/jadepics/ProgettoSDCCeML.git
cd ProgettoSDCCeML
```

### 2. Installare i pacchetti di sistema

Su Amazon Linux / EC2:

```bash
sudo dnf install -y python3 python3-pip python3-devel gcc gcc-c++ make
```

Se Docker non è installato:

```bash
sudo dnf install -y docker
sudo systemctl enable --now docker
sudo usermod -aG docker $USER
```

Dopo `usermod`, effettuare logout/login oppure riaprire la sessione SSH.

### 3. Creare l'ambiente Python

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 4. Installare le dipendenze

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 5. Rigenerare gli stub gRPC se il proto viene modificato

I file `rf_v2_pb2.py` e `rf_v2_pb2_grpc.py` sono già presenti nel progetto. Se viene modificato `proto/rf_v2.proto`, rigenerarli con:

```bash
python -m grpc_tools.protoc \
  -I proto \
  --python_out=. \
  --grpc_python_out=. \
  proto/rf_v2.proto
```

I Dockerfile rigenerano automaticamente gli stub durante la build.

---

## Preparazione dello storage condiviso

Il progetto assume che EFS sia montato in:

```text
/mnt/efs/gp_artifacts
```

## Configurazione

Il progetto fornisce due file di esempio:

```text
.env.master.example
.env.worker.example
```

Copiarli prima dell'esecuzione:

```bash
cp .env.master.example .env.master
cp .env.worker.example .env.worker
```

### Worker dinamici sulla stessa istanza

Nel path `scripts/worker.sh` supporta più container worker sulla stessa macchina tramite override runtime:

| Variabile | Significato |
|---|---|
| `WORKER_COUNT` | numero di container worker da avviare |
| `WORKER_ID_PREFIX` | prefisso degli ID worker |
| `WORKER_INDEX_START` | indice iniziale degli ID worker |
| `WORKER_BASE_PORT` | prima porta worker da usare |
| `MASTER_CLUSTER_HOST` | IP privato del nodo master/cluster master |

Esempio prima istanza worker:

```bash
./scripts/worker.sh rebuild-all \
  WORKER_COUNT=4 \
  WORKER_ID_PREFIX=worker \
  WORKER_INDEX_START=1 \
  WORKER_BASE_PORT=50061 \
  MASTER_CLUSTER_HOST=<PRIVATE_IP_MASTER>
```

Esempio seconda istanza worker:

```bash
./scripts/worker.sh rebuild-all \
  WORKER_COUNT=4 \
  WORKER_ID_PREFIX=worker \
  WORKER_INDEX_START=5 \
  WORKER_BASE_PORT=50065 \
  MASTER_CLUSTER_HOST=<PRIVATE_IP_MASTER>
```




## Esecuzione distribuita con Docker


## Esecuzione locale / sviluppo

Per una run locale semplificata è possibile usare il backend di consenso in memoria.

```
```

## Training distribuito

La CLI principale è sul Terminale client:

```bash
python training_debug_cli.py
```

Menu principale:

```text
1 -> Submit training
2 -> See job status
3 -> See experiments
4 -> Count saved trees
5 -> See validation metrics
6 -> Submit inference
7 -> Resume training job
8 -> Download trained model
9 -> Eliminate shared artifacts
0 -> Exit
```


Nel menu: 1 -> Submit training:

```text
1 -> Submit regression
2 -> Submit classification
```


Dataset principale:

```text
/mnt/efs/gp_artifacts/datasets/diabetes_dataset.csv
```


## Monitoraggio dei job

I job vengono salvati in:

```text
/mnt/efs/gp_artifacts/jobs/<job_id>
```

Struttura tipica:

```text
/mnt/efs/gp_artifacts/jobs/<job_id>/
├── job_record.json
├── task_ledger.json
├── prepared_dataset/
│   ├── schema.json
│   ├── dataset_scenario_report.json
│   ├── train_features.parquet
│   ├── train_labels.parquet
│   ├── validation_features.parquet
│   ├── validation_labels.parquet
│   ├── test_features.parquet
│   └── test_labels.parquet
└── experiments/
    └── <experiment_id>/
        ├── experiment_record.json
        ├── trees/
        │   ├── tree_0.joblib
        │   ├── tree_0.json
        │   └── ...
        ├── metrics/
        │   └── validation_metrics.json
        └── predictions/
```

Comandi utili:

```bash
cd /mnt/efs/gp_artifacts/jobs
ls
cat /mnt/efs/gp_artifacts/jobs/<job_id>/job_record.json
cat /mnt/efs/gp_artifacts/jobs/<job_id>/task_ledger.json
find /mnt/efs/gp_artifacts/jobs/<job_id>/experiments -path "*/trees/*.joblib" | wc -l
```

Per elencare job falliti:

```bash
python scripts/list_failed_jobs.py
```

---

## Inferenza distribuita

Avviare la CLI:

```bash
python training_debug_cli.py
```

Nel menu:

```text
6 -> Submit inference
```

La CLI richiede:

```text
model_id
split: validation / test / train
numero di righe
```

Output tipico:

```text
success: True
task_type: classification
prediction_uri: file:///mnt/efs/gp_artifacts/models/<model_id>/inference/<inference_id>/predictions.npy
n_rows: ...
n_cols: ...
```

---

## Download del modello

Dalla CLI:

```bash
python training_debug_cli.py
```

Selezionare:

```text
8 -> Download trained model
```

Il sistema invoca `DownloadModel` e produce un bundle del modello in formato esportabile.

---

# Le seguenti attività sono eseguibili per debug e test lato master o worker:

## Baseline locale

La baseline non distribuita si trova in:

```text
local_baseline/
```


Risultati eseguiti in locale sono salvati in:

```text
local_baseline/results
```

Confronto con job distribuiti:

```bash
python performance_evaluation/compare_existing_job_cli.py
```

---

## Dataset sintetici

Generazione dataset sintetici:

```bash
python -m synthetic_data.synthetic_dataset_cli
```

Output tipico:

```text
Dataset/synthetic_classification_<n_samples>_samples_<n_features>_features.csv
Dataset/synthetic_regression_<n_samples>_samples_<n_features>_features.csv
```

Copia su EFS:

```bash
cp Dataset/synthetic_classification_100000_samples_40_features.csv \
  /mnt/efs/gp_artifacts/datasets/

cp Dataset/synthetic_regression_100000_samples_40_features.csv \
  /mnt/efs/gp_artifacts/datasets/
```

---

## Benchmark e performance evaluation

### Confronto job distribuito vs baseline locale

```bash
python performance_evaluation/compare_existing_job_cli.py
```

Risultati:

```text
performance_evaluation/results/
```

### Benchmark inferenza distribuita

```bash
python performance_evaluation/distributed_inference_benchmark.py \
  --model-id <model_id> \
  --split test \
  --rows all \
  --local-baseline-json local_baseline/results/<baseline>.json \
  --output-json performance_evaluation/results/inference/distributed_inference_<name>.json
```

### Scalabilità

Per valutare la scalabilità, abbiamo ripetuto lo stesso task variando solo il numero di worker

---

## Fault tolerance

Il progetto include script per raccogliere artifact e log degli esperimenti di fault tolerance.

---

## Struttura del progetto

```text
.
├── Dockerfile.master
├── Dockerfile.worker
├── README.md
├── ML_SCENARIOS.md
├── requirements.txt
├── master.py
├── run_worker.py
├── training_debug_cli.py
├── submit_training_classification.py
├── submit_training_regression.py
├── rf_v2_pb2.py
├── rf_v2_pb2_grpc.py
│
├── proto/
│   └── rf_v2.proto
│
├── common/
│   ├── contracts.py
│   ├── enums.py
│   ├── grpc_config.py
│   ├── ids.py
│   ├── prediction_io.py
│   ├── repositories.py
│   └── storage_layout.py
│
├── masterPackage/
│   ├── data/
│   │   ├── dataset_loader.py
│   │   ├── dataset_validator.py
│   │   ├── split_manager.py
│   │   └── data_preparation_service.py
│   ├── experiment_planner.py
│   ├── shard_planner.py
│   ├── training_orchestrator.py
│   ├── training_job_service.py
│   ├── validation_coordinator.py
│   ├── test_evaluator.py
│   ├── model_selector.py
│   ├── model_manifest_builder.py
│   ├── model_export_service.py
│   ├── inference_coordinator.py
│   ├── worker_client.py
│   ├── fault_tolerance.py
│   ├── worker_heartbeat_monitor.py
│   ├── task_lease_manager.py
│   ├── recovery_planner.py
│   └── retry_policy.py
│
├── worker/
│   ├── worker_config.py
│   ├── worker_node.py
│   ├── worker_service.py
│   ├── worker_state.py
│   ├── master_client/
│   │   └── master_client.py
│   ├── runtime/
│   │   └── heartbeat_loop.py
│   ├── training/
│   │   ├── bootstrap_sampler.py
│   │   ├── decision_tree_factory.py
│   │   ├── shard_trainer.py
│   │   └── tree_artifact_writer.py
│   ├── prediction/
│   │   └── shard_predictor.py
│   ├── progress/
│   │   └── worker_progress_store.py
│   ├── storage/
│   ├── mappers/
│   └── utils/
│
├── scripts/
│   ├── master.sh
│   ├── worker.sh
│   ├── monitor.sh
│   ├── docker-clean.sh
│   ├── list_failed_jobs.py
│   ├── collect_worker_crash_artifacts.sh
│   └── coolect_master_leade_crash_artifacts.sh
│
├── local_baseline/
│   ├── local_baseline_cli.py
│   ├── local_baseline_runner.py
│   ├── results/
│   └── models/
│
├── synthetic_data/
│   ├── synthetic_dataset_cli.py
│   └── synthetic_dataset_generator.py
│
├── performance_evaluation/
│   ├── compare_existing_job_cli.py
│   ├── distributed_inference_benchmark.py
│   ├── distributed_job_reader.py
│   ├── distributed_jobs/
│   └── results/
│
├── Dataset/
│   └── diabetes_dataset.csv
│
├── logs/
│   └── fault_tolerance/
│
└── shared_artifacts/
    ├── jobs/
    └── models/
```

---

## Ruolo dei file principali

| File / directory | Ruolo |
|---|---|
| `master.py` | entry point del master gRPC e implementazione della facciata `CoordinatorService` |
| `run_worker.py` | entry point del worker |
| `proto/rf_v2.proto` | definizione RPC e messaggi gRPC |
| `common/contracts.py` | dataclass condivise master/worker |
| `common/repositories.py` | repository JSON per job, task ledger e manifest |
| `common/storage_layout.py` | layout deterministico degli artifact |
| `masterPackage/training_job_service.py` | orchestrazione completa di un job di training |
| `masterPackage/training_orchestrator.py` | assegnazione shard, retry e recovery training |
| `masterPackage/shard_planner.py` | divisione della foresta in shard di alberi |
| `masterPackage/recovery_planner.py` | calcolo alberi mancanti e recovery shard |
| `masterPackage/fault_tolerance.py` | leadership guard e consenso Raft minimale |
| `worker/training/shard_trainer.py` | training effettivo degli alberi lato worker |
| `worker/training/tree_artifact_writer.py` | scrittura atomica e idempotente degli alberi |
| `worker/prediction/shard_predictor.py` | predizioni parziali per validation/test/inference |
| `scripts/master.sh` | build, start, restart e gestione cluster master |
| `scripts/worker.sh` | build, start, restart e gestione worker dinamici |
| `training_debug_cli.py` | CLI principale per training, inference, status e download |
| `local_baseline/` | baseline locale non distribuita |
| `performance_evaluation/` | confronto prestazionale e benchmark |
| `synthetic_data/` | generazione dataset sintetici |

---

## Scelte implementative rilevanti

### Master come control plane

Il master non addestra direttamente gli alberi. Coordina il ciclo di vita del job: data preparation, planning, scheduling, validation, test, model selection e manifest.

### Worker come execution plane

Il worker esegue training e predizione parziale. Gli alberi vengono salvati su storage condiviso e sono referenziati tramite URI persistenti.

### EFS come fonte durevole

Il sistema assume che EFS sia accessibile da master, worker e client. In caso di crash, il nuovo leader ricostruisce lo stato osservando repository JSON, ledger e artifact esistenti.

### Idempotenza e recovery minimale

Il sistema usa identificativi deterministici per task e alberi. Il recovery non ricrea necessariamente lo shard originale, ma genera nuovi recovery shard contenenti solo gli alberi mancanti.

---

## Troubleshooting:

### No alive workers available


### Not leader

### Dataset non trovato

### Worker duplicati

### Pulizia Docker

### Reset stato Raft


---

## Limitazioni note

- Raft viene usato per leader election, non per una replica completa dello stato applicativo.
- EFS è assunto come componente affidabile e condiviso.
- L'inferenza richiede feature coerenti con il manifest del modello.
- Il preprocessing viene applicato centralmente dal master durante la data preparation.
- Gli script sono ottimizzati per Linux/AWS EC2 con EFS montato in `/mnt/efs/gp_artifacts`.
- Il deployment usa `--network host`, quindi bisogna evitare collisioni tra porte.
- I file `.env.master`, `.env.worker`, `.env.client` reali non devono essere versionati.

---
