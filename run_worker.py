from worker.worker_config import WorkerConfig
from worker.worker_node import WorkerNode

if __name__ == "__main__":
    WorkerNode(WorkerConfig.from_env()).start()