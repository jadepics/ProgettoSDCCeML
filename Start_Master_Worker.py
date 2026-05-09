import os
import subprocess
import sys
from pathlib import Path


# =========================================================
# MASTER LAUNCHER
# =========================================================

class MasterLauncher:

    def __init__(self):

        self.master_host = "0.0.0.0"

        self.master_port = "50051"

        self.artifact_root = (
            Path("shared_artifacts")
            .resolve()
        )

    def build_environment(self):

        env = os.environ.copy()

        env["MASTER_HOST"] = (
            self.master_host
        )

        env["MASTER_PORT"] = (
            self.master_port
        )

        env["ARTIFACT_ROOT"] = str(
            self.artifact_root
        )

        return env

    def start_master(self):

        env = self.build_environment()

        print()
        print("===================================")
        print("MASTER CONFIGURATION")
        print("===================================")

        print("MASTER_HOST  :", self.master_host)
        print("MASTER_PORT  :", self.master_port)
        print("ARTIFACT_ROOT:", self.artifact_root)

        print()
        print("[MASTER] starting...")
        print()

        subprocess.run(
            [
                sys.executable,
                "master.py",
            ],
            env=env,
        )

    def launch(self):

        self.start_master()


# =========================================================
# WORKER LAUNCHER
# =========================================================

class WorkerLauncher:

    def __init__(
        self,
        worker_id: str,
        worker_port: int,
    ):

        self.worker_id = worker_id

        self.worker_bind_host = "0.0.0.0"

        self.worker_port = str(
            worker_port
        )

        self.worker_advertise_host = (
            "127.0.0.1"
        )

        self.master_host = "127.0.0.1"

        self.master_port = "50051"

        self.artifact_root = (
            Path("shared_artifacts")
            .resolve()
        )

    def configure_environment(self):

        os.environ["WORKER_ID"] = (
            self.worker_id
        )

        os.environ["WORKER_BIND_HOST"] = (
            self.worker_bind_host
        )

        os.environ["WORKER_PORT"] = (
            self.worker_port
        )

        os.environ["WORKER_ADVERTISE_HOST"] = (
            self.worker_advertise_host
        )

        os.environ["MASTER_HOST"] = (
            self.master_host
        )

        os.environ["MASTER_PORT"] = (
            self.master_port
        )

        os.environ["ARTIFACT_ROOT"] = str(
            self.artifact_root
        )

    def start_worker(self):

        self.configure_environment()

        print()
        print("===================================")
        print("WORKER CONFIGURATION")
        print("===================================")

        print("WORKER_ID            :", self.worker_id)
        print("WORKER_BIND_HOST     :", self.worker_bind_host)
        print("WORKER_PORT          :", self.worker_port)
        print("WORKER_ADVERTISE_HOST:", self.worker_advertise_host)

        print("MASTER_HOST          :", self.master_host)
        print("MASTER_PORT          :", self.master_port)

        print("ARTIFACT_ROOT        :", self.artifact_root)

        print()
        print("[WORKER] starting...")
        print()

        from worker.worker_config import (
            WorkerConfig
        )

        from worker.worker_node import (
            WorkerNode
        )

        config = (
            WorkerConfig
            .from_env()
        )

        worker = WorkerNode(config)

        worker.start()

    def launch(self):

        self.start_worker()


# =========================================================
# MASTER ENTRYPOINT
# =========================================================

def launch_master():

    launcher = MasterLauncher()

    launcher.launch()


# =========================================================
# WORKER ENTRYPOINT
# =========================================================

def launch_worker():

    print()
    print("===================================")
    print("CREATE WORKER")
    print("===================================")

    worker_id = input(
        "\nInsert worker_id: "
    ).strip()

    worker_port = int(
        input(
            "Insert worker_port: "
        ).strip()
    )

    launcher = WorkerLauncher(
        worker_id=worker_id,
        worker_port=worker_port,
    )

    launcher.launch()


# =========================================================
# MAIN MENU
# =========================================================

def main():

    while True:

        print()
        print("===================================")
        print("MASTER / WORKER LAUNCHER")
        print("===================================")

        print("1 -> Launch master")
        print("2 -> Launch worker")
        print("0 -> Exit")

        choice = input(
            "\nSelect option: "
        ).strip()

        if choice == "1":

            launch_master()

        elif choice == "2":

            launch_worker()

        elif choice == "0":

            print()
            print("Closing launcher...")
            print()

            break

        else:

            print()
            print("[ERROR] Invalid option")
            print()


if __name__ == "__main__":

    main()