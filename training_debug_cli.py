from pathlib import Path
import json
import submit_training_classification, submit_training_regression


# =========================================================
# CONFIG
# =========================================================

##########################################################
#
#utilizzare l'ip privato del master per fare l'allenamento
#Correggere per rendere tutto automatico
#
###########################################################
MASTER_ADDRESS = "172.31.37.47:50051"

###########################################################
#
#Modificare, se non funziona, con il mount comune
#
###########################################################
ARTIFACT_ROOT = Path("/mnt/efs/gp_artifacts").resolve()

# =========================================================
# SUBMIT TRAINING
# =========================================================

def submit_training():

    print("CHOOSE BETWEEN CLASSIFICATION AND REGRESSION")
    print("1 -> CLASSIFICATION")
    print("2 -> REGRESSION")
    print("3 -> GO BACK")

    choice = input(
        "\nSelect option: "
    ).strip()

    if choice == "1":
        submit_training_classification.main(MASTER_ADDRESS)

    elif choice == "2":
        submit_training_regression.main(MASTER_ADDRESS)

    elif choice == '3':
        return


def submit_training_launcher():

    print()
    print("===================================")
    print("SUBMIT TRAINING")
    print("===================================")

    submit_training()


# =========================================================
# JOB STATUS
# =========================================================

def see_job_status(job_id: str):

    job_record_path = (
        ARTIFACT_ROOT
        / "jobs"
        / job_id
        / "job_record.json"
    )

    if not job_record_path.exists():

        print()
        print("[ERROR] job_record.json not found")
        print(job_record_path)
        print()

        return

    with open(
        job_record_path,
        "r",
        encoding="utf-8"
    ) as f:

        job_record = json.load(f)

    print()
    print("status:")
    print(job_record.get("status"))

    print()
    print("message:")
    print(job_record.get("message"))

    print()
    print("selected_experiment_id:")
    print(job_record.get("selected_experiment_id"))

    print()
    print("model_id:")
    print(job_record.get("model_id"))
    print()


def see_job_status_launcher():

    print()
    print("===================================")
    print("SEE JOB STATUS")
    print("===================================")

    job_id = input(
        "\nInsert job_id: "
    ).strip()

    see_job_status(job_id)


# =========================================================
# SEE EXPERIMENTS
# =========================================================

def see_experiments(job_id: str):

    experiments_root = (
        ARTIFACT_ROOT
        / "jobs"
        / job_id
        / "experiments"
    )

    if not experiments_root.exists():

        print()
        print("[ERROR] experiments folder not found")
        print(experiments_root)
        print()

        return

    print()

    for experiment_dir in experiments_root.iterdir():

        if not experiment_dir.is_dir():
            continue

        experiment_record_path = (
            experiment_dir
            / "experiment_record.json"
        )

        if not experiment_record_path.exists():
            continue

        with open(
            experiment_record_path,
            "r",
            encoding="utf-8"
        ) as f:

            experiment_record = json.load(f)

        print("===================================")

        print("experiment_id:")
        print(
            experiment_record.get(
                "experiment_id"
            )
        )

        print()
        print("status:")
        print(
            experiment_record.get(
                "status"
            )
        )

        print()
        print("expected_tree_count:")
        print(
            experiment_record.get(
                "expected_tree_count"
            )
        )

        print()
        print("completed_tree_count:")
        print(
            experiment_record.get(
                "completed_tree_count"
            )
        )

        print()
        print("assigned_workers:")
        print(
            experiment_record.get(
                "assigned_workers"
            )
        )

        print()


def see_experiments_launcher():

    print()
    print("===================================")
    print("SEE EXPERIMENTS")
    print("===================================")

    job_id = input(
        "\nInsert job_id: "
    ).strip()

    see_experiments(job_id)


# =========================================================
# COUNT TREES
# =========================================================

def count_saved_trees(job_id: str):

    experiments_root = (
        ARTIFACT_ROOT
        / "jobs"
        / job_id
        / "experiments"
    )

    if not experiments_root.exists():

        print()
        print("[ERROR] experiments folder not found")
        print(experiments_root)
        print()

        return

    print()

    for experiment_dir in experiments_root.iterdir():

        if not experiment_dir.is_dir():
            continue

        trees_dir = (
            experiment_dir
            / "trees"
        )

        tree_count = len(
            list(
                trees_dir.glob("*.joblib")
            )
        )

        print("===================================")

        print("Experiment:")
        print(experiment_dir.name)

        print()
        print("Saved trees:")
        print(tree_count)

        print()


def count_saved_trees_launcher():

    print()
    print("===================================")
    print("COUNT SAVED TREES")
    print("===================================")

    job_id = input(
        "\nInsert job_id: "
    ).strip()

    count_saved_trees(job_id)


# =========================================================
# SEE METRICS
# =========================================================

def see_validation_metrics(
    job_id: str,
    experiment_id: str,
):

    experiment_record_path = (
        ARTIFACT_ROOT
        / "jobs"
        / job_id
        / "experiments"
        / experiment_id
        / "experiment_record.json"
    )

    if not experiment_record_path.exists():

        print()
        print("[ERROR] experiment_record.json not found")
        print(experiment_record_path)
        print()

        return

    with open(
        experiment_record_path,
        "r",
        encoding="utf-8"
    ) as f:

        experiment_record = json.load(f)

    print()

    print("status:")
    print(
        experiment_record.get(
            "status"
        )
    )

    print()
    print("expected_tree_count:")
    print(
        experiment_record.get(
            "expected_tree_count"
        )
    )

    print()
    print("completed_tree_count:")
    print(
        experiment_record.get(
            "completed_tree_count"
        )
    )

    print()
    print("assigned_workers:")
    print(
        experiment_record.get(
            "assigned_workers"
        )
    )

    print()
    print("validation_metrics:")
    print(
        experiment_record.get(
            "validation_metrics"
        )
    )

    print()


def see_validation_metrics_launcher():

    print()
    print("===================================")
    print("SEE VALIDATION METRICS")
    print("===================================")

    job_id = input(
        "\nInsert job_id: "
    ).strip()

    experiment_id = input(
        "Insert experiment_id: "
    ).strip()

    see_validation_metrics(
        job_id,
        experiment_id,
    )


from pathlib import Path
import shutil

#RIVEDERE QUESTO CODICE PERCHé NON è PIù CONFORME CON IL PATHING DELL'ARCH
def reset_shared_artifacts(root_path: str = "./shared_artifacts") -> None:
    """
    Rimuove completamente la directory shared_artifacts
    e la ricrea vuota.

    Equivalente Python di:

        Remove-Item -Recurse -Force .\\shared_artifacts
        New-Item -ItemType Directory -Force .\\shared_artifacts

    Sicuro da chiamare prima di un test end-to-end.
    """

    root = Path(root_path)

    # rimozione completa se esiste
    if root.exists():
        shutil.rmtree(root, ignore_errors=True)

    # ricreazione directory
    root.mkdir(parents=True, exist_ok=True)

def reset_shared_artifacts_launcher():
    print()
    print("===================================")
    print("ELIMINATING PAST SHARED ARTIFACTS")
    print("===================================")
    print("1 -> ELIMINATE ARTIFACTS")
    print("2 -> GO BACK TO MENU")

    choice = input(
        "\nInsert job_id: "
    ).strip()

    if choice == "1":
        reset_shared_artifacts()

    elif choice == "2":
        return

# =========================================================
# MAIN MENU
# =========================================================

def main():

    while True:

        print()
        print("===================================")
        print("TRAINING DEBUG CLI")
        print("===================================")

        print("1 -> Submit training")
        print("2 -> See job status")
        print("3 -> See experiments")
        print("4 -> Count saved trees")
        print("5 -> See validation metrics")
        print("6 -> Eliminate shared artifacts")
        print("0 -> Exit")

        choice = input(
            "\nSelect option: "
        ).strip()

        if choice == "1":

            submit_training_launcher()

        elif choice == "2":

            see_job_status_launcher()

        elif choice == "3":

            see_experiments_launcher()

        elif choice == "4":

            count_saved_trees_launcher()

        elif choice == "5":

            see_validation_metrics_launcher()

        elif choice == "6":

            reset_shared_artifacts_launcher()

        elif choice == "0":

            print()
            print("Closing CLI...")
            print()

            break

        else:

            print()
            print("[ERROR] Invalid option")
            print()


if __name__ == "__main__":

    main()