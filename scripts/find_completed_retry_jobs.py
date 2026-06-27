#!/usr/bin/env python3

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


DEFAULT_EFS_ROOT = Path("/mnt/efs/gp_artifacts")
DEFAULT_JOBS_DIR = DEFAULT_EFS_ROOT / "jobs"


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}

    try:
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)
    except Exception:
        return {}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []

    rows = []

    try:
        with path.open("r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()

                if not line:
                    continue

                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except Exception:
        return []

    return rows


def as_number(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)):
        return float(value)

    return None


def format_timestamp(value: Any) -> str:
    number = as_number(value)

    if number is None:
        return ""

    try:
        return datetime.fromtimestamp(number).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return ""


def find_first_numeric(obj: Any, candidate_keys: set[str]) -> Optional[float]:
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key in candidate_keys and isinstance(value, (int, float)):
                return float(value)

        for value in obj.values():
            found = find_first_numeric(value, candidate_keys)

            if found is not None:
                return found

    if isinstance(obj, list):
        for item in obj:
            found = find_first_numeric(item, candidate_keys)

            if found is not None:
                return found

    return None


def get_nested(obj: dict[str, Any], *keys: str) -> Any:
    current: Any = obj

    for key in keys:
        if not isinstance(current, dict):
            return None

        current = current.get(key)

    return current


def get_dataset_uri(job_record: dict[str, Any]) -> str:
    candidates = [
        get_nested(job_record, "training_request", "dataset_uri"),
        get_nested(job_record, "prepared_dataset", "schema", "dataset_uri"),
        get_nested(job_record, "prepared_dataset", "schema", "source_uri"),
    ]

    for value in candidates:
        if isinstance(value, str) and value:
            return value

    return ""


def get_dataset_rows(job_record: dict[str, Any], events: list[dict[str, Any]]) -> Optional[int]:
    prepared_dataset = job_record.get("prepared_dataset", {})

    metadata = prepared_dataset.get("preparation_metadata", {})

    for key in [
        "final_row_count",
        "original_row_count",
    ]:
        value = metadata.get(key)

        if isinstance(value, int):
            return value

    n_train = prepared_dataset.get("n_train")
    n_validation = prepared_dataset.get("n_validation")
    n_test = prepared_dataset.get("n_test")

    if all(isinstance(v, int) for v in [n_train, n_validation, n_test]):
        return int(n_train + n_validation + n_test)

    for event in events:
        if event.get("event") == "dataset_ready":
            n_train = event.get("n_train")
            n_validation = event.get("n_validation")
            n_test = event.get("n_test")

            if all(isinstance(v, int) for v in [n_train, n_validation, n_test]):
                return int(n_train + n_validation + n_test)

    return None


def is_2m_dataset(job_record: dict[str, Any], events: list[dict[str, Any]]) -> bool:
    rows = get_dataset_rows(job_record, events)

    if rows == 2_000_000:
        return True

    dataset_uri = get_dataset_uri(job_record).lower()

    return (
        "2000000" in dataset_uri
        or "2m" in dataset_uri
        or "2_m" in dataset_uri
    )


def count_retry_events(events: list[dict[str, Any]]) -> tuple[int, int, int]:
    shard_retry_events = 0
    shard_retry_scheduled_events = 0
    retry_attempt_events = 0

    for event in events:
        event_name = str(event.get("event", ""))

        if event_name == "shard_retry":
            shard_retry_events += 1

        if event_name == "shard_retry_scheduled":
            shard_retry_scheduled_events += 1

        attempt_id = event.get("attempt_id")

        if isinstance(attempt_id, int) and attempt_id > 1:
            retry_attempt_events += 1

    return shard_retry_events, shard_retry_scheduled_events, retry_attempt_events


def extract_retry_details(events: list[dict[str, Any]], limit: int = 5) -> str:
    details = []

    for event in events:
        event_name = str(event.get("event", ""))

        if event_name not in {
            "shard_retry",
            "shard_retry_scheduled",
            "shard_retry_not_allowed",
            "shard_retry_skipped_no_candidate",
        }:
            continue

        task_id = event.get("task_id", "")
        worker_id = event.get("worker_id") or event.get("previous_worker_id") or ""
        retry_worker_id = event.get("retry_worker_id", "")
        reason = event.get("reason") or event.get("error_message") or ""

        text = event_name

        if task_id:
            text += f" task={task_id}"

        if worker_id:
            text += f" worker={worker_id}"

        if retry_worker_id:
            text += f" retry_worker={retry_worker_id}"

        if reason:
            text += f" reason={reason}"

        details.append(text)

        if len(details) >= limit:
            break

    return " | ".join(details)


def get_summary_counters(summary: dict[str, Any]) -> dict[str, Any]:
    counters = summary.get("counters", {})

    if isinstance(counters, dict):
        return counters

    return {}


def analyze_job(job_dir: Path) -> Optional[dict[str, Any]]:
    job_record = load_json(job_dir / "job_record.json")
    summary = load_json(job_dir / "metrics" / "scalability_summary.json")
    events = load_jsonl(job_dir / "metrics" / "scalability_events.jsonl")

    if not job_record:
        return None

    job_id = job_record.get("job_id") or job_dir.name
    status = str(job_record.get("status", "")).upper()

    counters = get_summary_counters(summary)

    retry_count_summary = counters.get("retry_count", summary.get("retry_count", 0))

    if not isinstance(retry_count_summary, int):
        retry_count_summary = 0

    shard_retry_events, shard_retry_scheduled_events, retry_attempt_events = count_retry_events(events)

    effective_retry_count = max(
        retry_count_summary,
        shard_retry_events,
        shard_retry_scheduled_events,
        retry_attempt_events,
    )

    failed_tree_count = counters.get("failed_tree_count", summary.get("failed_tree_count", 0))

    if not isinstance(failed_tree_count, int):
        failed_tree_count = 0

    completed_tree_count = counters.get(
        "completed_tree_count",
        summary.get("completed_tree_count"),
    )

    if not isinstance(completed_tree_count, int):
        completed_tree_count = None

    dataset_rows = get_dataset_rows(job_record, events)
    dataset_uri = get_dataset_uri(job_record)

    training_request = job_record.get("training_request", {})

    task_type = training_request.get("task_type", "")
    dataset_scenario = training_request.get("dataset_scenario", "")
    target_column = training_request.get("target_column", "")

    worker_count = summary.get("worker_count")
    n_estimators_total = summary.get("n_estimators_total") or training_request.get("n_estimators_total")

    total_time_seconds = summary.get("total_time_seconds")
    training_time_seconds = summary.get("training_time_seconds")
    throughput = summary.get("throughput_trees_per_second")

    return {
        "job_id": job_id,
        "status": status,
        "created_at": format_timestamp(job_record.get("created_at")),
        "updated_at": format_timestamp(job_record.get("updated_at")),
        "dataset_rows": dataset_rows,
        "is_2m_dataset": is_2m_dataset(job_record, events),
        "dataset_uri": dataset_uri,
        "dataset_scenario": dataset_scenario,
        "task_type": task_type,
        "target_column": target_column,
        "worker_count": worker_count,
        "n_estimators_total": n_estimators_total,
        "completed_tree_count": completed_tree_count,
        "failed_tree_count": failed_tree_count,
        "retry_count_summary": retry_count_summary,
        "shard_retry_events": shard_retry_events,
        "shard_retry_scheduled_events": shard_retry_scheduled_events,
        "retry_attempt_events": retry_attempt_events,
        "effective_retry_count": effective_retry_count,
        "total_time_seconds": total_time_seconds,
        "training_time_seconds": training_time_seconds,
        "throughput_trees_per_second": throughput,
        "selected_experiment_id": job_record.get("selected_experiment_id", ""),
        "model_id": job_record.get("model_id", ""),
        "message": job_record.get("message", ""),
        "retry_details": extract_retry_details(events),
        "job_dir": str(job_dir),
    }


def scan_jobs(jobs_dir: Path, only_2m: bool) -> list[dict[str, Any]]:
    rows = []

    for job_dir in sorted(jobs_dir.glob("job_*")):
        if not job_dir.is_dir():
            continue

        row = analyze_job(job_dir)

        if row is None:
            continue

        if row["status"] != "COMPLETED":
            continue

        if row["effective_retry_count"] < 1:
            continue

        if only_2m and not row["is_2m_dataset"]:
            continue

        rows.append(row)

    rows.sort(
        key=lambda row: (
            int(row["worker_count"] or 0),
            str(row["created_at"]),
            row["job_id"],
        )
    )

    return rows


def print_rows(rows: list[dict[str, Any]]) -> None:
    if not rows:
        print("[INFO] Nessun job COMPLETED con retry trovato.")
        return

    print()
    print("=== JOB COMPLETED CON ALMENO 1 RETRY ===")
    print()

    for row in rows:
        print(
            f"{row['job_id']} | "
            f"workers={row['worker_count']} | "
            f"trees={row['n_estimators_total']} | "
            f"retry={row['effective_retry_count']} | "
            f"failed_trees={row['failed_tree_count']} | "
            f"dataset_rows={row['dataset_rows']} | "
            f"time={row['training_time_seconds']}s"
        )

        print(f"  dataset: {row['dataset_uri']}")

        if row["retry_details"]:
            print(f"  retry details: {row['retry_details']}")

        print()


def save_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "job_id",
        "status",
        "created_at",
        "updated_at",
        "dataset_rows",
        "is_2m_dataset",
        "dataset_uri",
        "dataset_scenario",
        "task_type",
        "target_column",
        "worker_count",
        "n_estimators_total",
        "completed_tree_count",
        "failed_tree_count",
        "retry_count_summary",
        "shard_retry_events",
        "shard_retry_scheduled_events",
        "retry_attempt_events",
        "effective_retry_count",
        "total_time_seconds",
        "training_time_seconds",
        "throughput_trees_per_second",
        "selected_experiment_id",
        "model_id",
        "message",
        "retry_details",
        "job_dir",
    ]

    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow(row)

    print(f"[OK] CSV salvato in: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Trova job COMPLETED su EFS che hanno eseguito almeno un retry."
    )

    parser.add_argument(
        "--jobs-dir",
        type=Path,
        default=DEFAULT_JOBS_DIR,
        help="Directory dei job su EFS. Default: /mnt/efs/gp_artifacts/jobs",
    )

    parser.add_argument(
        "--only-2m",
        action="store_true",
        help="Mostra solo job riferiti a dataset da 2 milioni di righe.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_EFS_ROOT / "completed_retry_jobs.csv",
        help="Percorso CSV di output.",
    )

    args = parser.parse_args()

    if not args.jobs_dir.exists():
        raise SystemExit(f"[ERROR] jobs-dir non trovata: {args.jobs_dir}")

    rows = scan_jobs(
        jobs_dir=args.jobs_dir,
        only_2m=args.only_2m,
    )

    print_rows(rows)
    save_csv(rows, args.output)


if __name__ == "__main__":
    main()