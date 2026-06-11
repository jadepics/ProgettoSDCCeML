#!/usr/bin/env python3

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_JOBS_ROOT = Path("/mnt/efs/gp_artifacts/jobs")


def load_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def format_timestamp(ts: Any) -> str:
    if ts is None:
        return "-"

    try:
        value = float(ts)
        return datetime.fromtimestamp(value).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return str(ts)


def count_tree_joblibs(job_dir: Path) -> int:
    return len(list(job_dir.glob("experiments/*/trees/*.joblib")))


def count_tree_metadata(job_dir: Path) -> int:
    candidates = set()

    for path in job_dir.glob("experiments/*/trees/*.json"):
        name = path.name.lower()

        # Evita di contare file non relativi ai singoli alberi, se presenti.
        if "tree" in name:
            candidates.add(path)

    return len(candidates)


def read_expected_tree_count(job_dir: Path) -> int | None:
    counts: list[int] = []

    for experiment_record_path in job_dir.glob("experiments/*/experiment_record.json"):
        data = load_json(experiment_record_path)
        if not data:
            continue

        value = data.get("expected_tree_count")
        if isinstance(value, int):
            counts.append(value)

    if not counts:
        return None

    # In caso di più esperimenti, restituiamo la somma.
    return sum(counts)


def iter_failed_jobs(jobs_root: Path):
    for job_dir in jobs_root.glob("job_*"):
        if not job_dir.is_dir():
            continue

        job_record_path = job_dir / "job_record.json"
        if not job_record_path.exists():
            continue

        job_record = load_json(job_record_path)
        if not job_record:
            continue

        status = job_record.get("status")

        if status != "FAILED":
            continue

        tree_joblibs = count_tree_joblibs(job_dir)
        tree_metadata = count_tree_metadata(job_dir)
        expected_tree_count = read_expected_tree_count(job_dir)

        yield {
            "job_id": job_dir.name,
            "status": status,
            "tree_joblibs": tree_joblibs,
            "tree_metadata": tree_metadata,
            "expected_tree_count": expected_tree_count,
            "message": job_record.get("message", ""),
            "created_at": job_record.get("created_at"),
            "updated_at": job_record.get("updated_at"),
        }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="List FAILED training jobs ordered by updated_at."
    )

    parser.add_argument(
        "--jobs-root",
        type=Path,
        default=DEFAULT_JOBS_ROOT,
        help=f"Jobs root directory. Default: {DEFAULT_JOBS_ROOT}",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Maximum number of jobs to show. Default: 0 means no limit.",
    )

    args = parser.parse_args()

    jobs_root: Path = args.jobs_root

    if not jobs_root.exists():
        print(f"[ERROR] jobs root does not exist: {jobs_root}")
        return

    rows = list(iter_failed_jobs(jobs_root))

    rows.sort(
        key=lambda row: float(row["updated_at"] or 0),
        reverse=True,
    )

    if args.limit > 0:
        rows = rows[: args.limit]

    if not rows:
        print("No FAILED jobs found.")
        return

    print()
    print("FAILED JOBS ORDERED BY UPDATED_AT")
    print("=" * 80)
    print()

    for row in rows:
        expected = row["expected_tree_count"]

        if expected is None:
            progress = f"{row['tree_joblibs']}/?"
        else:
            progress = f"{row['tree_joblibs']}/{expected}"

        print(f"job_id:        {row['job_id']}")
        print(f"status:        {row['status']}")
        print(f"saved trees:   {progress}")
        print(f"tree metadata: {row['tree_metadata']}")
        print(f"created_at:    {format_timestamp(row['created_at'])}")
        print(f"updated_at:    {format_timestamp(row['updated_at'])}")
        print(f"message:       {row['message']}")
        print("-" * 80)


if __name__ == "__main__":
    main()