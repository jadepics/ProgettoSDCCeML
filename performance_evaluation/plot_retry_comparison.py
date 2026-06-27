import csv
import json
from pathlib import Path
from datetime import datetime
from typing import Any, Optional

import pandas as pd
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parent
JOBS_DIR = BASE_DIR / "distributed_jobs"
OUTPUT_DIR = BASE_DIR / "plots"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


RUNS = [
    {
        "config": "8x1",
        "retry_label": "con evento",
        "event_kind": "failure parziale",
        "job_id": "job_0c79c0847f294325a843439ad53eda43",
        "instances": 8,
        "workers_per_instance": 1,
        "total_workers": 8,
    },
    {
        "config": "8x1",
        "retry_label": "run pulita",
        "event_kind": "nessuna anomalia",
        "job_id": "job_5a445e641aee47228340290d1129c0dd",
        "instances": 8,
        "workers_per_instance": 1,
        "total_workers": 8,
    },
    {
        "config": "8x2",
        "retry_label": "con evento",
        "event_kind": "failure parziale",
        "job_id": "job_ad2a4f8fb7c84b00b430112347e4af46",
        "instances": 8,
        "workers_per_instance": 2,
        "total_workers": 16,
    },
    {
        "config": "8x2",
        "retry_label": "run pulita",
        "event_kind": "nessuna anomalia",
        "job_id": "job_356f180845df4b17a2c55c6ec9f22f70",
        "instances": 8,
        "workers_per_instance": 2,
        "total_workers": 16,
    },
    {
        "config": "3x3",
        "retry_label": "con evento",
        "event_kind": "retry reale",
        "job_id": "job_b7a0557e7eee400aa5473aaaedffee8d",
        "instances": 3,
        "workers_per_instance": 3,
        "total_workers": 9,
    },
    {
        "config": "3x3",
        "retry_label": "run pulita",
        "event_kind": "nessuna anomalia",
        "job_id": "job_8694995956964f2baf51c5a5201265d6",
        "instances": 3,
        "workers_per_instance": 3,
        "total_workers": 9,
    },
]

def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        print(f"[WARN] File mancante: {path}")
        return {}

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        print(f"[WARN] File mancante: {path}")
        return []

    events = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                pass

    return events


def find_first_numeric(obj: Any, candidate_keys: list[str]) -> Optional[float]:
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key in candidate_keys and isinstance(value, (int, float)):
                return float(value)

        for value in obj.values():
            result = find_first_numeric(value, candidate_keys)
            if result is not None:
                return result

    elif isinstance(obj, list):
        for item in obj:
            result = find_first_numeric(item, candidate_keys)
            if result is not None:
                return result

    return None


def parse_timestamp(value: Any) -> Optional[datetime]:
    if not isinstance(value, str):
        return None

    value = value.strip()

    if not value:
        return None

    value = value.replace("Z", "+00:00")

    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def extract_event_timestamp(event: dict[str, Any]) -> Optional[datetime]:
    for key in [
        "timestamp",
        "created_at",
        "time",
        "event_time",
        "recorded_at",
        "started_at",
        "completed_at",
    ]:
        ts = parse_timestamp(event.get(key))

        if ts is not None:
            return ts

    return None


def infer_duration_from_events(events: list[dict[str, Any]]) -> Optional[float]:
    timestamps = []

    for event in events:
        ts = extract_event_timestamp(event)

        if ts is not None:
            timestamps.append(ts)

    if len(timestamps) < 2:
        return None

    duration = (max(timestamps) - min(timestamps)).total_seconds()

    if duration <= 0:
        return None

    return duration


def count_failed_attempts(events: list[dict[str, Any]]) -> int:
    count = 0

    for event in events:
        text = json.dumps(event).lower()

        if (
            "attempt failed" in text
            or "attempt_failed" in text
            or "shard_failed" in text
            or "deadline_exceeded" in text
            or "unavailable" in text
            or "stream removed" in text
            or "oom" in text
        ):
            count += 1

    return count


def get_summary_counters(summary: dict[str, Any]) -> dict[str, Any]:
    counters = summary.get("counters", {})

    if isinstance(counters, dict):
        return counters

    return {}


def count_real_retry_events(summary: dict[str, Any], events: list[dict[str, Any]]) -> int:
    counters = get_summary_counters(summary)

    retry_count = counters.get("retry_count", summary.get("retry_count", 0))

    if not isinstance(retry_count, int):
        retry_count = 0

    shard_retry_events = 0
    retry_attempt_events = 0

    for event in events:
        event_name = str(event.get("event", ""))

        if event_name in {"shard_retry", "shard_retry_scheduled"}:
            shard_retry_events += 1

        attempt_id = event.get("attempt_id")

        if isinstance(attempt_id, int) and attempt_id > 1:
            retry_attempt_events += 1

    return max(retry_count, shard_retry_events, retry_attempt_events)

def count_retried_shards(events: list[dict[str, Any]]) -> int:
    retried_tasks = set()

    for event in events:
        event_name = str(event.get("event", ""))

        if event_name not in {"shard_retry", "shard_retry_scheduled"}:
            continue

        task_id = event.get("task_id")

        if isinstance(task_id, str) and task_id:
            retried_tasks.add(task_id)

    return len(retried_tasks)

def count_failure_recovery_events(events: list[dict[str, Any]]) -> int:
    count = 0

    for event in events:
        event_name = str(event.get("event", ""))
        text = json.dumps(event).lower()

        if (
            event_name in {"shard_failed", "shard_completed_with_failures"}
            or "some trees failed" in text
            or '"result_success": false' in text
            or "failed_tree_count" in text
        ):
            count += 1

    return count



def extract_duration_seconds(
    summary: dict[str, Any],
    job_record: dict[str, Any],
    experiment_record: dict[str, Any],
    events: list[dict[str, Any]],
) -> Optional[float]:
    value = summary.get("total_time_seconds")

    if isinstance(value, (int, float)) and value > 0:
        return float(value)

    value = summary.get("training_time_seconds")

    if isinstance(value, (int, float)) and value > 0:
        return float(value)

    candidates = [
        "total_time_seconds",
        "training_time_seconds",
        "total_training_seconds",
        "total_duration_seconds",
        "duration_seconds",
        "elapsed_seconds",
        "wall_clock_seconds",
        "wall_time_seconds",
        "job_duration_seconds",
        "makespan_seconds",
        "runtime_seconds",
        "total_wall_time_seconds",
        "fit_time_seconds",
        "execution_time_seconds",
        "elapsed_time_seconds",
    ]

    for source in [summary, job_record, experiment_record]:
        value = find_first_numeric(source, candidates)

        if value is not None and value > 0:
            return float(value)

    return infer_duration_from_events(events)


def extract_completed_trees(
    summary: dict[str, Any],
    experiment_record: dict[str, Any],
    task_ledger: dict[str, Any],
) -> Optional[int]:
    candidates = [
        "completed_tree_count",
        "completed_trees",
        "trees_completed",
        "n_completed_trees",
    ]

    for source in [summary, experiment_record, task_ledger]:
        value = find_first_numeric(source, candidates)

        if value is not None:
            return int(value)

    return None


def extract_expected_trees(
    summary: dict[str, Any],
    experiment_record: dict[str, Any],
) -> Optional[int]:
    candidates = [
        "n_estimators_total",
        "expected_tree_count",
        "expected_trees",
        "total_trees",
        "n_estimators",
        "tree_count",
    ]

    for source in [summary, experiment_record]:
        value = find_first_numeric(source, candidates)

        if value is not None:
            return int(value)

    return None


def find_experiment_record(job_dir: Path, job_id: str) -> dict[str, Any]:
    expected_path = job_dir / "experiments" / f"{job_id}_exp_000" / "experiment_record.json"

    if expected_path.exists():
        return load_json(expected_path)

    experiment_records = list((job_dir / "experiments").glob("*/experiment_record.json"))

    if experiment_records:
        return load_json(experiment_records[0])

    return {}


def load_run(run: dict[str, Any]) -> dict[str, Any]:
    job_id = run["job_id"]
    job_dir = JOBS_DIR / job_id

    job_record_path = job_dir / "job_record.json"
    task_ledger_path = job_dir / "task_ledger.json"
    summary_path = job_dir / "metrics" / "scalability_summary.json"
    events_path = job_dir / "metrics" / "scalability_events.jsonl"

    job_record = load_json(job_record_path)
    task_ledger = load_json(task_ledger_path)
    summary = load_json(summary_path)
    events = load_jsonl(events_path)
    experiment_record = find_experiment_record(job_dir, job_id)

    duration_seconds = extract_duration_seconds(
        summary=summary,
        job_record=job_record,
        experiment_record=experiment_record,
        events=events,
    )

    training_time_seconds = summary.get("training_time_seconds")

    if not isinstance(training_time_seconds, (int, float)):
        training_time_seconds = None

    total_time_seconds = summary.get("total_time_seconds")

    if not isinstance(total_time_seconds, (int, float)):
        total_time_seconds = duration_seconds

    completed_trees = extract_completed_trees(
        summary=summary,
        experiment_record=experiment_record,
        task_ledger=task_ledger,
    )

    expected_trees = extract_expected_trees(
        summary=summary,
        experiment_record=experiment_record,
    )

    if expected_trees is None:
        expected_trees = 400

    if completed_trees is None:
        completed_trees = expected_trees

    throughput = summary.get("throughput_trees_per_second")

    if not isinstance(throughput, (int, float)):
        throughput = None

        if duration_seconds is not None and duration_seconds > 0:
            throughput = completed_trees / duration_seconds

    real_retry_events = count_real_retry_events(summary, events)
    retried_shards = count_retried_shards(events)
    failure_recovery_events = count_failure_recovery_events(events)
    failed_attempts = count_failed_attempts(events)

    row = dict(run)
    row.update(
        {
            "duration_seconds": duration_seconds,
            "duration_minutes": None if duration_seconds is None else duration_seconds / 60,
            "total_time_seconds": total_time_seconds,
            "total_time_minutes": None if total_time_seconds is None else total_time_seconds / 60,
            "training_time_seconds": training_time_seconds,
            "training_time_minutes": None if training_time_seconds is None else training_time_seconds / 60,
            "expected_trees": expected_trees,
            "completed_trees": completed_trees,
            "throughput_trees_per_second": throughput,
            "real_retry_events": real_retry_events,
            "retried_shards": retried_shards,
            "failure_recovery_events": failure_recovery_events,
            "retry_events": real_retry_events,
            "failed_attempts": failed_attempts,

        }
    )

    return row


def build_dataframe() -> pd.DataFrame:
    rows = [load_run(run) for run in RUNS]
    df = pd.DataFrame(rows)

    df["config_order"] = df["config"].map({"8x1": 1, "8x2": 2, "3x3": 3})
    df["retry_order"] = df["retry_label"].map({"con evento": 1, "run pulita": 2})

    df = df.sort_values(
        by=["config_order", "retry_order"]
    ).reset_index(drop=True)

    return df


def add_overhead_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["overhead_vs_no_retry_seconds"] = None
    df["overhead_vs_no_retry_percent"] = None

    for config in sorted(df["config"].unique()):
        group = df[df["config"] == config]

        retry_row = group[group["retry_label"] == "con evento"]
        no_retry_row = group[group["retry_label"] == "run pulita"]

        if retry_row.empty or no_retry_row.empty:
            continue

        retry_time = retry_row.iloc[0]["duration_seconds"]
        no_retry_time = no_retry_row.iloc[0]["duration_seconds"]

        if pd.isna(retry_time) or pd.isna(no_retry_time) or no_retry_time <= 0:
            continue

        overhead_seconds = float(retry_time) - float(no_retry_time)
        overhead_percent = (overhead_seconds / float(no_retry_time)) * 100.0

        df.loc[
            (df["config"] == config) & (df["retry_label"] == "con retry"),
            "overhead_vs_no_retry_seconds",
        ] = overhead_seconds

        df.loc[
            (df["config"] == config) & (df["retry_label"] == "con retry"),
            "overhead_vs_no_retry_percent",
        ] = overhead_percent

    return df


def format_minutes_seconds(minutes: float) -> str:
    total_seconds = int(round(float(minutes) * 60))
    mins = total_seconds // 60
    secs = total_seconds % 60

    return f"{mins}m {secs:02d}s"


def plot_retry_comparison(df: pd.DataFrame) -> None:
    plot_df = df.dropna(subset=["duration_minutes"]).copy()

    if plot_df.empty:
        print("[WARN] Grafico non generato: nessuna durata valida trovata.")
        return

    configs = ["8x1", "8x2", "3x3"]
    x_base = {config: index for index, config in enumerate(configs)}

    bar_width = 0.34
    offsets = {
        "con evento": -bar_width / 2,
        "run pulita": bar_width / 2,
    }

    plt.figure(figsize=(11, 6))

    for retry_label in ["con evento", "run pulita"]:
        subset = plot_df[plot_df["retry_label"] == retry_label].copy()

        x_positions = [
            x_base[row["config"]] + offsets[retry_label]
            for _, row in subset.iterrows()
        ]

        plt.bar(
            x_positions,
            subset["duration_minutes"],
            width=bar_width,
            label=retry_label,
        )

    y_max = plot_df["duration_minutes"].max()
    label_offset = y_max * 0.025

    for _, row in plot_df.iterrows():
        x = x_base[row["config"]] + offsets[row["retry_label"]]
        y = row["duration_minutes"]

        plt.text(
            x,
            y + label_offset,
            row["retry_label"],
            ha="center",
            va="bottom",
            fontsize=9,
        )

        plt.text(
            x,
            y * 0.5,
            format_minutes_seconds(y),
            ha="center",
            va="center",
            fontsize=8,
            rotation=90,
            fontweight="bold",
        )

        if row["retry_label"] == "con evento":
            retried_shards = int(row["retried_shards"])

            if row["event_kind"] == "retry reale":
                event_text = f"crash controllato\nretry shard: {retried_shards}"
            else:
                event_text = f"failure parziale\nretry shard: {retried_shards}"

            plt.text(
                x,
                y * 0.08,
                event_text,
                ha="center",
                va="bottom",
                fontsize=7,
            )

    plt.xticks(
        [x_base[config] for config in configs],
        configs,
    )

    plt.xlabel("Configurazione distribuita")
    plt.ylabel("Tempo totale training (minuti)")
    plt.ylim(0, y_max * 1.22)
    plt.grid(axis="y", alpha=0.25)
    plt.title("Impatto di failure parziali e retry reali sul tempo di training")
    plt.legend(title="Scenario")
    plt.tight_layout()

    output = OUTPUT_DIR / "retry_overhead_training_time_grouped.png"
    plt.savefig(output, dpi=200)
    plt.close()

    print(f"[OK] Grafico salvato in: {output}")


def save_results(df: pd.DataFrame) -> None:
    output = BASE_DIR / "retry_overhead_results.csv"
    df.to_csv(output, index=False, encoding="utf-8")
    print(f"[OK] CSV salvato in: {output}")


def print_summary(df: pd.DataFrame) -> None:
    columns = [
        "config",
        "retry_label",
        "event_kind",
        "job_id",
        "duration_minutes",
        "training_time_minutes",
        "throughput_trees_per_second",
        "retried_shards",
        "real_retry_events",
        "failure_recovery_events",
        "failed_attempts",
        "overhead_vs_no_retry_seconds",
        "overhead_vs_no_retry_percent",
    ]

    print("\n=== CONFRONTO RETRY VS NO RETRY ===")
    print(df[columns].to_string(index=False))

    print("\n=== OVERHEAD ===")

    for config in ["8x1", "8x2", "3x3"]:
        row = df[
            (df["config"] == config)
            & (df["retry_label"] == "con evento")
        ]

        if row.empty:
            continue

        overhead_seconds = row.iloc[0]["overhead_vs_no_retry_seconds"]
        overhead_percent = row.iloc[0]["overhead_vs_no_retry_percent"]

        if pd.isna(overhead_seconds) or pd.isna(overhead_percent):
            print(f"{config}: overhead non calcolabile")
            continue

        print(
            f"{config}: evento +{overhead_seconds:.2f}s "
            f"({overhead_percent:.2f}% rispetto alla run senza retry)"
        )


def main() -> None:
    df = build_dataframe()
    df = add_overhead_columns(df)

    save_results(df)
    print_summary(df)
    plot_retry_comparison(df)


if __name__ == "__main__":
    main()