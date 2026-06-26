import csv
import json
import math
from asyncio.windows_events import NULL
from pathlib import Path
from datetime import datetime
from typing import Any, Optional

import pandas as pd
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parent
RUNS_CSV = BASE_DIR / "scalability_runs.csv"
JOBS_DIR = BASE_DIR / "distributed_jobs"
OUTPUT_DIR = BASE_DIR / "plots"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


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
    """
    Cerca ricorsivamente il primo valore numerico associato a una delle chiavi candidate.
    Serve perché i JSON possono avere nomi leggermente diversi.
    """
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

    # Gestione ISO con Z finale
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

    start = min(timestamps)
    end = max(timestamps)

    duration = (end - start).total_seconds()
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


def count_retry_events(events: list[dict[str, Any]]) -> int:
    count = 0

    for event in events:
        text = json.dumps(event).lower()
        if "retry" in text:
            count += 1

    return count


def extract_completed_trees(
    summary: dict[str, Any],
    experiment_record: dict[str, Any],
    task_ledger: dict[str, Any],
) -> Optional[int]:
    counters = summary.get("counters", {})

    if isinstance(counters, dict):
        value = counters.get("completed_tree_count")
        if isinstance(value, (int, float)):
            return int(value)

    value = summary.get("completed_tree_count")
    if isinstance(value, (int, float)):
        return int(value)

    value = find_first_numeric(
        summary,
        [
            "completed_tree_count",
            "completed_trees",
            "trees_completed",
            "n_completed_trees",
        ],
    )
    if value is not None:
        return int(value)

    value = find_first_numeric(
        experiment_record,
        [
            "completed_tree_count",
            "completed_trees",
            "trees_completed",
            "n_completed_trees",
        ],
    )
    if value is not None:
        return int(value)

    value = find_first_numeric(
        task_ledger,
        [
            "completed_tree_count",
            "completed_trees",
            "trees_completed",
            "n_completed_trees",
        ],
    )
    if value is not None:
        return int(value)

    return None


def extract_expected_trees(
    summary: dict[str, Any],
    experiment_record: dict[str, Any],
) -> Optional[int]:
    value = summary.get("n_estimators_total")
    if isinstance(value, (int, float)):
        return int(value)

    value = summary.get("expected_tree_count")
    if isinstance(value, (int, float)):
        return int(value)

    value = find_first_numeric(
        summary,
        [
            "n_estimators_total",
            "expected_tree_count",
            "expected_trees",
            "total_trees",
            "n_estimators",
            "tree_count",
        ],
    )
    if value is not None:
        return int(value)

    value = find_first_numeric(
        experiment_record,
        [
            "n_estimators_total",
            "expected_tree_count",
            "expected_trees",
            "total_trees",
            "n_estimators",
            "tree_count",
        ],
    )
    if value is not None:
        return int(value)

    return None



def extract_duration_seconds(
    summary: dict[str, Any],
    job_record: dict[str, Any],
    experiment_record: dict[str, Any],
    events: list[dict[str, Any]],
) -> Optional[float]:
    """
    Durata principale usata per il confronto end-to-end.
    Nel nostro progetto il campo corretto è summary["total_time_seconds"].
    """

    # Caso principale del tuo progetto
    value = summary.get("total_time_seconds")
    if isinstance(value, (int, float)) and value > 0:
        return float(value)

    # Fallback: solo tempo di training distribuito
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

    millisecond_candidates = [
        "duration_ms",
        "elapsed_ms",
        "wall_time_ms",
        "runtime_ms",
        "training_duration_ms",
        "total_duration_ms",
    ]

    for source in [summary, job_record, experiment_record]:
        value = find_first_numeric(source, millisecond_candidates)
        if value is not None and value > 0:
            return float(value) / 1000.0

    return infer_duration_from_events(events)


def load_runs() -> list[dict[str, str]]:
    with RUNS_CSV.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def analyze() -> pd.DataFrame:
    rows = []

    for run in load_runs():
        include = run.get("include", "true").strip().lower()
        if include not in ["true", "1", "yes", "y", "si", "sì"]:
            continue

        label = run["label"].strip()
        job_id = run["job_id"].strip()

        if not job_id or job_id == "...":
            continue

        instances = int(run["instances"])
        workers_per_instance = int(run["workers_per_instance"])
        total_workers = int(run["total_workers"])

        job_dir = JOBS_DIR / job_id
        exp_id = f"{job_id}_exp_000"

        job_record_path = job_dir / "job_record.json"
        task_ledger_path = job_dir / "task_ledger.json"
        summary_path = job_dir / "metrics" / "scalability_summary.json"
        events_path = job_dir / "metrics" / "scalability_events.jsonl"
        experiment_record_path = job_dir / "experiments" / exp_id / "experiment_record.json"

        job_record = load_json(job_record_path)
        task_ledger = load_json(task_ledger_path)
        summary = load_json(summary_path)
        experiment_record = load_json(experiment_record_path)
        events = load_jsonl(events_path)

        duration_seconds = extract_duration_seconds(
            summary=summary,
            job_record=job_record,
            experiment_record=experiment_record,
            events=events,
        )

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

        training_time_seconds = summary.get("training_time_seconds")
        if not isinstance(training_time_seconds, (int, float)):
            training_time_seconds = None

        total_time_seconds = summary.get("total_time_seconds")
        if not isinstance(total_time_seconds, (int, float)):
            total_time_seconds = duration_seconds

        throughput = summary.get("throughput_trees_per_second")
        if not isinstance(throughput, (int, float)):
            throughput = None
            if duration_seconds is not None and duration_seconds > 0:
                throughput = completed_trees / duration_seconds

        failed_attempts = count_failed_attempts(events)
        retry_events = count_retry_events(events)

        rows.append(
            {
                "label": label,
                "job_id": job_id,
                "instances": instances,
                "workers_per_instance": workers_per_instance,
                "total_workers": total_workers,
                "expected_trees": expected_trees,
                "completed_trees": completed_trees,
                "duration_seconds": duration_seconds,
                "duration_minutes": None if duration_seconds is None else duration_seconds / 60,
                "total_time_seconds": total_time_seconds,
                "training_time_seconds": training_time_seconds,
                "training_time_minutes": None if training_time_seconds is None else training_time_seconds / 60,
                "throughput_trees_per_second": throughput,
                "failed_attempts": failed_attempts,
                "retry_events": retry_events,
                "notes": run.get("notes", ""),
            }
        )

    df = pd.DataFrame(rows)

    if df.empty:
        return df

    df = df.sort_values(
        by=["total_workers", "instances", "workers_per_instance", "label"]
    ).reset_index(drop=True)

    # Baseline: preferisce 1x1, altrimenti usa il job più lento/piccolo disponibile
    baseline_rows = df[df["label"] == "1x1"]

    if not baseline_rows.empty and not pd.isna(baseline_rows.iloc[0]["duration_seconds"]):
        baseline_time = float(baseline_rows.iloc[0]["duration_seconds"])
    else:
        baseline_time = float(df["duration_seconds"].dropna().max())

    df["speedup"] = baseline_time / df["duration_seconds"]
    df["efficiency"] = df["speedup"] / df["total_workers"]

    return df


def save_results(df: pd.DataFrame) -> None:
    output_csv = BASE_DIR / "scalability_results.csv"
    df.to_csv(output_csv, index=False, encoding="utf-8")
    print(f"[OK] Risultati salvati in: {output_csv}")

def add_grouped_bar_positions(
    plot_df: pd.DataFrame,
        spacing: float,
        group_column: str = "total_workers",
) -> pd.DataFrame:
    """
    Aggiunge una colonna x_position per disegnare barre affiancate
    quando più configurazioni hanno lo stesso numero totale di worker.
    """
    plot_df = plot_df.copy()

    group_sizes = plot_df.groupby(group_column)["label"].transform("count")
    group_index = plot_df.groupby(group_column).cumcount()

    plot_df["x_position"] = (
        plot_df[group_column]
        + (group_index - (group_sizes - 1) / 2) * spacing
    )

    return plot_df

def format_minutes_seconds(minutes: float) -> str:
    total_seconds = int(round(float(minutes) * 60))
    mins = total_seconds // 60
    secs = total_seconds % 60
    return f"{mins}m {secs:02d}s"

def plot_duration(df: pd.DataFrame) -> None:
    plot_df = df.dropna(subset=["duration_minutes", "total_workers"]).copy()

    if plot_df.empty:
        print("[WARN] Grafico tempo non generato: nessuna durata valida trovata.")
        return

    plot_df["total_workers"] = pd.to_numeric(plot_df["total_workers"]).astype(int)
    plot_df["duration_minutes"] = pd.to_numeric(plot_df["duration_minutes"])

    plot_df = plot_df.sort_values(
        by=["total_workers", "instances", "workers_per_instance", "label"]
    ).reset_index(drop=True)

    bar_width = 0.40

    plot_df = add_grouped_bar_positions(plot_df, bar_width+0.10)

    plt.figure(figsize=(13, 6))

    plt.bar(
        plot_df["x_position"],
        plot_df["duration_minutes"],
        width=bar_width,
    )

    y_max = plot_df["duration_minutes"].max()
    label_offset = y_max * 0.015

    for _, row in plot_df.iterrows():
        x = row["x_position"]
        y = row["duration_minutes"]

        plt.text(
            x,
            y + label_offset,
            row["label"],
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

    worker_ticks = sorted(plot_df["total_workers"].unique())

    plt.xticks(
        worker_ticks,
        [str(worker) for worker in worker_ticks],
    )

    plt.xlabel("Numero totale di worker")
    plt.ylabel("Tempo totale training (minuti)")
    plt.title("Tempo di training al crescere del numero di worker")
    plt.ylim(0, y_max * 1.15)
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()

    output = OUTPUT_DIR / "training_time_by_workers_grouped.png"
    plt.savefig(output, dpi=200)
    plt.close()

    print(f"[OK] Grafico salvato in: {output}")

def plot_speedup(df: pd.DataFrame) -> None:
    plot_df = df.dropna(subset=["speedup", "total_workers"]).copy()

    if plot_df.empty:
        print("[WARN] Grafico speedup non generato: nessun valore valido trovato.")
        return

    plot_df["total_workers"] = pd.to_numeric(plot_df["total_workers"]).astype(int)
    plot_df["speedup"] = pd.to_numeric(plot_df["speedup"])

    plot_df = plot_df.sort_values(
        by=["total_workers", "instances", "workers_per_instance", "label"]
    ).reset_index(drop=True)

    bar_width = 0.40

    plot_df = add_grouped_bar_positions(plot_df, bar_width+0.10)

    plt.figure(figsize=(13, 6))

    plt.bar(
        plot_df["x_position"],
        plot_df["speedup"],
        width=bar_width,
        label="Speedup reale",
    )

    y_max = plot_df["speedup"].max()
    label_offset = y_max * 0.02

    for _, row in plot_df.iterrows():
        x = row["x_position"]
        y = row["speedup"]

        plt.text(
            x,
            y + label_offset,
            row["label"],
            ha="center",
            va="bottom",
            fontsize=9,
        )

        plt.text(
            x,
            y * 0.5,
            f"{y:.2f}x",
            ha="center",
            va="center",
            fontsize=8,
            rotation=90,
            fontweight="bold",
        )

    worker_ticks = sorted(plot_df["total_workers"].unique())

    plt.xticks(
        worker_ticks,
        [str(worker) for worker in worker_ticks],
    )

    # Linea ideale: speedup = numero worker.
    # È utile come riferimento teorico.
    ideal_x = worker_ticks
    ideal_y = worker_ticks

    plt.plot(
        ideal_x,
        ideal_y,
        linestyle="--",
        linewidth=1,
        label="Speedup ideale",
    )

    plt.xlabel("Numero totale di worker")
    plt.ylabel("Speedup")
    plt.title("Speedup rispetto alla baseline locale")
    plt.ylim(0, max(max(ideal_y), y_max) * 1.15)
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()

    output = OUTPUT_DIR / "speedup_by_workers.png"
    plt.savefig(output, dpi=200)
    plt.close()

    print(f"[OK] Grafico salvato in: {output}")


def plot_throughput(df: pd.DataFrame) -> None:
    plot_df = df.dropna(subset=["throughput_trees_per_second", "total_workers"]).copy()

    if plot_df.empty:
        print("[WARN] Grafico throughput non generato: nessun valore valido trovato.")
        return

    plot_df["total_workers"] = pd.to_numeric(plot_df["total_workers"]).astype(int)
    plot_df["throughput_trees_per_second"] = pd.to_numeric(
        plot_df["throughput_trees_per_second"]
    )

    plot_df = plot_df.sort_values(
        by=["total_workers", "instances", "workers_per_instance", "label"]
    ).reset_index(drop=True)

    bar_width = 0.40

    plot_df = add_grouped_bar_positions(plot_df,bar_width+0.10)

    plt.figure(figsize=(13, 6))

    plt.bar(
        plot_df["x_position"],
        plot_df["throughput_trees_per_second"],
        width=bar_width,
    )

    y_max = plot_df["throughput_trees_per_second"].max()
    label_offset = y_max * 0.02

    for _, row in plot_df.iterrows():
        x = row["x_position"]
        y = row["throughput_trees_per_second"]

        plt.text(
            x,
            y + label_offset,
            row["label"],
            ha="center",
            va="bottom",
            fontsize=9,
        )

        plt.text(
            x,
            y * 0.5,
            f"{y:.2f}",
            ha="center",
            va="center",
            fontsize=8,
            rotation=90,
            fontweight="bold",
        )

    worker_ticks = sorted(plot_df["total_workers"].unique())

    plt.xticks(
        worker_ticks,
        [str(worker) for worker in worker_ticks],
    )

    plt.xlabel("Numero totale di worker")
    plt.ylabel("Throughput (alberi/sec)")
    plt.title("Throughput al crescere del numero di worker")
    plt.ylim(0, y_max * 1.18)
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()

    output = OUTPUT_DIR / "throughput_by_workers.png"
    plt.savefig(output, dpi=200)
    plt.close()

    print(f"[OK] Grafico salvato in: {output}")

def plot_efficiency(df: pd.DataFrame) -> None:
    plot_df = df.dropna(subset=["efficiency", "total_workers"]).copy()

    if plot_df.empty:
        print("[WARN] Grafico efficienza non generato: nessun valore valido trovato.")
        return

    plot_df["total_workers"] = pd.to_numeric(plot_df["total_workers"]).astype(int)
    plot_df["efficiency"] = pd.to_numeric(plot_df["efficiency"])

    plot_df["efficiency_percent"] = plot_df["efficiency"] * 100.0

    plot_df = plot_df.sort_values(
        by=["total_workers", "instances", "workers_per_instance", "label"]
    ).reset_index(drop=True)

    bar_width = 0.40

    plot_df = add_grouped_bar_positions(plot_df, bar_width+0.10)

    plt.figure(figsize=(13, 6))

    plt.bar(
        plot_df["x_position"],
        plot_df["efficiency_percent"],
        width=bar_width,
        label="Efficienza reale",
    )

    y_max = plot_df["efficiency_percent"].max()
    label_offset = y_max * 0.02

    for _, row in plot_df.iterrows():
        x = row["x_position"]
        y = row["efficiency_percent"]

        plt.text(
            x,
            y + label_offset,
            row["label"],
            ha="center",
            va="bottom",
            fontsize=9,
        )

        plt.text(
            x,
            y * 0.5,
            f"{y:.1f}%",
            ha="center",
            va="center",
            fontsize=8,
            rotation=90,
            fontweight="bold",
        )

    worker_ticks = sorted(plot_df["total_workers"].unique())

    plt.xticks(
        worker_ticks,
        [str(worker) for worker in worker_ticks],
    )

    plt.axhline(
        100,
        linestyle="--",
        linewidth=1,
        label="Efficienza ideale 100%",
    )

    plt.xlabel("Numero totale di worker")
    plt.ylabel("Efficienza (%)")
    plt.title("Efficienza al crescere del numero di worker")
    plt.ylim(0, max(y_max * 1.18, 110))
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()

    output = OUTPUT_DIR / "efficiency_by_workers.png"
    plt.savefig(output, dpi=200)
    plt.close()

    print(f"[OK] Grafico salvato in: {output}")


def main() -> None:
    df = analyze()

    if df.empty:
        print("[ERRORE] Nessun job valido trovato. Controlla scalability_runs.csv e distributed_jobs/")
        return

    print("\n=== RISULTATI ESTRATTI ===")
    print(
        df[
            [
                "label",
                "job_id",
                "instances",
                "workers_per_instance",
                "total_workers",
                "duration_minutes",
                "throughput_trees_per_second",
                "speedup",
                "efficiency",
                "failed_attempts",
                "retry_events",
            ]
        ].to_string(index=False)
    )

    save_results(df)

    plot_duration(df)
    plot_speedup(df)
    plot_throughput(df)
    plot_efficiency(df)


if __name__ == "__main__":
    main()