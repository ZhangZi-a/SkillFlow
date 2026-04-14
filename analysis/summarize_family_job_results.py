#!/usr/bin/env python3
"""Summarize family_job_runner.py results under a jobs/<run_name> directory.

Expected layout:
    jobs/<run_name>/
      <base_job_name>__<family_name>/
        result.json
        config.json
        <trial_name>/
          result.json
          ...

The script writes:
- summary.json: structured summary for all family groups
- reward_matrix.csv: one row per family, task columns ordered by difficulty ranking
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from job_summary_common import REPO_ROOT, TrialRecord, load_json, load_trial_record


SUMMARY_FILENAME = "summary.json"
REWARD_MATRIX_FILENAME = "reward_matrix.csv"
RANKING_FILENAME = "ALL_TASK_DIFFICULTY_RANKING.json"


def extract_family_name(dir_name: str) -> str:
    parts = dir_name.split("__")
    return parts[-1] if len(parts) > 1 else dir_name


def direct_result_dirs(jobs_dir: Path) -> list[Path]:
    if not jobs_dir.is_dir():
        return []
    return sorted(
        path
        for path in jobs_dir.iterdir()
        if path.is_dir() and (path / "result.json").exists()
    )


def has_nested_trial_results(path: Path) -> bool:
    try:
        return any(child.is_dir() and (child / "result.json").exists() for child in path.iterdir())
    except FileNotFoundError:
        return False


def find_family_group_dirs(jobs_dir: Path) -> list[Path]:
    return [path for path in direct_result_dirs(jobs_dir) if has_nested_trial_results(path)]


def find_group_trial_dirs(group_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in group_dir.iterdir()
        if path.is_dir() and (path / "result.json").exists()
    )


def resolve_dataset_root(group_dir: Path) -> Path | None:
    config = load_json(group_dir / "config.json") or {}

    tasks = config.get("tasks")
    if isinstance(tasks, list) and tasks:
        first_task = tasks[0]
        if isinstance(first_task, dict):
            task_path = first_task.get("path")
            if isinstance(task_path, str) and task_path.strip():
                task_dir = Path(task_path).expanduser()
                if not task_dir.is_absolute():
                    task_dir = (REPO_ROOT / task_dir).resolve()
                return task_dir.parent

    datasets = config.get("datasets")
    if isinstance(datasets, list) and datasets:
        first_dataset = datasets[0]
        if isinstance(first_dataset, dict):
            dataset_path = first_dataset.get("path")
            if isinstance(dataset_path, str) and dataset_path.strip():
                dataset_dir = Path(dataset_path).expanduser()
                if not dataset_dir.is_absolute():
                    dataset_dir = (REPO_ROOT / dataset_dir).resolve()
                return dataset_dir

    return None


def load_difficulty_ranking(dataset_root: Path | None) -> list[str]:
    if dataset_root is None:
        return []
    ranking_path = dataset_root / RANKING_FILENAME
    if not ranking_path.exists():
        return []

    data = json.loads(ranking_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        return []

    ranking: list[str] = []
    seen: set[str] = set()
    for item in data:
        if isinstance(item, str):
            task_name = item.strip()
            if task_name and task_name not in seen:
                ranking.append(task_name)
                seen.add(task_name)
    return ranking


def task_name_from_trial(trial_record: TrialRecord) -> str:
    task_path = ((trial_record.result.get("task_id") or {}).get("path") or "")
    if isinstance(task_path, str) and task_path.strip():
        return Path(task_path).name

    task_cfg = (trial_record.result.get("config") or {}).get("task") or {}
    cfg_task_path = task_cfg.get("path")
    if isinstance(cfg_task_path, str) and cfg_task_path.strip():
        return Path(cfg_task_path).name

    raw_name = trial_record.task_name or trial_record.trial_name
    if "__" in raw_name:
        return raw_name.split("__", 1)[0]
    return raw_name


def sort_trial_records_by_difficulty(
    trial_records: list[TrialRecord], ranking: list[str]
) -> list[TrialRecord]:
    ranking_index = {task_name: idx for idx, task_name in enumerate(ranking)}

    return sorted(
        trial_records,
        key=lambda record: (
            ranking_index.get(task_name_from_trial(record), len(ranking_index)),
            task_name_from_trial(record),
            record.trial_name,
        ),
    )


def load_group_trial_records(group_dir: Path, ranking: list[str]) -> list[TrialRecord]:
    trial_records: list[TrialRecord] = []
    for trial_dir in find_group_trial_dirs(group_dir):
        trial_record = load_trial_record(trial_dir)
        if trial_record:
            trial_records.append(trial_record)
    return sort_trial_records_by_difficulty(trial_records, ranking)


def summarize_jobs_dir(jobs_dir: Path) -> dict[str, Any]:
    group_dirs = find_family_group_dirs(jobs_dir)

    summary: dict[str, Any] = {
        "jobs_dir": str(jobs_dir),
        "structure": "family_groups",
        "total_groups": 0,
        "total_tasks": 0,
        "completed_tasks": 0,
        "error_count": 0,
        "avg_completion_rate": 0.0,
        "avg_interaction_turns": 0.0,
        "avg_input_tokens_k": 0.0,
        "avg_cache_tokens_k": 0.0,
        "avg_output_tokens_k": 0.0,
        "avg_cost_usd": None,
        "cost_task_count": 0,
        "avg_skill_usage_rate": 0.0,
        "groups": [],
        "reward_matrix": {},
        "task_details": {},
    }

    total_turns = 0.0
    total_input_tokens = 0
    total_cache_tokens = 0
    total_output_tokens = 0
    total_cost_usd = 0.0
    total_cost_tasks = 0

    for group_dir in group_dirs:
        group_name = extract_family_name(group_dir.name)
        dataset_root = resolve_dataset_root(group_dir)
        difficulty_ranking = load_difficulty_ranking(dataset_root)
        trial_records = load_group_trial_records(group_dir, difficulty_ranking)

        rewards: list[float] = []
        group_tasks: list[dict[str, Any]] = []
        group_completed = 0
        group_error_count = 0
        group_turns = 0.0
        group_input_tokens = 0
        group_cache_tokens = 0
        group_output_tokens = 0
        group_used_skill_tasks = 0
        group_cost_usd = 0.0
        group_cost_tasks = 0

        for trial_record in trial_records:
            reward = trial_record.reward
            task_name = task_name_from_trial(trial_record)
            rewards.append(reward)
            total_turns += trial_record.turns
            total_input_tokens += trial_record.input_tokens
            total_cache_tokens += trial_record.cache_tokens
            total_output_tokens += trial_record.output_tokens
            summary["total_tasks"] += 1
            group_turns += trial_record.turns
            group_input_tokens += trial_record.input_tokens
            group_cache_tokens += trial_record.cache_tokens
            group_output_tokens += trial_record.output_tokens
            if trial_record.cost_usd is not None:
                total_cost_usd += trial_record.cost_usd
                total_cost_tasks += 1
                group_cost_usd += trial_record.cost_usd
                group_cost_tasks += 1

            used_skill = trial_record.skill_use_calls > 0 or bool(trial_record.skills_used)
            if used_skill:
                group_used_skill_tasks += 1

            if reward >= 1.0:
                summary["completed_tasks"] += 1
                group_completed += 1
            if trial_record.exception_type:
                summary["error_count"] += 1
                group_error_count += 1

            rank_index = (
                difficulty_ranking.index(task_name) + 1 if task_name in difficulty_ranking else None
            )
            task_summary = {
                "trial_name": trial_record.trial_name,
                "task_name": task_name,
                "source": trial_record.source,
                "reward": reward,
                "turns": trial_record.turns,
                "input_tokens": trial_record.input_tokens,
                "cache_tokens": trial_record.cache_tokens,
                "output_tokens": trial_record.output_tokens,
                "model_name": trial_record.model_name,
                "pricing_model_name": trial_record.pricing_model_name,
                "cost_usd": trial_record.cost_usd,
                "reported_cost_usd": trial_record.reported_cost_usd,
                "cost_source": trial_record.cost_source,
                "difficulty_rank": rank_index,
                "skill_usage_rate": trial_record.skill_usage_rate,
                "skill_use_steps": trial_record.skill_use_steps,
                "skill_use_calls": trial_record.skill_use_calls,
                "skills_used": trial_record.skills_used,
                "exception_type": trial_record.exception_type,
                "exception_message": (trial_record.exception_message or "")[:200],
            }
            group_tasks.append(task_summary)

        n_tasks = len(trial_records)
        avg_reward = round(sum(rewards) / n_tasks, 4) if n_tasks else 0.0
        completion_rate = round(group_completed / n_tasks, 4) if n_tasks else 0.0
        avg_turns = round(group_turns / n_tasks, 2) if n_tasks else 0.0
        avg_input_tokens_k = round(group_input_tokens / n_tasks / 1000, 2) if n_tasks else 0.0
        avg_cache_tokens_k = round(group_cache_tokens / n_tasks / 1000, 2) if n_tasks else 0.0
        avg_output_tokens_k = round(group_output_tokens / n_tasks / 1000, 2) if n_tasks else 0.0
        avg_group_skill_usage = round(group_used_skill_tasks / n_tasks, 4) if n_tasks else 0.0
        avg_cost_usd = round(group_cost_usd / group_cost_tasks, 6) if group_cost_tasks else None

        group_summary = {
            "name": group_name,
            "dataset_root": str(dataset_root) if dataset_root else None,
            "difficulty_ranking": difficulty_ranking,
            "n_tasks": n_tasks,
            "completed_tasks": group_completed,
            "error_count": group_error_count,
            "completion_rate": completion_rate,
            "avg_reward": avg_reward,
            "avg_interaction_turns": avg_turns,
            "avg_input_tokens_k": avg_input_tokens_k,
            "avg_cache_tokens_k": avg_cache_tokens_k,
            "avg_output_tokens_k": avg_output_tokens_k,
            "avg_cost_usd": avg_cost_usd,
            "cost_task_count": group_cost_tasks,
            "avg_skill_usage_rate": avg_group_skill_usage,
            "tasks": group_tasks,
        }
        summary["groups"].append(group_summary)
        summary["reward_matrix"][group_name] = [task["reward"] for task in group_tasks]
        summary["task_details"][group_name] = group_tasks

    summary["total_groups"] = len(summary["groups"])
    if summary["total_tasks"] > 0:
        total_tasks = summary["total_tasks"]
        summary["avg_completion_rate"] = round(summary["completed_tasks"] / total_tasks, 4)
        summary["avg_interaction_turns"] = round(total_turns / total_tasks, 2)
        summary["avg_input_tokens_k"] = round(total_input_tokens / total_tasks / 1000, 2)
        summary["avg_cache_tokens_k"] = round(total_cache_tokens / total_tasks / 1000, 2)
        summary["avg_output_tokens_k"] = round(total_output_tokens / total_tasks / 1000, 2)
        summary["avg_cost_usd"] = round(total_cost_usd / total_cost_tasks, 6) if total_cost_tasks else None
        summary["cost_task_count"] = total_cost_tasks
        used_skill_tasks = sum(
            1
            for group in summary["groups"]
            for task in group["tasks"]
            if task.get("skill_use_calls", 0) > 0
        )
        summary["avg_skill_usage_rate"] = round(used_skill_tasks / total_tasks, 4)

    return summary


def write_summary_json(jobs_dir: Path, summary: dict[str, Any]) -> Path:
    output_path = jobs_dir / SUMMARY_FILENAME
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path


def write_reward_matrix_csv(jobs_dir: Path, summary: dict[str, Any]) -> Path:
    output_path = jobs_dir / REWARD_MATRIX_FILENAME
    max_tasks = max((len(group.get("tasks", [])) for group in summary["groups"]), default=0)

    with output_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.writer(csv_file)
        header = ["group"]
        for index in range(max_tasks):
            suffix = index + 1
            header.extend(
                [
                    f"task_{suffix}_name",
                    f"task_{suffix}_rank",
                    f"task_{suffix}_reward",
                    f"task_{suffix}_turns",
                    f"task_{suffix}_trial_name",
                ]
            )
        writer.writerow(header)

        for group in summary["groups"]:
            row: list[Any] = [group["name"]]
            for task in group.get("tasks", []):
                row.extend(
                    [
                        task.get("task_name"),
                        task.get("difficulty_rank"),
                        task.get("reward"),
                        task.get("turns"),
                        task.get("trial_name"),
                    ]
                )
            while len(row) < len(header):
                row.append("")
            writer.writerow(row)

    return output_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Summarize family_job_runner.py results with difficulty-ranked tables"
    )
    parser.add_argument(
        "jobs_dir",
        type=str,
        help="Run directory such as jobs/kimi-cli-vllm-new-all",
    )
    args = parser.parse_args(argv)

    jobs_dir = Path(args.jobs_dir).expanduser()
    if not jobs_dir.is_absolute():
        jobs_dir = (REPO_ROOT / jobs_dir).resolve()

    if not jobs_dir.is_dir():
        print(f"[ERROR] Directory not found: {jobs_dir}")
        return 1

    group_dirs = find_family_group_dirs(jobs_dir)
    if not group_dirs:
        print(f"[ERROR] No family job group directories found under: {jobs_dir}")
        return 1

    summary = summarize_jobs_dir(jobs_dir)
    summary_path = write_summary_json(jobs_dir, summary)
    reward_matrix_path = write_reward_matrix_csv(jobs_dir, summary)

    print(f"[INFO] Wrote summary: {summary_path}")
    print(f"[INFO] Wrote reward matrix: {reward_matrix_path}")
    print()
    print("=== Summary ===")
    print(f"Total groups: {summary['total_groups']}")
    print(f"Total tasks: {summary['total_tasks']}")
    print(f"Completed tasks: {summary['completed_tasks']}")
    print(f"Error count: {summary['error_count']}")
    print(f"Average completion rate: {summary['avg_completion_rate']:.2%}")
    print(f"Average interaction turns: {summary['avg_interaction_turns']:.2f}")
    print(f"Average input tokens (K): {summary['avg_input_tokens_k']:.2f}")
    print(f"Average cache tokens (K): {summary['avg_cache_tokens_k']:.2f}")
    print(f"Average output tokens (K): {summary['avg_output_tokens_k']:.2f}")
    if summary['avg_cost_usd'] is None:
        print("Average cost (USD): n/a")
    else:
        print(f"Average cost (USD): {summary['avg_cost_usd']:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
