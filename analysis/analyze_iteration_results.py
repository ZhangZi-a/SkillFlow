#!/usr/bin/env python3
"""Summarize iterative skill-job results under a jobs/<run_name> directory.

This script is for group-based runs whose layout looks like:
    jobs/kimi-cli-vllm-few-task-skill-1/
      <group_name>/
        <trial_name>/
        shared_skills/
        skill_patch_history.jsonl
        result.json

This script summarizes iterative/grouped runs only.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path, PurePosixPath
from typing import Any

from job_summary_common import (
    REPO_ROOT,
    TrialRecord,
    collect_relative_files,
    extract_group_name,
    find_flat_trial_dirs,
    find_iterative_group_dirs,
    load_json,
    load_trial_record,
    normalize_rel_path,
    resolve_task_skills_dir,
    resolve_template_skills_dir,
    skill_names_from_state_files,
    sort_trial_records,
    write_summary_json,
)


def find_group_trial_dirs(group_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in group_dir.iterdir()
        if path.is_dir() and (path / "result.json").exists()
    )


def load_group_trial_records(group_dir: Path) -> list[TrialRecord]:
    trial_records: list[TrialRecord] = []
    for trial_dir in find_group_trial_dirs(group_dir):
        trial_record = load_trial_record(trial_dir)
        if trial_record:
            trial_records.append(trial_record)
    return sort_trial_records(trial_records)


def apply_trial_patch(state_files: set[str], trial_dir: Path) -> set[str]:
    next_state = set(state_files)

    applied = load_json(trial_dir / "skill_evolution" / "applied.json") or {}
    patch = load_json(trial_dir / "skill_evolution" / "patch.json") or {}

    deleted_paths = applied.get("deleted")
    upserted_paths = applied.get("upserted")

    if not isinstance(deleted_paths, list):
        deleted_paths = patch.get("delete_paths") or []
    if not isinstance(upserted_paths, list):
        raw_upserts = patch.get("upsert_files") or {}
        upserted_paths = list(raw_upserts.keys()) if isinstance(raw_upserts, dict) else []

    for deleted_path in deleted_paths:
        rel_path = normalize_rel_path(str(deleted_path))
        if not rel_path:
            continue
        prefix = f"{rel_path}/"
        next_state = {
            existing
            for existing in next_state
            if existing != rel_path and not existing.startswith(prefix)
        }

    for upserted_path in upserted_paths:
        rel_path = normalize_rel_path(str(upserted_path))
        if rel_path:
            next_state.add(rel_path)

    return next_state


def simulate_final_skill_counts(
    group_config: dict[str, Any],
    trial_records: list[TrialRecord],
) -> dict[str, tuple[int, list[str]]]:
    environment = group_config.get("environment") or {}
    env_kwargs = environment.get("kwargs") or {}
    copy_task_skills = bool(env_kwargs.get("copy_task_skills"))

    current_state_files = collect_relative_files(resolve_template_skills_dir(group_config))
    skill_state_by_trial: dict[str, tuple[int, list[str]]] = {}

    for trial_record in trial_records:
        if copy_task_skills:
            current_state_files |= collect_relative_files(resolve_task_skills_dir(trial_record.result))

        current_state_files = apply_trial_patch(current_state_files, trial_record.trial_dir)
        final_skill_names = skill_names_from_state_files(current_state_files)
        skill_state_by_trial[trial_record.trial_name] = (
            len(final_skill_names),
            final_skill_names,
        )

    return skill_state_by_trial


def summarize_jobs_dir(jobs_dir: Path) -> dict[str, Any]:
    group_dirs = find_iterative_group_dirs(jobs_dir)

    summary: dict[str, Any] = {
        "jobs_dir": str(jobs_dir),
        "structure": "iterative_groups",
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
        "avg_final_skill_count": 0.0,
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
    total_group_final_skill_count = 0.0

    for group_dir in group_dirs:
        group_name = extract_group_name(group_dir.name)
        group_config = load_json(group_dir / "config.json") or {}
        trial_records = load_group_trial_records(group_dir)
        skill_states = simulate_final_skill_counts(group_config, trial_records)

        rewards: list[float] = []
        group_tasks: list[dict[str, Any]] = []
        group_completed = 0
        group_error_count = 0
        group_used_skill_tasks = 0
        group_turns = 0.0
        group_input_tokens = 0
        group_cache_tokens = 0
        group_output_tokens = 0
        group_cost_usd = 0.0
        group_cost_tasks = 0

        for trial_record in trial_records:
            final_skill_count, final_skill_names = skill_states.get(
                trial_record.trial_name,
                (0, []),
            )
            trial_record.final_skill_count = final_skill_count
            trial_record.final_skill_names = final_skill_names

            used_skill = (
                trial_record.skill_use_steps > 0
                or trial_record.skill_use_calls > 0
                or bool(trial_record.skills_used)
            )

            reward = trial_record.reward
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
            if used_skill:
                group_used_skill_tasks += 1

            if reward >= 1.0:
                summary["completed_tasks"] += 1
                group_completed += 1
            if trial_record.exception_type:
                summary["error_count"] += 1
                group_error_count += 1

            task_summary = {
                "trial_name": trial_record.trial_name,
                "task_name": trial_record.task_name,
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
                "final_skill_count": final_skill_count,
                "skill_count": final_skill_count,
                "skill_usage_rate": 1.0 if used_skill else 0.0,
                "skill_use_steps": trial_record.skill_use_steps,
                "skill_use_calls": trial_record.skill_use_calls,
                "skills_used": trial_record.skills_used,
                "final_skill_names": final_skill_names,
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
        final_group_skill_names = group_tasks[-1]["final_skill_names"] if group_tasks else []
        final_group_skill_count = len(final_group_skill_names)
        total_group_final_skill_count += final_group_skill_count

        group_summary = {
            "name": group_name,
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
            "final_skill_count": final_group_skill_count,
            "final_skill_names": final_group_skill_names,
            "tasks": group_tasks,
        }
        summary["groups"].append(group_summary)
        summary["reward_matrix"][group_name] = rewards
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
            if task.get("skill_usage_rate", 0.0) > 0
        )
        summary["avg_skill_usage_rate"] = round(used_skill_tasks / total_tasks, 4)
    if summary["total_groups"] > 0:
        summary["avg_final_skill_count"] = round(
            total_group_final_skill_count / summary["total_groups"], 2
        )

    return summary


def write_reward_matrix_csv(jobs_dir: Path, summary: dict[str, Any]) -> Path:
    output_path = jobs_dir / "reward_matrix.csv"
    max_tasks = max((len(rewards) for rewards in summary["reward_matrix"].values()), default=0)

    with output_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["group", *[f"task_{index + 1}" for index in range(max_tasks)]])
        for group_name, rewards in summary["reward_matrix"].items():
            row: list[Any] = [group_name]
            row.extend(rewards)
            while len(row) < max_tasks + 1:
                row.append("")
            writer.writerow(row)

    return output_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize iterative skill-job results")
    parser.add_argument(
        "jobs_dir",
        type=str,
        help="Run directory such as jobs/kimi-cli-vllm-few-task-skill-1",
    )
    args = parser.parse_args(argv)

    jobs_dir = Path(args.jobs_dir).expanduser()
    if not jobs_dir.is_absolute():
        jobs_dir = (REPO_ROOT / jobs_dir).resolve()

    if not jobs_dir.is_dir():
        print(f"[ERROR] Directory not found: {jobs_dir}")
        return 1

    group_dirs = find_iterative_group_dirs(jobs_dir)
    flat_trial_dirs = find_flat_trial_dirs(jobs_dir)

    if not group_dirs:
        if flat_trial_dirs:
            print(
                "[ERROR] This directory looks like a flat trial run. "
                "Use the flat trial analysis tooling instead."
            )
            return 1
        print(f"[ERROR] No iterative group directories found under: {jobs_dir}")
        return 1

    summary = summarize_jobs_dir(jobs_dir)
    summary_path = write_summary_json(jobs_dir, summary)
    reward_matrix_path = write_reward_matrix_csv(jobs_dir, summary)

    print(f"[INFO] Wrote summary: {summary_path}")
    print(f"[INFO] Wrote reward matrix: {reward_matrix_path}")
    if flat_trial_dirs:
        print(
            f"[INFO] Ignored {len(flat_trial_dirs)} flat trial directories; "
            "this script only summarizes iterative group directories."
        )
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
    print(f"Average final skill count: {summary['avg_final_skill_count']:.2f}")
    print(f"Average skill usage rate: {summary['avg_skill_usage_rate']:.2%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
