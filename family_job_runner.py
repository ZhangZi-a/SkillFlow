#!/usr/bin/env python3
"""Run each dataset family as an isolated Harbor job without skill evolution.

This runner is similar to ``iterative_shared_skills_runner.py`` in job layout,
but it does not use the shared-skills environment or any post-trial patching.

Behavior:
- Treat each dataset in the base config as one family/group.
- Expand every valid task under that dataset into explicit ``tasks`` entries.
- Write outputs under ``jobs/<base_job_name>/<base_job_name>__<family_name>/``.
- Reuse ``-g`` as the total Docker/trial concurrency budget for the whole run.
- Run families one by one so the total number of concurrent Docker tasks never
  exceeds that global budget.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import os
import shutil
import subprocess
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urlparse

import yaml
from harbor import Job
from harbor.models.job.config import JobConfig
from harbor.models.task.paths import TaskPaths
from harbor.trial.hooks import TrialHookEvent

from libs.skill_evolution.patcher import ensure_standard_trajectory


ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

SHARED_ENV_IMPORT_PATH = "libs.terminus_env.environments.shared_skills_env:SharedSkillsDockerEnvironment"


@dataclass
class RunnerConfig:
    config_path: Path
    run_root_dir: Path | None = None
    max_parallel_trials: int | None = None
    dry_run: bool = False


@dataclass
class GroupResult:
    group_name: str
    job_name: str
    job_dir: Path
    task_count: int
    parallel_trials: int
    success: bool
    message: str


def load_job_config(path: Path) -> JobConfig:
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    return JobConfig.model_validate(data)


def sanitize_name(name: str) -> str:
    return name.replace("/", "-").replace(" ", "_").strip("-_")


def resolve_run_root_dir(base_config: JobConfig, requested_root: Path | None) -> Path:
    if requested_root is not None:
        if requested_root.is_absolute():
            return requested_root
        if requested_root.parts and requested_root.parts[0] == "jobs":
            return (ROOT_DIR / requested_root).resolve()
        return (ROOT_DIR / "jobs" / requested_root).resolve()

    base_job_name = sanitize_name(base_config.job_name or "job")
    return (ROOT_DIR / "jobs" / base_job_name).resolve()


def resolve_dataset_path(dataset_path: Path) -> Path:
    expanded = dataset_path.expanduser()
    if expanded.is_absolute():
        return expanded.resolve()
    return (ROOT_DIR / expanded).resolve()


def load_group_task_ranking(dataset_path: Path) -> list[str] | None:
    ranking_path = resolve_dataset_path(dataset_path) / "ALL_TASK_DIFFICULTY_RANKING.json"
    if not ranking_path.exists():
        return None

    ranking = json.loads(ranking_path.read_text(encoding="utf-8"))
    if not isinstance(ranking, list) or any(not isinstance(item, str) for item in ranking):
        raise ValueError(
            f"Invalid ranking file format: {ranking_path}. Expected a JSON array of task names."
        )

    ordered_names: list[str] = []
    seen: set[str] = set()
    for raw_name in ranking:
        task_name = raw_name.strip()
        if not task_name or task_name in seen:
            continue
        ordered_names.append(task_name)
        seen.add(task_name)
    return ordered_names


def resolve_group_task_paths(dataset_path: Path, disable_verification: bool) -> list[Path]:
    dataset_root = resolve_dataset_path(dataset_path)
    task_paths = sorted(
        [
            path.resolve()
            for path in dataset_root.iterdir()
            if TaskPaths(path).is_valid(disable_verification=disable_verification)
        ],
        key=lambda path: path.name,
    )
    if not task_paths:
        raise ValueError(f"No valid tasks found under dataset: {dataset_root}")

    ranking = load_group_task_ranking(dataset_root)
    if not ranking:
        return task_paths

    task_by_name = {path.name: path for path in task_paths}
    ordered_paths: list[Path] = []
    seen: set[str] = set()
    missing_in_dataset: list[str] = []

    for task_name in ranking:
        task_path = task_by_name.get(task_name)
        if task_path is None:
            missing_in_dataset.append(task_name)
            continue
        ordered_paths.append(task_path)
        seen.add(task_name)

    if missing_in_dataset:
        print(
            f"[group-order] {dataset_root.name}: ranking file contains unknown tasks: {missing_in_dataset}"
        )

    remaining_paths = [path for path in task_paths if path.name not in seen]
    if remaining_paths:
        print(
            f"[group-order] {dataset_root.name}: appending unranked tasks after ranking file order: "
            f"{[path.name for path in remaining_paths]}"
        )

    return ordered_paths + remaining_paths


def resolve_trial_dir(trial_uri: str) -> Path:
    if trial_uri.startswith("file://"):
        parsed = urlparse(trial_uri)
        return Path(unquote(parsed.path))
    return Path(trial_uri)


def render_group_progress(completed: int, total: int, width: int = 30) -> str:
    if total <= 0:
        return "[------------------------------] 0/0 groups (0.0%)"

    ratio = min(max(completed / total, 0.0), 1.0)
    filled = int(width * ratio)
    bar = "#" * filled + "-" * (width - filled)
    return f"[{bar}] {completed}/{total} groups ({ratio * 100:5.1f}%)"


def clear_progress_line(progress_line: str) -> None:
    print("\r" + " " * len(progress_line) + "\r", end="", flush=True)


def resolve_group_name(dataset_path: Path) -> str:
    return dataset_path.name


def validate_unique_group_names(datasets: list[object]) -> None:
    seen: dict[str, Path] = {}
    duplicates: list[str] = []

    for dataset in datasets:
        dataset_path = Path(dataset.path)
        group_name = resolve_group_name(dataset_path)
        if group_name in seen:
            duplicates.append(
                f"{group_name} -> {seen[group_name]} and {resolve_dataset_path(dataset_path)}"
            )
            continue
        seen[group_name] = resolve_dataset_path(dataset_path)

    if duplicates:
        duplicate_text = "\n".join(duplicates)
        raise ValueError(
            "Duplicate dataset group names detected. Please avoid using multiple datasets "
            f"with the same leaf directory name:\n{duplicate_text}"
        )


def resolve_effective_concurrency(config_dict: dict[str, object], override: int | None) -> int:
    if override is not None:
        return override

    orchestrator = config_dict.get("orchestrator")
    if isinstance(orchestrator, dict):
        raw_value = orchestrator.get("n_concurrent_trials")
        if isinstance(raw_value, int) and raw_value > 0:
            return raw_value
    return 1


def normalize_environment_config(environment_config: object) -> dict[str, object]:
    env = copy.deepcopy(environment_config) if isinstance(environment_config, dict) else {}

    if env.get("import_path") == SHARED_ENV_IMPORT_PATH:
        env.pop("import_path", None)
        env.setdefault("type", "docker")

    kwargs = env.get("kwargs")
    if isinstance(kwargs, dict):
        kwargs.pop("project_template_dir", None)
        kwargs.pop("copy_task_skills", None)
        kwargs.pop("shared_skills_root", None)
        if not kwargs:
            env.pop("kwargs", None)
        else:
            env["kwargs"] = kwargs

    return env


def load_json_if_exists(path: Path) -> dict[str, object] | None:
    if not path.exists() or not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def collect_completed_task_names(job_dir: Path) -> set[str]:
    completed: set[str] = set()
    if not job_dir.exists():
        return completed

    for child in job_dir.iterdir():
        if not child.is_dir() or not (child / "result.json").exists():
            continue

        config_data = load_json_if_exists(child / "config.json") or {}
        task_config = config_data.get("task") if isinstance(config_data.get("task"), dict) else {}
        task_path = task_config.get("path")
        if isinstance(task_path, str) and task_path.strip():
            completed.add(Path(task_path).name)
            continue

        task_name = config_data.get("task_name")
        if isinstance(task_name, str) and task_name.strip():
            completed.add(task_name.strip())
    return completed


def is_existing_job_complete(job_dir: Path, task_paths: list[Path]) -> tuple[bool, str]:
    if not job_dir.exists():
        return False, "no existing job directory"

    job_result = load_json_if_exists(job_dir / "result.json")
    if job_result is None:
        return False, "missing top-level result.json"

    planned_tasks = {task_path.name for task_path in task_paths}
    completed_tasks = collect_completed_task_names(job_dir)
    missing_tasks = sorted(planned_tasks - completed_tasks)
    if missing_tasks:
        preview = ", ".join(missing_tasks[:5])
        suffix = "..." if len(missing_tasks) > 5 else ""
        return False, f"missing completed trials for {len(missing_tasks)} task(s): {preview}{suffix}"

    recorded_trials = job_result.get("n_total_trials")
    if isinstance(recorded_trials, int) and recorded_trials < len(planned_tasks):
        return False, (
            "top-level result.json records only "
            f"{recorded_trials}/{len(planned_tasks)} trials"
        )

    return True, f"found completed job with {len(planned_tasks)} planned task(s)"


def prepare_group_job_dir(job_dir: Path, task_paths: list[Path], dry_run: bool) -> tuple[bool, str]:
    is_complete, reason = is_existing_job_complete(job_dir, task_paths)
    if is_complete:
        return False, f"Skip existing job: {reason}"

    if job_dir.exists():
        if dry_run:
            return False, f"Dry-run: would delete incomplete job directory because {reason}"
        shutil.rmtree(job_dir)
        return True, f"Deleted incomplete job directory because {reason}"

    return True, "No previous job directory found; starting fresh"


def build_group_job_config(
    base_config: JobConfig,
    group_name: str,
    dataset_path: Path,
    run_root_dir: Path,
    max_parallel_trials: int | None,
) -> tuple[JobConfig, list[Path], int]:
    config_dict = copy.deepcopy(base_config.model_dump())
    dataset_root = resolve_dataset_path(dataset_path)
    disable_verification = bool((config_dict.get("verifier") or {}).get("disable", False))
    ordered_task_paths = resolve_group_task_paths(
        dataset_root,
        disable_verification=disable_verification,
    )

    base_job_name = config_dict.get("job_name", "job")
    new_job_name = f"{base_job_name}__{sanitize_name(group_name)}"
    config_dict["job_name"] = new_job_name
    config_dict["jobs_dir"] = str(run_root_dir)
    config_dict["tasks"] = [
        {"path": str(task_path), "source": group_name}
        for task_path in ordered_task_paths
    ]
    config_dict["datasets"] = [{"path": str(dataset_root), "n_tasks": 0}]

    effective_concurrency = resolve_effective_concurrency(config_dict, max_parallel_trials)
    orchestrator = config_dict.get("orchestrator") or {}
    orchestrator["n_concurrent_trials"] = effective_concurrency
    config_dict["orchestrator"] = orchestrator

    config_dict["environment"] = normalize_environment_config(config_dict.get("environment"))

    return JobConfig.model_validate(config_dict), ordered_task_paths, effective_concurrency


def run_group_job(
    base_config_path: Path,
    group_name: str,
    dataset_path: Path,
    runner_cfg: RunnerConfig,
) -> GroupResult:
    base_config = load_job_config(base_config_path)
    run_root_dir = runner_cfg.run_root_dir or resolve_run_root_dir(base_config, None)
    run_root_dir.mkdir(parents=True, exist_ok=True)

    group_config, task_paths, effective_concurrency = build_group_job_config(
        base_config=base_config,
        group_name=group_name,
        dataset_path=dataset_path,
        run_root_dir=run_root_dir,
        max_parallel_trials=runner_cfg.max_parallel_trials,
    )
    job_dir = run_root_dir / (group_config.job_name or sanitize_name(group_name))
    should_run, preparation_message = prepare_group_job_dir(
        job_dir=job_dir,
        task_paths=task_paths,
        dry_run=runner_cfg.dry_run,
    )

    if not should_run:
        return GroupResult(
            group_name=group_name,
            job_name=group_config.job_name or group_name,
            job_dir=job_dir,
            task_count=len(task_paths),
            parallel_trials=effective_concurrency,
            success=True,
            message=preparation_message,
        )

    if runner_cfg.dry_run:
        return GroupResult(
            group_name=group_name,
            job_name=group_config.job_name or group_name,
            job_dir=job_dir,
            task_count=len(task_paths),
            parallel_trials=effective_concurrency,
            success=True,
            message=(
                f"Dry-run: planned {len(task_paths)} tasks "
                f"(global docker budget={effective_concurrency}); {preparation_message}"
            ),
        )

    try:
        def on_trial_ended_hook_sync(event: TrialHookEvent) -> None:
            if event.result is None:
                return

            trial_name = event.result.trial_name
            trial_dir = resolve_trial_dir(event.result.trial_uri)
            trajectory_path = ensure_standard_trajectory(trial_dir)
            if trajectory_path is None:
                print(
                    f"[trajectory] {trial_name}: could not materialize agent/trajectory.json "
                    f"under {trial_dir}"
                )
                return
            print(f"[trajectory] {trial_name}: wrote {trajectory_path}")

        async def on_trial_ended_hook(event: TrialHookEvent) -> None:
            await asyncio.to_thread(on_trial_ended_hook_sync, event)

        job = Job(config=group_config)
        job.on_trial_ended(on_trial_ended_hook)
        result = asyncio.run(job.run())
        return GroupResult(
            group_name=group_name,
            job_name=group_config.job_name or group_name,
            job_dir=job_dir,
            task_count=len(task_paths),
            parallel_trials=effective_concurrency,
            success=True,
            message=(
                f"Job completed, {len(result.trial_results)} trials "
                f"(global docker budget={effective_concurrency}); {preparation_message}"
            ),
        )
    except Exception as exc:
        return GroupResult(
            group_name=group_name,
            job_name=group_config.job_name or group_name,
            job_dir=job_dir,
            task_count=len(task_paths),
            parallel_trials=effective_concurrency,
            success=False,
            message=f"Job failed: {exc}\n{traceback.format_exc()}",
        )


def parse_group_result(stdout: str) -> GroupResult | None:
    lines = stdout.strip().splitlines()
    if not lines:
        return None

    last = lines[-1]
    if not last.startswith("{"):
        return None

    data = json.loads(last)
    return GroupResult(
        group_name=data.get("group_name", ""),
        job_name=data.get("job_name", ""),
        job_dir=Path(data.get("job_dir", "")),
        task_count=int(data.get("task_count", 0)),
        parallel_trials=int(data.get("parallel_trials", 0)),
        success=bool(data.get("success", False)),
        message=data.get("message", ""),
    )


def run_group_in_subprocess(
    base_config_path: Path,
    group_name: str,
    dataset_path: Path,
    runner_cfg: RunnerConfig,
) -> GroupResult:
    base_config = load_job_config(base_config_path)
    base_job_name = base_config.job_name or "job"
    group_job_name = f"{base_job_name}__{sanitize_name(group_name)}"
    fallback_run_root = runner_cfg.run_root_dir or resolve_run_root_dir(base_config, None)
    fallback_job_dir = fallback_run_root / group_job_name

    args = [
        sys.executable,
        __file__,
        "--only-group",
        group_name,
        "--config",
        str(base_config_path),
        "--dataset-path",
        str(dataset_path),
    ]
    if runner_cfg.dry_run:
        args.append("--dry-run")
    if runner_cfg.max_parallel_trials is not None:
        args.extend(["-g", str(runner_cfg.max_parallel_trials)])
    if runner_cfg.run_root_dir:
        args.extend(["--run-root-dir", str(runner_cfg.run_root_dir)])

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    result = subprocess.run(
        args,
        cwd=ROOT_DIR,
        env=env,
        capture_output=True,
        text=True,
    )

    parsed_result = parse_group_result(result.stdout)
    if parsed_result is not None:
        return parsed_result

    if result.returncode == 0:
        return GroupResult(
            group_name=group_name,
            job_name=group_job_name,
            job_dir=fallback_job_dir,
            task_count=0,
            parallel_trials=0,
            success=True,
            message=result.stdout,
        )

    return GroupResult(
        group_name=group_name,
        job_name=group_job_name,
        job_dir=fallback_job_dir,
        task_count=0,
        parallel_trials=0,
        success=False,
        message=f"Subprocess failed:\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run each dataset family as an isolated Harbor job")
    parser.add_argument("-c", "--config", type=Path, default=Path("config.yaml"), help="Base job config YAML")
    parser.add_argument(
        "-g",
        "--max-parallel-trials",
        "--max-parallel-groups",
        dest="max_parallel_trials",
        type=int,
        default=None,
        help=(
            "Global Docker/trial budget for the whole run. Families are executed one by one, "
            "and each family job uses this n_concurrent_trials value. "
            "--max-parallel-groups is kept as a deprecated alias."
        ),
    )
    parser.add_argument("--run-root-dir", type=Path, default=None, help="Output root; default is jobs/<job_name>")
    parser.add_argument("--dry-run", action="store_true", help="Only print planned jobs without running Harbor")

    parser.add_argument("--only-group", type=str, default=None, help="Subprocess mode: run only one group")
    parser.add_argument("--dataset-path", type=Path, default=None, help="Subprocess mode: dataset path for the selected group")

    args = parser.parse_args()

    if args.max_parallel_trials is not None and args.max_parallel_trials <= 0:
        raise SystemExit("-g/--max-parallel-trials must be >= 1")

    runner_cfg = RunnerConfig(
        config_path=args.config,
        run_root_dir=args.run_root_dir,
        max_parallel_trials=args.max_parallel_trials,
        dry_run=args.dry_run,
    )

    base_config = load_job_config(args.config)
    runner_cfg.run_root_dir = resolve_run_root_dir(base_config, runner_cfg.run_root_dir)

    if args.only_group:
        if not args.dataset_path:
            print("Error: --dataset-path required in --only-group mode", file=sys.stderr)
            sys.exit(1)
        result = run_group_job(
            base_config_path=args.config,
            group_name=args.only_group,
            dataset_path=args.dataset_path,
            runner_cfg=runner_cfg,
        )
        print(
            json.dumps(
                {
                    "group_name": result.group_name,
                    "job_name": result.job_name,
                    "job_dir": str(result.job_dir),
                    "task_count": result.task_count,
                    "parallel_trials": result.parallel_trials,
                    "success": result.success,
                    "message": result.message,
                }
            )
        )
        sys.exit(0 if result.success else 1)

    runner_cfg.run_root_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run output directory: {runner_cfg.run_root_dir}")

    datasets = base_config.datasets or []
    if not datasets:
        print("No datasets in config.", file=sys.stderr)
        sys.exit(1)

    validate_unique_group_names(datasets)

    results: list[GroupResult] = []
    total_groups = len(datasets)
    completed_groups = 0
    progress_line = f"Group progress: {render_group_progress(completed_groups, total_groups)}"
    if total_groups > 0:
        print(progress_line, end="", flush=True)

    for dataset in datasets:
        dataset_path = Path(dataset.path)
        group_name = resolve_group_name(dataset_path)
        result = run_group_in_subprocess(
            args.config,
            group_name,
            dataset_path,
            runner_cfg,
        )
        results.append(result)
        completed_groups += 1

        if total_groups > 0:
            clear_progress_line(progress_line)

        status = "✓" if result.success else "✗"
        short_message = result.message[:200] if result.success else result.message
        print(
            f"{status} Group: {result.group_name} | Job: {result.job_name} | "
            f"Tasks: {result.task_count} | Parallel: {result.parallel_trials} | {short_message}"
        )

        if total_groups > 0:
            progress_line = f"Group progress: {render_group_progress(completed_groups, total_groups)}"
            print(progress_line, end="", flush=True)

    if total_groups > 0:
        print()

    success_count = sum(1 for result in results if result.success)
    total_tasks = sum(result.task_count for result in results)
    print(
        f"\nCompleted: {success_count}/{len(results)} groups succeeded, total planned tasks: {total_tasks}."
    )


if __name__ == "__main__":
    main()
