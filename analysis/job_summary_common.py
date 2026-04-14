#!/usr/bin/env python3
"""Shared helpers for analysis job summary scripts."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any

from skill_usage_parser import analyze_skill_usage, load_trajectory

REPO_ROOT = Path(__file__).resolve().parent.parent


MODEL_PRICING_USD_PER_MILLION: dict[str, dict[str, float]] = {
    "claude-opus-4.5": {"input": 5.0, "output": 25.0, "cache": 0.5},
    "claude-opus-4.6": {"input": 5.0, "output": 25.0, "cache": 0.5},
    "claude-sonnet-4.5": {"input": 3.0, "output": 15.0, "cache": 0.3},
    "claude-sonnet-4.6": {"input": 3.0, "output": 15.0, "cache": 0.3},
    "kimi-k2.5": {"input": 0.3827, "output": 1.72, "cache": 0.0},
    "qwen3-coder-next": {"input": 0.15, "output": 1.2, "cache": 0.0},
    "qwen3-coder-480b-a35b-instruct": {"input": 0.22, "output": 1.8, "cache": 0.022},
    # MiniMax official pay-as-you-go pricing is published in CNY.
    # Converted here using 2026-04-08 FX: 1 CNY ~= 0.145514 USD.
    "minimax-m2.5": {"input": 0.3056, "output": 1.2223, "cache": 0.0306},
    "minimax-m2.7": {"input": 0.3056, "output": 1.2223, "cache": 0.0611},
    "gpt-5.4": {"input": 2.5, "output": 15.0, "cache": 0.25},
    "gpt-5.3-codex": {"input": 2.5, "output": 10.0, "cache": 0.0},
    "gpt-5.2-codex": {"input": 1.0, "output": 3.0, "cache": 0.0},
}

MODEL_NAME_ALIASES = {
    "claude-opus-4-5": "claude-opus-4.5",
    "claude-opus-4.5": "claude-opus-4.5",
    "claude-opus-4-6": "claude-opus-4.6",
    "claude-opus-4.6": "claude-opus-4.6",
    "claude-sonnet-4-5": "claude-sonnet-4.5",
    "claude-sonnet-4.5": "claude-sonnet-4.5",
    "claude-sonnet-4-6": "claude-sonnet-4.6",
    "claude-sonnet-4.6": "claude-sonnet-4.6",
    "claude-haiku-4-5": "claude-haiku-4.5",
    "claude-haiku-4.5": "claude-haiku-4.5",
    "kimi-k2.5": "kimi-k2.5",
    "kimi-k2-5": "kimi-k2.5",
    "qwen3-coder-next": "qwen3-coder-next",
    "qwen3-next-80b-instruct": "qwen3-coder-next",
    "qwen3-coder-480b-a35b-instruct": "qwen3-coder-480b-a35b-instruct",
    "qwen3-coder-480b-a35b-instruct-maas": "qwen3-coder-480b-a35b-instruct",
    "minimax-m2.5": "minimax-m2.5",
    "minimax-m2.7": "minimax-m2.7",
    "gpt-5.4": "gpt-5.4",
    "gpt-5-4": "gpt-5.4",
    "gpt-5.4-2026-03-05": "gpt-5.4",
    "gpt-5-4-2026-03-05": "gpt-5.4",
    "gpt-5.3-codex": "gpt-5.3-codex",
    "gpt-5-3-codex": "gpt-5.3-codex",
    "gpt-5.2-codex": "gpt-5.2-codex",
    "gpt-5-2-codex": "gpt-5.2-codex",
}


@dataclass
class TrialRecord:
    trial_dir: Path
    trial_name: str
    task_name: str
    source: str
    reward: float
    turns: int
    input_tokens: int
    cache_tokens: int
    output_tokens: int
    model_name: str | None
    pricing_model_name: str | None
    cost_usd: float | None
    reported_cost_usd: float | None
    cost_source: str | None
    started_at: datetime | None
    exception_type: str | None
    exception_message: str | None
    trajectory: dict[str, Any] | None
    result: dict[str, Any]
    skill_use_steps: int = 0
    skill_use_calls: int = 0
    skill_usage_rate: float = 0.0
    skills_used: list[str] = field(default_factory=list)
    task_skill_count: int = 0
    task_skill_names: list[str] = field(default_factory=list)
    final_skill_count: int = 0
    final_skill_names: list[str] = field(default_factory=list)


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def extract_group_name(dir_name: str) -> str:
    parts = dir_name.split("__")
    return parts[-1] if len(parts) > 1 else dir_name


def parse_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    normalized = value.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(normalized)
    except ValueError:
        return None


def extract_reward(trial_result: dict[str, Any]) -> float:
    verifier_result = trial_result.get("verifier_result") or {}
    rewards = verifier_result.get("rewards") or {}
    reward = rewards.get("reward")
    if reward is None:
        reward = verifier_result.get("reward")
    if reward is None:
        reward = trial_result.get("reward")
    try:
        return float(reward)
    except (TypeError, ValueError):
        return 0.0


def normalize_rel_path(path_value: str) -> str:
    normalized = str(path_value).strip().replace("\\", "/").lstrip("/").rstrip("/")
    return "" if normalized in {"", "."} else normalized


def extract_optional_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def canonicalize_model_name(model_name: str | None) -> str | None:
    if not model_name:
        return None

    normalized = model_name.strip().lower()
    for prefix in ("openai/", "moonshot/", "google/", "anthropic/"):
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix) :]
            break

    for prefix in ("aws.", "vertex."):
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix) :]
            break

    aliased = MODEL_NAME_ALIASES.get(normalized)
    if aliased is not None:
        return aliased

    # Some providers append release dates, e.g. gpt-5.3-codex-2026-02-24.
    for priced_model_name in MODEL_PRICING_USD_PER_MILLION:
        if normalized == priced_model_name or normalized.startswith(f"{priced_model_name}-"):
            return priced_model_name

    return normalized


def estimate_cost_usd(
    model_name: str | None,
    input_tokens: int,
    cache_tokens: int,
    output_tokens: int,
) -> tuple[str | None, float | None]:
    pricing_model_name = canonicalize_model_name(model_name)
    if pricing_model_name is None:
        return None, None

    pricing = MODEL_PRICING_USD_PER_MILLION.get(pricing_model_name)
    if pricing is None:
        return pricing_model_name, None

    estimated_cost = (
        input_tokens * pricing["input"]
        + cache_tokens * pricing["cache"]
        + output_tokens * pricing["output"]
    ) / 1_000_000
    return pricing_model_name, round(estimated_cost, 6)


def collect_relative_files(root_dir: Path | None) -> set[str]:
    if root_dir is None or not root_dir.is_dir():
        return set()

    files: set[str] = set()
    for path in root_dir.rglob("*"):
        if path.is_file():
            files.add(path.relative_to(root_dir).as_posix())
    return files


def skill_names_from_state_files(state_files: set[str]) -> list[str]:
    skill_names: set[str] = set()
    for rel_path in state_files:
        parts = PurePosixPath(rel_path).parts
        if len(parts) == 2 and parts[1] == "SKILL.md":
            skill_names.add(parts[0])
    return sorted(skill_names)


def resolve_template_skills_dir(group_config: dict[str, Any]) -> Path | None:
    environment = group_config.get("environment") or {}
    kwargs = environment.get("kwargs") or {}
    template_dir_value = kwargs.get("project_template_dir")
    if not template_dir_value:
        return None

    template_dir = Path(template_dir_value).expanduser()
    if not template_dir.is_absolute():
        template_dir = (REPO_ROOT / template_dir).resolve()

    nested_skills_dir = template_dir / "skills"
    if nested_skills_dir.is_dir():
        return nested_skills_dir
    if template_dir.is_dir():
        return template_dir
    return None


def resolve_task_skills_dir(trial_result: dict[str, Any]) -> Path | None:
    config = trial_result.get("config") or {}
    task_cfg = config.get("task") or {}
    task_path = task_cfg.get("path") or (trial_result.get("task_id") or {}).get("path")
    if not task_path:
        return None

    task_dir = Path(task_path).expanduser()
    if not task_dir.is_absolute():
        task_dir = (REPO_ROOT / task_dir).resolve()

    skills_dir = task_dir / "environment" / "skills"
    return skills_dir if skills_dir.is_dir() else None


def direct_result_dirs(jobs_dir: Path) -> list[Path]:
    if not jobs_dir.is_dir():
        return []
    return sorted(
        path
        for path in jobs_dir.iterdir()
        if path.is_dir() and (path / "result.json").exists()
    )


def _has_nested_result_dirs(path: Path) -> bool:
    try:
        return any(child.is_dir() and (child / "result.json").exists() for child in path.iterdir())
    except FileNotFoundError:
        return False


def find_flat_trial_dirs(jobs_dir: Path) -> list[Path]:
    return [
        path
        for path in direct_result_dirs(jobs_dir)
        if (path / "agent" / "trajectory.json").exists()
        or (path / "agent" / "claude-code.txt").exists()
    ]


def find_iterative_group_dirs(jobs_dir: Path) -> list[Path]:
    group_dirs: list[Path] = []
    for path in direct_result_dirs(jobs_dir):
        if _has_nested_result_dirs(path):
            group_dirs.append(path)
            continue
        if (path / "skill_patch_history.jsonl").exists() or (path / "shared_skills").exists():
            group_dirs.append(path)
    return group_dirs


def load_trial_record(trial_dir: Path) -> TrialRecord | None:
    trial_result = load_json(trial_dir / "result.json")
    if not trial_result:
        return None

    trajectory = load_trajectory(trial_dir / "agent" / "trajectory.json")
    if trajectory is None:
        trajectory = load_trajectory(trial_dir / "agent" / "claude-code.txt")

    turns = 0
    if trajectory and isinstance(trajectory.get("steps"), list):
        turns = len(trajectory["steps"])

    exc_info = trial_result.get("exception_info") or {}
    started_at = parse_timestamp(
        trial_result.get("started_at")
        or (trial_result.get("agent_execution") or {}).get("started_at")
    )
    reward = extract_reward(trial_result)
    skill_stats = analyze_skill_usage(trajectory)

    task_skill_names = skill_names_from_state_files(
        collect_relative_files(resolve_task_skills_dir(trial_result))
    )

    config = trial_result.get("config") or {}
    task_cfg = config.get("task") or {}
    agent_cfg = config.get("agent") or {}
    agent_info = trial_result.get("agent_info") or {}
    model_info = agent_info.get("model_info") or {}
    agent_result = trial_result.get("agent_result") or {}
    source = (
        trial_result.get("source")
        or task_cfg.get("source")
        or trial_result.get("task_name")
        or trial_dir.name
    )

    input_tokens = int(agent_result.get("n_input_tokens") or 0)
    cache_tokens = int(agent_result.get("n_cache_tokens") or 0)
    output_tokens = int(agent_result.get("n_output_tokens") or 0)
    model_name = agent_cfg.get("model_name") or model_info.get("name")
    pricing_model_name, estimated_cost_usd = estimate_cost_usd(
        model_name,
        input_tokens,
        cache_tokens,
        output_tokens,
    )
    reported_cost_usd = extract_optional_float(agent_result.get("cost_usd"))
    cost_usd = estimated_cost_usd
    cost_source = "pricing_map" if estimated_cost_usd is not None else None

    return TrialRecord(
        trial_dir=trial_dir,
        trial_name=trial_result.get("trial_name", trial_dir.name),
        task_name=trial_result.get("task_name") or source,
        source=source,
        reward=reward,
        turns=turns,
        input_tokens=input_tokens,
        cache_tokens=cache_tokens,
        output_tokens=output_tokens,
        model_name=model_name,
        pricing_model_name=pricing_model_name,
        cost_usd=cost_usd,
        reported_cost_usd=reported_cost_usd,
        cost_source=cost_source,
        started_at=started_at,
        exception_type=exc_info.get("exception_type"),
        exception_message=exc_info.get("exception_message"),
        trajectory=trajectory,
        result=trial_result,
        skill_use_steps=skill_stats.skill_use_steps,
        skill_use_calls=skill_stats.skill_use_calls,
        skill_usage_rate=skill_stats.skill_usage_rate,
        skills_used=skill_stats.skills_used,
        task_skill_count=len(task_skill_names),
        task_skill_names=task_skill_names,
    )


def sort_trial_records(trial_records: list[TrialRecord]) -> list[TrialRecord]:
    trial_records.sort(
        key=lambda record: (
            record.started_at is None,
            record.started_at or datetime.max,
            record.trial_name,
        )
    )
    return trial_records


def write_summary_json(jobs_dir: Path, summary: dict[str, Any]) -> Path:
    output_path = jobs_dir / "summary.json"
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path
