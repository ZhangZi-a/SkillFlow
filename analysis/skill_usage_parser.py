#!/usr/bin/env python3
"""Parse real Skill tool usage from agent trajectories.

This module detects actual `Skill` tool invocations from trajectory files instead
of inferring usage from filesystem path references. It can also be run as a
standalone script against one or more trajectory files, trial directories, or
run directories.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

SKILL_NAME_KEYS: tuple[str, ...] = ("skill", "skill_name", "name")
READ_TOOL_NAMES: tuple[str, ...] = ("read", "readfile")
RAW_CLAUDE_LOG_FILENAMES: tuple[str, ...] = ("claude-code.txt",)
SHELL_READ_COMMAND_PATTERN = re.compile(
    r"(?:^|[;&|]\s*|\b)(?:cat|sed|awk|grep|head|tail|less|more)\b",
    re.IGNORECASE,
)
SKILL_PATH_PATTERN = re.compile(
    r"(?:/|^)(?:\.claude/skills|\.codex/skills|\.agents/skills|skills)/([^/\\\"'\s]+)(?:/SKILL\.md|/|$)"
)


@dataclass(frozen=True)
class SkillUsageStats:
    skill_use_steps: int
    skill_use_calls: int
    skill_usage_rate: float
    skills_used: list[str]


def _iter_string_values(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
        return

    if isinstance(value, dict):
        for nested_value in value.values():
            yield from _iter_string_values(nested_value)
        return

    if isinstance(value, (list, tuple, set)):
        for nested_value in value:
            yield from _iter_string_values(nested_value)


def _normalize_skill_names(value: Any) -> set[str]:
    if isinstance(value, str):
        normalized = value.strip()
        return {normalized} if normalized else set()

    if isinstance(value, (list, tuple, set)):
        names: set[str] = set()
        for item in value:
            if isinstance(item, str) and item.strip():
                names.add(item.strip())
        return names

    return set()


def is_skill_tool_call(tool_call: dict[str, Any]) -> bool:
    function_name = tool_call.get("function_name") or tool_call.get("name") or ""
    return isinstance(function_name, str) and function_name.strip().lower() == "skill"


def _has_shell_skill_read(arguments: dict[str, Any] | None) -> bool:
    if not isinstance(arguments, dict):
        return False

    for text in _iter_string_values(arguments):
        if SKILL_PATH_PATTERN.search(text) and SHELL_READ_COMMAND_PATTERN.search(text):
            return True
    return False


def is_skill_file_read_call(tool_call: dict[str, Any]) -> bool:
    function_name = tool_call.get("function_name") or tool_call.get("name") or ""
    if isinstance(function_name, str) and function_name.strip().lower() in READ_TOOL_NAMES:
        return bool(extract_skill_names(tool_call.get("arguments")))
    return _has_shell_skill_read(tool_call.get("arguments"))


def is_skill_usage_call(tool_call: dict[str, Any]) -> bool:
    return is_skill_tool_call(tool_call) or is_skill_file_read_call(tool_call)


def extract_skill_names(arguments: dict[str, Any] | None) -> list[str]:
    if not isinstance(arguments, dict):
        return []

    skill_names: set[str] = set()

    for key in SKILL_NAME_KEYS:
        if key in arguments:
            skill_names.update(_normalize_skill_names(arguments.get(key)))

    if skill_names:
        return sorted(skill_names)

    for text in _iter_string_values(arguments):
        for match in SKILL_PATH_PATTERN.finditer(text):
            skill_name = match.group(1).strip()
            if skill_name:
                skill_names.add(skill_name)

    return sorted(skill_names)


def analyze_skill_usage(trajectory: dict[str, Any] | None) -> SkillUsageStats:
    if not isinstance(trajectory, dict):
        return SkillUsageStats(0, 0, 0.0, [])

    steps = trajectory.get("steps")
    if not isinstance(steps, list) or not steps:
        return SkillUsageStats(0, 0, 0.0, [])

    skill_use_steps = 0
    skill_use_calls = 0
    skills_used: set[str] = set()

    for step in steps:
        if not isinstance(step, dict):
            continue

        tool_calls = step.get("tool_calls")
        if not isinstance(tool_calls, list):
            continue

        step_has_skill_call = False
        for tool_call in tool_calls:
            if not isinstance(tool_call, dict) or not is_skill_usage_call(tool_call):
                continue

            step_has_skill_call = True
            skill_use_calls += 1
            skills_used.update(extract_skill_names(tool_call.get("arguments")))

        if step_has_skill_call:
            skill_use_steps += 1

    skill_usage_rate = round(skill_use_steps / len(steps), 4)
    return SkillUsageStats(
        skill_use_steps=skill_use_steps,
        skill_use_calls=skill_use_calls,
        skill_usage_rate=skill_usage_rate,
        skills_used=sorted(skills_used),
    )


def _stringify_content(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (bool, int, float)):
        return str(value)
    if isinstance(value, list):
        parts = [_stringify_content(item).strip() for item in value]
        return "\n\n".join(part for part in parts if part)
    if isinstance(value, dict):
        item_type = value.get("type")
        if item_type == "text":
            return _stringify_content(value.get("text"))
        if item_type == "thinking":
            return _stringify_content(value.get("thinking"))
        if item_type == "tool_result":
            return _stringify_content(value.get("content"))
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def _normalize_tool_arguments(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if value is None:
        return {}
    return {"input": value}


def _extract_assistant_parts(event: dict[str, Any]) -> tuple[str, str, list[dict[str, Any]], str | None]:
    message = event.get("message")
    if not isinstance(message, dict):
        return "", "", [], None

    content = message.get("content")
    if not isinstance(content, list):
        return "", "", [], message.get("model") if isinstance(message.get("model"), str) else None

    text_parts: list[str] = []
    reasoning_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []

    for item in content:
        if not isinstance(item, dict):
            continue
        item_type = item.get("type")
        if item_type == "text":
            text = item.get("text")
            if isinstance(text, str) and text.strip():
                text_parts.append(text)
        elif item_type == "thinking":
            thinking = item.get("thinking")
            if isinstance(thinking, str) and thinking.strip():
                reasoning_parts.append(thinking)
        elif item_type == "tool_use":
            tool_id = item.get("id")
            tool_name = item.get("name")
            if isinstance(tool_id, str) and isinstance(tool_name, str):
                tool_calls.append(
                    {
                        "tool_call_id": tool_id,
                        "function_name": tool_name,
                        "arguments": _normalize_tool_arguments(item.get("input")),
                    }
                )

    model_name = message.get("model") if isinstance(message.get("model"), str) else None
    return "\n\n".join(text_parts), "\n\n".join(reasoning_parts), tool_calls, model_name


def _extract_user_text(event: dict[str, Any]) -> str:
    message = event.get("message")
    if not isinstance(message, dict):
        return ""
    content = message.get("content")
    if not isinstance(content, list):
        return ""

    text_parts: list[str] = []
    for item in content:
        if not isinstance(item, dict):
            continue
        if item.get("type") != "text":
            continue
        text = item.get("text")
        if isinstance(text, str) and text.strip():
            text_parts.append(text)
    return "\n\n".join(text_parts)


def _extract_tool_results(event: dict[str, Any]) -> list[dict[str, Any]]:
    message = event.get("message")
    if not isinstance(message, dict):
        return []
    content = message.get("content")
    if not isinstance(content, list):
        return []

    results: list[dict[str, Any]] = []
    for item in content:
        if not isinstance(item, dict) or item.get("type") != "tool_result":
            continue
        tool_use_id = item.get("tool_use_id") or item.get("toolUseId")
        if not isinstance(tool_use_id, str) or not tool_use_id:
            continue
        result: dict[str, Any] = {
            "source_call_id": tool_use_id,
            "content": _stringify_content(item.get("content")),
        }
        if "is_error" in item:
            result["is_error"] = bool(item.get("is_error"))
        results.append(result)
    return results


def parse_claude_code_log(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None

    events: list[dict[str, Any]] = []
    try:
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            data = json.loads(line)
            if isinstance(data, dict):
                events.append(data)
    except Exception:
        return None

    if not events:
        return None

    session_id = ""
    agent_name = "claude-code"
    agent_version = ""
    default_model_name = ""
    cwd = ""
    steps: list[dict[str, Any]] = []
    pending_tool_steps: dict[str, dict[str, Any]] = {}
    step_id = 1

    for event in events:
        event_type = event.get("type")
        subtype = event.get("subtype")

        if event_type == "system" and subtype == "init":
            session_id = event.get("session_id") or session_id
            cwd = event.get("cwd") or cwd
            agent_name = "claude-code"
            agent_version = event.get("claude_code_version") or agent_version
            default_model_name = event.get("model") or default_model_name
            continue

        if event_type == "assistant":
            message_text, reasoning_text, tool_calls, model_name = _extract_assistant_parts(event)
            if not (message_text or reasoning_text or tool_calls):
                continue

            step: dict[str, Any] = {
                "step_id": step_id,
                "timestamp": event.get("timestamp"),
                "source": "agent",
                "model_name": model_name or default_model_name or None,
            }
            if message_text:
                step["message"] = message_text
            if reasoning_text:
                step["reasoning_content"] = reasoning_text
            if tool_calls:
                step["tool_calls"] = tool_calls
                for tool_call in tool_calls:
                    pending_tool_steps[tool_call["tool_call_id"]] = step

            extra = {
                "is_sidechain": bool(event.get("parent_tool_use_id")),
            }
            if cwd:
                extra["cwd"] = cwd
            step["extra"] = extra
            steps.append(step)
            step_id += 1
            continue

        if event_type == "user":
            tool_results = _extract_tool_results(event)
            attached_tool_result = False
            for tool_result in tool_results:
                step = pending_tool_steps.get(tool_result["source_call_id"])
                if step is None:
                    continue
                observation = step.setdefault("observation", {"results": []})
                observation.setdefault("results", []).append(
                    {
                        "source_call_id": tool_result["source_call_id"],
                        "content": tool_result["content"],
                    }
                )
                attached_tool_result = True
            if attached_tool_result:
                continue

            user_text = _extract_user_text(event)
            if not user_text:
                continue

            step = {
                "step_id": step_id,
                "timestamp": event.get("timestamp"),
                "source": "user",
                "message": user_text,
                "extra": {
                    "is_sidechain": bool(event.get("parent_tool_use_id")),
                },
            }
            steps.append(step)
            step_id += 1

    agent_info: dict[str, Any] = {"name": agent_name}
    if agent_version:
        agent_info["version"] = agent_version
    if default_model_name:
        agent_info["model_name"] = default_model_name
    extra: dict[str, Any] = {}
    if cwd:
        extra["cwds"] = [cwd]
    if extra:
        agent_info["extra"] = extra

    return {
        "schema_version": "ATIF-v1.2",
        "session_id": session_id or path.stem,
        "agent": agent_info,
        "steps": steps,
    }


def load_trajectory(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    if path.name in RAW_CLAUDE_LOG_FILENAMES:
        return parse_claude_code_log(path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def resolve_trajectory_paths(target: Path) -> list[Path]:
    if target.is_file():
        return [target] if target.name in {"trajectory.json", *RAW_CLAUDE_LOG_FILENAMES} else []

    if not target.is_dir():
        return []

    direct_trajectory = target / "agent" / "trajectory.json"
    if direct_trajectory.exists():
        return [direct_trajectory]

    direct_raw_log = target / "agent" / "claude-code.txt"
    if direct_raw_log.exists():
        return [direct_raw_log]

    trajectory_paths = sorted(target.glob("**/agent/trajectory.json"))
    if trajectory_paths:
        return trajectory_paths
    return sorted(target.glob("**/agent/claude-code.txt"))


def summarize_target(target: Path) -> dict[str, Any]:
    trajectory_paths = resolve_trajectory_paths(target)
    records: list[dict[str, Any]] = []

    for trajectory_path in trajectory_paths:
        trajectory = load_trajectory(trajectory_path)
        stats = analyze_skill_usage(trajectory)
        records.append(
            {
                "trajectory_path": str(trajectory_path),
                **asdict(stats),
            }
        )

    total_steps = sum(record["skill_use_steps"] for record in records)
    total_calls = sum(record["skill_use_calls"] for record in records)
    merged_skills = sorted(
        {
            skill_name
            for record in records
            for skill_name in record.get("skills_used", [])
        }
    )

    return {
        "target": str(target),
        "trajectory_count": len(records),
        "skill_use_steps": total_steps,
        "skill_use_calls": total_calls,
        "skills_used": merged_skills,
        "records": records,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Parse real Skill tool usage from trajectories")
    parser.add_argument("targets", nargs="+", help="Trajectory file, trial dir, or jobs dir")
    parser.add_argument("--indent", type=int, default=2, help="JSON output indentation")
    args = parser.parse_args(argv)

    summaries = [summarize_target(Path(target).expanduser()) for target in args.targets]
    print(json.dumps(summaries if len(summaries) > 1 else summaries[0], ensure_ascii=False, indent=args.indent))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
