#!/usr/bin/env python3
"""
iterative_shared_skills_runner.py
----------------------------------
主调度脚本：并行运行多个任务集（group），每个 group 内串行执行 trial，
并在每次 trial 结束后调用 LLM 更新共享技能目录。

用法：
    python iterative_shared_skills_runner.py --config config_qwen_coder.yaml [--max-parallel-groups 2] [--dry-run]

依赖：
    - harbor: 已安装的 Harbor 包
    - litellm: 用于调用 LLM 生成 skill patch
    - libs.skill_evolution.patcher: 轨迹压缩 + skill patch 模块
"""

from __future__ import annotations

import argparse
import asyncio
import difflib
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

# 将项目根目录加入 sys.path 以导入 libs
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import yaml
from harbor import Job
from harbor.models.job.config import JobConfig
from harbor.models.task.paths import TaskPaths
from harbor.models.trial.result import TrialResult
from harbor.trial.hooks import TrialHookEvent

from libs.skill_evolution.patcher import (
    CompactionConfig,
    SkillPatchEvolver,
    SkillPatchResult,
    SkillSnapshotter,
    TrajectoryCompactor,
    ensure_standard_trajectory,
)


@dataclass
class RunnerConfig:
    """主调度器配置。"""
    config_path: Path
    run_root_dir: Path | None = None
    max_parallel_groups: int = 2
    dry_run: bool = False
    # 是否强制启用 SharedSkillsDockerEnvironment（若配置中未指定）
    force_shared_env: bool = True
    # 共享技能模板目录（用于首次初始化）
    project_template_dir: Path | None = None
    copy_task_skills: bool = False
    # 轨迹压缩参数
    max_steps: int = 20
    max_obs_chars: int = 3000
    # LLM patch 温度
    patch_temperature: float = 0.2
    # LLM patch 最大输出 tokens
    patch_max_tokens: int = 8192


@dataclass
class GroupResult:
    """一个 group 的执行结果。"""
    group_name: str
    job_name: str
    job_dir: Path
    success: bool
    message: str


def load_job_config(path: Path) -> JobConfig:
    """从 YAML 加载 JobConfig。"""
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return JobConfig.model_validate(data)


def sanitize_name(name: str) -> str:
    """将名称转为安全的目录名。"""
    return name.replace("/", "-").replace(" ", "_").strip("-_")


def resolve_run_root_dir(base_config: JobConfig, requested_root: Path | None) -> Path:
    """解析单次命令的总输出目录，默认落在 jobs/<job_name>/。"""
    if requested_root is not None:
        if requested_root.is_absolute():
            return requested_root
        if requested_root.parts and requested_root.parts[0] == "jobs":
            return (ROOT_DIR / requested_root).resolve()
        return (ROOT_DIR / "jobs" / requested_root).resolve()

    base_job_name = sanitize_name(base_config.job_name or "job")
    return (ROOT_DIR / "jobs" / base_job_name).resolve()


def resolve_dataset_path(dataset_path: Path) -> Path:
    """将 dataset path 解析为绝对路径，便于读取 ranking 文件和显式展开 tasks。"""
    expanded = dataset_path.expanduser()
    if expanded.is_absolute():
        return expanded.resolve()
    return (ROOT_DIR / expanded).resolve()


def load_group_task_ranking(dataset_path: Path) -> list[str] | None:
    """读取 group 内的 ALL_TASK_DIFFICULTY_RANKING.json，按文件顺序返回任务名。"""
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
    """解析 group 内任务路径；若存在 ranking 文件，则按文件顺序返回。"""
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
    """将 Harbor trial_uri 转为本地路径，兼容 file:// URI。"""
    if trial_uri.startswith("file://"):
        parsed = urlparse(trial_uri)
        return Path(unquote(parsed.path))
    return Path(trial_uri)


def render_group_progress(completed: int, total: int, width: int = 30) -> str:
    """渲染 group 级进度条。"""
    if total <= 0:
        return "[------------------------------] 0/0 groups (0.0%)"

    ratio = min(max(completed / total, 0.0), 1.0)
    filled = int(width * ratio)
    bar = "#" * filled + "-" * (width - filled)
    return f"[{bar}] {completed}/{total} groups ({ratio * 100:5.1f}%)"


def clear_progress_line(progress_line: str) -> None:
    """清除当前终端中的进度条行。"""
    print("\r" + " " * len(progress_line) + "\r", end="", flush=True)


def read_text_snapshot(path: Path) -> str | None:
    """读取文本文件内容，用于记录 patch 变更前状态。"""
    if not path.exists() or not path.is_file():
        return None

    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return None


def load_json_if_exists(path: Path) -> dict[str, Any] | None:
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


def prepare_group_job_dir(job_dir: Path, task_paths: list[Path]) -> tuple[bool, str]:
    is_complete, reason = is_existing_job_complete(job_dir, task_paths)
    if is_complete:
        return False, f"Skip existing job: {reason}"

    if job_dir.exists():
        shutil.rmtree(job_dir)
        return True, f"Deleted incomplete job directory because {reason}"

    return True, "No previous job directory found; starting fresh"


def build_patch_change_details(
    shared_skills_dir: Path,
    patch: SkillPatchResult,
) -> tuple[list[dict[str, Any]], str]:
    """构造 patch 的逐文件变更记录和 unified diff。"""
    change_records: list[dict[str, Any]] = []
    diff_blocks: list[str] = []

    for rel in patch.delete_paths:
        target = shared_skills_dir / rel
        before_exists = target.exists()
        before_content = read_text_snapshot(target)
        change_records.append(
            {
                "action": "delete",
                "path": rel,
                "exists_before": before_exists,
                "had_text_before": before_content is not None,
            }
        )
        if before_content is not None:
            diff_blocks.append(
                "".join(
                    difflib.unified_diff(
                        before_content.splitlines(keepends=True),
                        [],
                        fromfile=f"a/{rel}",
                        tofile="/dev/null",
                    )
                )
            )

    for rel, content in patch.upsert_files.items():
        target = shared_skills_dir / rel
        before_exists = target.exists()
        before_content = read_text_snapshot(target)
        after_content = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False, indent=2)
        change_records.append(
            {
                "action": "update" if before_exists else "create",
                "path": rel,
                "exists_before": before_exists,
                "had_text_before": before_content is not None,
                "new_content_chars": len(after_content),
            }
        )
        diff_blocks.append(
            "".join(
                difflib.unified_diff(
                    (before_content or "").splitlines(keepends=True),
                    after_content.splitlines(keepends=True),
                    fromfile=f"a/{rel}" if before_exists else "/dev/null",
                    tofile=f"b/{rel}",
                )
            )
        )

    diff_text = "\n".join(block for block in diff_blocks if block)
    return change_records, diff_text


def append_patch_history(history_path: Path, entry: dict[str, Any]) -> None:
    """将每次 patch 的摘要追加到 group 级历史日志。"""
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with history_path.open("a", encoding="utf-8") as history_file:
        history_file.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")


def resolve_patch_api_config(first_agent: Any) -> tuple[str | None, str | None, str | None, dict[str, str]]:
    """从 agent.kwargs / agent.env 解析 patch 阶段需要的 base_url、api_key、provider 与额外请求头。"""
    kwargs = first_agent.kwargs or {}
    env = getattr(first_agent, "env", None) or {}

    def infer_provider(base_url: str | None) -> str | None:
        if not base_url:
            return None
        api_base_lower = base_url.rstrip("/").lower()
        if "/anthropic" in api_base_lower:
            return "anthropic"
        if "/google/" in api_base_lower:
            return "gemini"
        if "/openai" in api_base_lower or api_base_lower.endswith("/v1"):
            return "openai"
        return None

    api_base = kwargs.get("base_url") or kwargs.get("api_base")
    api_key = kwargs.get("api_key")
    provider_hint: str | None = infer_provider(api_base)
    extra_headers: dict[str, str] = {}

    env_sources = (
        ("anthropic", ("ANTHROPIC_BASE_URL",), ("ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_API_KEY")),
        ("gemini", ("GOOGLE_API_BASE",), ("GEMINI_API_KEY", "GOOGLE_API_KEY")),
        ("openai", ("OPENAI_BASE_URL", "OPENAI_API_BASE"), ("OPENAI_API_KEY",)),
    )
    for provider, base_env_keys, key_env_keys in env_sources:
        if not api_base:
            for base_env_key in base_env_keys:
                if env.get(base_env_key):
                    api_base = env[base_env_key]
                    provider_hint = provider
                    break
        if not api_key:
            for key_env_key in key_env_keys:
                if env.get(key_env_key):
                    api_key = env[key_env_key]
                    provider_hint = provider_hint or provider
                    break

    # Some Harbor configs use generic LLM_* env vars instead of provider-specific names.
    if not api_base and env.get("LLM_BASE_URL"):
        api_base = env["LLM_BASE_URL"]
        provider_hint = provider_hint or infer_provider(api_base) or "openai"
    if not api_key and env.get("LLM_API_KEY"):
        api_key = env["LLM_API_KEY"]
        provider_hint = provider_hint or infer_provider(api_base) or "openai"

    if provider_hint == "anthropic" and api_key:
        # Friday Anthropic gateway expects Authorization and does not accept LiteLLM's default auth headers.
        extra_headers["Authorization"] = api_key

    return api_base, api_key, provider_hint, extra_headers


def resolve_patch_model_name(
    model_name: str,
    api_base: str | None,
    provider_hint: str | None = None,
) -> str:
    """为 LiteLLM 补全 provider 前缀，避免 provider 未识别。"""
    normalized = model_name.strip()
    if not normalized:
        return normalized

    if "/" in normalized:
        return normalized

    normalized_lower = normalized.lower()
    api_base_lower = api_base.rstrip("/").lower() if api_base else ""
    provider_lower = provider_hint.lower() if provider_hint else ""

    if (
        provider_lower == "anthropic"
        or "/anthropic" in api_base_lower
        or normalized_lower.startswith(("claude", "vertex.claude"))
    ):
        return f"anthropic/{normalized}"

    if (
        provider_lower == "gemini"
        or "/google/" in api_base_lower
        or normalized_lower.startswith("gemini")
    ):
        return f"gemini/{normalized}"

    if provider_lower == "openai" or "/openai" in api_base_lower:
        return f"openai/{normalized}"

    return f"openai/{normalized}"


def resolve_patch_temperature(model_name: str, temperature: float) -> float:
    """兼容 provider 侧的采样限制，避免 patch LLM 因参数非法直接失败。"""
    normalized = model_name.strip().lower()
    if normalized.startswith("moonshot/"):
        return 1.0
    return temperature


def prepare_shared_skills_dir(
    job_dir: Path,
    group_name: str,
    project_template_dir: Path | None,
) -> Path:
    """
    计算并准备共享技能目录。
    目录结构：jobs/<run_root>/<job_name>/shared_skills/<group_name>/
    如果目录为空且提供了 project_template_dir，则复制模板内容。
    """
    shared_dir = job_dir / "shared_skills" / group_name
    shared_dir.mkdir(parents=True, exist_ok=True)

    # 若目录为空且模板存在，则复制
    if not any(shared_dir.iterdir()) and project_template_dir and project_template_dir.exists():
        import shutil
        for item in project_template_dir.iterdir():
            if item.is_dir():
                dst = shared_dir / item.name
                if not dst.exists():
                    shutil.copytree(item, dst)
            else:
                shutil.copy2(item, shared_dir / item.name)
    return shared_dir


def build_group_job_config(
    base_config: JobConfig,
    group_name: str,
    dataset_path: Path,
    ordered_task_paths: list[Path],
    run_root_dir: Path,
    shared_skills_dir: Path,
    force_shared_env: bool,
    project_template_dir: Path | None,
    copy_task_skills: bool,
) -> JobConfig:
    """
    基于 base_config 构造单个 group 的 JobConfig。
    - job_name: <base_job_name>__<group_name>
    - tasks: 按 group 内 ALL_TASK_DIFFICULTY_RANKING.json（若存在）顺序展开
    - datasets: 保留一个空 dataset 以复用 Harbor 的 dataset 级 metrics 命名
    - orchestrator.n_concurrent_trials: 1（串行）
    - environment: 强制启用 SharedSkillsDockerEnvironment（若 force_shared_env）
    """
    import copy

    # 深拷贝 base_config dict
    config_dict = base_config.model_dump()
    dataset_root = resolve_dataset_path(dataset_path)

    # job_name
    base_job_name = config_dict.get("job_name", "job")
    new_job_name = f"{base_job_name}__{sanitize_name(group_name)}"
    config_dict["job_name"] = new_job_name
    config_dict["jobs_dir"] = str(run_root_dir)

    # tasks：按 ranking 文件顺序显式展开，确保同 group 内迭代顺序稳定可控
    config_dict["tasks"] = [
        {"path": str(task_path), "source": group_name}
        for task_path in ordered_task_paths
    ]

    # datasets：保留 dataset 名称用于 Harbor 汇总统计，但不再由 dataset 自动发现任务
    config_dict["datasets"] = [{"path": str(dataset_root), "n_tasks": 0}]

    # orchestrator：强制串行
    orchestrator = config_dict.get("orchestrator") or {}
    orchestrator["n_concurrent_trials"] = 1
    config_dict["orchestrator"] = orchestrator

    # environment：若未配置 shared environment，强制注入
    env = config_dict.get("environment") or {}
    if force_shared_env and not env.get("import_path"):
        env["import_path"] = "libs.terminus_env.environments.shared_skills_env:SharedSkillsDockerEnvironment"
        kwargs = env.get("kwargs") or {}
        kwargs["project_template_dir"] = str(project_template_dir) if project_template_dir else None
        kwargs["copy_task_skills"] = copy_task_skills
        env["kwargs"] = kwargs
        config_dict["environment"] = env

    return JobConfig.model_validate(config_dict)


def run_group_job(
    base_config_path: Path,
    group_name: str,
    dataset_path: Path,
    runner_cfg: RunnerConfig,
) -> GroupResult:
    """
    在子进程中被调用：运行单个 group 的 job。
    使用 Harbor Job.on_trial_ended hook 在每次 trial 后调用 skill patch。
    """
    base_config = load_job_config(base_config_path)

    # 准备共享技能目录
    # Harbor 会创建 <run_root_dir>/<job_name>
    run_root_dir = runner_cfg.run_root_dir or resolve_run_root_dir(base_config, None)
    run_root_dir.mkdir(parents=True, exist_ok=True)
    base_job_name = base_config.job_name or "job"
    job_dir = run_root_dir / f"{base_job_name}__{sanitize_name(group_name)}"

    base_config_dict = base_config.model_dump()
    disable_verification = bool((base_config_dict.get("verifier") or {}).get("disable", False))
    task_paths = resolve_group_task_paths(dataset_path, disable_verification=disable_verification)
    should_run, preparation_message = prepare_group_job_dir(job_dir, task_paths)
    if not should_run:
        return GroupResult(
            group_name=group_name,
            job_name=f"{base_job_name}__{sanitize_name(group_name)}",
            job_dir=job_dir,
            success=True,
            message=preparation_message,
        )

    shared_skills_dir = prepare_shared_skills_dir(
        job_dir,
        group_name,
        runner_cfg.project_template_dir,
    )

    # 构造 group job config
    group_config = build_group_job_config(
        base_config,
        group_name,
        dataset_path,
        task_paths,
        run_root_dir,
        shared_skills_dir,
        runner_cfg.force_shared_env,
        runner_cfg.project_template_dir,
        runner_cfg.copy_task_skills,
    )

    # 从第一个 agent 获取 patch LLM 配置（优先 kwargs，其次 env）
    first_agent = base_config.agents[0] if base_config.agents else None
    if not first_agent:
        return GroupResult(
            group_name=group_name,
            job_name=group_config.job_name or group_name,
            job_dir=job_dir,
            success=False,
            message="No agent configured in base config.",
        )

    model_name = first_agent.model_name or ""
    api_base, api_key, provider_hint, extra_headers = resolve_patch_api_config(first_agent)
    patch_model_name = resolve_patch_model_name(model_name, api_base, provider_hint)
    patch_temperature = resolve_patch_temperature(patch_model_name, runner_cfg.patch_temperature)
    patch_max_tokens = runner_cfg.patch_max_tokens

    # 初始化 patcher 组件
    compactor = TrajectoryCompactor(
        CompactionConfig(
            max_steps=runner_cfg.max_steps,
            max_obs_chars=runner_cfg.max_obs_chars,
        )
    )
    evolver = SkillPatchEvolver(
        model_name=patch_model_name,
        api_base=api_base,
        api_key=api_key,
        temperature=patch_temperature,
        max_tokens=patch_max_tokens,
        extra_headers=extra_headers,
    )

    # 定义 hook：在每次 trial 结束后调用 patch
    def on_trial_ended_hook_sync(trial_result: TrialResult) -> None:
        """
        Harbor Job.on_trial_ended 的回调处理逻辑。
        在容器停止后、下一个 trial 前执行，更新共享技能目录。
        """
        # 从 trial_result 获取路径信息
        trial_name = trial_result.trial_name
        trial_uri = trial_result.trial_uri  # 例如 "jobs/<job_name>/<trial_name>" 或 file:// URI
        trial_dir = resolve_trial_dir(trial_uri)
        debug_dir = trial_dir / "skill_evolution"
        debug_dir.mkdir(parents=True, exist_ok=True)

        # 读取标准 trajectory.json；若缺失则尝试从 claude-code.txt 物化生成
        trajectory_path = ensure_standard_trajectory(trial_dir)
        if trajectory_path is None:
            (debug_dir / "status.json").write_text(
                json.dumps(
                    {
                        "status": "skipped",
                        "reason": "trajectory_not_found",
                        "trial_uri": trial_uri,
                        "resolved_trial_dir": str(trial_dir),
                        "trajectory_path": str(trial_dir / "agent" / "trajectory.json"),
                        "raw_claude_log_path": str(trial_dir / "agent" / "claude-code.txt"),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            print(f"[hook] {trial_name}: trajectory not found, skip patch")
            return

        # 读取 result.json（trial_result 的 JSON dump）
        trial_result_path = trial_dir / "result.json"
        trial_result_dict: dict[str, Any] | None = None
        if trial_result_path.exists():
            trial_result_dict = json.loads(trial_result_path.read_text(encoding="utf-8"))

        # 读取 verifier/ctrf.json
        verifier_path = trial_dir / "verifier" / "ctrf.json"
        verifier_ctr: dict[str, Any] | None = None
        if verifier_path.exists():
            verifier_ctr = json.loads(verifier_path.read_text(encoding="utf-8"))

        # 提取 task 信息
        # trial_config.json 包含 task_name, source 等
        trial_config_path = trial_dir / "config.json"
        task_name = trial_name
        task_source = ""
        if trial_config_path.exists():
            trial_config = json.loads(trial_config_path.read_text(encoding="utf-8"))
            task_name = trial_config.get("task_name", trial_name)
            task_source = trial_config.get("source", "")

        # 提取 reward：来自 verifier_result.rewards
        reward = None
        verifier_passed = False
        if trial_result.verifier_result and trial_result.verifier_result.rewards:
            # 取第一个 reward 值
            rewards = trial_result.verifier_result.rewards
            first_reward = next(iter(rewards.values()), None)
            reward = float(first_reward) if first_reward is not None else None
            verifier_passed = reward is not None and reward >= 1.0

        # 补充 reward 到 trial_result_dict
        if trial_result_dict is not None and reward is not None:
            trial_result_dict["reward"] = reward

        # 构造精简 outcome
        outcome = compactor.extract_trial_outcome(
            trajectory_path=trajectory_path,
            trial_name=trial_name,
            task_name=task_name,
            task_source=task_source,
            trial_result=trial_result_dict,
            verifier_ctr=verifier_ctr,
        )
        # 覆盖 outcome 中的 reward 和 verifier_passed（优先使用 TrialResult）
        outcome.reward = reward
        outcome.verifier_passed = verifier_passed

        # 获取当前技能快照
        snapshot = SkillSnapshotter.snapshot(shared_skills_dir)

        # 调用 LLM 生成 patch
        patch = evolver.generate_patch(snapshot, outcome)
        change_records, diff_text = build_patch_change_details(shared_skills_dir, patch)
        print(f"[hook] {trial_name}: patch summary: {patch.summary[:200]}")

        # 应用 patch
        applied = SkillPatchEvolver.apply_patch(
            shared_skills_dir,
            patch,
            dry_run=runner_cfg.dry_run,
        )
        if runner_cfg.dry_run:
            print(f"[hook] {trial_name}: dry-run, patch not applied")
        else:
            print(f"[hook] {trial_name}: applied {len(applied.get('upserted', []))} upserts, {len(applied.get('deleted', []))} deletes")

        # 写调试文件
        (debug_dir / "outcome.json").write_text(
            json.dumps(outcome.__dict__, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        (debug_dir / "patch.json").write_text(
            json.dumps(patch.__dict__, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (debug_dir / "applied.json").write_text(
            json.dumps(applied, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (debug_dir / "changes.json").write_text(
            json.dumps(change_records, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        if diff_text:
            (debug_dir / "changes.diff").write_text(diff_text, encoding="utf-8")

        append_patch_history(
            job_dir / "skill_patch_history.jsonl",
            {
                "trial_name": trial_name,
                "task_name": task_name,
                "reward": reward,
                "verifier_passed": verifier_passed,
                "patch_model_name": patch_model_name,
                "patch_temperature": patch_temperature,
                "patch_max_tokens": patch_max_tokens,
                "attempt_count": patch.attempt_count,
                "attempt_modes": patch.attempt_modes,
                "successful_attempt": patch.successful_attempt,
                "successful_prompt_mode": patch.successful_prompt_mode,
                "successful_attempt_kind": patch.successful_attempt_kind,
                "summary": patch.summary,
                "upsert_paths": sorted(patch.upsert_files.keys()),
                "delete_paths": patch.delete_paths,
                "applied": applied,
                "debug_dir": str(debug_dir),
            },
        )

    async def on_trial_ended_hook(event: TrialHookEvent) -> None:
        if event.result is None:
            return
        await asyncio.to_thread(on_trial_ended_hook_sync, event.result)

    # 创建 Job 并注册 hook
    try:
        job = Job(config=group_config)
        job.on_trial_ended(on_trial_ended_hook)

        # Harbor Job.run 是 async，需要在同步入口中执行
        result = asyncio.run(job.run())
        return GroupResult(
            group_name=group_name,
            job_name=group_config.job_name or group_name,
            job_dir=job_dir,
            success=True,
            message=(
                f"Job completed, {len(result.trial_results)} trials; {preparation_message}"
            ),
        )
    except Exception as e:
        import traceback
        return GroupResult(
            group_name=group_name,
            job_name=group_config.job_name or group_name,
            job_dir=job_dir,
            success=False,
            message=f"Job failed: {e}\n{traceback.format_exc()}",
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
        success=data.get("success", False),
        message=data.get("message", ""),
    )


def run_group_in_subprocess(
    base_config_path: Path,
    group_name: str,
    dataset_path: Path,
    runner_cfg: RunnerConfig,
) -> GroupResult:
    """
    在子进程中运行 group job，隔离 logger 与状态。
    """
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
        "--max-steps",
        str(runner_cfg.max_steps),
        "--max-obs-chars",
        str(runner_cfg.max_obs_chars),
        "--patch-temperature",
        str(runner_cfg.patch_temperature),
        "--patch-max-tokens",
        str(runner_cfg.patch_max_tokens),
    ]
    if runner_cfg.dry_run:
        args.append("--dry-run")
    if runner_cfg.force_shared_env:
        args.append("--force-shared-env")
    if runner_cfg.project_template_dir:
        args.extend(["--project-template-dir", str(runner_cfg.project_template_dir)])
    if runner_cfg.copy_task_skills:
        args.append("--copy-task-skills")
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
            success=True,
            message=result.stdout,
        )

    return GroupResult(
        group_name=group_name,
        job_name=group_job_name,
        job_dir=fallback_job_dir,
        success=False,
        message=f"Subprocess failed:\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Iterative Shared Skills Runner")
    parser.add_argument("-c", "--config", type=Path, default=Path("config.yaml"), help="Base job config YAML")
    parser.add_argument("-g", "--max-parallel-groups", type=int, default=4, help="最大并行 group 数")
    parser.add_argument("--dry-run", action="store_true", help="不实际应用 patch，仅打印")
    parser.add_argument("--force-shared-env", action="store_true", default=True, help="若配置未指定，强制启用 SharedSkillsDockerEnvironment")
    parser.add_argument("--project-template-dir", type=Path, default=None, help="共享技能模板目录")
    parser.add_argument("--copy-task-skills", action="store_true", help="是否复制 task skills")
    parser.add_argument("--run-root-dir", type=Path, default=None, help="本次命令的总输出目录；默认自动使用 jobs/<job_name>")
    parser.add_argument("--max-steps", type=int, default=60, help="轨迹压缩最大步数")
    parser.add_argument("--max-obs-chars", type=int, default=3000, help="单条环境输出最大字符数")
    parser.add_argument("--patch-temperature", type=float, default=0.2, help="LLM patch 温度")
    parser.add_argument("--patch-max-tokens", type=int, default=16384, help="LLM patch 最大输出 tokens；Kimi 之类模型建议适当调大")

    # 子进程模式参数
    parser.add_argument("--only-group", type=str, default=None, help="子进程模式：仅运行指定 group")
    parser.add_argument("--dataset-path", type=Path, default=None, help="子进程模式：指定 dataset 路径")

    args = parser.parse_args()

    runner_cfg = RunnerConfig(
        config_path=args.config,
        run_root_dir=args.run_root_dir,
        max_parallel_groups=args.max_parallel_groups,
        dry_run=args.dry_run,
        force_shared_env=args.force_shared_env,
        project_template_dir=args.project_template_dir,
        copy_task_skills=args.copy_task_skills,
        max_steps=args.max_steps,
        max_obs_chars=args.max_obs_chars,
        patch_temperature=args.patch_temperature,
        patch_max_tokens=args.patch_max_tokens,
    )

    # 默认 project_template_dir
    if runner_cfg.project_template_dir is None:
        default_template = ROOT_DIR / "shared_skills_template" / "skills"
        if default_template.exists():
            runner_cfg.project_template_dir = default_template

    base_config = load_job_config(args.config)
    runner_cfg.run_root_dir = resolve_run_root_dir(base_config, runner_cfg.run_root_dir)

    # 子进程模式
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
        # 打印 JSON 供父进程解析
        print(json.dumps({
            "group_name": result.group_name,
            "job_name": result.job_name,
            "job_dir": str(result.job_dir),
            "success": result.success,
            "message": result.message,
        }))
        sys.exit(0 if result.success else 1)

    # 主进程模式
    runner_cfg.run_root_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run output directory: {runner_cfg.run_root_dir}")
    datasets = base_config.datasets or []

    if not datasets:
        print("No datasets in config.", file=sys.stderr)
        sys.exit(1)

    # 并发运行所有 group
    import concurrent.futures

    results: list[GroupResult] = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=runner_cfg.max_parallel_groups) as executor:
        futures = []
        for ds in datasets:
            ds_path = Path(ds.path)
            group_name = ds_path.name
            fut = executor.submit(
                run_group_in_subprocess,
                args.config,
                group_name,
                ds_path,
                runner_cfg,
            )
            futures.append(fut)

        total_groups = len(futures)
        completed_groups = 0
        progress_line = f"Group progress: {render_group_progress(completed_groups, total_groups)}"
        if total_groups > 0:
            print(progress_line, end="", flush=True)

        for fut in concurrent.futures.as_completed(futures):
            result = fut.result()
            results.append(result)
            completed_groups += 1

            if total_groups > 0:
                clear_progress_line(progress_line)

            status = "✓" if result.success else "✗"
            message = result.message[:200] if result.success else result.message
            print(f"{status} Group: {result.group_name} | Job: {result.job_name} | {message}")

            if total_groups > 0:
                progress_line = f"Group progress: {render_group_progress(completed_groups, total_groups)}"
                print(progress_line, end="", flush=True)

        if total_groups > 0:
            print()

    # 汇总
    success_count = sum(1 for r in results if r.success)
    print(f"\nCompleted: {success_count}/{len(results)} groups succeeded.")


if __name__ == "__main__":
    main()
