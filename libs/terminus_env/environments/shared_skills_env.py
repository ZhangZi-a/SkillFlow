from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Iterable

import yaml

from harbor.environments.docker.docker import DockerEnvironment
from harbor.models.task.config import EnvironmentConfig
from harbor.models.trial.paths import TrialPaths


class SharedSkillsDockerEnvironment(DockerEnvironment):
    """Bind-mount shared skills and optionally rewrite task envs for overlay builds."""

    DEFAULT_MOUNT_TARGETS: tuple[str, ...] = (
        "/root/.claude/skills",
        "/etc/claude-code/.claude/skills",
        "/root/.qwen/skills",
        "/root/.codex/skills",
        "/root/.gemini/skills",
        "/root/.agents/skills",
        "/root/.goose/skills",
        "/root/.factory/skills",
        "/root/.opencode/skill",
        "/root/.kimi/skills",
        "/root/.config/agent/skills",
    )

    def __init__(
        self,
        environment_dir: Path,
        environment_name: str,
        session_id: str,
        trial_paths: TrialPaths,
        task_env_config: EnvironmentConfig,
        *args,
        project_template_dir: str | None = None,
        mount_targets: Iterable[str] | str | None = None,
        shared_skills_root: str = "shared_skills",
        copy_task_skills: bool = False,
        docker_image: str | None = None,
        allow_internet: bool | None = None,
        overlay_base_image: str | None = None,
        base_image_from: str = "skillflow/harbor-cli-base:ubuntu24.04",
        **kwargs,
    ):
        self._project_template_dir = self._resolve_host_path(project_template_dir)
        self._mount_targets = self._normalize_mount_targets(mount_targets)
        self._shared_skills_root = shared_skills_root
        self._copy_task_skills = copy_task_skills
        self._original_environment_dir = Path(environment_dir).resolve()
        self._overlay_base_image = overlay_base_image
        self._base_image_from = base_image_from

        if allow_internet is not None:
            task_env_config.allow_internet = allow_internet
        if docker_image and not overlay_base_image:
            task_env_config.docker_image = docker_image

        effective_environment_dir = self._original_environment_dir
        if overlay_base_image:
            task_env_config.docker_image = None
            effective_environment_dir = self._prepare_overlay_environment_dir(
                environment_dir=self._original_environment_dir,
                trial_paths=trial_paths,
                source_base_image=base_image_from,
                target_base_image=overlay_base_image,
            )

        super().__init__(
            effective_environment_dir,
            environment_name,
            session_id,
            trial_paths,
            task_env_config,
            *args,
            **kwargs,
        )

    @property
    def _docker_compose_paths(self) -> list[Path]:
        build_or_prebuilt = (
            self._DOCKER_COMPOSE_PREBUILT_PATH
            if self._use_prebuilt
            else self._DOCKER_COMPOSE_BUILD_PATH
        )
        paths = [self._DOCKER_COMPOSE_BASE_PATH, build_or_prebuilt]

        if self._environment_docker_compose_path.exists():
            paths.append(self._environment_docker_compose_path)

        custom_compose_path = (
            self.trial_paths.trial_dir / "docker-compose-shared-skills.yaml"
        )
        self._write_shared_skills_override(custom_compose_path)
        paths.append(custom_compose_path)

        if not self.task_env_config.allow_internet:
            paths.append(self._DOCKER_COMPOSE_NO_NETWORK_PATH)

        return paths

    def _write_shared_skills_override(self, target_compose_path: Path) -> None:
        shared_skills_dir = self._prepare_shared_skills_dir()
        shared_skills_path = shared_skills_dir.resolve().as_posix()
        volumes: list[str] = []

        for mount_target in self._mount_targets:
            mount = f"{shared_skills_path}:{mount_target}"
            if mount not in volumes:
                volumes.append(mount)

        compose_data = {
            "services": {
                "main": {
                    "volumes": volumes,
                }
            }
        }
        target_compose_path.write_text(yaml.safe_dump(compose_data, sort_keys=False))

    @property
    def _template_skills_dir(self) -> Path | None:
        if self._project_template_dir is None:
            return None

        nested_skills_dir = self._project_template_dir / "skills"
        if nested_skills_dir.is_dir():
            return nested_skills_dir

        return self._project_template_dir

    def _prepare_shared_skills_dir(self) -> Path:
        shared_skills_dir = self._shared_skills_dir
        shared_skills_dir.mkdir(parents=True, exist_ok=True)

        self._copy_missing_tree(self._template_skills_dir, shared_skills_dir)
        if self._copy_task_skills:
            self._copy_missing_tree(self._original_environment_dir / "skills", shared_skills_dir)

        return shared_skills_dir

    @property
    def _shared_skills_dir(self) -> Path:
        return self.trial_paths.trial_dir.parent / self._shared_skills_root / self._task_group_name

    @property
    def _task_group_name(self) -> str:
        default_name = self.environment_name or "adhoc"
        config_path = self.trial_paths.config_path
        if not config_path.exists():
            return self._sanitize_name(default_name)

        try:
            config_data = json.loads(config_path.read_text())
        except json.JSONDecodeError as exc:
            self.logger.warning("Failed to parse %s: %s", config_path, exc)
            return self._sanitize_name(default_name)

        task = config_data.get("task") or {}
        source_name = task.get("source") or default_name
        return self._sanitize_name(source_name)

    @staticmethod
    def _sanitize_name(value: str) -> str:
        sanitized = value.strip().replace("/", "_").replace("\\", "_")
        return sanitized or "adhoc"

    @staticmethod
    def _normalize_mount_targets(
        mount_targets: Iterable[str] | str | None,
    ) -> tuple[str, ...]:
        if mount_targets is None:
            return SharedSkillsDockerEnvironment.DEFAULT_MOUNT_TARGETS
        if isinstance(mount_targets, str):
            return (mount_targets,)
        return tuple(mount_targets)

    @staticmethod
    def _resolve_host_path(path_value: str | None) -> Path | None:
        if not path_value:
            return None

        path = Path(path_value).expanduser()
        if not path.is_absolute():
            path = Path.cwd() / path
        return path.resolve()

    def _copy_missing_tree(self, source_dir: Path | None, target_dir: Path) -> None:
        if source_dir is None or not source_dir.exists():
            return
        if not source_dir.is_dir():
            self.logger.warning("Skip non-directory skills template: %s", source_dir)
            return

        for source_child in source_dir.iterdir():
            target_child = target_dir / source_child.name

            if source_child.is_dir():
                if target_child.exists() and not target_child.is_dir():
                    self.logger.warning(
                        "Skip skills dir %s because %s already exists as a file",
                        source_child,
                        target_child,
                    )
                    continue
                if not target_child.exists():
                    shutil.copytree(source_child, target_child)
                    continue
                self._copy_missing_tree(source_child, target_child)
                continue

            if target_child.exists():
                continue

            target_child.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_child, target_child)

    def _prepare_overlay_environment_dir(
        self,
        environment_dir: Path,
        trial_paths: TrialPaths,
        source_base_image: str,
        target_base_image: str,
    ) -> Path:
        overlay_dir = trial_paths.trial_dir / "environment-overlay"
        if overlay_dir.exists():
            shutil.rmtree(overlay_dir)
        shutil.copytree(environment_dir, overlay_dir)

        dockerfile_path = overlay_dir / "Dockerfile"
        if dockerfile_path.exists():
            original = dockerfile_path.read_text(encoding="utf-8")
            rewritten = original.replace(source_base_image, target_base_image)
            if rewritten != original:
                dockerfile_path.write_text(rewritten, encoding="utf-8")
            else:
                self.logger.warning(
                    "Overlay base image rewrite found no matches in %s", dockerfile_path
                )

        compose_path = overlay_dir / "docker-compose.yaml"
        if compose_path.exists():
            compose_data = yaml.safe_load(compose_path.read_text(encoding="utf-8"))
            rewritten = self._replace_in_structure(
                compose_data,
                source_value=source_base_image,
                target_value=target_base_image,
            )
            if rewritten != compose_data:
                compose_path.write_text(
                    yaml.safe_dump(rewritten, sort_keys=False),
                    encoding="utf-8",
                )

        return overlay_dir

    @classmethod
    def _replace_in_structure(
        cls,
        data: Any,
        source_value: str,
        target_value: str,
    ) -> Any:
        if isinstance(data, str):
            return data.replace(source_value, target_value)
        if isinstance(data, list):
            return [
                cls._replace_in_structure(item, source_value, target_value)
                for item in data
            ]
        if isinstance(data, dict):
            return {
                key: cls._replace_in_structure(value, source_value, target_value)
                for key, value in data.items()
            }
        return data


JobSharedSkillsDockerEnvironment = SharedSkillsDockerEnvironment
