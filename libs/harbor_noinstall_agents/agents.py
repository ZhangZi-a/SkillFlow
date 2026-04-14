from __future__ import annotations

import json
import os
import shlex

from harbor.agents.installed.base import ExecInput
from harbor.agents.installed.claude_code import ClaudeCode
from harbor.agents.installed.qwen_code import QwenCode
from harbor.environments.base import BaseEnvironment

try:
    from harbor.agents.installed.kimi_cli import (
        KimiCli,
        _OUTPUT_FILENAME as _KIMI_OUTPUT_FILENAME,
        _PROVIDER_CONFIG as _KIMI_PROVIDER_CONFIG,
    )
except ModuleNotFoundError:  # pragma: no cover - depends on installed Harbor version
    KimiCli = None
    _KIMI_OUTPUT_FILENAME = "kimi-cli.txt"
    _PROVIDER_CONFIG = {}

try:
    from harbor.agents.installed.codex import Codex
except ModuleNotFoundError:  # pragma: no cover - depends on installed Harbor version
    Codex = None


class _NoInstallSetupMixin:
    """Skip Harbor's install.sh setup and only do best-effort version detection."""

    async def setup(self, environment: BaseEnvironment) -> None:
        setup_dir = self.logs_dir / "setup"
        setup_dir.mkdir(parents=True, exist_ok=True)
        (setup_dir / "mode.txt").write_text(
            "skip install script; use preinstalled CLI in image\n",
            encoding="utf-8",
        )

        if getattr(self, "_version", None) is None:
            get_version_command = getattr(self, "get_version_command", None)
            parse_version = getattr(self, "parse_version", None)
            version_cmd = get_version_command() if callable(get_version_command) else None
            if version_cmd:
                try:
                    result = await environment.exec(command=version_cmd)
                    (setup_dir / "version-return-code.txt").write_text(
                        str(result.return_code), encoding="utf-8"
                    )
                    if result.stdout:
                        (setup_dir / "version-stdout.txt").write_text(
                            result.stdout, encoding="utf-8"
                        )
                    if result.stderr:
                        (setup_dir / "version-stderr.txt").write_text(
                            result.stderr, encoding="utf-8"
                        )
                    if result.return_code == 0 and result.stdout:
                        self._version = (
                            parse_version(result.stdout)
                            if callable(parse_version)
                            else result.stdout.strip()
                        )
                except Exception as exc:  # pragma: no cover
                    (setup_dir / "version-error.txt").write_text(
                        str(exc), encoding="utf-8"
                    )


class NoInstallClaudeCode(_NoInstallSetupMixin, ClaudeCode):
    """Claude Code agent that assumes `claude` is already available in the image."""

    @staticmethod
    def name() -> str:
        return ClaudeCode.name()

    def create_run_agent_commands(self, instruction: str):
        original_env = {key: os.environ.get(key) for key in self._extra_env}
        try:
            os.environ.update(self._extra_env)
            return super().create_run_agent_commands(instruction)
        finally:
            for key, original_value in original_env.items():
                if original_value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = original_value


class NoInstallQwenCode(_NoInstallSetupMixin, QwenCode):
    """Qwen Code agent that assumes `qwen` is already available in the image."""

    @staticmethod
    def name() -> str:
        return QwenCode.name()

    def create_run_agent_commands(self, instruction: str):
        original_env = {key: os.environ.get(key) for key in self._extra_env}
        try:
            os.environ.update(self._extra_env)
            return super().create_run_agent_commands(instruction)
        finally:
            for key, original_value in original_env.items():
                if original_value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = original_value


if KimiCli is not None:

    class NoInstallKimiCli(_NoInstallSetupMixin, KimiCli):
        """Kimi CLI agent that assumes `kimi` is already available in the image."""

        _STDERR_FILENAME = "kimi-cli.stderr.txt"

        @staticmethod
        def _build_safe_wire_command(escaped_prompt: str, mcp_enabled: bool) -> str:
            mcp_flag = "--mcp-config-file /tmp/kimi-mcp.json" if mcp_enabled else ""
            output_log = f"/logs/agent/{_KIMI_OUTPUT_FILENAME}"
            stderr_log = f"/logs/agent/{NoInstallKimiCli._STDERR_FILENAME}"
            return (
                "bash -lc "
                + shlex.quote(
                    f"""
set -u
PROMPT_FIFO="/tmp/kimi-prompt-$$.fifo"
OUTPUT_FIFO="/tmp/kimi-output-$$.fifo"
OUTPUT_LOG="{output_log}"
STDERR_LOG="{stderr_log}"
rm -f "$PROMPT_FIFO" "$OUTPUT_FIFO"
mkfifo "$PROMPT_FIFO" "$OUTPUT_FIFO"
cleanup() {{
  rm -f "$PROMPT_FIFO" "$OUTPUT_FIFO"
}}
trap cleanup EXIT
: > "$OUTPUT_LOG"
: > "$STDERR_LOG"
kimi --config-file /tmp/kimi-config.json --wire --yolo {mcp_flag} < "$PROMPT_FIFO" > "$OUTPUT_FIFO" 2>> "$STDERR_LOG" &
KIMI_PID=$!
{{
  printf '%s\\n' {escaped_prompt}
  while kill -0 "$KIMI_PID" 2>/dev/null; do
    sleep 1
  done
}} > "$PROMPT_FIFO" &
WRITER_PID=$!
SAW_FINISHED=0
while IFS= read -r line; do
  echo "$line" >> "$OUTPUT_LOG"
  case "$line" in
    *'"id":"1","result":{{"status":"finished"'* )
      SAW_FINISHED=1
      kill "$WRITER_PID" 2>/dev/null || true
      break
      ;;
  esac
done < "$OUTPUT_FIFO"
wait "$WRITER_PID" 2>/dev/null || true
if [ "$SAW_FINISHED" -eq 1 ] && kill -0 "$KIMI_PID" 2>/dev/null; then
  kill "$KIMI_PID" 2>/dev/null || true
fi
wait "$KIMI_PID"
KIMI_STATUS=$?
if [ "$SAW_FINISHED" -eq 1 ]; then
  if [ "$KIMI_STATUS" -eq 0 ] || [ "$KIMI_STATUS" -eq 143 ]; then
    exit 0
  fi
fi
exit "$KIMI_STATUS"
"""
                )
            )

        def create_run_agent_commands(self, instruction: str):
            if not self.model_name or "/" not in self.model_name:
                raise ValueError("Model name must be in format provider/model_name")

            provider, model = self.model_name.split("/", 1)

            original_env = {key: os.environ.get(key) for key in self._extra_env}
            try:
                os.environ.update(self._extra_env)
                config_json = self._build_config_json(provider, model)
                skills_cmd = self._build_register_skills_command()
                mcp_cmd = self._build_register_mcp_servers_command()
            finally:
                for key, original_value in original_env.items():
                    if original_value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = original_value

            escaped_config = shlex.quote(config_json)
            prompt_request = json.dumps(
                {
                    "jsonrpc": "2.0",
                    "method": "prompt",
                    "id": "1",
                    "params": {"user_input": instruction},
                }
            )
            escaped_prompt = shlex.quote(prompt_request)

            env: dict[str, str] = {}
            pcfg = _PROVIDER_CONFIG.get(provider, {})
            for key in pcfg.get("env_keys", []):
                val = self._extra_env.get(key) or os.environ.get(key)
                if val:
                    env[key] = val

            setup_parts = [f"echo {escaped_config} > /tmp/kimi-config.json"]
            if skills_cmd:
                setup_parts.append(skills_cmd)
            if mcp_cmd:
                setup_parts.append(mcp_cmd)

            commands = [ExecInput(command=" && ".join(setup_parts), env=env)]
            commands.append(
                ExecInput(
                    command=self._build_safe_wire_command(
                        escaped_prompt=escaped_prompt,
                        mcp_enabled=bool(mcp_cmd),
                    ),
                    env=env,
                )
            )
            return commands

        @staticmethod
        def name() -> str:
            return KimiCli.name()

else:

    class NoInstallKimiCli:
        """Compatibility placeholder when the installed Harbor build has no Kimi agent."""

        def __init__(self, *args, **kwargs):
            raise ModuleNotFoundError(
                "harbor.agents.installed.kimi_cli is not available in the current Harbor installation"
            )

        @staticmethod
        def name() -> str:
            return "kimi-cli"


if Codex is not None:

    class NoInstallCodex(_NoInstallSetupMixin, Codex):
        """Codex CLI agent that assumes `codex` is already available in the image."""

        @staticmethod
        def name() -> str:
            return Codex.name()

        def create_run_agent_commands(self, instruction: str):
            original_env = {key: os.environ.get(key) for key in self._extra_env}
            try:
                os.environ.update(self._extra_env)
                return super().create_run_agent_commands(instruction)
            finally:
                for key, original_value in original_env.items():
                    if original_value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = original_value

else:

    class NoInstallCodex:
        """Compatibility placeholder when the installed Harbor build has no Codex agent."""

        def __init__(self, *args, **kwargs):
            raise ModuleNotFoundError(
                "harbor.agents.installed.codex is not available in the current Harbor installation"
            )

        @staticmethod
        def name() -> str:
            return "codex"
