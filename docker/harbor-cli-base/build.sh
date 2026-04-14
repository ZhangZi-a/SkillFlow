#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
image_tag="${1:-skillflow/harbor-cli-base:ubuntu24.04}"

docker build \
    --build-arg BASE_IMAGE="${BASE_IMAGE:-ubuntu:24.04}" \
    --build-arg NODE_VERSION="${NODE_VERSION:-22}" \
    --build-arg NVM_VERSION="${NVM_VERSION:-v0.40.2}" \
    --build-arg CLAUDE_CODE_VERSION="${CLAUDE_CODE_VERSION:-latest}" \
    --build-arg QWEN_CODE_VERSION="${QWEN_CODE_VERSION:-latest}" \
    --build-arg GEMINI_CLI_VERSION="${GEMINI_CLI_VERSION:-latest}" \
    --build-arg CODEX_CLI_VERSION="${CODEX_CLI_VERSION:-latest}" \
    --build-arg KIMI_CLI_VERSION="${KIMI_CLI_VERSION:-latest}" \
    -t "$image_tag" \
    "$script_dir"

printf 'Built %s\n' "$image_tag"
printf 'Downstream Dockerfiles can use: FROM %s\n' "$image_tag"
