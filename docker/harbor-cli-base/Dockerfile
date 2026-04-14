# Build locally with ./build.sh
# Downstream images can simply use: FROM skillflow/harbor-cli-base:ubuntu24.04
ARG BASE_IMAGE=ubuntu:24.04
FROM ${BASE_IMAGE}

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ARG NODE_VERSION=22
ARG NVM_VERSION=v0.40.2
ARG CLAUDE_CODE_VERSION=latest
ARG QWEN_CODE_VERSION=latest
ARG GEMINI_CLI_VERSION=latest
ARG CODEX_CLI_VERSION=latest
ARG KIMI_CLI_VERSION=latest

ENV HOME=/root \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    DEBIAN_FRONTEND=noninteractive \
    PATH=/root/.local/bin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    NVM_DIR=/root/.nvm \
    NODE_VERSION=${NODE_VERSION} \
    NVM_VERSION=${NVM_VERSION} \
    CLAUDE_CODE_VERSION=${CLAUDE_CODE_VERSION} \
    QWEN_CODE_VERSION=${QWEN_CODE_VERSION} \
    GEMINI_CLI_VERSION=${GEMINI_CLI_VERSION} \
    CODEX_CLI_VERSION=${CODEX_CLI_VERSION} \
    KIMI_CLI_VERSION=${KIMI_CLI_VERSION}

COPY install-harbor-clis.sh /tmp/install-harbor-clis.sh

RUN chmod +x /tmp/install-harbor-clis.sh \
    && /tmp/install-harbor-clis.sh \
    && rm -f /tmp/install-harbor-clis.sh

WORKDIR /workspace
