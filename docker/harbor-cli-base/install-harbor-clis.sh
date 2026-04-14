#!/usr/bin/env bash
set -euo pipefail

: "${NODE_VERSION:=22}"
: "${NVM_VERSION:=v0.40.2}"
: "${CLAUDE_CODE_VERSION:=latest}"
: "${QWEN_CODE_VERSION:=latest}"
: "${GEMINI_CLI_VERSION:=latest}"
: "${CODEX_CLI_VERSION:=latest}"
: "${KIMI_CLI_VERSION:=latest}"

export HOME=/root
export DEBIAN_FRONTEND="${DEBIAN_FRONTEND:-noninteractive}"
export PATH="/root/.local/bin:/usr/local/bin:/usr/bin:/bin:${PATH:-}"
export NVM_DIR="${NVM_DIR:-$HOME/.nvm}"
export PIP_DISABLE_PIP_VERSION_CHECK="${PIP_DISABLE_PIP_VERSION_CHECK:-1}"
export PIP_NO_CACHE_DIR="${PIP_NO_CACHE_DIR:-1}"

install_system_packages() {
    apt-get update
    apt-get install -y --no-install-recommends \
        bash \
        ca-certificates \
        curl \
        git \
        python3 \
        python3-pip \
        python3-venv \
        ripgrep \
        xz-utils
    rm -rf /var/lib/apt/lists/*
}

ensure_nvm() {
    if [ ! -s "$NVM_DIR/nvm.sh" ]; then
        rm -rf "$NVM_DIR"
        git clone --branch "$NVM_VERSION" --depth 1 https://github.com/nvm-sh/nvm.git "$NVM_DIR"
    fi

    . "$NVM_DIR/nvm.sh"
    command -v nvm >/dev/null 2>&1 || {
        echo "Error: nvm failed to load" >&2
        exit 1
    }

    nvm install "$NODE_VERSION"
    nvm use "$NODE_VERSION"
    nvm alias default "$NODE_VERSION"
}

npm_install_global() {
    local package_name="$1"
    local package_version="$2"

    if [ -z "$package_version" ] || [ "$package_version" = "latest" ]; then
        npm install -g "${package_name}@latest"
    else
        npm install -g "${package_name}@${package_version}"
    fi
}

link_binary() {
    local name="$1"
    local bin_path

    bin_path="$(command -v "$name" 2>/dev/null || true)"
    if [ -z "$bin_path" ]; then
        return 0
    fi

    mkdir -p "$HOME/.local/bin"
    ln -sf "$bin_path" "$HOME/.local/bin/$name"
    ln -sf "$bin_path" "/usr/local/bin/$name"
}

configure_gemini_settings() {
    mkdir -p "$HOME/.gemini"
    cat > "$HOME/.gemini/settings.json" <<'EOF'
{
  "experimental": {
    "skills": true
  }
}
EOF
}

resolve_kimi_bin_path() {
    local venv_dir="$HOME/.local/share/kimi-cli-venv"
    local script_name

    if [ ! -x "$venv_dir/bin/python" ]; then
        return 1
    fi

    script_name="$($venv_dir/bin/python - <<'PY'
import importlib.metadata as md

try:
    dist = md.distribution("kimi-cli")
except md.PackageNotFoundError:
    print("")
    raise SystemExit(0)

names = [ep.name for ep in dist.entry_points if ep.group == "console_scripts"]
for preferred in ("kimi", "kimi-cli"):
    if preferred in names:
        print(preferred)
        break
else:
    print(names[0] if names else "")
PY
)"

    if [ -n "$script_name" ] && [ -x "$venv_dir/bin/$script_name" ]; then
        printf '%s\n' "$venv_dir/bin/$script_name"
        return 0
    fi

    return 1
}

install_kimi_cli() {
    local venv_dir="$HOME/.local/share/kimi-cli-venv"
    local kimi_bin

    rm -rf "$venv_dir"
    python3 -m venv "$venv_dir"

    if [ -z "$KIMI_CLI_VERSION" ] || [ "$KIMI_CLI_VERSION" = "latest" ]; then
        "$venv_dir/bin/pip" install --upgrade --prefer-binary kimi-cli
    else
        "$venv_dir/bin/pip" install --upgrade --prefer-binary "kimi-cli==$KIMI_CLI_VERSION"
    fi

    kimi_bin="$(resolve_kimi_bin_path)"
    if [ -z "$kimi_bin" ]; then
        echo "Error: could not resolve kimi-cli entry point" >&2
        exit 1
    fi

    mkdir -p "$HOME/.local/bin"
    ln -sf "$kimi_bin" "$HOME/.local/bin/kimi"
    ln -sf "$kimi_bin" /usr/local/bin/kimi
}

cleanup_caches() {
    npm cache clean --force >/dev/null 2>&1 || true
    rm -rf "$HOME/.cache/pip" "$HOME/.npm/_cacache"
}

verify_installations() {
    node --version
    npm --version
    claude --version
    qwen --version
    gemini --version
    codex --version
    kimi --version
}

install_system_packages
ensure_nvm
npm_install_global "@anthropic-ai/claude-code" "$CLAUDE_CODE_VERSION"
npm_install_global "@qwen-code/qwen-code" "$QWEN_CODE_VERSION"
npm_install_global "@google/gemini-cli" "$GEMINI_CLI_VERSION"
npm_install_global "@openai/codex" "$CODEX_CLI_VERSION"
configure_gemini_settings
link_binary node
link_binary npm
link_binary npx
link_binary claude
link_binary qwen
link_binary gemini
link_binary codex
install_kimi_cli
cleanup_caches
verify_installations
