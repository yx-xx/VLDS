#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LIBERO_ROOT="$ROOT/examples/libero/.cache/LIBERO"
LIBERO_REVISION="8f1084e3132a39270c3a13ebe37270a43ece2a01"
SIM_PYTHON="$ROOT/examples/libero/.venv/bin/python"

cd "$ROOT"
command -v uv >/dev/null || { echo "Install uv before running this script." >&2; exit 1; }
GIT_LFS_SKIP_SMUDGE=1 uv sync --frozen

if [[ ! -d "$LIBERO_ROOT/.git" ]]; then
    mkdir -p "$(dirname "$LIBERO_ROOT")"
    git init "$LIBERO_ROOT"
fi
if ! git -C "$LIBERO_ROOT" rev-parse --verify HEAD >/dev/null 2>&1; then
    for FETCH_ATTEMPT in 1 2 3; do
        if git -c http.version=HTTP/1.1 -c http.lowSpeedLimit=1024 -c http.lowSpeedTime=120 \
            -C "$LIBERO_ROOT" fetch --progress --depth 1 \
            https://github.com/Lifelong-Robot-Learning/LIBERO.git "$LIBERO_REVISION"; then
            break
        fi
        if [[ "$FETCH_ATTEMPT" -eq 3 ]]; then
            echo "LIBERO download failed after 3 attempts; rerun this script to retry." >&2
            exit 1
        fi
        echo "LIBERO download failed; retrying ($FETCH_ATTEMPT/3)." >&2
        sleep "$((FETCH_ATTEMPT * 10))"
    done
    git -C "$LIBERO_ROOT" checkout --detach FETCH_HEAD
elif [[ "$(git -C "$LIBERO_ROOT" rev-parse HEAD)" != "$LIBERO_REVISION" ]]; then
    echo "LIBERO revision differs from $LIBERO_REVISION; refusing to overwrite the checkout." >&2
    exit 1
fi

if [[ ! -x "$SIM_PYTHON" ]]; then
    uv venv --python 3.8 "$ROOT/examples/libero/.venv"
fi
uv pip install --python "$SIM_PYTHON" \
    -r examples/libero/requirements.txt -r "$LIBERO_ROOT/requirements.txt" \
    --overrides examples/libero/franka-overrides.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    --index-strategy unsafe-best-match
uv pip install --python "$SIM_PYTHON" -e packages/openpi-client
uv pip install --python "$SIM_PYTHON" \
    --reinstall-package libero --config-setting editable_mode=compat -e "$LIBERO_ROOT"

echo "Franka Panda simulation environments are ready."
echo "Run: uv run --frozen scripts/run_franka_sim.py --models pi05"
