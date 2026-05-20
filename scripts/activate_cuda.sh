#!/usr/bin/env bash
# Source this file to activate the project venv with TensorFlow CUDA libraries.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV_DIR="$PROJECT_DIR/.venv"

if [ ! -f "$VENV_DIR/bin/activate" ]; then
    echo "Missing virtual environment at $VENV_DIR" >&2
    return 1 2> /dev/null || exit 1
fi

# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"

CUDA_LIB_PATHS="$(find "$VIRTUAL_ENV/lib/python3.12/site-packages/nvidia" -type d -name lib 2> /dev/null | paste -sd: -)"
if [ -n "$CUDA_LIB_PATHS" ]; then
    if [ "${LD_LIBRARY_PATH#"$CUDA_LIB_PATHS"}" = "$LD_LIBRARY_PATH" ]; then
        if [ -n "${LD_LIBRARY_PATH:-}" ]; then
            export LD_LIBRARY_PATH="$CUDA_LIB_PATHS:$LD_LIBRARY_PATH"
        else
            export LD_LIBRARY_PATH="$CUDA_LIB_PATHS"
        fi
    fi
fi
unset CUDA_LIB_PATHS
