#!/usr/bin/env sh
set -eu

cd "$(dirname "$0")"

VENV=.venv
PY="$VENV/bin/python"
STAMP="$VENV/.requirements.installed"

if [ ! -x "$PY" ]; then
    echo "Creating the virtual environment in $VENV ..."
    if command -v uv >/dev/null 2>&1; then
        uv venv "$VENV" --python ">=3.12,<3.14"
    else
        PYTHON3=""
        for candidate in python3.13 python3.12 python3; do
            if command -v "$candidate" >/dev/null 2>&1 &&
                "$candidate" -c 'import sys; sys.exit(sys.version_info < (3, 12))' 2>/dev/null; then
                PYTHON3="$candidate"
                break
            fi
        done
        if [ -z "$PYTHON3" ]; then
            echo "Python 3.12 or newer is required. Install it, or install uv (https://docs.astral.sh/uv/)." >&2
            exit 1
        fi
        "$PYTHON3" -m venv "$VENV"
    fi
fi

if ! cmp -s requirements.txt "$STAMP" 2>/dev/null; then
    echo "Installing requirements ..."
    if command -v uv >/dev/null 2>&1; then
        uv pip install --python "$PY" -r requirements.txt
    else
        "$PY" -m pip install --disable-pip-version-check -r requirements.txt
    fi
    cp requirements.txt "$STAMP"
fi

if [ "${GD_SETUP_ONLY:-}" = "1" ]; then
    echo "Environment ready. Start the app with ./run.sh"
    exit 0
fi

exec "$PY" -m streamlit run app/main.py "$@"
