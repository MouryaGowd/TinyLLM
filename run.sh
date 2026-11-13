#!/usr/bin/env bash
# macOS / Linux launcher. Usage: ./run.sh [chat|generate|train|evaluate] [extra args]
# Creates .venv and installs dependencies on first run. Default action: chat.
set -e
cd "$(dirname "$0")"

if [ ! -x .venv/bin/python ] || ! .venv/bin/python -c "import torch" 2>/dev/null; then
    echo "First-time setup: creating virtual environment and installing PyTorch (a few minutes)..."
    python3 -m venv .venv
    .venv/bin/python -m pip install --upgrade pip
    .venv/bin/python -m pip install -r requirements.txt
fi

action="${1:-chat}"
[ $# -gt 0 ] && shift
case "$action" in
    chat|generate|train|evaluate) exec .venv/bin/python "$action.py" "$@" ;;
    *) echo "usage: ./run.sh [chat|generate|train|evaluate] [args]"; exit 1 ;;
esac
