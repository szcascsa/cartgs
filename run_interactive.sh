#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
python_bin="${CONDA_PREFIX:-}/bin/python"
if [[ -x "$HOME/anaconda3/envs/zwj/bin/python" ]]; then
    python_bin="$HOME/anaconda3/envs/zwj/bin/python"
elif [[ ! -x "$python_bin" ]]; then
    python_bin="$(command -v python3)"
fi
exec "$python_bin" "$project_dir/scripts/interactive_run.py" "$@"
