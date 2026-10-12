#!/usr/bin/env bash
set -euo pipefail
candidate_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
python3 -m venv "$candidate_root/.venv"
"$candidate_root/.venv/bin/python" -m pip install -e "$candidate_root/packages/protocol" -e "$candidate_root"
"$candidate_root/.venv/bin/python" -m unittest discover -s "$candidate_root/tests" -v
