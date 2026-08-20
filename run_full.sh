#!/usr/bin/env bash
# Full experimental run (Phase 6 main run).
#
# ~6.4 days on 8 performance cores at 300 s x 20 seeds (14,941 jobs). Results are written
# one JSON per (instance, method, seed) to results/raw/full/. The runner is RESUMABLE:
# already-completed combinations are skipped, so if the run is interrupted just run this
# script again to continue.
#
#   ./run_full.sh                  # launch / resume the full run
#   ./run_full.sh --budget 60      # override knobs (passed through to the runner)
#
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"$DIR/.venv/Scripts/python.exe" -m src.run.runner --config config/full.yaml "$@"
