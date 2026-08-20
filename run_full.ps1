# Full experimental run (Phase 6 main run).
#
# ~6.4 days on 8 performance cores at 300 s x 20 seeds (14,941 jobs). Results are written
# one JSON per (instance, method, seed) to results/raw/full/. The runner is RESUMABLE:
# already-completed combinations are skipped, so if the run is interrupted (reboot, etc.)
# just run this script again to continue.
#
#   ./run_full.ps1                 # launch / resume the full run
#   ./run_full.ps1 --budget 60     # override the budget knob, etc. (passes args through)
#
& "$PSScriptRoot\.venv\Scripts\python.exe" -m src.run.runner --config config/full.yaml @args
