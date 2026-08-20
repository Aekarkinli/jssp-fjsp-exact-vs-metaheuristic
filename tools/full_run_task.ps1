# Wrapper launched by the "SolverAwareFullRun" scheduled task.
# ASCII-only (no path literals) so the file encoding cannot corrupt the non-ASCII project
# path; the task sets the working directory. Uses `uv run` so the interpreter is resolved
# robustly regardless of session context. The runner is resumable, so this is safe to start
# and restart at any time. All output is appended to results/full_run.log.
$ErrorActionPreference = "Continue"
$log = Join-Path (Get-Location) "results\full_run.log"
Add-Content -LiteralPath $log -Value ("===== full run (re)started " + (Get-Date -Format "yyyy-MM-dd HH:mm:ss") + " =====")
$uv = (Get-Command uv -ErrorAction SilentlyContinue).Source
if (-not $uv) { $uv = "C:\Python314\Scripts\uv.exe" }  # run-machine fallback
& $uv run --no-sync python -m src.run.runner --config config/full.yaml *>> $log
