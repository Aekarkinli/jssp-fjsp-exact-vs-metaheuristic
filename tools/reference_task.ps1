# Wrapper launched by the "SolverAwareReference" scheduled task.
# Multi-thread CP-SAT reference on the hard subset (instances the single-thread fair run did
# not prove). RESUMABLE: run_reference skips instances whose result file already exists and
# writes atomically, so restarting after a connected-standby/sleep interruption resumes
# exactly where it stopped and never duplicates an output. 900 s budget, all cores, runs alone.
# ASCII-only (no path literals) so the file encoding cannot corrupt the non-ASCII project
# path; the scheduled task sets the working directory. Output appended to
# results/reference_run.log.
$ErrorActionPreference = "Continue"
$log = Join-Path (Get-Location) "results\reference_run.log"
Add-Content -LiteralPath $log -Value ("===== reference (re)started " + (Get-Date -Format "yyyy-MM-dd HH:mm:ss") + " =====")
$uv = (Get-Command uv -ErrorAction SilentlyContinue).Source
if (-not $uv) { $uv = "C:\Python314\Scripts\uv.exe" }  # run-machine fallback
& $uv run --no-sync python -m src.run.run_reference --budget 900 *>> $log

# Self-remove the dedicated task once the reference is complete (sentinel written by the
# runner only when every hard-subset instance has a result). If the run was interrupted by
# standby, the sentinel is absent, this wrapper process was already torn down, and
# restart-on-failure relaunches the resumable runner until it completes.
if (Test-Path (Join-Path (Get-Location) "results\raw\reference\_COMPLETE")) {
    Add-Content -LiteralPath $log -Value ("reference COMPLETE; unregistering SolverAwareReference " + (Get-Date -Format "yyyy-MM-dd HH:mm:ss"))
    try { Unregister-ScheduledTask -TaskName "SolverAwareReference" -Confirm:$false -ErrorAction Stop }
    catch { Add-Content -LiteralPath $log -Value ("unregister failed: " + $_.Exception.Message) }
}
