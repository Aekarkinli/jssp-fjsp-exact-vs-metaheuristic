# Drives the revision-round experiment queue to completion.
#
# Invoked by the scheduled task at startup or logon and every 15 minutes. The task refuses a
# second copy, so a repeated trigger is a no-op while this runs; if the process died, the
# next trigger revives it. The queue driver resumes from the result files on disk.

$ErrorActionPreference = 'Continue'
$root = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $root

$python = Join-Path $root '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) {
    Write-Output "FATAL: interpreter not found at $python"
    exit 1
}

$stateDir = Join-Path $root 'results\run_state_rev1'
$sentinel = Join-Path $stateDir '_QUEUE_COMPLETE'
$wrapperLog = Join-Path $stateDir 'wrapper.log'
New-Item -ItemType Directory -Force -Path $stateDir | Out-Null

function Write-WrapperLog([string]$message) {
    $line = "[{0}] {1}" -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $message
    Write-Output $line
    Add-Content -LiteralPath $wrapperLog -Value $line -Encoding utf8
}

Write-WrapperLog "wrapper started (pid $PID)"
$attempt = 0
while (-not (Test-Path -LiteralPath $sentinel)) {
    $attempt++
    Write-WrapperLog "invoking queue driver (attempt $attempt)"
    & $python -m tools.queue_rev1
    $code = $LASTEXITCODE
    if (Test-Path -LiteralPath $sentinel) {
        Write-WrapperLog "queue complete"
        break
    }
    Write-WrapperLog "queue driver returned $code; retrying in 60 s"
    Start-Sleep -Seconds 60
}

& $python -m tools.watchdog_rev1
Write-WrapperLog "wrapper finished; removing the scheduled tasks"
foreach ($t in @('SolverAwareRev1Queue', 'SolverAwareRev1Watchdog')) {
    try {
        Unregister-ScheduledTask -TaskName $t -Confirm:$false -ErrorAction Stop
        Write-WrapperLog "scheduled task $t removed"
    } catch {
        Write-WrapperLog "could not remove $t : $($_.Exception.Message)"
    }
}
