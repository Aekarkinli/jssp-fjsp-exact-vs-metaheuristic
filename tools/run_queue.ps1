# Drives the v2 experiment queue to completion.
#
# Invoked by the scheduled task at system startup and every 15 minutes thereafter. The
# scheduled task refuses to start a second copy, so a repeated trigger while this is already
# running is a no-op; if the process died for any reason, the next trigger revives it.
#
# The queue driver is resumable at the level of a single result file, so this script simply
# calls it in a loop until the completion sentinel appears. It calls the virtual
# environment's interpreter directly rather than going through uv, so it works identically
# under the SYSTEM account, which has no user package cache.

$ErrorActionPreference = 'Continue'
$root = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $root

$python = Join-Path $root '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) {
    Write-Output "FATAL: interpreter not found at $python"
    exit 1
}

$stateDir = Join-Path $root 'results\run_state'
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
    & $python -m tools.queue
    $code = $LASTEXITCODE
    if (Test-Path -LiteralPath $sentinel) {
        Write-WrapperLog "queue complete"
        break
    }
    Write-WrapperLog "queue driver returned $code; retrying in 60 s"
    Start-Sleep -Seconds 60
}

Write-WrapperLog "wrapper finished; removing the scheduled task"
try {
    Unregister-ScheduledTask -TaskName 'SolverAwareV2Queue' -Confirm:$false -ErrorAction Stop
    Write-WrapperLog "scheduled task removed"
} catch {
    Write-WrapperLog "could not remove the scheduled task: $($_.Exception.Message)"
}
