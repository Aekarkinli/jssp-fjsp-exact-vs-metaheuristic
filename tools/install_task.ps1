# Registers the scheduled task that keeps the v2 experiment queue running.
#
# Two triggers. One fires at system startup, so a reboot or a power cut resumes the run
# without anyone logging in. The other repeats every fifteen minutes indefinitely, so a
# process killed by a suspend, a driver reset or anything else is revived at the next tick.
# Multiple instances are refused, which makes the repeating trigger a safe no-op while the
# run is healthy.
#
# The task runs as SYSTEM: no stored password, starts before logon, and survives the user
# signing out. The wrapper unregisters the task itself once the queue completes.
#
#   Run from an elevated PowerShell:  .\tools\install_task.ps1

$ErrorActionPreference = 'Stop'
$taskName = 'SolverAwareV2Queue'
$root = Split-Path -Parent $PSScriptRoot
$wrapper = Join-Path $root 'tools\run_queue.ps1'

if (-not (Test-Path -LiteralPath $wrapper)) { throw "wrapper not found at $wrapper" }

$identity = [Security.Principal.WindowsPrincipal]::new([Security.Principal.WindowsIdentity]::GetCurrent())
if (-not $identity.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    throw 'This script must be run from an elevated PowerShell session.'
}

try { Unregister-ScheduledTask -TaskName $taskName -Confirm:$false -ErrorAction Stop }
catch { }

# Stamp the code version now, from a session that can reach the version-control client.
# The task runs as SYSTEM, which has no client on its path, and every result file records
# the commit it was produced by.
$stateDir = Join-Path $root 'results\run_state'
New-Item -ItemType Directory -Force -Path $stateDir | Out-Null
$commit = (& git -C $root rev-parse HEAD 2>$null)
if ($LASTEXITCODE -eq 0 -and $commit) {
    Set-Content -LiteralPath (Join-Path $stateDir 'git_commit.txt') -Value $commit.Trim() -Encoding utf8 -NoNewline
    Write-Output "stamped code version $($commit.Trim())"
} else {
    Write-Warning 'could not determine the code version; result files will record "unknown"'
}

$action = New-ScheduledTaskAction -Execute 'powershell.exe' `
    -Argument ('-NoProfile -NonInteractive -ExecutionPolicy Bypass -WindowStyle Hidden -File "{0}"' -f $wrapper) `
    -WorkingDirectory $root

$atStartup = New-ScheduledTaskTrigger -AtStartup
# A year of fifteen-minute repetitions. The scheduler rejects the "indefinite" sentinel on
# current Windows builds, and a year is two orders of magnitude longer than the run.
$repeating = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(1) `
    -RepetitionInterval (New-TimeSpan -Minutes 15) -RepetitionDuration (New-TimeSpan -Days 365)

$principal = New-ScheduledTaskPrincipal -UserId 'SYSTEM' -LogonType ServiceAccount -RunLevel Highest

$settings = New-ScheduledTaskSettingsSet `
    -MultipleInstances IgnoreNew `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -DontStopOnIdleEnd `
    -RestartCount 999 `
    -RestartInterval (New-TimeSpan -Minutes 2) `
    -ExecutionTimeLimit ([TimeSpan]::Zero)
$settings.WakeToRun = $true
$settings.DisallowStartIfOnBatteries = $false

Register-ScheduledTask -TaskName $taskName -Action $action -Trigger @($atStartup, $repeating) `
    -Principal $principal -Settings $settings `
    -Description 'Runs the solver-aware v2 experiment queue, resuming automatically after reboots and suspends.' | Out-Null

Write-Output "registered scheduled task '$taskName'"
Start-ScheduledTask -TaskName $taskName
Write-Output 'task started; progress: .venv\Scripts\python.exe -m tools.queue --status'
