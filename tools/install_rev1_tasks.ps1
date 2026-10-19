# Registers the two scheduled tasks of the revision-round run.
#
#   SolverAwareRev1Queue     runs the queue; restarted at boot or logon and every 15 minutes
#   SolverAwareRev1Watchdog  hourly health check with progress and estimated completion
#
# From an elevated PowerShell the tasks run as SYSTEM with a startup trigger, so the run
# resumes after a reboot before anyone logs in. From a normal session they run as the current
# user with a logon trigger, so the run resumes when the user logs in again.
#
#   .\tools\install_rev1_tasks.ps1

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$wrapper = Join-Path $root 'tools\run_queue_rev1.ps1'
$python = Join-Path $root '.venv\Scripts\python.exe'

$identity = [Security.Principal.WindowsPrincipal]::new([Security.Principal.WindowsIdentity]::GetCurrent())
$isAdmin = $identity.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)

foreach ($t in @('SolverAwareRev1Queue', 'SolverAwareRev1Watchdog')) {
    try { Unregister-ScheduledTask -TaskName $t -Confirm:$false -ErrorAction Stop } catch { }
}

# Stamp the code version from a session that can reach git; workers under SYSTEM read it.
$stateDir = Join-Path $root 'results\run_state'
New-Item -ItemType Directory -Force -Path $stateDir | Out-Null
$commit = (& git -C $root rev-parse HEAD 2>$null)
if ($LASTEXITCODE -eq 0 -and $commit) {
    Set-Content -LiteralPath (Join-Path $stateDir 'git_commit.txt') -Value $commit.Trim() -Encoding utf8 -NoNewline
    Write-Output "stamped code version $($commit.Trim())"
}

if ($isAdmin) {
    $principal = New-ScheduledTaskPrincipal -UserId 'SYSTEM' -LogonType ServiceAccount -RunLevel Highest
    $boot = New-ScheduledTaskTrigger -AtStartup
    Write-Output 'mode: SYSTEM, resumes at system startup'
} else {
    $user = "$env:USERDOMAIN\$env:USERNAME"
    $principal = New-ScheduledTaskPrincipal -UserId $user -LogonType Interactive -RunLevel Limited
    $boot = New-ScheduledTaskTrigger -AtLogOn -User $user
    Write-Output "mode: user $user, resumes at logon"
}

$settings = New-ScheduledTaskSettingsSet -MultipleInstances IgnoreNew -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries -StartWhenAvailable -DontStopOnIdleEnd `
    -RestartCount 999 -RestartInterval (New-TimeSpan -Minutes 2) -ExecutionTimeLimit ([TimeSpan]::Zero)
$settings.DisallowStartIfOnBatteries = $false

$queueAction = New-ScheduledTaskAction -Execute 'powershell.exe' `
    -Argument ('-NoProfile -NonInteractive -ExecutionPolicy Bypass -WindowStyle Hidden -File "{0}"' -f $wrapper) `
    -WorkingDirectory $root
$every15 = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(1) `
    -RepetitionInterval (New-TimeSpan -Minutes 15) -RepetitionDuration (New-TimeSpan -Days 365)
Register-ScheduledTask -TaskName 'SolverAwareRev1Queue' -Action $queueAction `
    -Trigger @($boot, $every15) -Principal $principal -Settings $settings `
    -Description 'Runs the revision-round experiment queue and resumes it after reboots.' | Out-Null

$watchAction = New-ScheduledTaskAction -Execute $python -Argument '-m tools.watchdog_rev1' `
    -WorkingDirectory $root
$hourly = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(60) `
    -RepetitionInterval (New-TimeSpan -Hours 1) -RepetitionDuration (New-TimeSpan -Days 365)
$wsettings = New-ScheduledTaskSettingsSet -MultipleInstances IgnoreNew -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries -StartWhenAvailable -ExecutionTimeLimit (New-TimeSpan -Minutes 10)
Register-ScheduledTask -TaskName 'SolverAwareRev1Watchdog' -Action $watchAction `
    -Trigger @($hourly) -Principal $principal -Settings $wsettings `
    -Description 'Hourly health check of the revision-round queue.' | Out-Null

Write-Output 'registered SolverAwareRev1Queue and SolverAwareRev1Watchdog'
Start-ScheduledTask -TaskName 'SolverAwareRev1Queue'
Write-Output 'queue started; status: .venv\Scripts\python.exe -m tools.queue_rev1 --status'
