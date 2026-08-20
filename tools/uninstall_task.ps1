# Removes the scheduled task that drives the v2 experiment queue.
#
# The wrapper does this itself when the queue finishes; this script is for stopping the run
# early. Result files already on disk are untouched, so re-installing the task resumes from
# exactly where it stopped.
#
#   Run from an elevated PowerShell:  .\tools\uninstall_task.ps1

$ErrorActionPreference = 'Stop'
$taskName = 'SolverAwareV2Queue'

try {
    Stop-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
    Unregister-ScheduledTask -TaskName $taskName -Confirm:$false
    Write-Output "removed scheduled task '$taskName'"
} catch {
    Write-Output "task '$taskName' is not registered"
}

Get-Process -Name python -ErrorAction SilentlyContinue |
    Where-Object { $_.Path -like '*Solver-Aware-Evidence-Standards-v2*' } |
    ForEach-Object {
        Write-Output "stopping worker pid $($_.Id)"
        Stop-Process -Id $_.Id -Force -ErrorAction SilentlyContinue
    }
