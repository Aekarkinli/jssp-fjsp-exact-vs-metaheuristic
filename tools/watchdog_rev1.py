"""Hourly health check of the revision-round experiment queue.

Run by a scheduled task once an hour (and on demand). It writes one summary line to
``results/run_state_rev1/health.log`` and the full report to ``health.json``:

- progress of every stage and the estimated time of completion, from the throughput
  actually observed over the last hour where available and from the per-job estimate
  otherwise;
- whether worker processes are alive;
- whether the queue has stalled, meaning no new result file for longer than any single job
  can take; a stalled queue is restarted by stopping its processes, after which the queue
  wrapper resumes from the files on disk;
- crashed or infeasible results written during the last hour;
- free disk space.

Any problem is logged with the word ALERT so a reader of the log, or the monitoring session,
can find it at once.

    .venv/Scripts/python.exe -m tools.watchdog_rev1
"""
from __future__ import annotations

import json
import shutil
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import psutil  # noqa: E402

from tools.queue_rev1 import (  # noqa: E402
    COMPLETE,
    JOB_SECONDS,
    RAW,
    STATE_DIR,
    TIMED_WORKERS,
    _count,
    _read_state,
    stages,
)

HEALTH_LOG = STATE_DIR / "health.log"
HEALTH_JSON = STATE_DIR / "health.json"
STALL_MINUTES = 45      # a 300 s job plus model building never takes this long
MIN_FREE_GB = 5.0


def _workers() -> list[psutil.Process]:
    out = []
    for p in psutil.process_iter(["name", "cmdline"]):
        try:
            cmd = " ".join(p.info.get("cmdline") or [])
        except Exception:
            continue
        if "python" in (p.info.get("name") or "").lower() and (
                "src.run.runner" in cmd or "src.run.run_cec" in cmd
                or "tools.queue_rev1" in cmd):
            out.append(p)
    return out


def _recent(stage, since: float) -> list[Path]:
    d = RAW / stage.run_dir
    if not d.exists():
        return []
    return [f for f in d.glob("*.json") if f.stat().st_mtime >= since]


def check(act: bool = True) -> dict:
    now = time.time()
    state = _read_state()
    done = set(state.get("done", []))
    report: dict = {"time": datetime.now().strftime("%Y-%m-%d %H:%M"), "alerts": [],
                    "actions": [], "stages": {}}
    if COMPLETE.exists():
        report["status"] = "complete"
        return report

    current = next((s for s in stages() if s.name not in done), None)
    remaining_h = 0.0
    for s in stages():
        have = _count(s)
        report["stages"][s.name] = f"{have}/{s.expected}"
        if s.name in done:
            continue
        left_jobs = max(0, s.expected - have)
        if s is current and s.timed:
            rate = len(_recent(s, now - 3600))  # results per hour observed
            if rate > 0:
                remaining_h += left_jobs / rate
                report["observed_rate_per_hour"] = rate
                continue
        remaining_h += s.est_hours * (left_jobs / s.expected if s.expected else 1)
    report["current_stage"] = current.name if current else None
    report["remaining_hours"] = round(remaining_h, 1)
    report["eta"] = (datetime.now() + timedelta(hours=remaining_h)).strftime("%Y-%m-%d %H:%M")

    workers = _workers()
    report["worker_processes"] = len(workers)
    if not workers:
        report["alerts"].append("no queue or worker process is running")

    if current is not None and current.timed:
        d = RAW / current.run_dir
        files = list(d.glob("*.json")) if d.exists() else []
        newest = max((f.stat().st_mtime for f in files), default=None)
        started = (STATE_DIR / "heartbeat.json")
        # the heartbeat is rewritten whenever a stage starts or restarts, so a stage that
        # was just reset is not taken for a stalled one because its newest file is old
        hb = started.stat().st_mtime if started.exists() else None
        ref = max(t for t in (newest, hb) if t is not None) if (newest or hb) else now
        idle_min = (now - ref) / 60.0
        report["minutes_since_last_result"] = round(idle_min, 1)
        if idle_min > STALL_MINUTES:
            report["alerts"].append(
                f"stalled: no new result in {idle_min:.0f} min in stage {current.name}")
            if act and workers:
                for p in workers:
                    try:
                        p.kill()
                    except Exception:
                        pass
                report["actions"].append(
                    f"stopped {len(workers)} process(es); the wrapper resumes the queue")

        bad = []
        for f in _recent(current, now - 3600):  # noqa: B007
            try:
                r = json.loads(f.read_text(encoding="utf-8"))
            except Exception:
                continue
            # The exact solver may legitimately hold no schedule; every heuristic must.
            no_schedule = r.get("feasible_final") is False and r.get("method") != "cpsat"
            # CMA-ES stops on its own convergence test on tiny instances and the run is
            # flagged as stopped early while holding a feasible schedule. That is expected,
            # reported in the paper, and not a failure.
            early_stop = (r.get("crashed") and r.get("feasible_final")
                          and not r.get("error") and r.get("method") == "cmaes")
            if (r.get("crashed") and not early_stop) or r.get("error") or no_schedule:
                bad.append(f.name)
        if bad:
            report["alerts"].append(f"{len(bad)} crashed or infeasible result(s) in the "
                                    f"last hour, e.g. {bad[:3]}")

    # on battery the processor is power-limited and timed budgets are no longer comparable;
    # results written in such a period are found and rerun by tools/power_audit.py
    try:
        bat = psutil.sensors_battery()
        if bat is not None and not bat.power_plugged:
            report["alerts"].append("running on battery: timed results are not comparable, "
                                    "plug in and run tools/power_audit.py --delete")
    except Exception:  # noqa: BLE001
        pass

    free_gb = shutil.disk_usage(ROOT).free / 1e9
    report["free_disk_gb"] = round(free_gb, 1)
    if free_gb < MIN_FREE_GB:
        report["alerts"].append(f"low disk space: {free_gb:.1f} GB free")

    load = psutil.cpu_percent(interval=2.0)
    report["cpu_percent"] = load
    report["status"] = "ALERT" if report["alerts"] else "ok"
    report["expected_parallel_jobs"] = TIMED_WORKERS
    report["job_seconds_estimate"] = JOB_SECONDS
    return report


def main() -> None:
    report = check(act="--dry-run" not in sys.argv)
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    HEALTH_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if report["status"] == "complete":
        line = f"[{report['time']}] complete"
    else:
        prog = ", ".join(f"{k} {v}" for k, v in report["stages"].items())
        line = (f"[{report['time']}] {report['status']} | stage {report['current_stage']} | "
                f"{prog} | remaining ~{report['remaining_hours']} h, ETA {report['eta']} | "
                f"workers {report['worker_processes']} | cpu {report['cpu_percent']}% | "
                f"disk {report['free_disk_gb']} GB")
        for a in report["alerts"]:
            line += f"\n    ALERT {a}"
        for a in report["actions"]:
            line += f"\n    ACTION {a}"
    with HEALTH_LOG.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")
    print(line)


if __name__ == "__main__":
    main()
