"""Ordered, resumable driver for the targeted experiments.

Same mechanism as ``tools/queue.py``, with its own stages and its own state directory
(``results/run_state_rev1``) so the completed main queue is never touched. Every stage is
resumable at the level of a single result file. A scheduled task re-invokes this driver after
a reboot or a crash, and ``tools/watchdog_rev1.py`` reports progress, the estimated time of
completion and any sign of trouble once an hour.

Timed stages run one job per performance core and never concurrently with each other, as in
the main study, so their wall-clock budgets are comparable with the main results. The
continuous-benchmark stage is budgeted in evaluations, not time, and is the only stage that
uses every core.

    .venv/Scripts/python.exe -m tools.queue_rev1            # run to completion
    .venv/Scripts/python.exe -m tools.queue_rev1 --status   # progress and estimate
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.run.revision_sets import (  # noqa: E402
    DECODED_METHODS,
    FULL_PANEL,
    HURINK_METHODS,
    HURINK_SEEDS,
    MEMETIC_METHODS,
    REVISION_SEEDS,
    VARIANT_METHODS,
    hurink_set,
    large_set,
    revision_subset,
)

STATE_DIR = ROOT / "results" / "run_state_rev1"
STATE_FILE = STATE_DIR / "queue_state.json"
HEARTBEAT = STATE_DIR / "heartbeat.json"
COMPLETE = STATE_DIR / "_QUEUE_COMPLETE"
LOG = STATE_DIR / "queue.log"
RAW = ROOT / "results" / "raw"
PY = [sys.executable]
TIMED_WORKERS = 8
JOB_SECONDS = 300 + 12  # budget plus measured per-job overhead (load, model build, write)


@dataclass
class Stage:
    name: str
    description: str
    command: list[str]
    run_dir: str
    expected: int
    timed: bool = True

    @property
    def est_hours(self) -> float:
        if not self.timed:
            return 11.0  # the main CEC stage needed about two core-hours per cell
        return self.expected * JOB_SECONDS / TIMED_WORKERS / 3600.0


def _runner(run_name, instances, methods, seeds, mapping="eligible"):
    return PY + ["-m", "src.run.runner", "--config", "config/full.yaml",
                 "--run-name", run_name, "--instances", ",".join(instances),
                 "--methods", ",".join(methods), "--seeds", str(seeds),
                 "--mapping", mapping]


def stages() -> list[Stage]:
    sub = revision_subset()
    large = large_set()
    hur = hurink_set()
    det = {"dispatching", "greedy"}
    n_large = len(large) * sum(1 if m in det else REVISION_SEEDS for m in FULL_PANEL)
    return [
        Stage("rev_hurink",
              "Hurink natural experiment: 66 bases x 4 flexibility levels, exact and tabu",
              _runner("rev_hurink", hur, HURINK_METHODS, HURINK_SEEDS), "rev_hurink",
              len(hur) * len(HURINK_METHODS) * HURINK_SEEDS),
        Stage("rev_memetic",
              "Memetic variants: four decoded optimisers with critical-block descent",
              _runner("rev_memetic", sub, MEMETIC_METHODS, REVISION_SEEDS, "eligible_ls"),
              "rev_memetic", len(sub) * len(MEMETIC_METHODS) * REVISION_SEEDS),
        Stage("rev_variants",
              "Improved variants: JADE, SHADE, CL-PSO, I-GWO through the shared decoder",
              _runner("rev_variants", sub, VARIANT_METHODS, REVISION_SEEDS), "rev_variants",
              len(sub) * len(VARIANT_METHODS) * REVISION_SEEDS),
        Stage("rev_cec",
              "Improved variants on CEC2017 (evaluation budget, all cores)",
              PY + ["-m", "src.run.run_cec", "--methods", ",".join(VARIANT_METHODS),
                    "--workers", "24"],
              "cec", 29 * len(VARIANT_METHODS), timed=False),
        Stage("rev_opperm",
              "Alternative representation: operation-based permutation, 12 optimisers",
              _runner("rev_opperm", sub, DECODED_METHODS, REVISION_SEEDS, "opperm"),
              "rev_opperm", len(sub) * len(DECODED_METHODS) * REVISION_SEEDS),
        Stage("rev_large",
              "Large-scale extension: 16 instances x full panel",
              _runner("rev_large", large, FULL_PANEL, REVISION_SEEDS), "rev_large", n_large),
    ]


def _count(stage: Stage) -> int:
    d = RAW / stage.run_dir
    if not d.exists():
        return 0
    if stage.name == "rev_cec":
        from src.run.revision_sets import VARIANT_METHODS as vm
        return sum(len(list(d.glob(f"*__{m}.json"))) for m in vm)
    return len(list(d.glob("*.json")))


def _read_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"done": [], "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}


def _write_state(state: dict) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")


def _log(message: str) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def remaining_hours(state: dict | None = None) -> float:
    state = state or _read_state()
    done = set(state.get("done", []))
    left = 0.0
    for s in stages():
        if s.name in done:
            continue
        frac = 1.0 - min(1.0, _count(s) / s.expected) if s.expected else 1.0
        left += s.est_hours * frac
    return left


def _heartbeat(stage_name: str, state: dict) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "current_stage": stage_name,
        "completed_stages": state.get("done", []),
        "progress": {s.name: {"have": _count(s), "expected": s.expected} for s in stages()},
        "remaining_hours_estimate": round(remaining_hours(state), 2),
    }
    HEARTBEAT.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def print_status() -> None:
    state = _read_state()
    done = set(state.get("done", []))
    print(f"{'stage':14s} {'status':8s} {'progress':>12s} {'est. h':>7s}")
    for s in stages():
        status = "done" if s.name in done else "pending"
        print(f"{s.name:14s} {status:8s} {f'{_count(s)}/{s.expected}':>12s} {s.est_hours:7.1f}")
    left = remaining_hours(state)
    print(f"\nremaining estimate: {left:.1f} h ({left / 24:.1f} days)")
    if COMPLETE.exists():
        print("QUEUE COMPLETE")


def run() -> int:
    if COMPLETE.exists():
        _log("queue already complete; nothing to do")
        return 0
    state = _read_state()
    done = set(state.get("done", []))
    from src.run.provenance import stamp_commit

    _log(f"code version for this invocation: {stamp_commit()}")
    for stage in stages():
        if stage.name in done:
            continue
        _log(f"START stage '{stage.name}' ({stage.description}); "
             f"progress {_count(stage)}/{stage.expected}, estimate {stage.est_hours:.1f} h")
        _heartbeat(stage.name, state)
        t0 = time.perf_counter()
        proc = subprocess.run(stage.command, cwd=ROOT)
        elapsed = (time.perf_counter() - t0) / 3600.0
        if proc.returncode != 0:
            _log(f"stage '{stage.name}' exited with code {proc.returncode} after "
                 f"{elapsed:.2f} h; it is resumable, so the next invocation retries it")
            _heartbeat(stage.name, state)
            return proc.returncode
        done.add(stage.name)
        state["done"] = [s.name for s in stages() if s.name in done]
        _write_state(state)
        _log(f"DONE stage '{stage.name}' in {elapsed:.2f} h; "
             f"progress {_count(stage)}/{stage.expected}")
        _heartbeat(stage.name, state)
    COMPLETE.write_text(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), encoding="utf-8")
    _log("QUEUE COMPLETE - every revision stage finished")
    return 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--reset-stage", default=None)
    args = ap.parse_args()
    if args.status:
        print_status()
        return
    if args.reset_stage:
        state = _read_state()
        state["done"] = [s for s in state.get("done", []) if s != args.reset_stage]
        _write_state(state)
        COMPLETE.unlink(missing_ok=True)
        print(f"stage '{args.reset_stage}' reset")
        return
    raise SystemExit(run())


if __name__ == "__main__":
    main()
