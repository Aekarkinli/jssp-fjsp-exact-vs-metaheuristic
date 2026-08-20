"""Ordered, resumable driver for the whole v2 experiment queue.

The queue is a list of stages. Each stage is a command that is itself resumable at the level
of a single result file, so interrupting the machine at any moment costs at most the one job
in flight. The driver runs the stages in order, records where it is in a state file, and
writes a sentinel when everything is finished. A scheduled task registered at system startup
re-invokes this driver after a reboot, a power cut or a suspend, and the driver simply picks
up where the files on disk say it stopped.

Stage order matters. The main comparison comes first because the hybrid and the oracle bound
read its results. The multi-threaded reference uses every core and therefore has to run
alone, which the sequential queue guarantees. The continuous-benchmark study is budgeted in
evaluations rather than wall-clock, so it is the only stage that may use all cores.

    uv run python -m tools.queue            # run the queue to completion
    uv run python -m tools.queue --status   # print progress and exit
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATE_DIR = ROOT / "results" / "run_state"
STATE_FILE = STATE_DIR / "queue_state.json"
HEARTBEAT = STATE_DIR / "heartbeat.json"
COMPLETE = STATE_DIR / "_QUEUE_COMPLETE"
LOG = STATE_DIR / "queue.log"

PY = [sys.executable]

# ---------------------------------------------------------------- stage definitions
# Subsets are fixed here, before the run, and are deterministic functions of the panel:
# the ablation spans every flexible family, the sensitivity subset spans both problem
# classes and the whole size range.
ABLATION_INSTANCES = ["mk01", "mk03", "mk06", "mk08", "mk10",
                      "edata_la01", "edata_la21", "rdata_la16", "vdata_la06", "fattahi05"]
ABLATION_METHODS = ["ga", "brkga", "de", "pso", "lshade", "cmaes", "gwo", "mde"]
SENSITIVITY_INSTANCES = ["ft10", "la36", "abz7", "swv06", "ta21",
                         "mk01", "mk06", "edata_la21", "rdata_la16", "fattahi05"]
SENSITIVITY_METHODS = ["ga", "brkga", "de", "pso", "abc", "gwo",
                       "lshade", "imode", "cmaes", "rime", "mde", "csa"]


@dataclass
class Stage:
    name: str
    description: str
    command: list[str]
    est_hours: float
    counts: tuple[Path, str] | None = None  # (directory, glob) for a progress readout
    expected: int = 0
    env: dict = field(default_factory=dict)


def stages() -> list[Stage]:
    raw = ROOT / "results" / "raw"
    return [
        Stage(
            "main",
            "Main comparison: 67 instances x 17 methods x 20 seeds x 300 s",
            PY + ["-m", "src.run.runner", "--config", "config/full.yaml",
                  "--run-name", "full"],
            est_hours=201.0,
            counts=(raw / "full", "*.json"),
            expected=67 * (2 + 15 * 20),
        ),
        Stage(
            "hybrid",
            "Warm-start hybrids, budgeted end to end, 3 variants x 4 budgets x 20 seeds",
            PY + ["-m", "src.run.run_hybrid", "--variant", "all", "--workers", "8"],
            est_hours=48.0,
            counts=(raw / "hybrid", "*.json"),
            expected=3 * 4 * 67 * 20,
        ),
        Stage(
            "reference",
            "Multi-threaded exact reference on the unproved subset, runs alone",
            PY + ["-m", "src.run.run_reference"],
            est_hours=8.0,
            counts=(raw / "reference", "*.json"),
        ),
        Stage(
            "ablation",
            "Decoder ablation: the discarded machine-key mapping, same instances and seeds",
            PY + ["-m", "src.run.runner", "--config", "config/full.yaml",
                  "--run-name", "ablation_legacy", "--mapping", "legacy",
                  "--instances", ",".join(ABLATION_INSTANCES),
                  "--methods", ",".join(ABLATION_METHODS), "--seeds", "10"],
            est_hours=8.5,
            counts=(raw / "ablation_legacy", "*.json"),
            expected=len(ABLATION_INSTANCES) * len(ABLATION_METHODS) * 10,
        ),
        Stage(
            "sens_pop20",
            "Population sensitivity: 20 individuals",
            PY + ["-m", "src.run.runner", "--config", "config/full.yaml",
                  "--run-name", "sens_pop20", "--pop-size", "20", "--budget", "60",
                  "--instances", ",".join(SENSITIVITY_INSTANCES),
                  "--methods", ",".join(SENSITIVITY_METHODS), "--seeds", "10"],
            est_hours=2.5,
            counts=(raw / "sens_pop20", "*.json"),
            expected=len(SENSITIVITY_INSTANCES) * len(SENSITIVITY_METHODS) * 10,
        ),
        Stage(
            "sens_pop100",
            "Population sensitivity: 100 individuals",
            PY + ["-m", "src.run.runner", "--config", "config/full.yaml",
                  "--run-name", "sens_pop100", "--pop-size", "100", "--budget", "60",
                  "--instances", ",".join(SENSITIVITY_INSTANCES),
                  "--methods", ",".join(SENSITIVITY_METHODS), "--seeds", "10"],
            est_hours=2.5,
            counts=(raw / "sens_pop100", "*.json"),
            expected=len(SENSITIVITY_INSTANCES) * len(SENSITIVITY_METHODS) * 10,
        ),
        Stage(
            "sens_recommended",
            "Population sensitivity: each method's own recommended rule",
            PY + ["-m", "src.run.runner", "--config", "config/full.yaml",
                  "--run-name", "sens_recommended", "--pop-policy", "recommended",
                  "--budget", "60",
                  "--instances", ",".join(SENSITIVITY_INSTANCES),
                  "--methods", ",".join(SENSITIVITY_METHODS), "--seeds", "10"],
            est_hours=2.5,
            counts=(raw / "sens_recommended", "*.json"),
            expected=len(SENSITIVITY_INSTANCES) * len(SENSITIVITY_METHODS) * 10,
        ),
        Stage(
            "cec",
            "Continuous-benchmark study: the same optimisers on CEC2017",
            PY + ["-m", "src.run.run_cec"],
            est_hours=4.0,
            counts=(raw / "cec", "*.json"),
        ),
    ]


# ---------------------------------------------------------------- state handling
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
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{stamp}] {message}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def _progress(stage: Stage) -> tuple[int, int]:
    if stage.counts is None:
        return (0, 0)
    directory, pattern = stage.counts
    have = len(list(directory.glob(pattern))) if directory.exists() else 0
    return (have, stage.expected)


def _heartbeat(stage_name: str, state: dict) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "current_stage": stage_name,
        "completed_stages": state.get("done", []),
        "progress": {s.name: dict(zip(("have", "expected"), _progress(s)))
                     for s in stages() if s.counts},
    }
    HEARTBEAT.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def print_status() -> None:
    state = _read_state()
    done = set(state.get("done", []))
    total_left = 0.0
    print(f"{'stage':20s} {'status':10s} {'progress':>16s} {'est. hours':>11s}")
    for s in stages():
        have, expected = _progress(s)
        prog = f"{have}/{expected}" if expected else (str(have) if have else "-")
        status = "done" if s.name in done else "pending"
        if s.name not in done:
            total_left += s.est_hours
        print(f"{s.name:20s} {status:10s} {prog:>16s} {s.est_hours:11.1f}")
    print(f"\nremaining estimate: {total_left:.0f} h ({total_left / 24:.1f} days)")
    if COMPLETE.exists():
        print("QUEUE COMPLETE")


# ---------------------------------------------------------------- driver
def run() -> int:
    if COMPLETE.exists():
        _log("queue already complete; nothing to do")
        return 0

    state = _read_state()
    done = set(state.get("done", []))

    from src.run.provenance import stamp_commit

    commit = stamp_commit()
    _log(f"code version for this invocation: {commit}")

    for stage in stages():
        if stage.name in done:
            continue
        have, expected = _progress(stage)
        _log(f"START stage '{stage.name}' ({stage.description}); "
             f"progress {have}/{expected or '?'}, estimate {stage.est_hours:.1f} h")
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
        have, expected = _progress(stage)
        _log(f"DONE stage '{stage.name}' in {elapsed:.2f} h; progress {have}/{expected or '?'}")
        _heartbeat(stage.name, state)

    COMPLETE.write_text(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), encoding="utf-8")
    _log("QUEUE COMPLETE - every stage finished")
    return 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--status", action="store_true", help="print progress and exit")
    ap.add_argument("--reset-stage", default=None, help="mark a stage as not done")
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
