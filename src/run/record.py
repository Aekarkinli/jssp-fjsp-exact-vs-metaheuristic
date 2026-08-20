"""Standalone result-record builder for the auxiliary runs (hybrid, multi-thread reference).

Kept separate from runner.py so editing it never disturbs an in-progress main run. Produces
the same record schema the aggregation reads.
"""
from __future__ import annotations

import importlib.metadata as md
import platform
import time

from src.core.feasibility import check_feasibility
from src.run.provenance import git_commit


def env_meta() -> dict:
    commit = git_commit()
    versions = {}
    for pkg in ("ortools", "pyjobshop", "mealpy", "numpy", "cma", "scipy", "opfunu"):
        try:
            versions[pkg] = md.version(pkg)
        except Exception:
            versions[pkg] = "unknown"
    return {"git_commit": commit, "versions": versions,
            "python": platform.python_version(), "hw_cpu": platform.processor()}


def build_record(result, instance, seed, budget, meta, **extra) -> dict:
    r = result
    sched = r.schedule
    feasible = sched is not None and check_feasibility(sched).feasible
    return {
        "instance_id": instance.name, "family": instance.family, "type": instance.problem_type,
        "n_jobs": instance.num_jobs, "n_machines": instance.num_machines,
        "n_op": instance.num_operations, "method": r.method, "seed": seed, "budget_s": budget,
        "status": r.status, "best_obj": None if r.best_obj == float("inf") else r.best_obj,
        "best_bound": r.best_bound, "rel_gap": r.rel_gap,
        "time_to_first": r.time_to_first, "time_to_best": r.time_to_best,
        "n_decoder_calls": r.n_decoder_calls, "n_repairs": r.n_repairs,
        "n_rejected": r.n_rejected, "n_restarts": r.n_restarts,
        "feasible_final": feasible, "crashed": r.crashed, "wall_time": r.wall_time,
        "peak_mem_bytes": None,
        "anytime": [[round(p.t, 4), p.obj, p.bound, p.evals] for p in r.anytime],
        "schedule": None if sched is None else {
            "machine": [a.machine for a in sched.assignments],
            "start": [a.start for a in sched.assignments]},
        "extra": r.extra, "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        **meta, **extra,
    }
