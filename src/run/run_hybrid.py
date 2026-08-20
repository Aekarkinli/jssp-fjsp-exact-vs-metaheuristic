"""Warm-started exact hybrid, budgeted end to end.

The first version of this study gave the warm-started exact solver a heuristic incumbent for
free and then granted it the full budget, so the seeding cost never appeared in the account
and a one-second hybrid was in truth a one-second exact run on top of however long the
seeding had taken. It also chose the seeding solution as the best heuristic result seen
afterwards, per instance and per budget, which is information no practitioner has in advance.

This runner fixes both. One total budget B covers the whole composite process. The clock
starts when the seeder starts; the seeder consumes b seconds; the warm-started exact phase
receives what is left, B - b. The reported anytime trace is the seed value at time b followed
by the exact solver's own trace shifted by b, so the composite can be read at any checkpoint
exactly as a single method is.

Three variants:

``hyb_cheap``   seeded by the dispatching-rule bank, which costs milliseconds.
``hyb_tabu``    seeded by the tabu search given a fixed fifth of the budget.
``hyb_oracle``  seeded by the best heuristic schedule found anywhere in the main run whose
                time-to-best fits inside B, charged at that time-to-best.

The first two are fixed by rule and depend on no result, so they carry the necessity class.
The third depends on results and is reported only as an unattainable upper bound; the
distance between it and the other two measures what the first version's selection rule was
worth. Every variant runs with the full seed list, so hybrid-versus-exact and
hybrid-versus-heuristic are comparisons of two distributions rather than of a point against
a distribution.

    uv run python -m src.run.run_hybrid --variant hyb_tabu --budget 300 --workers 8
"""
from __future__ import annotations

import os

# Force single-threaded numerics before NumPy is imported anywhere in this process tree.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import multiprocessing as mp  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import psutil  # noqa: E402
import yaml  # noqa: E402
from pyjobshop import Solution  # noqa: E402
from pyjobshop.Solution import ScheduledTask  # noqa: E402

from src.core.schedule import Assignment, Schedule  # noqa: E402
from src.methods.base import AnytimePoint, MethodResult  # noqa: E402
from src.methods.constructive import solve_dispatching  # noqa: E402
from src.methods.exact_cpsat import build_model, solve_cpsat  # noqa: E402
from src.methods.tabu import solve_tabu  # noqa: E402
from src.run.record import build_record, env_meta  # noqa: E402
from src.run.runner import INDEX, _load  # noqa: E402

RAW = Path("results/raw")
HYBRID = RAW / "hybrid"
MAIN_RUN = "full"
TABU_BUDGET_FRACTION = 0.2
VARIANTS = ("hyb_cheap", "hyb_tabu", "hyb_oracle")

_META: dict = {}
_CORES: list[int] = []


def _oracle_seed(instance_id: str, budget: float):
    """Best heuristic schedule from the main run that was reached within the budget.

    Returns (objective, time_to_best, machines, starts, source_method) or None. Only runs
    whose time-to-best fits inside the budget qualify, so the oracle is charged honestly for
    the time it would have needed.
    """
    best = None
    for p in (RAW / MAIN_RUN).glob(f"{instance_id}__*.json"):
        r = json.loads(p.read_text(encoding="utf-8"))
        if (r.get("method") == "cpsat" or r.get("crashed") or not r.get("feasible_final")
                or r.get("schedule") is None or r.get("best_obj") is None):
            continue
        ttb = r.get("time_to_best")
        if ttb is None or ttb > budget:
            continue
        if best is None or r["best_obj"] < best["best_obj"]:
            best = r
    if best is None:
        return None
    return (float(best["best_obj"]), float(best["time_to_best"]),
            best["schedule"]["machine"], best["schedule"]["start"], best["method"])


def _warm_solution(instance, machines, starts) -> Solution:
    data = build_model(instance).data()
    tasks = []
    for op in instance.operations:
        g = op.global_index
        mch = machines[g]
        mode = next(mi for mi in data.task2modes(g) if data.modes[mi].resources[0] == mch)
        dur = op.duration_on(mch)
        tasks.append(ScheduledTask(mode, [mch], starts[g], starts[g] + dur, 0, 0, True))
    return Solution(data, tasks)


def result_path(instance_id: str, variant: str, budget: float, seed: int) -> Path:
    return HYBRID / f"{instance_id}__{variant}_b{int(budget)}__seed{seed:03d}.json"


def _seed_phase(variant: str, instance, budget: float, seed: int):
    """Run the seeding phase. Returns (objective, elapsed, machines, starts, source)."""
    if variant == "hyb_cheap":
        t0 = time.perf_counter()
        r = solve_dispatching(instance)
        elapsed = time.perf_counter() - t0
        sched = r.schedule
        return (float(r.best_obj), elapsed,
                [a.machine for a in sched.assignments],
                [a.start for a in sched.assignments], "dispatching")

    if variant == "hyb_tabu":
        t0 = time.perf_counter()
        r = solve_tabu(instance, TABU_BUDGET_FRACTION * budget, seed=seed)
        elapsed = time.perf_counter() - t0
        sched = r.schedule
        return (float(r.best_obj), elapsed,
                [a.machine for a in sched.assignments],
                [a.start for a in sched.assignments], "tabu")

    oracle = _oracle_seed(instance.name, budget)
    if oracle is None:
        return None
    obj, ttb, machines, starts, source = oracle
    return (obj, ttb, machines, starts, source)


def _run_one(job):
    instance_id, variant, budget, seed = job
    dest = result_path(instance_id, variant, budget, seed)
    if dest.exists():
        return ("skip", instance_id, variant, budget, seed, None)

    instance = _load(instance_id)
    seeded = _seed_phase(variant, instance, budget, seed)
    if seeded is None:
        return ("noseed", instance_id, variant, budget, seed, None)
    seed_obj, seed_time, machines, starts, source = seeded

    remaining = budget - seed_time
    anytime = [AnytimePoint(t=min(seed_time, budget), obj=seed_obj, bound=None, evals=None)]
    best_obj, best_bound, status = seed_obj, None, "FEASIBLE"
    schedule = None
    exact_wall = 0.0

    if remaining > 0:
        warm = _warm_solution(instance, machines, starts)
        r = solve_cpsat(instance, remaining, seed=seed, num_workers=1, initial_solution=warm)
        exact_wall = r.wall_time
        for p in r.anytime:
            anytime.append(AnytimePoint(t=seed_time + p.t, obj=p.obj, bound=p.bound, evals=None))
        if r.feasible_final and r.best_obj < best_obj:
            best_obj = float(r.best_obj)
        best_bound = r.best_bound
        status = r.status
        schedule = r.schedule
    if schedule is None:
        # the exact phase found nothing better; keep the seeding schedule as the answer
        schedule = Schedule(instance, tuple(
            Assignment(op.global_index, op.job, machines[op.global_index],
                       starts[op.global_index], op.duration_on(machines[op.global_index]))
            for op in instance.operations))

    res = MethodResult(
        method=variant,
        instance=instance.name,
        status=status,
        best_obj=float(best_obj),
        best_bound=best_bound,
        schedule=schedule,
        feasible_final=True,
        anytime=anytime,
        time_to_first=min(seed_time, budget),
        time_to_best=anytime[-1].t,
        wall_time=seed_time + exact_wall,
        extra={"seeder": source, "seed_objective": seed_obj,
               "seed_time_s": seed_time, "exact_time_s": max(0.0, remaining),
               "oracle_free": variant != "hyb_oracle"},
    )

    rec = build_record(res, instance, seed, budget, _META, variant=variant)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(rec), encoding="utf-8")
    tmp.replace(dest)
    return ("ok", instance_id, variant, budget, seed, rec["best_obj"])


def _init_worker(cores: list[int], meta: dict) -> None:
    global _META
    _META = meta
    try:
        idx = mp.current_process()._identity[0] - 1
        psutil.Process().cpu_affinity([cores[idx % len(cores)]])
    except Exception:
        pass


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="all", help="hyb_cheap | hyb_tabu | hyb_oracle | all")
    ap.add_argument("--budgets", default=None, help="comma-separated, default from config")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--config", default="config/full.yaml")
    ap.add_argument("--instances", default=None, help="comma-separated subset, for checks")
    ap.add_argument("--seeds", type=int, default=None, help="use only the first N seeds")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    cores = cfg["execution"]["timed_affinity"]["logical_processors"]
    seeds = list(cfg["seeds"]["stochastic"])
    budgets = ([float(b) for b in args.budgets.split(",")] if args.budgets
               else [float(b) for b in cfg["hybrid"]["budgets_seconds"]])
    variants = VARIANTS if args.variant == "all" else (args.variant,)
    if args.seeds is not None:
        seeds = seeds[: args.seeds]
    instances = ([i for i in args.instances.split(",") if i in INDEX] if args.instances
                 else list(INDEX))

    jobs = [(iid, v, b, int(s))
            for v in variants for b in budgets for iid in instances for s in seeds]
    pending = [j for j in jobs if not result_path(j[0], j[1], j[2], j[3]).exists()]
    HYBRID.mkdir(parents=True, exist_ok=True)
    workers = max(1, min(args.workers, len(cores)))
    print(f"hybrid: {len(jobs)} jobs ({len(pending)} pending), variants={variants}, "
          f"budgets={budgets}, workers={workers}")
    if not pending:
        print("nothing to do")
        return

    meta = env_meta()
    counts = {"ok": 0, "skip": 0, "noseed": 0}
    start = time.perf_counter()
    with mp.Pool(processes=workers, initializer=_init_worker,
                 initargs=(cores[:workers], meta)) as pool:
        for i, (status, iid, variant, budget, seed, obj) in enumerate(
            pool.imap_unordered(_run_one, pending), start=1
        ):
            counts[status] = counts.get(status, 0) + 1
            if i % 50 == 0 or i == len(pending):
                el = time.perf_counter() - start
                print(f"  [{i}/{len(pending)}] ok={counts['ok']} noseed={counts['noseed']} "
                      f"({el:.0f}s) last={iid}/{variant}/b{int(budget)}/s{seed}={obj}")
    print(f"hybrid done: {counts}")


if __name__ == "__main__":
    main()
