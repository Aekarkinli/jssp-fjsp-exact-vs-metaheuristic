"""Config-driven, parallel, resumable runner for the fair comparison.

Enumerates (instance, method, seed), runs each job, and writes one result file per
combination to ``results/raw/``. Already-completed combinations are skipped, so the run is
resumable and the analysis reads whatever is on disk.

Fairness and isolation:
- single-threaded numerics are forced by setting the BLAS thread environment variables to
  one *before* NumPy is imported, here at module top, so spawned workers inherit it;
- each worker pins itself to one performance core (no SMT sibling), so no timed job shares
  a core or competes for one;
- CP-SAT runs with a single search worker in this fair comparison.

Deterministic methods (CP-SAT single-thread, the dispatching bank, the greedy constructor)
run once; stochastic methods run once per seed.

    uv run python -m src.run.runner --config config/pilot.yaml
    uv run python -m src.run.runner --config config/pilot.yaml --budget 2 --seeds 1   # smoke
"""
from __future__ import annotations

import os

# Force single-threaded numerics before NumPy is imported anywhere in this process tree.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import importlib.metadata as _md  # noqa: E402
import json  # noqa: E402
import multiprocessing as mp  # noqa: E402
import platform  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import psutil  # noqa: E402
import yaml  # noqa: E402

from src.core.decoder import DEFAULT_MAPPING  # noqa: E402
from src.core.feasibility import check_feasibility  # noqa: E402
from src.io.instance_sources import FJSP_INSTANCES, JSSP_INSTANCES  # noqa: E402
from src.io.loaders import load_fjsp_file, load_jssp_file  # noqa: E402
from src.methods.anneal import solve_anneal  # noqa: E402
from src.methods.brkga import solve_brkga  # noqa: E402
from src.methods.cmaes import solve_cmaes  # noqa: E402
from src.methods.constructive import solve_dispatching, solve_greedy  # noqa: E402
from src.methods.exact_cpsat import solve_cpsat  # noqa: E402
from src.methods.metaheuristic import (  # noqa: E402
    OPTIMIZERS,
    recommended_pop_size,
    solve_metaheuristic,
)
from src.methods.ported import solve_ported  # noqa: E402
from src.methods.tabu import solve_tabu  # noqa: E402

ALL_METHODS = ["cpsat", "dispatching", "greedy", "tabu", "sa", "ga", "brkga", "de", "pso",
               "abc", "gwo", "lshade", "imode", "cmaes", "rime", "mde", "csa"]
# Only the two constructive methods are genuinely deterministic. The exact solver's random
# seed changes its search, so it runs with the full seed list like every other method;
# representing it by a single run would make the treatment of uncertainty asymmetric.
DETERMINISTIC = {"dispatching", "greedy"}
DECODED = {"ga", "brkga", "de", "pso", "abc", "gwo", "lshade", "imode", "cmaes", "rime",
           "mde", "csa"}
ALIAS = {"recent_extra": "rime", "greedy_gt": "greedy"}
RAW = Path("results/raw")
DATA = Path("data/instances")

# instance id -> (kind, local relative path, family); computed once at import
INDEX: dict[str, tuple[str, str, str]] = {}
for _s in JSSP_INSTANCES:
    INDEX[_s.id] = ("jssp", _s.local, _s.family)
for _s in FJSP_INSTANCES:
    INDEX[_s.id] = ("fjsp", _s.local, _s.family)

_META: dict = {}  # set per worker by the pool initializer


def _load(instance_id: str):
    kind, local, family = INDEX[instance_id]
    loader = load_jssp_file if kind == "jssp" else load_fjsp_file
    return loader(DATA / local, instance_id, family)


def _run_method(method, instance, budget, seed, pop_size, mapping=DEFAULT_MAPPING,
                pop_policy="common"):
    if method == "cpsat":
        return solve_cpsat(instance, budget, seed=seed, num_workers=1)
    if method == "dispatching":
        return solve_dispatching(instance)
    if method == "greedy":
        return solve_greedy(instance)
    if method == "tabu":
        return solve_tabu(instance, budget, seed=seed, mapping=mapping)
    if method == "sa":
        return solve_anneal(instance, budget, seed=seed, mapping=mapping)
    if pop_policy == "recommended":
        from src.core.decoder import vector_length
        pop_size = recommended_pop_size(method, vector_length(instance))
    if method == "brkga":
        return solve_brkga(instance, budget, seed=seed, pop_size=pop_size, mapping=mapping)
    if method == "cmaes":
        return solve_cmaes(instance, budget, seed=seed, pop_size=pop_size, mapping=mapping)
    if method in ("mde", "csa"):
        return solve_ported(instance, method, budget, seed=seed, pop_size=pop_size,
                            mapping=mapping)
    if method in OPTIMIZERS:
        return solve_metaheuristic(instance, method, budget, seed=seed, pop_size=pop_size,
                                   mapping=mapping)
    raise ValueError(f"unknown method {method!r}")


def result_path(instance_id: str, method: str, seed: int, run_name: str) -> Path:
    return RAW / run_name / f"{instance_id}__{method}__seed{seed:03d}.json"


def _serialize(r, instance, seed, budget, peak_mem, run_wall, settings=None) -> dict:
    sched = r.schedule
    feasible = sched is not None and check_feasibility(sched).feasible
    rec = {
        "instance_id": instance.name, "family": instance.family, "type": instance.problem_type,
        "n_jobs": instance.num_jobs, "n_machines": instance.num_machines,
        "n_op": instance.num_operations, "method": r.method, "seed": seed, "budget_s": budget,
        "status": r.status,
        "best_obj": None if r.best_obj == float("inf") else r.best_obj,
        "best_bound": r.best_bound, "rel_gap": r.rel_gap,
        "time_to_first": r.time_to_first, "time_to_best": r.time_to_best,
        "n_decoder_calls": r.n_decoder_calls, "n_repairs": r.n_repairs,
        "n_rejected": r.n_rejected, "n_restarts": r.n_restarts,
        "feasible_final": feasible, "crashed": r.crashed,
        "wall_time": r.wall_time, "run_wall": run_wall, "peak_mem_bytes": peak_mem,
        # [wall-clock, objective, dual bound, decoder evaluations]; the last two are null
        # where the notion does not apply to the method
        "anytime": [[round(p.t, 4), p.obj, p.bound, p.evals] for p in r.anytime],
        "settings": settings or {},
        "schedule": None if sched is None else {
            "machine": [a.machine for a in sched.assignments],
            "start": [a.start for a in sched.assignments],
        },
        "extra": r.extra,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        **_META,
    }
    return rec


def run_job(job) -> tuple:
    instance_id, method, seed, budget, pop_size, run_name, mapping, pop_policy = job
    out = result_path(instance_id, method, seed, run_name)
    if out.exists():
        return ("skip", instance_id, method, seed, None)
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        affinity = psutil.Process().cpu_affinity()
    except Exception:
        affinity = None
    try:
        instance = _load(instance_id)
        t0 = time.perf_counter()
        r = _run_method(method, instance, budget, seed, pop_size, mapping, pop_policy)
        run_wall = time.perf_counter() - t0
        try:
            mi = psutil.Process().memory_info()
            peak = getattr(mi, "peak_wset", mi.rss)
        except Exception:
            peak = None
        rec = _serialize(r, instance, seed, budget, peak, run_wall,
                         settings={"mapping": mapping, "pop_policy": pop_policy,
                                   "pop_size": int(pop_size)})
        rec["affinity_core"] = affinity
        rec["run_name"] = run_name
        status = "ok" if (rec["feasible_final"] and not r.crashed) else "fail"
        best = rec["best_obj"]
    except Exception as exc:  # noqa: BLE001
        rec = {"instance_id": instance_id, "method": method, "seed": seed,
               "crashed": True, "error": repr(exc), "affinity_core": affinity, **_META}
        status, best = "fail", None

    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(rec), encoding="utf-8")
    tmp.replace(out)  # atomic publish, so a partial file is never seen as complete
    return (status, instance_id, method, seed, best)


def _init_worker(cores: list[int], meta: dict) -> None:
    global _META
    _META = meta
    try:
        idx = mp.current_process()._identity[0] - 1
        psutil.Process().cpu_affinity([cores[idx % len(cores)]])
    except Exception:
        pass


def _git_commit() -> str:
    from src.run.provenance import git_commit

    return git_commit()


def _versions() -> dict:
    out = {}
    for pkg in ("ortools", "pyjobshop", "mealpy", "numpy", "scipy", "opfunu", "cma",
                "statsmodels"):
        try:
            out[pkg] = _md.version(pkg)
        except Exception:
            out[pkg] = "unknown"
    return out


def _method_ids(config, override) -> list[str]:
    if override:
        raw = override
    else:
        raw = config["methods"]
    ids = []
    for m in raw:
        mid = m if isinstance(m, str) else m.get("id")
        mid = ALIAS.get(mid, mid)
        if mid in ALL_METHODS and mid not in ids:
            ids.append(mid)
    return ids


def _instance_ids(config, override) -> list[str]:
    if override:
        return [i for i in override if i in INDEX]
    inst = config.get("instances", "all")
    if isinstance(inst, list):
        return [i for i in inst if i in INDEX]
    return list(INDEX)  # "all" or a documentation dict -> the full curated set


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--budget", type=float, default=None)
    ap.add_argument("--seeds", type=int, default=None, help="use only the first N seeds")
    ap.add_argument("--instances", default=None, help="comma-separated instance ids")
    ap.add_argument("--methods", default=None, help="comma-separated method ids")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--run-name", default=None, help="output subdir (default: config stem)")
    ap.add_argument("--mapping", default=DEFAULT_MAPPING,
                    help="FJSP machine-key mapping: eligible (primary) or legacy (ablation)")
    ap.add_argument("--pop-policy", default="common", choices=["common", "recommended"],
                    help="common: one population size for all; recommended: each method's rule")
    ap.add_argument("--pop-size", type=int, default=None, help="override the common population")
    args = ap.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    run_name = args.run_name or Path(args.config).stem
    budget = args.budget if args.budget is not None else config["budget"]["t_max_seconds"]
    methods = _method_ids(config, args.methods.split(",") if args.methods else None)
    instances = _instance_ids(config, args.instances.split(",") if args.instances else None)
    seeds = list(config["seeds"]["stochastic"])
    if args.seeds is not None:
        seeds = seeds[: args.seeds]
    pop_size = args.pop_size or config.get("parameters", {}).get("population_size", 50)
    cores = config["execution"]["timed_affinity"]["logical_processors"]
    workers = args.workers or min(len(cores), config["execution"]["timed_affinity"]["workers"])

    jobs = []
    for instance_id in instances:
        for method in methods:
            chosen = [seeds[0]] if method in DETERMINISTIC else seeds
            for s in chosen:
                jobs.append((instance_id, method, int(s), float(budget), int(pop_size),
                             run_name, args.mapping, args.pop_policy))

    (RAW / run_name).mkdir(parents=True, exist_ok=True)
    pending = [j for j in jobs if not result_path(j[0], j[1], j[2], run_name).exists()]
    print(f"config={args.config} run={run_name} budget={budget}s workers={workers} | "
          f"{len(instances)} instances x {len(methods)} methods x {len(seeds)} seeds")
    print(f"{len(jobs)} jobs total, {len(jobs) - len(pending)} already done, {len(pending)} pending")
    if not pending:
        print("nothing to do (all results present)")
        return

    meta = {"git_commit": _git_commit(), "python": platform.python_version(),
            "versions": _versions(), "hw_cpu": platform.processor(),
            "hostname": platform.node()}

    done = {"ok": 0, "fail": 0, "skip": 0}
    failures = []
    start = time.perf_counter()
    with mp.Pool(processes=workers, initializer=_init_worker,
                 initargs=(cores[:workers], meta)) as pool:
        for i, (status, iid, method, seed, best) in enumerate(
            pool.imap_unordered(run_job, pending), start=1
        ):
            done[status] = done.get(status, 0) + 1
            if status == "fail":
                failures.append((iid, method, seed))
            if i % 10 == 0 or i == len(pending):
                el = time.perf_counter() - start
                print(f"  [{i}/{len(pending)}] ok={done['ok']} fail={done['fail']} "
                      f"({el:.0f}s) last={iid}/{method}/s{seed}={best}")

    print(f"done: ok={done['ok']} fail={done['fail']} skip={done['skip']}")
    if failures:
        print("FAILURES:", failures[:20])
        raise SystemExit(f"{len(failures)} job(s) failed")


if __name__ == "__main__":
    main()
