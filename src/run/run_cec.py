"""Continuous-benchmark study: the same optimisers, the same code, on CEC2017.

The transfer question the study asks is whether strength on continuous benchmark functions
predicts strength on a combinatorial scheduling problem. The first version of this work never
measured the continuous side; it cited the source papers and inferred the rest. This stage
measures it directly, with the same software versions, the same population setting and the
same implementations that run through the scheduling decoder, so the two rankings are
comparable and their correlation is an observation rather than an assumption.

Protocol follows the competition definition: dimension 30, a budget of 10,000 x D function
evaluations, 51 independent runs per function, and the error to the known optimum as the
performance measure. Twenty-nine functions are used, which is the suite minus the one
withdrawn from the official definition.

Function numbering. The benchmark library indexes the suite consecutively from one to
twenty-nine, having already removed the withdrawn function, so its index two is the official
third function and its index twenty-nine is the official thirtieth. Each record therefore
stores the library index, the official competition index, and the function name, so results
can be reported against the official numbering without ambiguity.

Termination is by evaluation count, not wall-clock, so this stage is the only one that may
use every core: nothing here is timing-sensitive.

    uv run python -m src.run.run_cec [--runs 51] [--dim 30] [--workers 16]
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import multiprocessing as mp  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

warnings.filterwarnings("ignore")

from src.methods.brkga import ELITE_FRACTION, MUTANT_FRACTION, RHO_ELITE  # noqa: E402
from src.methods.metaheuristic import DEFAULT_POP_SIZE, OPTIMIZERS  # noqa: E402
from src.methods.ported import csa_optimize, mde_optimize  # noqa: E402
from src.run.record import env_meta  # noqa: E402

OUT = Path("results/raw/cec")
# The library indexes the suite consecutively after removing the withdrawn function, so
# these twenty-nine indices are exactly the official first and third to thirtieth functions.
FUNCTION_IDS = list(range(1, 30))
METHODS = ["ga", "brkga", "de", "pso", "abc", "gwo", "lshade", "imode", "cmaes",
           "rime", "mde", "csa"]


def _official_id(fid: int) -> int:
    """Official competition index of library index `fid` (the second one was withdrawn)."""
    return fid if fid == 1 else fid + 1


def _function(fid: int, dim: int):
    from opfunu.cec_based import cec2017

    return getattr(cec2017, f"F{fid}2017")(ndim=dim)


class _Budget(Exception):
    """Raised to stop an optimiser that has spent its evaluation budget."""


class _Counter:
    """Wraps the benchmark function with an evaluation budget and a best-so-far record."""

    def __init__(self, fn, budget: int) -> None:
        self.fn = fn
        self.budget = budget
        self.n = 0
        self.best = float("inf")

    def __call__(self, x) -> float:
        if self.n >= self.budget:
            raise _Budget
        self.n += 1
        value = float(self.fn.evaluate(np.asarray(x, dtype=float)))
        if value < self.best:
            self.best = value
        return value

    def batch(self, population) -> np.ndarray:
        return np.array([self(row) for row in np.asarray(population, dtype=float)])


def _run_mealpy(method: str, fn, budget: int, seed: int, pop_size: int) -> float:
    from mealpy import FloatVar, Problem, Termination

    counter = _Counter(fn, budget)
    lb, ub = list(map(float, fn.lb)), list(map(float, fn.ub))

    class _P(Problem):
        def __init__(self):
            super().__init__(bounds=FloatVar(lb=lb, ub=ub), minmax="min", log_to=None)

        def obj_func(self, x):
            return counter(x)

    model = OPTIMIZERS[method](pop_size)
    try:
        model.solve(_P(), termination=Termination(max_fe=budget), mode="single", seed=seed)
    except (_Budget, Exception):  # noqa: BLE001
        pass
    return counter.best


def _run_brkga(fn, budget: int, seed: int, pop_size: int) -> float:
    """BRKGA on a continuous domain: keys in the unit box map linearly onto the box.

    The method is defined over random keys, so a decoder is part of it by construction. On a
    box-constrained continuous problem the natural decoder is the affine one, which is what
    is used here.
    """
    rng = np.random.default_rng(seed)
    lb, ub = np.asarray(fn.lb, dtype=float), np.asarray(fn.ub, dtype=float)
    dim = len(lb)
    counter = _Counter(fn, budget)

    def evaluate(keys):
        return np.array([counter(lb + k * (ub - lb)) for k in keys])

    n_elite = max(1, int(round(ELITE_FRACTION * pop_size)))
    n_mutant = max(1, int(round(MUTANT_FRACTION * pop_size)))
    n_cross = max(0, pop_size - n_elite - n_mutant)
    try:
        pop = rng.random((pop_size, dim))
        fit = evaluate(pop)
        while True:
            order = np.argsort(fit, kind="stable")
            elite, rest = pop[order[:n_elite]], pop[order[n_elite:]]
            elite_fit = fit[order[:n_elite]]
            offspring = rng.random((n_mutant, dim))
            if n_cross:
                ea = elite[rng.integers(0, n_elite, size=n_cross)]
                eb = rest[rng.integers(0, max(1, len(rest)), size=n_cross)]
                offspring = np.vstack(
                    [offspring, np.where(rng.random((n_cross, dim)) < RHO_ELITE, ea, eb)]
                )
            child_fit = evaluate(offspring)
            pop = np.vstack([elite, offspring])
            fit = np.concatenate([elite_fit, child_fit])
    except _Budget:
        pass
    return counter.best


def _run_cmaes(fn, budget: int, seed: int, pop_size: int) -> float:
    import cma

    lb, ub = np.asarray(fn.lb, dtype=float), np.asarray(fn.ub, dtype=float)
    counter = _Counter(fn, budget)
    options = {"bounds": [list(lb), list(ub)], "popsize": pop_size, "seed": seed + 1,
               "verbose": -9, "verb_log": 0, "maxiter": 10**9, "tolfun": 0.0,
               "tolx": 0.0, "tolflatfitness": 10**9, "tolstagnation": 10**9,
               "CMA_diagonal": True}
    try:
        es = cma.CMAEvolutionStrategy(list((lb + ub) / 2.0), float(np.mean(ub - lb) * 0.3), options)
        while True:
            xs = es.ask()
            es.tell(xs, [counter(x) for x in xs])
    except (_Budget, Exception):  # noqa: BLE001
        pass
    return counter.best


def _run_ported(method: str, fn, budget: int, seed: int, pop_size: int) -> float:
    counter = _Counter(fn, budget)
    optimiser = mde_optimize if method == "mde" else csa_optimize
    lb, ub = float(fn.lb[0]), float(fn.ub[0])
    try:
        optimiser(counter.batch, len(fn.lb), lb, ub, pop_size, seed, max_cycles=10**9)
    except (_Budget, Exception):  # noqa: BLE001
        pass
    return counter.best


def _one_cell(job):
    method, fid, dim, runs, budget, pop_size, seeds = job
    dest = OUT / f"cec2017_F{fid}_d{dim}__{method}.json"
    if dest.exists():
        return ("skip", method, fid, None)
    fn = _function(fid, dim)
    optimum = float(fn.f_global)
    errors = []
    t0 = time.perf_counter()
    for i in range(runs):
        seed = int(seeds[i % len(seeds)]) + 1000 * (i // len(seeds))
        if method == "brkga":
            best = _run_brkga(fn, budget, seed, pop_size)
        elif method == "cmaes":
            best = _run_cmaes(fn, budget, seed, pop_size)
        elif method in ("mde", "csa"):
            best = _run_ported(method, fn, budget, seed, pop_size)
        else:
            best = _run_mealpy(method, fn, budget, seed, pop_size)
        errors.append(max(0.0, best - optimum))
    record = {
        "suite": "CEC2017", "function": fid, "function_official": _official_id(fid),
        "function_name": str(getattr(fn, "name", "")), "dim": dim, "method": method,
        "runs": runs, "budget_evaluations": budget, "pop_size": pop_size,
        "f_optimum": optimum, "errors": errors,
        "mean_error": float(np.mean(errors)), "median_error": float(np.median(errors)),
        "std_error": float(np.std(errors, ddof=1)) if len(errors) > 1 else 0.0,
        "best_error": float(np.min(errors)), "worst_error": float(np.max(errors)),
        "wall_s": time.perf_counter() - t0,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        **env_meta(),
    }
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(record), encoding="utf-8")
    tmp.replace(dest)
    return ("ok", method, fid, record["median_error"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=51)
    ap.add_argument("--dim", type=int, default=30)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--pop-size", type=int, default=DEFAULT_POP_SIZE)
    args = ap.parse_args()

    budget = 10_000 * args.dim
    seeds = [11, 23, 37, 41, 53, 67, 79, 83, 97, 101, 113, 127, 131, 149, 157, 167, 173,
             181, 191, 199]
    jobs = [(m, f, args.dim, args.runs, budget, args.pop_size, seeds)
            for m in METHODS for f in FUNCTION_IDS]
    OUT.mkdir(parents=True, exist_ok=True)
    pending = [j for j in jobs
               if not (OUT / f"cec2017_F{j[1]}_d{j[2]}__{j[0]}.json").exists()]
    print(f"CEC2017: {len(jobs)} cells ({len(pending)} pending), dim={args.dim}, "
          f"budget={budget} evaluations, {args.runs} runs each, workers={args.workers}")
    if not pending:
        print("nothing to do")
        return

    start = time.perf_counter()
    with mp.Pool(processes=args.workers) as pool:
        for i, (status, method, fid, med) in enumerate(
            pool.imap_unordered(_one_cell, pending), start=1
        ):
            if i % 10 == 0 or i == len(pending):
                el = time.perf_counter() - start
                print(f"  [{i}/{len(pending)}] ({el:.0f}s) last={method}/F{fid} median={med}")
    print(f"CEC2017 done -> {OUT}")


if __name__ == "__main__":
    main()
