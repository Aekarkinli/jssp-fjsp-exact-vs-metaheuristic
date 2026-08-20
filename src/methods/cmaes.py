"""Covariance matrix adaptation evolution strategy through the shared decoder.

CMA-ES is the reference continuous optimiser against which new real-valued methods are
usually measured, so a comparison that decodes real vectors into schedules is incomplete
without it. The reference implementation by the method's own author is used rather than a
re-implementation.

The search operates in the unit box, box constraints are handled by the library's own
transformation, and the initial point is the centre of the box with an initial step size of
0.3, which is the library's recommended value for a unit-scaled search. Population size
follows the study-wide setting so that every population-based method carries the same value;
the method's own default rule is exercised separately in the population-size sensitivity
study.

Very high dimension. The scheduling vectors reach several thousand components on the largest
instances, where a full covariance model costs quadratic memory and cubic update time. The
library's separable mode is therefore enabled for every instance, uniformly, rather than for
selected ones, so no instance receives a different algorithm from another.
"""
from __future__ import annotations

import time

import cma
import numpy as np

from src.core.decoder import DEFAULT_MAPPING, decode_vector, vector_length
from src.core.instance import Instance
from src.methods.base import MethodResult, RunRecorder

SIGMA0 = 0.3


def solve_cmaes(
    instance: Instance,
    time_limit: float,
    seed: int,
    pop_size: int = 50,
    mapping: str = DEFAULT_MAPPING,
    separable: bool = True,
) -> MethodResult:
    dim = vector_length(instance)
    recorder = RunRecorder()

    def objective(x) -> float:
        result = decode_vector(instance, np.clip(x, 0.0, 1.0), mapping=mapping)
        makespan = float(result.schedule.makespan)
        recorder.record(makespan, result.n_repairs, np.clip(x, 0.0, 1.0))
        return makespan

    options = {
        "bounds": [0.0, 1.0],
        "popsize": int(pop_size),
        "seed": int(seed) + 1,  # the library reserves 0 for "no seeding"
        "verbose": -9,
        "verb_log": 0,
        "verb_disp": 0,
        "maxiter": 10**9,
        "tolfun": 0.0,
        "tolfunhist": 0.0,
        "tolx": 0.0,
        "tolflatfitness": 10**9,
        "tolstagnation": 10**9,
    }
    if separable:
        options["CMA_diagonal"] = True

    crashed = False
    start = recorder.start
    try:
        es = cma.CMAEvolutionStrategy([0.5] * dim, SIGMA0, options)
        while time.perf_counter() - start < time_limit:
            candidates = es.ask()
            es.tell(candidates, [objective(c) for c in candidates])
    except Exception:  # noqa: BLE001 - a crashing optimiser is logged, not fatal to the run
        crashed = True
    wall = time.perf_counter() - start

    schedule = None
    if recorder.best_x is not None:
        schedule = decode_vector(instance, recorder.best_x, mapping=mapping).schedule

    return MethodResult(
        method="cmaes",
        instance=instance.name,
        status="FEASIBLE" if schedule is not None else "UNKNOWN",
        best_obj=recorder.best,
        best_bound=None,
        schedule=schedule,
        feasible_final=schedule is not None,
        anytime=recorder.points(),
        time_to_first=recorder.time_to_first,
        time_to_best=recorder.time_to_best,
        wall_time=wall,
        n_decoder_calls=recorder.n_calls,
        n_repairs=recorder.n_repairs,
        crashed=crashed,
        extra={"separable": bool(separable), "sigma0": SIGMA0},
    )
