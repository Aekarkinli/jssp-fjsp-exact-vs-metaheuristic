"""Library optimisers behind the common interface, through the shared decoder.

Every real-valued optimiser searches a continuous vector in [0,1]^D that the shared decoder
turns into a schedule. The objective wrapper records the best-so-far trace stamped with both
wall-clock time and decoder-evaluation count, the evaluation count itself, and the repair
count, so every method produces the same log.

Termination is by wall-clock time; the epoch cap is set to the library maximum so the time
limit is the binding constraint. Evaluation runs single-threaded so no method spawns threads,
matching the single-thread fairness rule and the numerical-library pinning.

Parameter policy. Two configurations are defined. The *common* configuration gives every
population-based method the same population size and leaves all remaining parameters at their
library or source defaults; it is a comparability choice and the manuscript states it as such
rather than calling it a default configuration. The *self-configured* alternative lets each
method set its population size by its own recommended rule and is exercised in the
population-size sensitivity study. ``recommended_pop_size`` holds those rules.

Registry note. The genetic algorithm uses the library's real-coded variant. The library's
other genetic-algorithm entry never evaluates its offspring under default settings on a
continuous problem, which would silently reduce a core baseline to a random search.
"""
from __future__ import annotations

import math
import time

from mealpy import FloatVar, Problem, Termination
from mealpy.evolutionary_based.DE import OriginalDE
from mealpy.evolutionary_based.GA import BaseGA
from mealpy.evolutionary_based.SHADE import L_SHADE
from mealpy.physics_based.RIME import OriginalRIME
from mealpy.sota_based.IMODE import OriginalIMODE
from mealpy.swarm_based.ABC import OriginalABC
from mealpy.swarm_based.GWO import OriginalGWO
from mealpy.swarm_based.PSO import OriginalPSO

from src.core.decoder import DEFAULT_MAPPING, decode_vector, vector_length
from src.core.instance import Instance
from src.methods.base import MethodResult, RunRecorder

EPOCH_CAP = 100000  # library maximum; the wall-clock limit is the binding constraint
DEFAULT_POP_SIZE = 50

# name -> factory(pop_size). All remaining parameters stay at library defaults.
OPTIMIZERS = {
    "ga": lambda ps: BaseGA(epoch=EPOCH_CAP, pop_size=ps),
    "de": lambda ps: OriginalDE(epoch=EPOCH_CAP, pop_size=ps),
    "pso": lambda ps: OriginalPSO(epoch=EPOCH_CAP, pop_size=ps),
    "abc": lambda ps: OriginalABC(epoch=EPOCH_CAP, pop_size=ps),
    "gwo": lambda ps: OriginalGWO(epoch=EPOCH_CAP, pop_size=ps),
    "lshade": lambda ps: L_SHADE(epoch=EPOCH_CAP, pop_size=ps),
    "imode": lambda ps: OriginalIMODE(epoch=EPOCH_CAP, pop_size=ps),
    "rime": lambda ps: OriginalRIME(epoch=EPOCH_CAP, pop_size=ps),
}

# Population-size rules recommended by each method's own source, used only in the
# sensitivity study. A cap keeps the largest instances runnable: the scheduling vectors
# reach several thousand components, where an uncapped linear rule would spend the whole
# budget initialising a single population.
POP_CAP = 400


def recommended_pop_size(method: str, dim: int) -> int:
    """Population size under each method's own recommended rule, capped."""
    if method == "lshade":
        value = 18 * dim  # source rule for the initial population, before linear reduction
    elif method == "cmaes":
        value = 4 + int(3 * math.log(max(2, dim)))
    elif method in ("de", "imode"):
        value = 10 * dim
    elif method == "brkga":
        value = 2 * dim
    else:
        value = 50
    return int(max(8, min(POP_CAP, value)))


class _SchedulingProblem(Problem):
    def __init__(self, instance: Instance, recorder: RunRecorder, mapping: str) -> None:
        self.instance = instance
        self.recorder = recorder
        self.mapping = mapping
        dim = vector_length(instance)
        super().__init__(
            bounds=FloatVar(lb=[0.0] * dim, ub=[1.0] * dim), minmax="min", log_to=None
        )

    def obj_func(self, x) -> float:
        result = decode_vector(self.instance, x, mapping=self.mapping)
        makespan = result.schedule.makespan
        self.recorder.record(makespan, result.n_repairs, x)
        return float(makespan)


def solve_metaheuristic(
    instance: Instance,
    method: str,
    time_limit: float,
    seed: int,
    pop_size: int = DEFAULT_POP_SIZE,
    mapping: str = DEFAULT_MAPPING,
) -> MethodResult:
    """Run a library optimiser through the shared decoder under a wall-clock budget."""
    if method not in OPTIMIZERS:
        raise ValueError(f"unknown method {method!r}; options: {sorted(OPTIMIZERS)}")

    recorder = RunRecorder()
    problem = _SchedulingProblem(instance, recorder, mapping)
    model = OPTIMIZERS[method](pop_size)
    termination = Termination(max_time=float(time_limit))

    crashed = False
    try:
        model.solve(problem, termination=termination, mode="single", seed=int(seed))
    except Exception:  # noqa: BLE001 - a crashing optimiser is logged, not fatal to the run
        crashed = True
    wall = time.perf_counter() - recorder.start

    schedule = None
    if recorder.best_x is not None:
        schedule = decode_vector(instance, recorder.best_x, mapping=mapping).schedule
    feasible = schedule is not None

    return MethodResult(
        method=method,
        instance=instance.name,
        status="FEASIBLE" if feasible else "UNKNOWN",
        best_obj=recorder.best,
        best_bound=None,
        schedule=schedule,
        feasible_final=feasible,
        anytime=recorder.points(),
        time_to_first=recorder.time_to_first,
        time_to_best=recorder.time_to_best,
        wall_time=wall,
        n_decoder_calls=recorder.n_calls,
        n_repairs=recorder.n_repairs,
        crashed=crashed,
    )
