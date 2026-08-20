"""Common method interface and result record.

Every method (exact, constructive, metaheuristic) is called as
``solve(instance, time_limit, seed, ...) -> MethodResult`` and returns the same record.
The runner adds environment fields (peak memory, library versions, git commit, hardware,
affinity) around this; the method itself reports objective, bound, schedule, the anytime
trace, and the feasibility-burden counters. Field names live here only, never in the paper.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from math import isfinite

import numpy as np

from src.core.schedule import Schedule

INF = float("inf")


class RunRecorder:
    """Tracks best-so-far, the anytime trace, and feasibility-burden counters.

    Shared by every method that drives the shared decoder (the mealpy wrappers, BRKGA,
    CMA-ES and the two ported optimisers), so their logs are produced identically.

    Each improvement is stamped with both the elapsed wall-clock time and the number of
    decoder evaluations consumed so far. The second index lets the same run be read under an
    equal-evaluation budget as well as an equal-time budget, which separates search
    effectiveness from implementation throughput without running a second experiment.
    """

    def __init__(self) -> None:
        self.start = time.perf_counter()
        self.best = INF
        self.best_x: np.ndarray | None = None
        self.anytime: list[tuple[float, int, float]] = []
        self.n_calls = 0
        self.n_repairs = 0

    def record(self, makespan: float, repairs: int, x) -> None:
        self.n_calls += 1
        self.n_repairs += repairs
        if makespan < self.best:
            self.best = makespan
            self.best_x = np.array(x, dtype=float, copy=True)
            self.anytime.append(
                (time.perf_counter() - self.start, self.n_calls, float(makespan))
            )

    def points(self) -> list["AnytimePoint"]:
        return [AnytimePoint(t=t, obj=o, evals=e) for (t, e, o) in self.anytime]

    @property
    def time_to_first(self) -> float | None:
        return self.anytime[0][0] if self.anytime else None

    @property
    def time_to_best(self) -> float | None:
        return self.anytime[-1][0] if self.anytime else None


@dataclass(frozen=True)
class AnytimePoint:
    """Best-so-far objective at a wall-clock time and an evaluation count.

    ``bound`` carries the dual bound for exact methods; ``evals`` carries the decoder
    evaluation count for methods that search through the shared decoder. Each is None where
    the notion does not apply.
    """

    t: float  # seconds since solve start
    obj: float
    bound: float | None = None
    evals: int | None = None


@dataclass
class MethodResult:
    method: str
    instance: str
    status: str  # OPTIMAL | FEASIBLE | INFEASIBLE | TIME_LIMIT | UNKNOWN
    best_obj: float  # makespan; INF if no feasible solution
    best_bound: float | None  # dual bound (exact methods); None otherwise
    schedule: Schedule | None
    feasible_final: bool
    anytime: list[AnytimePoint] = field(default_factory=list)
    time_to_first: float | None = None  # time to first feasible solution
    time_to_best: float | None = None
    wall_time: float = 0.0
    # feasibility-burden and cost counters (metaheuristics fill these)
    n_decoder_calls: int = 0
    n_repairs: int = 0
    n_rejected: int = 0
    n_restarts: int = 0
    crashed: bool = False
    extra: dict = field(default_factory=dict)

    @property
    def rel_gap(self) -> float | None:
        """Relative optimality gap (obj - bound) / obj, when both are available."""
        if self.best_bound is None or not isfinite(self.best_obj) or self.best_obj <= 0:
            return None
        return (self.best_obj - self.best_bound) / self.best_obj

    @property
    def proven_optimal(self) -> bool:
        return self.status == "OPTIMAL"
