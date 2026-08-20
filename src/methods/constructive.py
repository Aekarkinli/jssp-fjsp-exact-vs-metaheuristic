"""Problem-specific constructive baselines: dispatching-rule bank and a greedy constructor.

Both operate directly on the combinatorial structure (not the continuous vector) using
Giffler-Thompson active-schedule generation. At each step every job's next operation is a
candidate, placed on the eligible machine giving its earliest completion (the
earliest-eligible-machine rule for the flexible case). The operation with the earliest
completion time defines a machine, and among the operations competing for that machine
within that completion time one is chosen by a priority rule. The result is an active
schedule and therefore feasible.

Rules: shortest processing time (SPT), most work remaining (MWKR), most operations
remaining (MOPNR). The dispatching bank reports the best schedule across the three rules
and logs each rule's makespan. The greedy constructor uses earliest completion time.
"""
from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

from src.core.instance import Instance, Operation
from src.core.schedule import Assignment, Schedule
from src.methods.base import AnytimePoint, MethodResult


@dataclass(frozen=True)
class Candidate:
    op: Operation
    machine: int
    duration: int
    est: int  # earliest start
    ect: int  # earliest completion
    job_remaining_work: int  # sum of min durations of unscheduled ops in this job
    job_remaining_ops: int


def _best_machine(op: Operation, job_ready: int, machine_ready: list[int]) -> tuple[int, int, int, int]:
    """Eligible (machine, duration, est, ect) with the earliest completion (ties to lower index)."""
    best: tuple[int, int, int, int] | None = None
    for machine, duration in op.modes:
        est = job_ready if job_ready > machine_ready[machine] else machine_ready[machine]
        ect = est + duration
        if best is None or ect < best[3] or (ect == best[3] and machine < best[0]):
            best = (machine, duration, est, ect)
    return best  # type: ignore[return-value]


def build_active_schedule(instance: Instance, priority: Callable[[Candidate], float]) -> Schedule:
    """Giffler-Thompson active schedule under a priority rule (smaller key is selected)."""
    jobs = instance.jobs
    machine_ready = [0] * instance.num_machines
    job_ready = [0] * instance.num_jobs
    job_next = [0] * instance.num_jobs

    # suffix sums of minimum durations, for the most-work-remaining rule
    suffix_work: list[list[int]] = []
    for job in jobs:
        s = [0] * (len(job) + 1)
        for k in range(len(job) - 1, -1, -1):
            s[k] = s[k + 1] + job[k].min_duration
        suffix_work.append(s)

    assignments: list[Assignment | None] = [None] * instance.num_operations
    remaining = instance.num_operations

    while remaining:
        ready: list[Candidate] = []
        best_ect = None
        m_star = 0
        for j, job in enumerate(jobs):
            pos = job_next[j]
            if pos >= len(job):
                continue
            op = job[pos]
            machine, duration, est, ect = _best_machine(op, job_ready[j], machine_ready)
            ready.append(
                Candidate(op, machine, duration, est, ect, suffix_work[j][pos], len(job) - pos)
            )
            if best_ect is None or ect < best_ect:
                best_ect = ect
                m_star = machine

        conflict = [c for c in ready if c.machine == m_star and c.est < best_ect]
        sel = min(conflict, key=lambda c: (priority(c), c.op.global_index))
        g = sel.op.global_index
        assignments[g] = Assignment(g, sel.op.job, sel.machine, sel.est, sel.duration)
        machine_ready[sel.machine] = sel.est + sel.duration
        job_ready[sel.op.job] = machine_ready[sel.machine]
        job_next[sel.op.job] += 1
        remaining -= 1

    return Schedule(instance, tuple(a for a in assignments))  # type: ignore[arg-type]


DISPATCHING_RULES: dict[str, Callable[[Candidate], float]] = {
    "SPT": lambda c: c.duration,
    "MWKR": lambda c: -c.job_remaining_work,
    "MOPNR": lambda c: -c.job_remaining_ops,
}


def solve_dispatching(instance: Instance, time_limit: float | None = None, seed: int = 0) -> MethodResult:
    """Run the dispatching-rule bank and return the best schedule across rules."""
    start = time.perf_counter()
    per_rule: dict[str, int] = {}
    best: tuple[Schedule, int, str] | None = None
    for name, rule in DISPATCHING_RULES.items():
        sched = build_active_schedule(instance, rule)
        mk = sched.makespan
        per_rule[name] = mk
        if best is None or mk < best[1]:
            best = (sched, mk, name)
    wall = time.perf_counter() - start
    sched, mk, rule_name = best  # type: ignore[misc]
    return MethodResult(
        method="dispatching",
        instance=instance.name,
        status="FEASIBLE",
        best_obj=float(mk),
        best_bound=None,
        schedule=sched,
        feasible_final=True,
        anytime=[AnytimePoint(wall, float(mk))],
        time_to_first=wall,
        time_to_best=wall,
        wall_time=wall,
        extra={"per_rule": per_rule, "best_rule": rule_name},
    )


def solve_greedy(instance: Instance, time_limit: float | None = None, seed: int = 0) -> MethodResult:
    """Greedy Giffler-Thompson constructor using earliest completion time."""
    start = time.perf_counter()
    sched = build_active_schedule(instance, lambda c: c.ect)
    wall = time.perf_counter() - start
    mk = sched.makespan
    return MethodResult(
        method="greedy",
        instance=instance.name,
        status="FEASIBLE",
        best_obj=float(mk),
        best_bound=None,
        schedule=sched,
        feasible_final=True,
        anytime=[AnytimePoint(wall, float(mk))],
        time_to_first=wall,
        time_to_best=wall,
        wall_time=wall,
    )
