"""Feasibility checker for a complete schedule.

A schedule is feasible when every operation is assigned an eligible machine for its correct
duration, operations within a job respect their precedence chain, and no machine processes
two operations at the same time. The checker runs on every final schedule produced in the
study and is the safety net behind the decoder, the repair step, and every method.
"""
from __future__ import annotations

from dataclasses import dataclass

from src.core.schedule import Schedule


@dataclass(frozen=True)
class FeasibilityResult:
    feasible: bool
    violations: tuple[str, ...]

    def __bool__(self) -> bool:
        return self.feasible


def check_feasibility(schedule: Schedule) -> FeasibilityResult:
    inst = schedule.instance
    A = schedule.assignments
    v: list[str] = []

    if len(A) != inst.num_operations:
        v.append(f"expected {inst.num_operations} assignments, got {len(A)}")
        return FeasibilityResult(False, tuple(v))

    # per-operation: indexing, eligibility, duration, non-negative start
    for g, a in enumerate(A):
        op = inst.operation(g)
        if a.operation != g:
            v.append(f"assignment {g} refers to operation {a.operation}")
            continue
        if a.machine not in op.eligible_machines:
            v.append(f"op {g}: machine {a.machine} not eligible {op.eligible_machines}")
            continue
        if a.start < 0:
            v.append(f"op {g}: negative start {a.start}")
        expected = op.duration_on(a.machine)
        if a.duration != expected:
            v.append(f"op {g}: duration {a.duration} != {expected} on machine {a.machine}")

    # precedence within each job
    for job in inst.jobs:
        for k in range(1, len(job)):
            prev = A[job[k - 1].global_index]
            cur = A[job[k].global_index]
            if cur.start < prev.end:
                v.append(
                    f"precedence: job {job[k].job} op@pos{k} starts {cur.start} "
                    f"before previous ends {prev.end}"
                )

    # machine capacity: no overlap on any machine
    for machine, ops in schedule.by_machine().items():
        for i in range(1, len(ops)):
            if ops[i].start < ops[i - 1].end:
                v.append(
                    f"overlap on machine {machine}: op {ops[i - 1].operation} "
                    f"[{ops[i - 1].start},{ops[i - 1].end}) and op {ops[i].operation} "
                    f"[{ops[i].start},{ops[i].end})"
                )

    return FeasibilityResult(len(v) == 0, tuple(v))
