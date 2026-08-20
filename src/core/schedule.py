"""Schedule representation and makespan.

A `Schedule` assigns each operation a machine and a start time. The duration is fixed by
the chosen machine. The schedule is the common output of every method (exact solver,
constructive heuristics, and decoded metaheuristics) so that one feasibility checker and
one objective apply uniformly.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from src.core.instance import Instance


@dataclass(frozen=True)
class Assignment:
    operation: int  # global operation index
    job: int
    machine: int
    start: int
    duration: int

    @property
    def end(self) -> int:
        return self.start + self.duration


@dataclass(frozen=True)
class Schedule:
    """A complete assignment of operations to (machine, start). Indexed by global index."""

    instance: Instance
    assignments: tuple[Assignment, ...]

    @property
    def makespan(self) -> int:
        return max((a.end for a in self.assignments), default=0)

    def by_machine(self) -> dict[int, list[Assignment]]:
        out: dict[int, list[Assignment]] = defaultdict(list)
        for a in self.assignments:
            out[a.machine].append(a)
        for lst in out.values():
            lst.sort(key=lambda a: a.start)
        return out
