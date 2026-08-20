"""Unified instance model for JSSP and FJSP.

A job-shop instance is a special case of a flexible job-shop instance in which every
operation has exactly one eligible machine. We use one model for both so the decoder,
the feasibility checker, the exact solver, and the problem-specific methods all consume
the same structure.

Conventions:
- Machines are 0-indexed in `[0, num_machines)`.
- Each job is an ordered chain of operations; operation `k` must complete before
  operation `k+1` of the same job starts (linear precedence).
- Each operation carries one or more *modes*, a `(machine, duration)` pair giving an
  eligible machine and its processing time. Durations are positive integers.
- `global_index` numbers operations 0..num_operations-1 in job-major order.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Operation:
    """A single operation of a job and its eligible (machine, duration) modes."""

    job: int
    position: int  # 0-based index within the job
    global_index: int  # 0-based index across all operations (job-major)
    modes: tuple[tuple[int, int], ...]  # ((machine, duration), ...), machine 0-indexed

    @property
    def eligible_machines(self) -> tuple[int, ...]:
        return tuple(m for m, _ in self.modes)

    @property
    def is_flexible(self) -> bool:
        return len(self.modes) > 1

    def duration_on(self, machine: int) -> int:
        for m, d in self.modes:
            if m == machine:
                return d
        raise ValueError(
            f"machine {machine} is not eligible for operation {self.global_index} "
            f"(job {self.job}, position {self.position})"
        )

    @property
    def min_duration(self) -> int:
        return min(d for _, d in self.modes)


@dataclass(frozen=True)
class Instance:
    """A JSSP or FJSP instance."""

    name: str
    family: str
    problem_type: str  # "JSSP" or "FJSP"
    num_machines: int
    jobs: tuple[tuple[Operation, ...], ...]
    source: str = ""

    @property
    def num_jobs(self) -> int:
        return len(self.jobs)

    @property
    def operations(self) -> tuple[Operation, ...]:
        return tuple(op for job in self.jobs for op in job)

    @property
    def num_operations(self) -> int:
        return sum(len(job) for job in self.jobs)

    @property
    def is_flexible(self) -> bool:
        return any(op.is_flexible for op in self.operations)

    @property
    def num_flexible_operations(self) -> int:
        return sum(1 for op in self.operations if op.is_flexible)

    def operation(self, global_index: int) -> Operation:
        return self.operations[global_index]

    def job_predecessor(self, op: Operation) -> Operation | None:
        if op.position == 0:
            return None
        return self.jobs[op.job][op.position - 1]


def build_instance(
    name: str,
    family: str,
    problem_type: str,
    num_machines: int,
    raw_jobs: list[list[list[tuple[int, int]]]],
    source: str = "",
) -> Instance:
    """Assemble and validate an Instance from raw (machine, duration) mode lists.

    `raw_jobs[j][k]` is the list of `(machine, duration)` modes for the k-th operation of
    job j. Raises ValueError on any structural defect so the manifest can mark the parse
    as failed rather than feeding a malformed instance into the experiments.
    """
    if problem_type not in ("JSSP", "FJSP"):
        raise ValueError(f"unknown problem_type {problem_type!r}")
    if num_machines <= 0:
        raise ValueError(f"{name}: num_machines must be positive, got {num_machines}")

    jobs: list[tuple[Operation, ...]] = []
    g = 0
    for j, raw_job in enumerate(raw_jobs):
        if not raw_job:
            raise ValueError(f"{name}: job {j} has no operations")
        ops: list[Operation] = []
        for k, modes in enumerate(raw_job):
            if not modes:
                raise ValueError(f"{name}: job {j} operation {k} has no eligible machines")
            seen: set[int] = set()
            for m, d in modes:
                if not (0 <= m < num_machines):
                    raise ValueError(
                        f"{name}: job {j} op {k} machine {m} out of range [0,{num_machines})"
                    )
                if d <= 0:
                    raise ValueError(f"{name}: job {j} op {k} non-positive duration {d}")
                if m in seen:
                    raise ValueError(f"{name}: job {j} op {k} duplicate machine {m}")
                seen.add(m)
            if problem_type == "JSSP" and len(modes) != 1:
                raise ValueError(
                    f"{name}: JSSP operation (job {j}, op {k}) must have exactly one mode"
                )
            ops.append(
                Operation(job=j, position=k, global_index=g, modes=tuple(modes))
            )
            g += 1
        jobs.append(tuple(ops))

    return Instance(
        name=name,
        family=family,
        problem_type=problem_type,
        num_machines=num_machines,
        jobs=tuple(jobs),
        source=source,
    )
