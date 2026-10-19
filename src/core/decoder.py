"""Shared random-key decoder: continuous vector -> active schedule.

Every real-valued optimiser acts on a continuous vector that this decoder turns into a
feasible schedule. Using one decoder for all of them removes the representation as a source
of difference *among the decoded methods*; it does not make them identical in every other
respect, and the manuscript scopes the claim accordingly.

Encoding:
- JSSP: a vector of `num_operations` priority keys.
- FJSP: a vector of `2 * num_operations`; the first half are operation priority keys, the
  second half are machine-selection keys.

Decoding uses the Giffler-Thompson active-schedule procedure. At each step the operation
with the earliest possible completion time defines a machine, and among the operations
competing for that machine within that completion time the one with the highest priority
(smallest key) is scheduled at its earliest start. The result is always an active schedule
and therefore feasible.

Machine selection (FJSP). A machine-selection key `k in [0,1)` indexes the operation's own
sorted eligible set,

    j = min(|E(o)| - 1, floor(k * |E(o)|)),   machine = E(o)[j],

so the mapping is uniform over the eligible machines for a uniformly distributed key and
structurally incapable of producing an ineligible choice. For a fixed key the choice still
depends on the order of the eligible set, so it is not invariant to machine relabelling. No repair
step is needed and the repair count is zero by construction.

A second mapping is retained for a controlled ablation only. The `legacy` mapping sends the
key to `floor(k * num_machines)` over all machines and, when that machine is ineligible,
moves it to the nearest eligible machine by index (ties to the lower index), counting the
move as a repair. Arithmetic distance between machine numbers carries no operational
meaning, so this mapping is biased by the labelling and manufactures repair work. It is
never used for the main comparison; running it against the primary mapping measures what
the naive choice costs.

Two further representations are used in the targeted experiments as controlled
alternatives, and neither is used by the main comparison.

- ``opperm``: an operation-based permutation with repetition (Bierwirth, 1995). The vector
  has one key per operation for both problem classes. Sorting the keys gives a sequence of
  job indices, and the k-th occurrence of a job stands for its k-th operation. Operations are
  placed in that order at the earliest time their job and machine allow, including idle gaps
  left earlier on the machine. In the flexible case the decoder, not the vector, chooses the
  machine: the eligible machine that gives the earliest completion, ties to the lower index.
- ``eligible_ls``: the primary decoder followed by a deterministic steepest descent in the
  critical-block neighbourhood of the tabu search (with machine reassignment of critical
  operations in the flexible case) until no move improves the makespan. The objective is the
  makespan after the descent and the vector itself is left unchanged, which is the
  Baldwinian form of a memetic algorithm.
"""
from __future__ import annotations

import bisect
from dataclasses import dataclass

import numpy as np

from src.core.instance import Instance, Operation
from src.core.schedule import Assignment, Schedule

ELIGIBLE = "eligible"  # primary mapping: index into the sorted eligible set, repair-free
LEGACY = "legacy"      # ablation only: global machine index + nearest-eligible repair
OPPERM = "opperm"      # revision: operation-based permutation, decoder-chosen machine
MEMETIC = "eligible_ls"  # revision: primary decoding + critical-block steepest descent
DEFAULT_MAPPING = ELIGIBLE


@dataclass(frozen=True)
class DecodeResult:
    schedule: Schedule
    n_repairs: int
    chosen_machines: tuple[int, ...]


def vector_length(instance: Instance, mapping: str = DEFAULT_MAPPING) -> int:
    """Length of the continuous vector this instance expects under a representation."""
    n = instance.num_operations
    if mapping == OPPERM:
        return n
    return 2 * n if instance.is_flexible else n


def split_vector(instance: Instance, vector) -> tuple[np.ndarray, np.ndarray | None]:
    """Split a flat vector into (priority keys, machine keys-or-None)."""
    n = instance.num_operations
    vec = np.asarray(vector, dtype=float)
    if instance.is_flexible:
        if vec.size < 2 * n:
            raise ValueError(f"FJSP expects 2*{n} keys, got {vec.size}")
        return vec[:n], vec[n : 2 * n]
    if vec.size < n:
        raise ValueError(f"JSSP expects {n} keys, got {vec.size}")
    return vec[:n], None


def _select_machine(
    op: Operation, key: float, num_machines: int, mapping: str
) -> tuple[int, bool]:
    """Return (machine, repaired) for an operation given its machine-selection key."""
    eligible = op.eligible_machines
    if len(op.modes) == 1:
        return op.modes[0][0], False

    if mapping == ELIGIBLE:
        # Index the operation's own eligible set. Uniform over eligible machines and
        # independent of how the machines are numbered, so no repair can arise.
        k = eligible[min(len(eligible) - 1, max(0, int(key * len(eligible))))]
        return k, False

    if mapping == LEGACY:
        idx = min(num_machines - 1, max(0, int(key * num_machines)))
        if idx in eligible:
            return idx, False
        nearest = min(eligible, key=lambda m: (abs(m - idx), m))
        return nearest, True

    raise ValueError(f"unknown machine mapping {mapping!r}")


def decode(
    instance: Instance,
    priorities,
    machine_keys=None,
    mapping: str = DEFAULT_MAPPING,
) -> DecodeResult:
    """Decode priority (and machine) keys into an active schedule."""
    inst = instance
    n_op = inst.num_operations
    priorities = np.asarray(priorities, dtype=float)
    if priorities.size < n_op:
        raise ValueError(f"need {n_op} priority keys, got {priorities.size}")
    if inst.is_flexible and machine_keys is None:
        raise ValueError("flexible instance requires machine_keys")
    mkeys = None if machine_keys is None else np.asarray(machine_keys, dtype=float)

    # 1) fix each operation's machine and duration up front
    chosen_machine = [0] * n_op
    chosen_duration = [0] * n_op
    n_repairs = 0
    for op in inst.operations:
        key = 0.0 if mkeys is None else float(mkeys[op.global_index])
        m, repaired = _select_machine(op, key, inst.num_machines, mapping)
        chosen_machine[op.global_index] = m
        chosen_duration[op.global_index] = op.duration_on(m)
        n_repairs += int(repaired)

    # 2) Giffler-Thompson active-schedule generation
    jobs = inst.jobs
    machine_ready = [0] * inst.num_machines
    job_ready = [0] * inst.num_jobs
    job_next = [0] * inst.num_jobs  # next unscheduled position per job
    assignments: list[Assignment | None] = [None] * n_op
    remaining = n_op

    while remaining:
        # schedulable operations: the next unscheduled op of each unfinished job
        best_ect = None
        ready: list[tuple[Operation, int, int, int]] = []  # (op, machine, est, ect)
        for j, job in enumerate(jobs):
            pos = job_next[j]
            if pos >= len(job):
                continue
            op = job[pos]
            g = op.global_index
            m = chosen_machine[g]
            est = job_ready[j] if job_ready[j] > machine_ready[m] else machine_ready[m]
            ect = est + chosen_duration[g]
            ready.append((op, m, est, ect))
            if best_ect is None or ect < best_ect:
                best_ect = ect
                m_star = m

        # operations competing for m_star that can start before best completion time
        conflict = [r for r in ready if r[1] == m_star and r[2] < best_ect]
        # highest priority = smallest key; tie -> smallest global index
        op, m, est, _ect = min(
            conflict, key=lambda r: (priorities[r[0].global_index], r[0].global_index)
        )
        g = op.global_index
        assignments[g] = Assignment(
            operation=g, job=op.job, machine=m, start=est, duration=chosen_duration[g]
        )
        machine_ready[m] = est + chosen_duration[g]
        job_ready[op.job] = machine_ready[m]
        job_next[op.job] += 1
        remaining -= 1

    schedule = Schedule(inst, tuple(a for a in assignments))  # type: ignore[arg-type]
    return DecodeResult(schedule, n_repairs, tuple(chosen_machine))


def decode_operation_permutation(instance: Instance, keys) -> DecodeResult:
    """Operation-based permutation with repetition, decoded with gap insertion.

    Sorting the keys (ties by position) yields a job sequence in which the k-th occurrence
    of a job denotes its k-th operation. Each operation starts at the earliest time at which
    its job predecessor has finished and an idle interval of sufficient length exists on the
    machine, so later operations may be inserted before earlier ones on a machine. In the
    flexible case every eligible machine is tried and the one with the earliest completion
    is taken, ties to the lower machine index. The schedule is feasible by construction.
    """
    inst = instance
    n_op = inst.num_operations
    keys = np.asarray(keys, dtype=float)
    if keys.size < n_op:
        raise ValueError(f"need {n_op} keys, got {keys.size}")
    job_of = [op.job for op in inst.operations]
    order = np.argsort(keys[:n_op], kind="stable")

    starts: list[list[int]] = [[] for _ in range(inst.num_machines)]
    ends: list[list[int]] = [[] for _ in range(inst.num_machines)]
    job_ready = [0] * inst.num_jobs
    job_next = [0] * inst.num_jobs
    assignments: list[Assignment | None] = [None] * n_op
    chosen = [0] * n_op

    def earliest(m: int, ready: int, p: int) -> int:
        s, e = starts[m], ends[m]
        i = bisect.bisect_right(e, ready)  # first interval that ends after `ready`
        t = ready
        while i < len(s):
            if t + p <= s[i]:
                return t
            if e[i] > t:
                t = e[i]
            i += 1
        return t

    for g_pos in order:
        j = job_of[int(g_pos)]
        op = inst.jobs[j][job_next[j]]
        g = op.global_index
        ready = job_ready[j]
        best = None
        for m, p in op.modes:
            t = earliest(m, ready, p)
            cand = (t + p, m, t, p)
            if best is None or cand < best:
                best = cand
        end, m, t, p = best
        k = bisect.bisect_left(starts[m], t)
        starts[m].insert(k, t)
        ends[m].insert(k, end)
        assignments[g] = Assignment(operation=g, job=j, machine=m, start=t, duration=p)
        chosen[g] = m
        job_ready[j] = end
        job_next[j] += 1

    schedule = Schedule(inst, tuple(a for a in assignments))  # type: ignore[arg-type]
    return DecodeResult(schedule, 0, tuple(chosen))


def decode_vector(
    instance: Instance, vector, mapping: str = DEFAULT_MAPPING
) -> DecodeResult:
    """Convenience: decode a single flat continuous vector under a representation."""
    if mapping == OPPERM:
        return decode_operation_permutation(instance, vector)
    if mapping == MEMETIC:
        from src.methods.memetic import steepest_descent  # local import: no cycle

        priorities, machine_keys = split_vector(instance, vector)
        base = decode(instance, priorities, machine_keys, mapping=ELIGIBLE)
        improved = steepest_descent(instance, base.schedule)
        machines = tuple(a.machine for a in improved.assignments)
        return DecodeResult(improved, 0, machines)
    priorities, machine_keys = split_vector(instance, vector)
    return decode(instance, priorities, machine_keys, mapping=mapping)


def n_flexible_operations(instance: Instance) -> int:
    """Operations with a genuine machine choice.

    The normaliser for any per-operation feasibility statistic: an instance where every
    operation has a single eligible machine offers the decoder no machine decision at all,
    so counting its repairs on the same scale as a large flexible instance is meaningless.
    """
    return sum(1 for op in instance.operations if len(op.modes) > 1)


def eligible_set_sizes(instance: Instance) -> list[int]:
    """Number of eligible machines per operation, used to characterise flexibility."""
    return [len(op.eligible_machines) for op in instance.operations]
