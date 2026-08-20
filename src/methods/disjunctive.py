"""Disjunctive-graph machinery shared by the problem-specific improvement methods.

A schedule is represented by the processing order on each machine together with, for the
flexible case, the machine assigned to each operation. The makespan is the longest path in
the disjunctive graph and is obtained from heads and tails in one topological sweep, which
also detects the cyclic orders that a move may create. The critical path is decomposed into
maximal blocks of consecutive operations on the same machine.

Both the tabu search and the annealing method are defined over exactly this structure and
this evaluator, so any difference between them comes from how they traverse the
neighbourhood rather than from how a candidate is scored.
"""
from __future__ import annotations

import random

import numpy as np

from src.core.decoder import DEFAULT_MAPPING, decode_vector, vector_length
from src.core.instance import Instance
from src.core.schedule import Assignment, Schedule


class Graph:
    """Fixed job structure; evaluates makespan/heads/tails for given machine sequences."""

    def __init__(self, instance: Instance) -> None:
        n = instance.num_operations
        self.n = n
        self.instance = instance
        self.job_pred = [-1] * n
        self.job_succ = [-1] * n
        for job in instance.jobs:
            for k, op in enumerate(job):
                g = op.global_index
                if k > 0:
                    self.job_pred[g] = job[k - 1].global_index
                if k < len(job) - 1:
                    self.job_succ[g] = job[k + 1].global_index

    def evaluate(self, machine_seq: list[list[int]], dur: list[int]):
        """Return (makespan, heads, tails, mach_pred, mach_succ, topo) or None if cyclic."""
        n = self.n
        jp, js = self.job_pred, self.job_succ
        mp = [-1] * n
        ms = [-1] * n
        for seq in machine_seq:
            prev = -1
            for g in seq:
                mp[g] = prev
                if prev >= 0:
                    ms[prev] = g
                prev = g

        indeg = [(1 if jp[g] >= 0 else 0) + (1 if mp[g] >= 0 else 0) for g in range(n)]
        heads = [0] * n
        stack = [g for g in range(n) if indeg[g] == 0]
        topo = []
        while stack:
            g = stack.pop()
            topo.append(g)
            e = heads[g] + dur[g]
            for s in (js[g], ms[g]):
                if s >= 0:
                    if e > heads[s]:
                        heads[s] = e
                    indeg[s] -= 1
                    if indeg[s] == 0:
                        stack.append(s)
        if len(topo) < n:
            return None  # cyclic sequence (infeasible)

        makespan = max(heads[g] + dur[g] for g in range(n))
        tails = [0] * n
        for g in reversed(topo):
            t = 0
            for s in (js[g], ms[g]):
                if s >= 0:
                    v = tails[s] + dur[s]
                    if v > t:
                        t = v
            tails[g] = t
        return makespan, heads, tails, mp, ms, topo


def critical_blocks(graph, makespan, heads, tails, mp, ms, dur):
    """A single critical path decomposed into maximal same-machine blocks."""
    n = graph.n
    js = graph.job_succ

    def crit(g):
        return heads[g] + dur[g] + tails[g] == makespan

    start = next((g for g in range(n) if crit(g) and heads[g] == 0), None)
    if start is None:
        return []
    path = [start]
    arcs = []  # 'm' or 'j' between consecutive path ops
    g = start
    while True:
        nxt = None
        for s, via in ((ms[g], "m"), (js[g], "j")):
            if s >= 0 and crit(s) and heads[s] == heads[g] + dur[g]:
                nxt, via_t = s, via
                break
        if nxt is None:
            break
        path.append(nxt)
        arcs.append(via_t)
        g = nxt

    blocks = []
    cur = [path[0]]
    for i, a in enumerate(arcs):
        if a == "m":
            cur.append(path[i + 1])
        else:
            blocks.append(cur)
            cur = [path[i + 1]]
    blocks.append(cur)
    return blocks


def n5_swaps(blocks):
    """N5 moves: reverse the first or last arc of each critical block."""
    moves = set()
    last = len(blocks) - 1
    for bi, block in enumerate(blocks):
        if len(block) < 2:
            continue
        first_block, last_block = bi == 0, bi == last
        if not first_block:
            moves.add((block[0], block[1]))  # swap first two
        if not last_block:
            moves.add((block[-2], block[-1]))  # swap last two
    return moves


def n1_arcs(blocks):
    """N1 moves: reverse any single arc inside a critical block."""
    moves = []
    for block in blocks:
        for i in range(len(block) - 1):
            moves.append((block[i], block[i + 1]))
    return moves


def reassign_moves(instance, blocks, assignment):
    """For each critical operation, propose moving it to each other eligible machine."""
    moves = []
    seen = set()
    for block in blocks:
        for g in block:
            if g in seen:
                continue
            seen.add(g)
            op = instance.operation(g)
            if not op.is_flexible:
                continue
            for m in op.eligible_machines:
                if m != assignment[g]:
                    moves.append((g, m))
    return moves


def state_from_schedule(schedule: Schedule):
    inst = schedule.instance
    assignment = [a.machine for a in schedule.assignments]
    order = sorted(range(inst.num_operations), key=lambda g: schedule.assignments[g].start)
    seqs: list[list[int]] = [[] for _ in range(inst.num_machines)]
    for g in order:
        seqs[assignment[g]].append(g)
    return assignment, seqs


def build_schedule(instance, assignment, dur, heads) -> Schedule:
    A = [
        Assignment(g, instance.operation(g).job, assignment[g], heads[g], dur[g])
        for g in range(instance.num_operations)
    ]
    return Schedule(instance, tuple(A))


def swap_positions(seq, a, b):
    ia, ib = seq.index(a), seq.index(b)
    seq[ia], seq[ib] = seq[ib], seq[ia]


def random_start(instance: Instance, rng: random.Random, mapping: str = DEFAULT_MAPPING):
    """A diversified feasible start: decode a random key vector into machine sequences."""
    vec = np.array([rng.random() for _ in range(vector_length(instance))], dtype=float)
    schedule = decode_vector(instance, vec, mapping=mapping).schedule
    assignment, seqs = state_from_schedule(schedule)
    dur = [
        instance.operation(g).duration_on(assignment[g])
        for g in range(instance.num_operations)
    ]
    return assignment, seqs, dur


def perturb(seqs, rng, k):
    """Apply k random adjacent swaps on random non-trivial machine sequences."""
    nonempty = [m for m, s in enumerate(seqs) if len(s) >= 2]
    if not nonempty:
        return
    for _ in range(k):
        m = rng.choice(nonempty)
        i = rng.randrange(len(seqs[m]) - 1)
        seqs[m][i], seqs[m][i + 1] = seqs[m][i + 1], seqs[m][i]
