"""Simulated annealing on the disjunctive graph.

The second problem-specific improvement method in the panel, included so that the
problem-specific side is not represented by a single hand-written algorithm. It follows the
job-shop annealing of van Laarhoven, Aarts and Lenstra (1992): a candidate is produced by
reversing one arc of the critical path, which always yields an acyclic and therefore feasible
order, and is accepted by the Metropolis rule.

Configuration, reported in full in the manuscript. The initial temperature is calibrated on
the instance itself from a short random walk, as the value at which a deteriorating move of
average size is accepted with probability 0.9. Chains have length ``max(100, n)`` at ``n``
operations. Cooling uses the source's own schedule,

    c_{k+1} = c_k / (1 + c_k ln(1 + delta) / (3 sigma_k)),

with ``delta = 0.1`` and ``sigma_k`` the standard deviation of the objective values observed
in chain ``k``. When the chain freezes, meaning three consecutive chains accept nothing, the
search restarts from the incumbent at half the initial temperature; the published method
would stop there, and the restart is what lets it use the whole wall-clock budget that every
other method receives. For the flexible job shop the move set is extended with reassignment
of a critical operation to another eligible machine, chosen with equal probability against an
arc reversal, matching the neighbourhood the tabu search uses.
"""
from __future__ import annotations

import math
import random
import time

from src.core.decoder import DEFAULT_MAPPING
from src.core.instance import Instance
from src.methods.base import AnytimePoint, MethodResult
from src.methods.constructive import DISPATCHING_RULES, build_active_schedule
from src.methods.disjunctive import (
    Graph,
    build_schedule,
    critical_blocks,
    n1_arcs,
    reassign_moves,
    state_from_schedule,
    swap_positions,
)

DELTA = 0.1
CHI0 = 0.9
FREEZE_CHAINS = 3


def _apply_swap(seqs, assignment, a, b):
    swap_positions(seqs[assignment[a]], a, b)


def _apply_reassign(instance, seqs, assignment, dur, g, new_m):
    seqs[assignment[g]].remove(g)
    seqs[new_m].append(g)
    assignment[g] = new_m
    dur[g] = instance.operation(g).duration_on(new_m)


def solve_anneal(
    instance: Instance,
    time_limit: float,
    seed: int,
    mapping: str = DEFAULT_MAPPING,
) -> MethodResult:
    rng = random.Random(seed)
    graph = Graph(instance)
    n = instance.num_operations
    flexible = instance.is_flexible
    chain_length = max(100, n)

    init = min(
        (build_active_schedule(instance, rule) for rule in DISPATCHING_RULES.values()),
        key=lambda s: s.makespan,
    )
    assignment, seqs = state_from_schedule(init)
    dur = [instance.operation(g).duration_on(assignment[g]) for g in range(n)]
    ev = graph.evaluate(seqs, dur)
    cur_mk, heads, tails, mp, ms, _ = ev

    best_mk = cur_mk
    best_seqs = [s[:] for s in seqs]
    best_assignment = assignment[:]
    best_dur = dur[:]
    best_heads = heads[:]

    start = time.perf_counter()
    anytime = [(0.0, float(cur_mk))]

    def propose():
        """Draw one neighbour; return (kind, move, makespan, state) or None."""
        blocks = critical_blocks(graph, cur_mk, heads, tails, mp, ms, dur)
        arcs = n1_arcs(blocks)
        reassigns = reassign_moves(instance, blocks, assignment) if flexible else []
        use_reassign = bool(reassigns) and (not arcs or rng.random() < 0.5)
        if use_reassign:
            g, new_m = reassigns[rng.randrange(len(reassigns))]
            old_m, old_dur = assignment[g], dur[g]
            old_seq_o, old_seq_n = seqs[old_m][:], seqs[new_m][:]
            _apply_reassign(instance, seqs, assignment, dur, g, new_m)
            res = graph.evaluate(seqs, dur)
            seqs[old_m], seqs[new_m] = old_seq_o, old_seq_n
            assignment[g], dur[g] = old_m, old_dur
            return ("reassign", (g, new_m), res)
        if not arcs:
            return None
        a, b = arcs[rng.randrange(len(arcs))]
        m = assignment[a]
        swap_positions(seqs[m], a, b)
        res = graph.evaluate(seqs, dur)
        swap_positions(seqs[m], a, b)
        return ("swap", (a, b), res)

    def commit(kind, move):
        if kind == "swap":
            _apply_swap(seqs, assignment, *move)
        else:
            _apply_reassign(instance, seqs, assignment, dur, *move)

    # initial temperature: accept an average deterioration with probability CHI0
    rises = []
    for _ in range(min(200, 20 * max(1, n // 10))):
        p = propose()
        if p is None or p[2] is None:
            continue
        d = p[2][0] - cur_mk
        if d > 0:
            rises.append(d)
    mean_rise = (sum(rises) / len(rises)) if rises else max(1.0, 0.01 * cur_mk)
    temperature = t0 = max(1e-6, -mean_rise / math.log(CHI0))

    frozen_chains = 0
    n_restarts = 0
    while time.perf_counter() - start < time_limit:
        values = []
        accepted = 0
        for _ in range(chain_length):
            if time.perf_counter() - start >= time_limit:
                break
            p = propose()
            if p is None or p[2] is None:
                continue
            kind, move, res = p
            new_mk = res[0]
            delta = new_mk - cur_mk
            if delta <= 0 or rng.random() < math.exp(-delta / temperature):
                commit(kind, move)
                ev = graph.evaluate(seqs, dur)
                cur_mk, heads, tails, mp, ms, _ = ev
                accepted += 1
                if cur_mk < best_mk:
                    best_mk = cur_mk
                    best_seqs = [s[:] for s in seqs]
                    best_assignment = assignment[:]
                    best_dur = dur[:]
                    best_heads = heads[:]
                    anytime.append((time.perf_counter() - start, float(cur_mk)))
            values.append(cur_mk)

        if len(values) > 1:
            mean = sum(values) / len(values)
            sigma = math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1))
        else:
            sigma = 0.0
        if sigma > 0:
            temperature = temperature / (1.0 + temperature * math.log(1.0 + DELTA) / (3.0 * sigma))
        else:
            temperature *= 0.95

        frozen_chains = frozen_chains + 1 if accepted == 0 else 0
        if frozen_chains >= FREEZE_CHAINS:
            n_restarts += 1
            seqs = [s[:] for s in best_seqs]
            assignment = best_assignment[:]
            dur = best_dur[:]
            ev = graph.evaluate(seqs, dur)
            cur_mk, heads, tails, mp, ms, _ = ev
            temperature = t0 / 2.0
            frozen_chains = 0

    schedule = build_schedule(instance, best_assignment, best_dur, best_heads)
    wall = time.perf_counter() - start
    return MethodResult(
        method="sa",
        instance=instance.name,
        status="FEASIBLE",
        best_obj=float(best_mk),
        best_bound=None,
        schedule=schedule,
        feasible_final=True,
        anytime=[AnytimePoint(t=t, obj=o) for (t, o) in anytime],
        time_to_first=0.0,
        time_to_best=anytime[-1][0],
        wall_time=wall,
        n_restarts=n_restarts,
        extra={"t0": t0},
    )
