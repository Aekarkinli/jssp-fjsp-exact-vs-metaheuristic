"""Tabu search with the critical-path block (N5) neighbourhood.

The strongest problem-specific baseline in the panel. A schedule is represented by the
operation order on each machine and, for the flexible case, the machine assignment. The
makespan is the longest path in the disjunctive graph; the critical path is decomposed into
machine blocks and the N5 neighbourhood of Nowicki and Smutnicki (1996) reverses the arcs at
block borders. For the flexible job shop the neighbourhood is augmented with reassignment of
a critical operation to another eligible machine.

Configuration, reported in full in the manuscript so the method can be rebuilt from the text:
a short-term tabu list of reversed arcs with tenure drawn uniformly from ``[T, 2T]`` at
``T = 8``; an aspiration criterion that admits a tabu move improving the incumbent; a
best-admissible move rule that, when every move is tabu and none improves the incumbent,
takes the least deteriorating move rather than restarting, so the trajectory keeps moving;
and a restart after ``max(200, 2n)`` non-improving iterations, alternating three
intensifying restarts from the incumbent perturbed by ``max(2, n/8)`` random adjacent swaps
with one diversifying restart from a fresh random-key solution. The initial solution is the
best of the dispatching-rule bank. One configuration is used for every instance, with no
per-instance tuning.

The search is stochastic through the randomised tenure, the perturbation and the diversifying
restart, so it is run with the same seed list as every other stochastic method.
"""
from __future__ import annotations

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
    n5_swaps,
    perturb,
    random_start,
    reassign_moves,
    state_from_schedule,
    swap_positions,
)

TENURE = 8


def solve_tabu(
    instance: Instance,
    time_limit: float,
    seed: int,
    tenure: int | None = None,
    restart_patience: int | None = None,
    mapping: str = DEFAULT_MAPPING,
) -> MethodResult:
    rng = random.Random(seed)
    graph = Graph(instance)
    n = instance.num_operations
    flexible = instance.is_flexible

    # initial solution: best of the dispatching rules
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
    base_tenure = tenure or TENURE  # small relative to the N5 neighbourhood; not scaled to n
    restart_patience = restart_patience or max(200, 2 * n)
    tabu: dict[tuple[int, int], int] = {}
    it = 0
    since_improve = 0
    restart_count = 0

    while time.perf_counter() - start < time_limit:
        it += 1
        blocks = critical_blocks(graph, cur_mk, heads, tails, mp, ms, dur)
        swaps = n5_swaps(blocks)
        reassigns = reassign_moves(instance, blocks, assignment) if flexible else []
        if not swaps and not reassigns:
            break

        # Track the best admissible move (non-tabu, or tabu but improving the incumbent)
        # and, separately, the best move overall. When every move is tabu we still move to
        # the least-bad one so the search keeps wandering instead of restarting.
        adm_move = adm_kind = None
        adm_mk = float("inf")
        any_move = any_kind = None
        any_mk = float("inf")

        def _consider(move, kind, mk2, is_tabu):
            nonlocal adm_move, adm_kind, adm_mk, any_move, any_kind, any_mk
            if mk2 < any_mk:
                any_mk, any_move, any_kind = mk2, move, kind
            if (not is_tabu) or mk2 < best_mk:
                if mk2 < adm_mk:
                    adm_mk, adm_move, adm_kind = mk2, move, kind

        for (a, b) in swaps:
            m = assignment[a]
            swap_positions(seqs[m], a, b)
            res = graph.evaluate(seqs, dur)
            swap_positions(seqs[m], a, b)
            if res is None:
                continue
            _consider((a, b), "swap", res[0], (a, b) in tabu or (b, a) in tabu)
        for (g, new_m) in reassigns:
            old_m = assignment[g]
            old_seq_m, old_seq_n, old_dur = seqs[old_m][:], seqs[new_m][:], dur[g]
            seqs[old_m].remove(g)
            seqs[new_m].append(g)
            dur[g] = instance.operation(g).duration_on(new_m)
            assignment[g] = new_m
            res = graph.evaluate(seqs, dur)
            seqs[old_m], seqs[new_m], dur[g], assignment[g] = old_seq_m, old_seq_n, old_dur, old_m
            if res is None:
                continue
            _consider((g, new_m), "reassign", res[0], (g, new_m) in tabu)

        best_move = adm_move if adm_move is not None else any_move
        best_move_kind = adm_kind if adm_move is not None else any_kind

        if best_move is None:
            since_improve = restart_patience  # no admissible move at all -> restart below
        elif best_move_kind == "swap":
            a, b = best_move
            swap_positions(seqs[assignment[a]], a, b)
            tabu[(a, b)] = it + base_tenure + rng.randint(0, base_tenure)
        else:
            g, new_m = best_move
            seqs[assignment[g]].remove(g)
            seqs[new_m].append(g)
            assignment[g] = new_m
            dur[g] = instance.operation(g).duration_on(new_m)
            tabu[(g, new_m)] = it + base_tenure + rng.randint(0, base_tenure)

        # purge expired tabu entries occasionally
        if it % 64 == 0:
            tabu = {k: v for k, v in tabu.items() if v > it}

        ev = graph.evaluate(seqs, dur)
        if ev is None:  # safety: should not happen for N5
            seqs = [s[:] for s in best_seqs]
            assignment = best_assignment[:]
            dur = best_dur[:]
            ev = graph.evaluate(seqs, dur)
        cur_mk, heads, tails, mp, ms, _ = ev

        if cur_mk < best_mk:
            best_mk = cur_mk
            best_seqs = [s[:] for s in seqs]
            best_assignment = assignment[:]
            best_dur = dur[:]
            best_heads = heads[:]
            anytime.append((time.perf_counter() - start, float(cur_mk)))
            since_improve = 0
        else:
            since_improve += 1

        if since_improve >= restart_patience:
            restart_count += 1
            if restart_count % 4 == 0:
                # diversify: a fresh random-key start explores a different basin
                assignment, seqs, dur = random_start(instance, rng, mapping=mapping)
            else:
                # intensify: restart from the incumbent with a medium kick
                seqs = [s[:] for s in best_seqs]
                assignment = best_assignment[:]
                dur = best_dur[:]
                perturb(seqs, rng, k=max(2, n // 8))
            ev = graph.evaluate(seqs, dur)
            if ev is None:
                seqs = [s[:] for s in best_seqs]
                assignment = best_assignment[:]
                dur = best_dur[:]
                ev = graph.evaluate(seqs, dur)
            cur_mk, heads, tails, mp, ms, _ = ev
            tabu.clear()
            since_improve = 0

    schedule = build_schedule(instance, best_assignment, best_dur, best_heads)
    wall = time.perf_counter() - start
    return MethodResult(
        method="tabu",
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
        n_restarts=restart_count,
        extra={"iterations": it},
    )
