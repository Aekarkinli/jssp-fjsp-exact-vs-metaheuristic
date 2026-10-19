"""Critical-block steepest descent used by the memetic variants of the decoded optimisers.

Used in the targeted experiments to measure how much of the distance between the decoded
optimisers and the tabu search is due to the problem-specific move set. The descent uses
exactly the neighbourhood and the evaluator of the tabu search: the critical path of the
current machine sequences is decomposed into blocks, the arcs at block borders are reversed
(and, in the flexible case, a critical operation may move to another eligible machine), and
the best improving move is applied until no move improves the makespan. There is no tabu
list, no aspiration and no restart, so the procedure is a pure local search and is
deterministic for a given starting schedule.

The decoded optimisers call it through the representation ``eligible_ls`` of the shared
decoder, so their own operators, populations and termination are unchanged and only the
evaluation of a candidate differs.
"""
from __future__ import annotations

from src.core.instance import Instance
from src.core.schedule import Schedule
from src.methods.disjunctive import (
    Graph,
    build_schedule,
    critical_blocks,
    n5_swaps,
    reassign_moves,
    state_from_schedule,
    swap_positions,
)

_GRAPHS: dict[int, Graph] = {}


def _graph(instance: Instance) -> Graph:
    key = id(instance)
    g = _GRAPHS.get(key)
    if g is None or g.instance is not instance:
        if len(_GRAPHS) > 8:
            _GRAPHS.clear()
        g = _GRAPHS[key] = Graph(instance)
    return g


def steepest_descent(instance: Instance, schedule: Schedule) -> Schedule:
    """Improve a feasible schedule to a local optimum of the critical-block neighbourhood."""
    graph = _graph(instance)
    n = instance.num_operations
    assignment, seqs = state_from_schedule(schedule)
    dur = [instance.operation(g).duration_on(assignment[g]) for g in range(n)]
    ev = graph.evaluate(seqs, dur)
    cur_mk, heads, tails, mp, ms, _ = ev

    while True:
        blocks = critical_blocks(graph, cur_mk, heads, tails, mp, ms, dur)
        best_mk, best_move, best_kind = cur_mk, None, None
        for (a, b) in sorted(n5_swaps(blocks)):
            m = assignment[a]
            swap_positions(seqs[m], a, b)
            res = graph.evaluate(seqs, dur)
            swap_positions(seqs[m], a, b)
            if res is not None and res[0] < best_mk:
                best_mk, best_move, best_kind = res[0], (a, b), "swap"
        if instance.is_flexible:
            for (g, new_m) in reassign_moves(instance, blocks, assignment):
                old_m = assignment[g]
                old_a, old_b, old_d = seqs[old_m][:], seqs[new_m][:], dur[g]
                seqs[old_m].remove(g)
                seqs[new_m].append(g)
                dur[g] = instance.operation(g).duration_on(new_m)
                assignment[g] = new_m
                res = graph.evaluate(seqs, dur)
                seqs[old_m], seqs[new_m], dur[g], assignment[g] = old_a, old_b, old_d, old_m
                if res is not None and res[0] < best_mk:
                    best_mk, best_move, best_kind = res[0], (g, new_m), "reassign"
        if best_move is None:
            break
        if best_kind == "swap":
            a, b = best_move
            swap_positions(seqs[assignment[a]], a, b)
        else:
            g, new_m = best_move
            seqs[assignment[g]].remove(g)
            seqs[new_m].append(g)
            assignment[g] = new_m
            dur[g] = instance.operation(g).duration_on(new_m)
        cur_mk, heads, tails, mp, ms, _ = graph.evaluate(seqs, dur)

    return build_schedule(instance, assignment, dur, heads)
