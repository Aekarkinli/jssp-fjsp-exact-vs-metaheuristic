"""Fast tabu-search tests: disjunctive evaluator, feasibility, and baseline strength."""
from __future__ import annotations

from pathlib import Path

from src.core.feasibility import check_feasibility
from src.io.loaders import load_fjsp_file, load_jssp_file
from src.methods.constructive import DISPATCHING_RULES, build_active_schedule
from src.methods.disjunctive import Graph as _Graph
from src.methods.disjunctive import state_from_schedule as _state_from_schedule
from src.methods.tabu import solve_tabu

DATA = Path("data/instances")


def test_evaluator_matches_active_schedule_makespan():
    """Re-evaluating an active schedule's sequences reproduces its makespan."""
    inst = load_jssp_file(DATA / "jssp/la01.txt", "la01", "f")
    sched = min((build_active_schedule(inst, r) for r in DISPATCHING_RULES.values()),
                key=lambda s: s.makespan)
    asg, seqs = _state_from_schedule(sched)
    dur = [inst.operation(g).duration_on(asg[g]) for g in range(inst.num_operations)]
    mk, *_ = _Graph(inst).evaluate(seqs, dur)
    assert mk == sched.makespan


def test_tabu_feasible_and_beats_dispatching_jssp():
    inst = load_jssp_file(DATA / "jssp/ft10.txt", "ft10", "f")
    r = solve_tabu(inst, time_limit=4, seed=11)
    assert check_feasibility(r.schedule).feasible
    assert r.best_obj == r.schedule.makespan
    dispatching_best = min(build_active_schedule(inst, rule).makespan
                           for rule in DISPATCHING_RULES.values())
    assert r.best_obj < dispatching_best  # the strong baseline must beat the constructive one
    assert r.best_obj >= 930  # cannot beat the proven optimum


def test_tabu_deterministic_under_seed():
    inst = load_jssp_file(DATA / "jssp/ft10.txt", "ft10", "f")
    a = solve_tabu(inst, time_limit=3, seed=23)
    b = solve_tabu(inst, time_limit=3, seed=23)
    assert a.anytime[0].obj == b.anytime[0].obj  # identical initial construction + first move


def test_tabu_feasible_on_fjsp_with_reassignment():
    inst = load_fjsp_file(DATA / "fjsp/mk01.txt", "mk01", "brandimarte")
    assert inst.is_flexible
    r = solve_tabu(inst, time_limit=4, seed=11)
    assert check_feasibility(r.schedule).feasible
    assert r.best_obj >= 40  # mk01 proven optimum
