"""Tests for the methods added in v2: BRKGA, CMA-ES and simulated annealing."""
from __future__ import annotations

from pathlib import Path

import pytest

from src.core.feasibility import check_feasibility
from src.io.loaders import load_fjsp_file, load_jssp_file
from src.methods.anneal import solve_anneal
from src.methods.brkga import solve_brkga
from src.methods.cmaes import solve_cmaes
from src.methods.constructive import DISPATCHING_RULES, build_active_schedule
from src.methods.metaheuristic import recommended_pop_size

DATA = Path("data/instances")


def _jssp():
    return load_jssp_file(DATA / "jssp/ft06.txt", "ft06", "f")


def _fjsp():
    return load_fjsp_file(DATA / "fjsp/mk01.txt", "mk01", "brandimarte")


@pytest.mark.parametrize("solver,name", [(solve_brkga, "brkga"), (solve_cmaes, "cmaes")])
def test_decoded_method_runs_end_to_end(solver, name):
    for inst, floor in ((_jssp(), 55), (_fjsp(), 40)):
        r = solver(inst, time_limit=3, seed=11)
        assert not r.crashed
        assert r.feasible_final and check_feasibility(r.schedule).feasible
        assert r.best_obj == r.schedule.makespan >= floor
        assert r.n_decoder_calls > 0
        assert r.n_repairs == 0
        assert r.method == name
        assert [p.evals for p in r.anytime] == sorted(p.evals for p in r.anytime)


def test_brkga_keeps_the_elite_and_improves():
    inst = load_jssp_file(DATA / "jssp/la01.txt", "la01", "l")
    r = solve_brkga(inst, time_limit=4, seed=11)
    assert r.anytime[0].obj >= r.best_obj  # best-so-far never worsens
    assert r.best_obj >= 666  # la01 proven optimum


def test_annealing_feasible_and_beats_its_own_start():
    inst = load_jssp_file(DATA / "jssp/ft10.txt", "ft10", "f")
    r = solve_anneal(inst, time_limit=5, seed=11)
    assert check_feasibility(r.schedule).feasible
    assert r.best_obj == r.schedule.makespan >= 930  # ft10 proven optimum
    start = min(build_active_schedule(inst, rule).makespan
                for rule in DISPATCHING_RULES.values())
    assert r.best_obj < start  # an improvement method must improve on its construction


def test_annealing_handles_machine_reassignment():
    r = solve_anneal(_fjsp(), time_limit=4, seed=23)
    assert check_feasibility(r.schedule).feasible
    assert r.best_obj >= 40


def test_recommended_population_rules_are_capped_and_positive():
    for method in ("lshade", "cmaes", "de", "imode", "brkga", "ga", "pso"):
        for dim in (36, 500, 2000):
            value = recommended_pop_size(method, dim)
            assert 8 <= value <= 400
    # the rules genuinely differ from the common setting where their source says so
    assert recommended_pop_size("cmaes", 2000) < 50
    assert recommended_pop_size("lshade", 2000) == 400
