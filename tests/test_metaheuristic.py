"""Fast metaheuristic-wrapper tests: every method runs end to end with a complete log."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.core.decoder import LEGACY
from src.core.feasibility import check_feasibility
from src.io.loaders import load_fjsp_file, load_jssp_file
from src.methods.metaheuristic import OPTIMIZERS, recommended_pop_size, solve_metaheuristic

DATA = Path("data/instances")


def test_methods_registered():
    assert set(OPTIMIZERS) == {"ga", "de", "pso", "abc", "gwo", "lshade", "imode", "rime"}


@pytest.mark.parametrize("method", list(OPTIMIZERS))
def test_method_runs_end_to_end_jssp(method):
    inst = load_jssp_file(DATA / "jssp/ft06.txt", "ft06", "f")
    r = solve_metaheuristic(inst, method, time_limit=2, seed=11)
    assert not r.crashed
    assert r.feasible_final and check_feasibility(r.schedule).feasible
    assert r.best_obj == r.schedule.makespan >= 55  # ft06 optimum
    assert r.n_decoder_calls > 0
    assert r.n_repairs == 0  # JSSP has no machine flexibility to repair
    assert len(r.anytime) >= 1
    assert r.time_to_first is not None and r.time_to_best is not None


def test_fjsp_needs_no_repair_under_the_primary_mapping():
    inst = load_fjsp_file(DATA / "fjsp/mk01.txt", "mk01", "b")
    r = solve_metaheuristic(inst, "de", time_limit=2, seed=11)
    assert r.feasible_final and check_feasibility(r.schedule).feasible
    assert r.n_repairs == 0  # the machine key indexes the eligible set, so nothing to repair


def test_fjsp_records_repairs_under_the_ablation_mapping():
    inst = load_fjsp_file(DATA / "fjsp/mk01.txt", "mk01", "b")
    r = solve_metaheuristic(inst, "de", time_limit=2, seed=11, mapping=LEGACY)
    assert r.feasible_final and check_feasibility(r.schedule).feasible
    assert r.n_repairs > 0  # the ablation mapping is what manufactures repair work


def test_anytime_trace_carries_an_evaluation_index():
    """Every improvement is stamped with the evaluation count, which is what makes the
    equal-effort comparison readable from the same run as the equal-time comparison."""
    inst = load_jssp_file(DATA / "jssp/ft06.txt", "ft06", "f")
    r = solve_metaheuristic(inst, "de", time_limit=2, seed=11)
    evals = [p.evals for p in r.anytime]
    assert all(e is not None for e in evals)
    assert evals == sorted(evals)
    assert evals[-1] <= r.n_decoder_calls


def test_seed_controls_initial_trajectory():
    """Wall-clock termination makes the final value timing-dependent, but a fixed seed
    fixes the search trajectory; the initial population's best is reproducible."""
    inst = load_jssp_file(DATA / "jssp/ft06.txt", "ft06", "f")
    a = solve_metaheuristic(inst, "gwo", time_limit=2, seed=41)
    b = solve_metaheuristic(inst, "gwo", time_limit=2, seed=41)
    assert a.anytime[0].obj == b.anytime[0].obj
