"""Tests for the MDE and CSA ports: convergence and decoder integration."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from src.core.feasibility import check_feasibility
from src.io.loaders import load_fjsp_file, load_jssp_file
from src.methods.ported import csa_optimize, mde_optimize, solve_ported

DATA = Path("data/instances")


def _sphere(pop):
    return np.sum(np.asarray(pop) ** 2, axis=1)


def test_mde_minimises_sphere():
    best, x = mde_optimize(_sphere, dim=5, lb=-5.0, ub=5.0, pop_size=20, seed=11, max_cycles=600)
    assert best < 1e-3  # converges close to the global minimum 0
    assert len(x) == 5


def test_csa_minimises_sphere():
    best, x = csa_optimize(_sphere, dim=5, lb=-5.0, ub=5.0, pop_size=20, seed=11, max_cycles=600)
    assert best < 1e-3
    assert len(x) == 5


def test_ports_run_through_decoder_feasibly():
    jssp = load_jssp_file(DATA / "jssp/ft06.txt", "ft06", "f")
    fjsp = load_fjsp_file(DATA / "fjsp/mk01.txt", "mk01", "b")
    for method in ("mde", "csa"):
        rj = solve_ported(jssp, method, time_limit=2, seed=11)
        assert not rj.crashed and check_feasibility(rj.schedule).feasible
        assert rj.best_obj == rj.schedule.makespan >= 55  # ft06 optimum
        assert rj.n_repairs == 0  # JSSP: nothing to repair
        rf = solve_ported(fjsp, method, time_limit=2, seed=11)
        assert not rf.crashed and check_feasibility(rf.schedule).feasible
        assert rf.n_repairs == 0  # the primary mapping cannot produce an ineligible choice
        assert rf.n_decoder_calls > 0 and len(rf.anytime) >= 1
