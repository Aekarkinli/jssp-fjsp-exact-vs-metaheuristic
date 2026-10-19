"""Tests for the representations and methods of the targeted experiments."""
from __future__ import annotations

import numpy as np
import pytest

from src.core.decoder import MEMETIC, OPPERM, decode_vector, vector_length
from src.core.feasibility import check_feasibility
from src.run.runner import _load

INSTANCES = ["ft06", "la01", "mk01", "edata_la01", "fattahi05", "hs_vdata_la01",
             "behnke11", "ta42"]


@pytest.mark.parametrize("iid", INSTANCES)
def test_operation_permutation_is_feasible_and_deterministic(iid):
    inst = _load(iid)
    assert vector_length(inst, OPPERM) == inst.num_operations
    rng = np.random.default_rng(3)
    for _ in range(5):
        x = rng.random(vector_length(inst, OPPERM))
        a = decode_vector(inst, x, mapping=OPPERM)
        b = decode_vector(inst, x, mapping=OPPERM)
        assert check_feasibility(a.schedule).feasible
        assert a.n_repairs == 0
        assert a.schedule.makespan == b.schedule.makespan


def test_operation_permutation_machine_choice_is_eligible():
    inst = _load("mk01")
    x = np.random.default_rng(1).random(inst.num_operations)
    res = decode_vector(inst, x, mapping=OPPERM)
    for a in res.schedule.assignments:
        assert a.machine in inst.operation(a.operation).eligible_machines


def test_operation_permutation_ft06_reaches_optimum_from_known_sequence():
    """Keys that list operations in the start order of an optimal schedule decode to it."""
    import json
    from pathlib import Path

    inst = _load("ft06")
    fix = json.loads(Path("tests/fixtures/ft06_optimal.json").read_text(encoding="utf-8"))
    starts = [pair[1] for pair in fix["assignments_by_global_index"]]
    keys = np.argsort(np.argsort(np.asarray(starts, dtype=float), kind="stable")).astype(float)
    res = decode_vector(inst, keys / len(keys), mapping=OPPERM)
    assert res.schedule.makespan <= 55


@pytest.mark.parametrize("iid", ["ft06", "la01", "mk01", "edata_la01"])
def test_memetic_descent_never_worsens_and_stays_feasible(iid):
    inst = _load(iid)
    rng = np.random.default_rng(7)
    for _ in range(3):
        x = rng.random(vector_length(inst))
        base = decode_vector(inst, x)
        improved = decode_vector(inst, x, mapping=MEMETIC)
        assert check_feasibility(improved.schedule).feasible
        assert improved.schedule.makespan <= base.schedule.makespan
        again = decode_vector(inst, x, mapping=MEMETIC)
        assert again.schedule.makespan == improved.schedule.makespan


@pytest.mark.parametrize("method", ["jade", "shade", "clpso", "igwo"])
def test_improved_variants_run_through_the_decoder(method):
    from src.methods.metaheuristic import solve_metaheuristic

    inst = _load("ft06")
    r = solve_metaheuristic(inst, method, time_limit=1.0, seed=11, pop_size=20)
    assert not r.crashed
    assert r.schedule is not None and check_feasibility(r.schedule).feasible
    assert r.n_decoder_calls > 20


@pytest.mark.parametrize("mapping", [OPPERM, MEMETIC])
def test_every_decoded_solver_accepts_the_new_representations(mapping):
    from src.run.runner import _run_method

    inst = _load("mk01")
    for method in ("ga", "brkga", "cmaes", "mde"):
        r = _run_method(method, inst, 1.0, 11, 10, mapping)
        assert r.schedule is not None and check_feasibility(r.schedule).feasible, method
