"""Gate test: both constructors return feasible schedules on every curated instance."""
from __future__ import annotations

from pathlib import Path

import pytest

from src.core.feasibility import check_feasibility
from src.io.instance_sources import FJSP_INSTANCES, JSSP_INSTANCES
from src.io.loaders import load_fjsp_file, load_jssp_file
from src.methods.constructive import (
    DISPATCHING_RULES,
    build_active_schedule,
    solve_dispatching,
    solve_greedy,
)

DATA = Path("data/instances")

ALL = [(s, "jssp") for s in JSSP_INSTANCES] + [(s, "fjsp") for s in FJSP_INSTANCES]


def _load(spec, kind):
    loader = load_jssp_file if kind == "jssp" else load_fjsp_file
    return loader(DATA / spec.local, spec.id, spec.family)


@pytest.mark.parametrize("spec,kind", ALL, ids=[s.id for s, _ in ALL])
def test_constructors_feasible_on_every_instance(spec, kind):
    inst = _load(spec, kind)
    for solver in (solve_dispatching, solve_greedy):
        r = solver(inst)
        assert r.feasible_final
        fr = check_feasibility(r.schedule)
        assert fr.feasible, (spec.id, r.method, fr.violations[:3])
        assert r.best_obj == r.schedule.makespan > 0


def test_dispatching_reports_all_rules():
    inst = _load(JSSP_INSTANCES[0], "jssp")  # ft06
    r = solve_dispatching(inst)
    assert set(r.extra["per_rule"]) == set(DISPATCHING_RULES) == {"SPT", "MWKR", "MOPNR"}
    assert r.best_obj == min(r.extra["per_rule"].values())
    assert r.extra["best_rule"] in r.extra["per_rule"]


def test_rules_change_outcome_somewhere():
    """The three rules are genuinely different policies on at least one instance."""
    inst = _load(JSSP_INSTANCES[1], "jssp")  # ft10
    values = {
        name: build_active_schedule(inst, rule).makespan
        for name, rule in DISPATCHING_RULES.items()
    }
    assert len(set(values.values())) > 1, values
