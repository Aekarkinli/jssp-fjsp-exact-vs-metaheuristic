"""Feasibility checker tests against the proven-optimal ft06 schedule and corruptions."""
from __future__ import annotations

import json
from pathlib import Path

from src.core.feasibility import check_feasibility
from src.core.schedule import Assignment, Schedule
from src.io.loaders import load_jssp_file

DATA = Path("data/instances")
FIXTURE = Path("tests/fixtures/ft06_optimal.json")


def _ft06_optimal():
    inst = load_jssp_file(DATA / "jssp/ft06.txt", "ft06", "fisher_thompson")
    fix = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assignments = []
    for g, (machine, start) in enumerate(fix["assignments_by_global_index"]):
        op = inst.operation(g)
        assignments.append(Assignment(g, op.job, machine, start, op.duration_on(machine)))
    return inst, Schedule(inst, tuple(assignments)), fix["makespan"]


def test_ft06_known_optimum_feasible_and_55():
    inst, sched, fixture_makespan = _ft06_optimal()
    assert fixture_makespan == 55
    assert sched.makespan == 55
    result = check_feasibility(sched)
    assert result.feasible, result.violations


def test_precedence_violation_rejected():
    inst, sched, _ = _ft06_optimal()
    a = list(sched.assignments)
    g = inst.jobs[0][1].global_index  # second operation of job 0
    a[g] = Assignment(a[g].operation, a[g].job, a[g].machine, 0, a[g].duration)  # start before predecessor
    result = check_feasibility(Schedule(inst, tuple(a)))
    assert not result.feasible
    assert any("precedence" in v for v in result.violations)


def test_machine_overlap_rejected():
    inst, sched, _ = _ft06_optimal()
    by_m = sched.by_machine()
    machine = next(m for m, ops in by_m.items() if len(ops) >= 2)
    first, second = by_m[machine][0], by_m[machine][1]
    a = list(sched.assignments)
    a[second.operation] = Assignment(
        second.operation, second.job, machine, first.start, second.duration
    )  # force overlap with the first operation on this machine
    result = check_feasibility(Schedule(inst, tuple(a)))
    assert not result.feasible
    assert any("overlap" in v for v in result.violations)


def test_ineligible_machine_rejected():
    inst, sched, _ = _ft06_optimal()
    a = list(sched.assignments)
    bad_machine = (a[0].machine + 1) % inst.num_machines  # JSSP op has a single eligible machine
    a[0] = Assignment(a[0].operation, a[0].job, bad_machine, a[0].start, a[0].duration)
    result = check_feasibility(Schedule(inst, tuple(a)))
    assert not result.feasible
    assert any("not eligible" in v for v in result.violations)
