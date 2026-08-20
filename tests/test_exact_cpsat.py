"""CP-SAT runner tests on small instances with known optima (kept fast)."""
from __future__ import annotations

from pathlib import Path

from src.core.feasibility import check_feasibility
from src.io.loaders import load_fjsp_file, load_jssp_file
from src.methods.exact_cpsat import solve_cpsat

DATA = Path("data/instances")


def test_ft06_proven_optimal_55_with_anytime_trace():
    inst = load_jssp_file(DATA / "jssp/ft06.txt", "ft06", "fisher_thompson")
    r = solve_cpsat(inst, time_limit=30, seed=11, num_workers=1)
    assert r.status == "OPTIMAL"
    assert r.best_obj == 55 == r.best_bound
    assert r.rel_gap == 0.0
    assert r.proven_optimal
    fr = check_feasibility(r.schedule)
    assert fr.feasible, fr.violations
    # anytime trace: non-empty, improving objective, final point equals the optimum
    assert len(r.anytime) >= 1
    objs = [p.obj for p in r.anytime]
    assert objs == sorted(objs, reverse=True)
    assert objs[-1] == 55
    assert r.time_to_first is not None and r.time_to_best is not None
    assert r.time_to_first <= r.time_to_best


def test_mk01_fjsp_proven_optimal_40():
    inst = load_fjsp_file(DATA / "fjsp/mk01.txt", "mk01", "brandimarte")
    r = solve_cpsat(inst, time_limit=30, seed=11, num_workers=1)
    assert r.status == "OPTIMAL"
    assert r.best_obj == 40 == r.best_bound
    assert check_feasibility(r.schedule).feasible


def test_warm_start_accepts_solution_hint():
    """The hybrid path: warm-starting from a heuristic schedule must not break solving."""
    from src.methods.constructive import solve_greedy
    from src.methods.exact_cpsat import build_model
    from pyjobshop import Solution
    from pyjobshop.Solution import ScheduledTask

    inst = load_jssp_file(DATA / "jssp/ft06.txt", "ft06", "fisher_thompson")
    greedy = solve_greedy(inst)
    data = build_model(inst).data()
    # task order == operation global-index order; mode index per task equals the position
    # of the chosen machine in data.task2modes (single mode per task for JSSP)
    tasks = []
    for op, a in zip(inst.operations, greedy.schedule.assignments):
        mode_idx = data.task2modes(op.global_index)[0]
        tasks.append(ScheduledTask(mode_idx, [a.machine], a.start, a.end, 0, 0, True))
    warm = Solution(data, tasks)
    r = solve_cpsat(inst, time_limit=30, seed=11, num_workers=1, initial_solution=warm)
    assert r.status == "OPTIMAL" and r.best_obj == 55
