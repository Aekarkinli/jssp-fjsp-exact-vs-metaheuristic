"""Exact solver: OR-Tools CP-SAT through PyJobShop, with an anytime callback.

PyJobShop builds the constraint-programming model; OR-Tools solves it. PyJobShop's own
solve returns only final values, so the model is handed to an OR-Tools `CpSolver` with a
solution callback that records every improving incumbent and the dual bound over time.
This yields the anytime trace the budget-checkpoint analysis needs, the proof status, and
the dual bound. The fair comparison uses a single search worker; a separate reference run
uses all cores.
"""
from __future__ import annotations

import time

from ortools.sat.python import cp_model as cp
from pyjobshop import Model
from pyjobshop.solvers.ortools import CPModel as ORToolsCPModel

from src.core.instance import Instance
from src.core.schedule import Assignment, Schedule
from src.methods.base import AnytimePoint, MethodResult


def build_model(instance: Instance) -> Model:
    """Build a PyJobShop makespan model. Tasks are added in operation global-index order."""
    model = Model()
    machines = [model.add_machine(name=f"M{k}") for k in range(instance.num_machines)]
    tasks = []
    for op in instance.operations:
        task = model.add_task(name=f"op{op.global_index}")
        tasks.append(task)
        for machine, duration in op.modes:
            model.add_mode(task, machines[machine], duration)
    for job in instance.jobs:
        for k in range(1, len(job)):
            model.add_end_before_start(
                tasks[job[k - 1].global_index], tasks[job[k].global_index]
            )
    return model


class _IncumbentRecorder(cp.CpSolverSolutionCallback):
    """Records (wall_time, objective, dual bound) for each improving incumbent."""

    def __init__(self) -> None:
        super().__init__()
        self.trace: list[tuple[float, float, float]] = []

    def on_solution_callback(self) -> None:
        self.trace.append(
            (self.wall_time, float(self.objective_value), float(self.best_objective_bound))
        )


def _extract_schedule(instance: Instance, data, variables, solver: cp.CpSolver) -> Schedule:
    assignments = []
    for op in instance.operations:
        g = op.global_index
        task_var = variables.task_vars[g]
        machine = None
        for mode_idx in data.task2modes(g):
            if solver.value(variables.mode_vars[mode_idx]):
                machine = data.modes[mode_idx].resources[0]
                break
        start = int(solver.value(task_var.start))
        assignments.append(Assignment(g, op.job, machine, start, op.duration_on(machine)))
    return Schedule(instance, tuple(assignments))


def solve_cpsat(
    instance: Instance,
    time_limit: float,
    seed: int = 0,
    num_workers: int = 1,
    initial_solution=None,
) -> MethodResult:
    """Solve to makespan optimality (or the time limit) and return an anytime result."""
    model = build_model(instance)
    data = model.data()
    cpm = ORToolsCPModel(data)

    if initial_solution is not None:
        cpm.variables.warmstart(initial_solution)

    solver = cp.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit)
    solver.parameters.num_workers = int(num_workers)
    solver.parameters.random_seed = int(seed)
    solver.parameters.log_search_progress = False

    recorder = _IncumbentRecorder()
    wall_start = time.perf_counter()
    status_code = solver.solve(cpm.model, recorder)
    wall = time.perf_counter() - wall_start
    status = solver.status_name(status_code)

    feasible = status in ("OPTIMAL", "FEASIBLE")
    if feasible:
        best_obj = float(solver.objective_value)
        schedule = _extract_schedule(instance, data, cpm.variables, solver)
    else:
        best_obj = float("inf")
        schedule = None
    # dual bound is meaningful unless the model is proven infeasible
    best_bound = None if status == "INFEASIBLE" else float(solver.best_objective_bound)

    anytime = [AnytimePoint(t=t, obj=o, bound=b) for (t, o, b) in recorder.trace]
    ttf = recorder.trace[0][0] if recorder.trace else None
    ttb = recorder.trace[-1][0] if recorder.trace else None

    return MethodResult(
        method="cpsat" if num_workers == 1 else "cpsat_mt",
        instance=instance.name,
        status=status,
        best_obj=best_obj,
        best_bound=best_bound,
        schedule=schedule,
        feasible_final=feasible,
        anytime=anytime,
        time_to_first=ttf,
        time_to_best=ttb,
        wall_time=solver.wall_time if solver.wall_time else wall,
    )
