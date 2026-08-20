"""One-off generator for the proven-optimal ft06 schedule fixture.

Not run by pytest. Documents how `tests/fixtures/ft06_optimal.json` was produced: the
six-by-six Fisher-Thompson instance is solved to proven optimality with OR-Tools CP-SAT
(via PyJobShop). The literature optimum is 55; the assertion guards against drift. The
test suite then validates the schedule machinery against this externally-known optimum
without invoking the solver.

    uv run python tests/_generate_ft06_fixture.py
"""
from __future__ import annotations

import json
from pathlib import Path

from pyjobshop import Model

from src.io.loaders import load_jssp_file


def main() -> None:
    inst = load_jssp_file("data/instances/jssp/ft06.txt", "ft06", "fisher_thompson")
    model = Model()
    machines = [model.add_machine(name=f"M{k}") for k in range(inst.num_machines)]
    tasks = []
    for op in inst.operations:
        task = model.add_task(name=f"op{op.global_index}")
        tasks.append(task)
        for machine, dur in op.modes:
            model.add_mode(task, machines[machine], dur)
    for job in inst.jobs:
        for k in range(1, len(job)):
            model.add_end_before_start(
                tasks[job[k - 1].global_index], tasks[job[k].global_index]
            )

    res = model.solve(solver="ortools", time_limit=60, display=False)
    assert res.status.name == "OPTIMAL", res.status
    sol = res.best
    assert sol.makespan == 55, f"ft06 optimum should be 55, got {sol.makespan}"

    assignments = [[int(st.resources[0]), int(st.start)] for st in sol.tasks]
    out = {
        "instance": "ft06",
        "makespan": 55,
        "note": "proven optimal via OR-Tools CP-SAT (PyJobShop); literature optimum 55",
        "assignments_by_global_index": assignments,  # [machine, start] per operation
    }
    path = Path("tests/fixtures/ft06_optimal.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {path} (makespan {sol.makespan})")


if __name__ == "__main__":
    main()
