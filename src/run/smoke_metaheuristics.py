"""Phase 3 gate artifact: every wrapped metaheuristic runs end to end.

Runs each mealpy method through the shared decoder on one JSSP and one FJSP instance at a
short budget, and records the complete per-run log (objective, decoder calls, repairs,
time-to-first, time-to-best, anytime points, feasibility) into
`results/calibration/metaheuristics_smoke.json`. Fails if any method does not produce a
feasible final schedule or a populated log.

    uv run python -m src.run.smoke_metaheuristics [budget_seconds]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from src.core.feasibility import check_feasibility
from src.io.loaders import load_fjsp_file, load_jssp_file
from src.methods.metaheuristic import OPTIMIZERS, solve_metaheuristic

OUT = Path("results/calibration/metaheuristics_smoke.json")


def main() -> None:
    budget = float(sys.argv[1]) if len(sys.argv) > 1 else 4.0
    instances = [
        ("jssp", "ft06", load_jssp_file("data/instances/jssp/ft06.txt", "ft06", "x")),
        ("fjsp", "mk01", load_fjsp_file("data/instances/fjsp/mk01.txt", "mk01", "x")),
    ]
    records = []
    failures = []
    for kind, name, inst in instances:
        for method in OPTIMIZERS:
            r = solve_metaheuristic(inst, method, time_limit=budget, seed=11)
            feasible = r.schedule is not None and check_feasibility(r.schedule).feasible
            log_ok = (
                r.best_obj < float("inf")
                and r.n_decoder_calls > 0
                and len(r.anytime) > 0
                and r.time_to_first is not None
                and r.time_to_best is not None
                and not r.crashed
            )
            rec = {
                "kind": kind, "instance": name, "method": method,
                "best_obj": r.best_obj, "n_decoder_calls": r.n_decoder_calls,
                "n_repairs": r.n_repairs, "time_to_first": round(r.time_to_first or -1, 3),
                "time_to_best": round(r.time_to_best or -1, 3), "n_anytime": len(r.anytime),
                "feasible_final": feasible, "crashed": r.crashed, "log_complete": log_ok,
            }
            records.append(rec)
            if not (feasible and log_ok):
                failures.append(rec)
            print(f"{kind} {name} {method:7} obj={r.best_obj:7.0f} calls={r.n_decoder_calls:6} "
                  f"repairs={r.n_repairs:7} feasible={feasible} log_ok={log_ok}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT}")
    if failures:
        raise SystemExit(f"{len(failures)} method/instance combination(s) FAILED")
    print(f"all {len(records)} method/instance runs produced feasible schedules and complete logs")


if __name__ == "__main__":
    main()
