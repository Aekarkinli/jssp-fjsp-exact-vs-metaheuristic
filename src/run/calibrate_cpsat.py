"""Phase 2 calibration: CP-SAT must reproduce known optima on small instances.

Runs the single-worker CP-SAT runner on a set of small instances with known proven optima
and records the comparison in `results/calibration/cpsat_small_optima.json`. The pass
criterion is value reproduction: the final objective must equal the literature optimum and
the schedule must pass the independent feasibility checker. Whether optimality is also
*proved* within the calibration budget is recorded separately, because proof availability
at a budget is a study subject (RQ1), not a correctness requirement of the runner.

    uv run python -m src.run.calibrate_cpsat
"""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path

from src.core.feasibility import check_feasibility
from src.io.loaders import load_fjsp_file, load_jssp_file
from src.methods.exact_cpsat import solve_cpsat

DATA = Path("data/instances")
OUT = Path("results/calibration/cpsat_small_optima.json")

# Small instances with proven optima; generous limits, expected to prove well within them.
CASES = [
    ("jssp/ft06.txt", "ft06", "jssp", 30),
    ("jssp/ft10.txt", "ft10", "jssp", 60),
    ("jssp/la01.txt", "la01", "jssp", 30),
    ("jssp/abz5.txt", "abz5", "jssp", 60),
    ("jssp/orb01.txt", "orb01", "jssp", 60),
    ("fjsp/mk01.txt", "mk01", "fjsp", 30),
    ("fjsp/mk02.txt", "mk02", "fjsp", 60),
    ("fjsp/edata_la01.txt", "edata_la01", "fjsp", 30),
    ("fjsp/fattahi10.txt", "fattahi10", "fjsp", 30),
]


def _reference() -> dict[str, dict]:
    with Path("data/reference_values.csv").open(encoding="utf-8") as fh:
        return {r["id"]: r for r in csv.DictReader(fh)}


def main() -> None:
    ref = _reference()
    records = []
    failures = []
    for rel, name, kind, limit in CASES:
        inst = (load_jssp_file if kind == "jssp" else load_fjsp_file)(DATA / rel, name, "calib")
        t0 = time.perf_counter()
        r = solve_cpsat(inst, time_limit=limit, seed=11, num_workers=1)
        wall = time.perf_counter() - t0
        expected = int(ref[name]["BKS"]) if ref[name]["proven_optimal"] == "True" else None
        feas = check_feasibility(r.schedule).feasible if r.schedule else False
        value_match = expected is None or int(r.best_obj) == expected
        rec = {
            "instance": name,
            "status": r.status,
            "objective": r.best_obj,
            "bound": r.best_bound,
            "expected_optimum": expected,
            "value_match": value_match,
            "proved_within_budget": r.status == "OPTIMAL",
            "schedule_feasible": feas,
            "time_limit_s": limit,
            "wall_time_s": round(wall, 2),
            "n_incumbents": len(r.anytime),
        }
        records.append(rec)
        if not (value_match and feas and r.status in ("OPTIMAL", "FEASIBLE")):
            failures.append(rec)
        print(f"{name:12} {r.status:8} obj={r.best_obj:7.0f} expected={expected} "
              f"feasible={feas} wall={wall:5.2f}s")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"cases": records, "failures": failures}, indent=2),
                   encoding="utf-8")
    print(f"\nwrote {OUT}")
    if failures:
        raise SystemExit(f"{len(failures)} calibration case(s) FAILED")
    n_proved = sum(1 for r in records if r["proved_within_budget"])
    print(f"all {len(records)} cases reproduce their literature optimum "
          f"({n_proved} also proved within the calibration budget)")


if __name__ == "__main__":
    main()
