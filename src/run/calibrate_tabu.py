"""Phase 3 calibration: tabu search against published JSSP makespans.

Runs the tabu search on the standard calibration instances (the ten-by-ten Fisher-Thompson
case, a Lawrence instance, an Adams-Balas-Zawack instance) across a few seeds and records
the best makespan, the gap to the published value, and the dispatching-bank baseline for
comparison, into `results/calibration/tabu_jssp.json`.

The pass criterion is that the tabu search is clearly stronger than the constructive
baseline on every case and lands within a documented small margin of the published value.
The full study runs the tabu at the main budget across twenty seeds, so these short-budget
few-seed numbers are a conservative lower bound on its study performance.

    uv run python -m src.run.calibrate_tabu [budget_seconds] [n_seeds]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from src.core.feasibility import check_feasibility
from src.io.loaders import load_jssp_file
from src.methods.constructive import solve_dispatching
from src.methods.tabu import solve_tabu

STUDY_SEEDS = [11, 23, 37, 41, 53, 67, 79, 83, 97, 101]
CASES = [("ft10", 930), ("la36", 1268), ("abz7", 656)]
OUT = Path("results/calibration/tabu_jssp.json")
MARGIN_PCT = 8.0  # documented small margin for a from-scratch N5 critical-block tabu


def main() -> None:
    budget = float(sys.argv[1]) if len(sys.argv) > 1 else 30.0
    n_seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    seeds = STUDY_SEEDS[:n_seeds]

    records = []
    for name, published in CASES:
        inst = load_jssp_file(f"data/instances/jssp/{name}.txt", name, "calibration")
        per_seed = []
        for s in seeds:
            r = solve_tabu(inst, time_limit=budget, seed=s)
            assert check_feasibility(r.schedule).feasible, (name, s)
            per_seed.append(int(r.best_obj))
        best = min(per_seed)
        disp = int(solve_dispatching(inst).best_obj)
        rec = {
            "instance": name,
            "published": published,
            "tabu_best": best,
            "tabu_per_seed": per_seed,
            "tabu_gap_pct": round(100 * (best - published) / published, 2),
            "dispatching": disp,
            "dispatching_gap_pct": round(100 * (disp - published) / published, 2),
            "seeds": seeds,
            "budget_s": budget,
        }
        records.append(rec)
        print(f"{name:6} tabu best={best:5d} (+{rec['tabu_gap_pct']:.2f}%)  "
              f"dispatching={disp:5d} (+{rec['dispatching_gap_pct']:.2f}%)  published={published}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT}")

    failures = [r for r in records if r["tabu_best"] >= r["dispatching"] or r["tabu_gap_pct"] > MARGIN_PCT]
    if failures:
        raise SystemExit(f"tabu calibration FAILED for: {[r['instance'] for r in failures]}")
    print("tabu is stronger than dispatching and within the documented margin on all cases")


if __name__ == "__main__":
    main()
