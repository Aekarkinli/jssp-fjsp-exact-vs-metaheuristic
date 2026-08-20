"""Pre-launch smoke and timing check for the v2 design.

Runs each new or changed component once, on the smallest and the largest instances, and
prints the numbers that decide whether the twelve-day run is safe to start: does every method
produce a feasible schedule and a complete log, does the corrected decoder really never
repair, and does any method collapse to a handful of evaluations on the largest instance.

    uv run python -m src.run.smoke_v2
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import time  # noqa: E402

from src.core.decoder import vector_length  # noqa: E402
from src.core.feasibility import check_feasibility  # noqa: E402
from src.run.runner import ALL_METHODS, DETERMINISTIC, INDEX, _load, _run_method  # noqa: E402

BUDGET = 8.0
PROBES = ["ft06", "mk01", "la36", "ta71", "mk10"]


def main() -> None:
    sizes = []
    for iid in INDEX:
        inst = _load(iid)
        sizes.append((vector_length(inst), iid, inst.num_operations, inst.is_flexible))
    sizes.sort(reverse=True)
    print("largest search dimensions:")
    for dim, iid, n_op, flex in sizes[:5]:
        print(f"  {iid:14s} dim={dim:5d} ops={n_op:5d} flexible={flex}")
    print()

    header = f"{'instance':12s} {'method':10s} {'obj':>8s} {'evals':>9s} {'repairs':>8s} {'feas':>5s} {'wall':>6s}"
    for iid in PROBES:
        if iid not in INDEX:
            continue
        inst = _load(iid)
        print(f"--- {iid}  (dim={vector_length(inst)}, flexible={inst.is_flexible}) ---")
        print(header)
        for method in ALL_METHODS:
            budget = BUDGET
            t0 = time.perf_counter()
            r = _run_method(method, inst, budget, 11, 50)
            wall = time.perf_counter() - t0
            feasible = r.schedule is not None and check_feasibility(r.schedule).feasible
            flag = "" if feasible and not r.crashed else "   <-- PROBLEM"
            print(f"{iid:12s} {method:10s} {r.best_obj:8.0f} {r.n_decoder_calls:9d} "
                  f"{r.n_repairs:8d} {str(feasible):>5s} {wall:6.1f}{flag}")
            if method in DETERMINISTIC:
                continue
        print()


if __name__ == "__main__":
    main()
