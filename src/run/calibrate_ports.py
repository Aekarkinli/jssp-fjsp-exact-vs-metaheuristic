"""Phase 4 port-validation gate: MDE and CSA reproduce their papers' CEC convergence.

Each ported optimiser is validated as a continuous minimiser on functions from its own
paper before it is attached to the scheduling decoder (decision D2). A weak port would
unfairly handicap the very methods the study tests, so convergence to the known global
optimum is required, not assumed. Results are written to
`results/calibration/ports_cec.json`.

- MDE (Karkinli 2023) is validated on CEC2013 F1 (sphere, optimum -1400) and F5 (-1000),
  matching its Table 2 where MDE reaches these optima.
- CSA (Civicioglu & Besdok 2024) is validated on CEC2017 F1 (Bent Cigar, optimum 100),
  matching its Table 3, with the CEC2013 sphere as an extra correctness anchor. CEC2017 F1
  is ill-conditioned and needs more cycles than the sphere; CSA's Levy-flight scaling
  reaches it where the sphere is solved quickly.

    uv run python -m src.run.calibrate_ports
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
from opfunu.cec_based import cec2013, cec2017

from src.methods.ported import csa_optimize, mde_optimize

OUT = Path("results/calibration/ports_cec.json")
TOL = 1.0  # absolute gap to the global optimum; the ports reach ~1e-12 in practice
SEED = 11

CASES = [
    (mde_optimize, "MDE", cec2013.F12013, 20, -1400, 3000, "CEC2013 F1 sphere"),
    (mde_optimize, "MDE", cec2013.F52013, 20, -1000, 3000, "CEC2013 F5"),
    (csa_optimize, "CSA", cec2013.F12013, 20, -1400, 3000, "CEC2013 F1 sphere (anchor)"),
    (csa_optimize, "CSA", cec2017.F12017, 30, 100, 15000, "CEC2017 F1 Bent Cigar"),
]


def _batch(func):
    return lambda pop: np.array([func.evaluate(np.asarray(row)) for row in pop])


def main() -> None:
    warnings.filterwarnings("ignore")
    records, failures = [], []
    for optimise, method, fcls, ndim, optimum, cycles, note in CASES:
        func = fcls(ndim=ndim)
        best, _ = optimise(_batch(func), ndim, func.lb[0], func.ub[0], 30, seed=SEED, max_cycles=cycles)
        gap = float(best - optimum)
        rec = {
            "method": method, "function": func.__class__.__name__, "note": note,
            "ndim": ndim, "optimum": optimum, "best": round(float(best), 6),
            "gap": round(gap, 6), "cycles": cycles, "seed": SEED, "tolerance": TOL,
            "converged": abs(gap) <= TOL,
        }
        records.append(rec)
        if not rec["converged"]:
            failures.append(rec)
        print(f"{method} {note:28} best={best:12.4f} optimum={optimum:6} gap={gap:.3e} "
              f"{'OK' if rec['converged'] else 'FAIL'}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT}")
    if failures:
        raise SystemExit(f"port validation FAILED: {[r['note'] for r in failures]}")
    print("both ports reproduce their papers' CEC convergence")


if __name__ == "__main__":
    main()
