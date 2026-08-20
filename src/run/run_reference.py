"""Multi-thread CP-SAT reference run (run ALONE, after the main run).

A separate, practical reference that establishes near-optima and bounds on the hard subset
(instances the single-thread CP-SAT did not prove in the main run). It uses all cores and a
longer budget and is NOT part of the fair single-thread comparison. It must run alone, never
concurrently with the timed jobs. Writes results/raw/reference/<instance>__cpsat_mt__seed011.json.

    uv run python -m src.run.run_reference [--budget 900]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.methods.exact_cpsat import solve_cpsat
from src.run.record import build_record, env_meta
from src.run.runner import INDEX, _load

RAW = Path("results/raw")


def hard_subset() -> list[str]:
    """Instances no single-thread exact run proved optimal in the main comparison.

    The exact solver now runs with the full seed list, so an instance counts as hard only
    when none of its seeds returned a proof. Reading a single seed would misclassify the
    instances whose proof is seed-dependent, which are exactly the interesting ones.
    """
    hard = []
    for iid in INDEX:
        results = list((RAW / "full").glob(f"{iid}__cpsat__seed*.json"))
        if not results:
            continue
        proved = False
        for p in results:
            r = json.loads(p.read_text(encoding="utf-8"))
            if r.get("status") == "OPTIMAL":
                proved = True
                break
        if not proved:
            hard.append(iid)
    return hard


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=float, default=900.0)
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()
    out = RAW / "reference"
    out.mkdir(parents=True, exist_ok=True)
    meta = env_meta()
    subset = hard_subset()
    print(f"multi-thread CP-SAT reference on {len(subset)} hard instances (all cores, {args.budget}s each)")
    for iid in subset:
        dest = out / f"{iid}__cpsat_mt__seed{args.seed:03d}.json"
        if dest.exists():
            continue
        inst = _load(iid)
        r = solve_cpsat(inst, args.budget, seed=args.seed, num_workers=0)  # all cores
        rec = build_record(r, inst, args.seed, args.budget, meta)
        tmp = dest.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(rec), encoding="utf-8")
        tmp.replace(dest)
        print(f"  {iid}: obj={rec['best_obj']} bound={rec['best_bound']} {r.status} ({r.wall_time:.0f}s)")
    # completion sentinel: written only when every hard-subset instance has a result, so the
    # dedicated scheduled task can detect completion and remove itself. Not matched by the
    # reference aggregator glob (*__cpsat_mt__*.json), so it is inert for the analysis.
    remaining = [iid for iid in subset
                 if not (out / f"{iid}__cpsat_mt__seed{args.seed:03d}.json").exists()]
    if not remaining:
        (out / "_COMPLETE").write_text(
            f"all {len(subset)} reference jobs complete\n", encoding="utf-8")
        print(f"reference COMPLETE ({len(subset)} instances) -> {out}")
    else:
        print(f"reference incomplete: {len(remaining)} remaining -> {remaining}")
    print(f"reference written -> {out}")


if __name__ == "__main__":
    main()
