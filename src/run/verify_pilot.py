"""Phase 5 pilot verification (the analysis-lock gate).

Reads every pilot result file and confirms the run is crash-free, every log field is
populated, and each stored schedule independently re-verifies (its makespan recomputes and
the feasibility checker accepts it). Then it confirms repeatability under fixed seeds and
that calibration held. Writes `results/calibration/pilot_verification.json` and exits
non-zero if any check fails.

    uv run python -m src.run.verify_pilot [--config config/pilot.yaml]
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import yaml

from src.core.feasibility import check_feasibility
from src.core.schedule import Assignment, Schedule
from src.methods.constructive import solve_dispatching, solve_greedy
from src.methods.exact_cpsat import solve_cpsat
from src.methods.metaheuristic import solve_metaheuristic
from src.methods.tabu import solve_tabu
from src.run.runner import DETERMINISTIC, _load, _method_ids, _instance_ids, result_path

OUT = Path("results/calibration/pilot_verification.json")
P_CORES = {0, 2, 4, 6, 8, 10, 12, 14}
# methods that drive the shared random-key decoder (and so accrue repairs on FJSP); tabu and
# the exact/constructive methods work on the combinatorial structure and never repair.
DECODER_METHODS = {"ga", "de", "pso", "abc", "gwo", "lshade", "imode", "rime", "mde", "csa"}
REQUIRED = [
    "instance_id", "family", "type", "n_jobs", "n_machines", "n_op", "method", "seed",
    "budget_s", "status", "best_obj", "feasible_final", "crashed", "wall_time",
    "n_decoder_calls", "n_repairs", "time_to_first", "time_to_best", "anytime", "schedule",
    "versions", "git_commit", "affinity_core", "peak_mem_bytes", "timestamp_utc",
]


def _reference() -> dict[str, dict]:
    with Path("data/reference_values.csv").open(encoding="utf-8") as fh:
        return {r["id"]: r for r in csv.DictReader(fh)}


def _rebuild(rec) -> tuple:
    inst = _load(rec["instance_id"])
    machine, start = rec["schedule"]["machine"], rec["schedule"]["start"]
    assignments = [
        Assignment(g, inst.operation(g).job, machine[g], start[g],
                   inst.operation(g).duration_on(machine[g]))
        for g in range(inst.num_operations)
    ]
    return inst, Schedule(inst, tuple(assignments))


def _checkpoint_values(anytime, checkpoints):
    """Best objective at or before each checkpoint time (the budget-extraction the analysis uses)."""
    out = {}
    for cp in checkpoints:
        best = None
        for t, obj, _bound in anytime:
            if t <= cp:
                best = obj if best is None else min(best, obj)
        out[cp] = best
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/pilot.yaml")
    args = ap.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    run_name = Path(args.config).stem
    instances = _instance_ids(config, None)
    methods = _method_ids(config, None)
    seeds = list(config["seeds"]["stochastic"])
    checkpoints = config["budget"]["checkpoints_seconds"]
    ref = _reference()

    errors: list[str] = []
    n_checked = 0

    # 1) completeness, crash-free, and independent re-verification of every schedule
    for iid in instances:
        for method in methods:
            for seed in ([seeds[0]] if method in DETERMINISTIC else seeds):
                path = result_path(iid, method, seed, run_name)
                if not path.exists():
                    errors.append(f"missing result: {path.name}")
                    continue
                rec = json.loads(path.read_text(encoding="utf-8"))
                n_checked += 1
                for f in REQUIRED:
                    if f not in rec:
                        errors.append(f"{path.name}: missing field {f}")
                if rec.get("crashed"):
                    errors.append(f"{path.name}: crashed")
                    continue
                if not rec.get("feasible_final"):
                    errors.append(f"{path.name}: not feasible_final")
                if rec.get("best_obj") is None:
                    errors.append(f"{path.name}: no best_obj")
                    continue
                if not rec.get("anytime"):
                    errors.append(f"{path.name}: empty anytime trace")
                aff = rec.get("affinity_core")
                if not (isinstance(aff, list) and set(aff) <= P_CORES):
                    errors.append(f"{path.name}: affinity {aff} not a performance core")
                # independent re-verification from the stored schedule
                inst, sched = _rebuild(rec)
                if sched.makespan != rec["best_obj"]:
                    errors.append(f"{path.name}: makespan {sched.makespan} != best_obj {rec['best_obj']}")
                if not check_feasibility(sched).feasible:
                    errors.append(f"{path.name}: stored schedule fails feasibility check")
                # decoder methods accrue repairs on FJSP and none on JSSP
                if method in DECODER_METHODS:
                    if inst.is_flexible and rec["n_repairs"] == 0:
                        errors.append(f"{path.name}: FJSP decoder run recorded zero repairs")
                    if not inst.is_flexible and rec["n_repairs"] != 0:
                        errors.append(f"{path.name}: JSSP decoder run recorded repairs")

    # 2) repeatability
    repeat = {}
    a = solve_cpsat(_load("ft06"), 30, seed=11, num_workers=1).best_obj
    b = solve_cpsat(_load("ft06"), 30, seed=11, num_workers=1).best_obj
    repeat["cpsat_ft06"] = (a, b)
    if a != b:
        errors.append(f"cpsat not repeatable: {a} != {b}")
    d1 = solve_dispatching(_load("mk01")).best_obj
    d2 = solve_dispatching(_load("mk01")).best_obj
    repeat["dispatching_mk01"] = (d1, d2)
    if d1 != d2:
        errors.append(f"dispatching not repeatable: {d1} != {d2}")
    # stochastic: a fixed seed fixes the trajectory; the initial-population best reproduces
    g1 = solve_metaheuristic(_load("ft06"), "ga", time_limit=2, seed=11).anytime[0].obj
    g2 = solve_metaheuristic(_load("ft06"), "ga", time_limit=2, seed=11).anytime[0].obj
    repeat["ga_ft06_initial"] = (g1, g2)
    if g1 != g2:
        errors.append(f"ga initial point not reproducible under seed: {g1} != {g2}")

    # 3) calibration held: CP-SAT reproduces the proven optima on pilot instances
    calib = {}
    for iid in instances:
        rec = json.loads(result_path(iid, "cpsat", seeds[0], run_name).read_text(encoding="utf-8"))
        r = ref.get(iid, {})
        if r.get("proven_optimal") == "True":
            bks = int(r["BKS"])
            proved = rec["status"] == "OPTIMAL"
            calib[iid] = {"cpsat": rec["best_obj"], "bks": bks, "status": rec["status"],
                          "proved_within_budget": proved}
            # an incumbent can never beat the true optimum (integrity check)
            if rec["best_obj"] < bks:
                errors.append(f"cpsat {iid}: incumbent {rec['best_obj']} below optimum {bks}")
            # where CP-SAT proves optimality the value must equal the literature optimum (D15)
            if proved and rec["best_obj"] != bks:
                errors.append(f"cpsat {iid}: proved {rec['best_obj']} != optimum {bks}")
        # tabu must beat the dispatching bank
        tabu_best = min(
            json.loads(result_path(iid, "tabu", s, run_name).read_text())["best_obj"] for s in seeds
        )
        disp = json.loads(result_path(iid, "dispatching", seeds[0], run_name).read_text())["best_obj"]
        if tabu_best > disp:
            errors.append(f"tabu {iid} ({tabu_best}) did not beat dispatching ({disp})")

    # 4) checkpoint extraction sanity (the anytime mechanism feeds the budget analysis)
    sample = json.loads(result_path(instances[0], "ga", seeds[0], run_name).read_text())
    cps = _checkpoint_values(sample["anytime"], checkpoints)

    report = {
        "n_results_checked": n_checked,
        "instances": instances, "methods": methods, "seeds": seeds,
        "repeatability": repeat,
        "calibration": calib,
        "example_checkpoints": {str(k): v for k, v in cps.items()},
        "n_errors": len(errors),
        "errors": errors[:50],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"checked {n_checked} results; {len(errors)} errors -> {OUT}")
    for e in errors[:20]:
        print("  ERROR:", e)
    if errors:
        raise SystemExit(f"pilot verification FAILED with {len(errors)} error(s)")
    print("PILOT VERIFIED: crash-free, complete logs, schedules re-verified, "
          "repeatable, calibration held")


if __name__ == "__main__":
    main()
