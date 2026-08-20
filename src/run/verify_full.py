"""Phase 6 full-run completeness and integrity verification.

Independent of the analysis pipeline. Reads every full result file and confirms:

1. completeness  -- every (instance, method, seed) expected by the locked full config is
   present, with no missing and no unexpected/extra files;
2. integrity     -- every required log field is populated, no job crashed, every final
   schedule is feasible, and every stored schedule re-verifies (its makespan recomputes from
   the stored start/machine assignment and the independent feasibility checker accepts it);
3. provenance    -- the git commit, library versions, hostname, affinity, and timing logs are
   present in the raw outputs, and the affinity of every timed job is a performance core
   (no oversubscription, no SMT sibling);
4. duplicates    -- the deterministic file naming admits exactly one file per combination, so
   a double launch cannot duplicate a job; this is confirmed and any stray file is reported;
5. calibration   -- from the stored values only (no solver re-run): CP-SAT never reports an
   incumbent below a proven optimum, a proved CP-SAT value equals the literature optimum, and
   the tabu search beats the dispatching bank on every instance.

Writes ``results/calibration/full_verification.json`` and exits non-zero on any error.

    uv run python -m src.run.verify_full [--config config/full.yaml] [--run full]
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import yaml

from src.core.feasibility import check_feasibility
from src.core.schedule import Assignment, Schedule
from src.run.runner import DETERMINISTIC, _instance_ids, _method_ids, _load, result_path

OUT = Path("results/calibration/full_verification.json")
P_CORES = {0, 2, 4, 6, 8, 10, 12, 14}
DECODER_METHODS = {"ga", "de", "pso", "abc", "gwo", "lshade", "imode", "rime", "mde", "csa"}
REQUIRED = [
    "instance_id", "family", "type", "n_jobs", "n_machines", "n_op", "method", "seed",
    "budget_s", "status", "best_obj", "feasible_final", "crashed", "wall_time",
    "n_decoder_calls", "n_repairs", "time_to_first", "time_to_best", "anytime", "schedule",
    "versions", "git_commit", "affinity_core", "peak_mem_bytes", "timestamp_utc",
]

_INST_CACHE: dict = {}


def _inst(iid):
    if iid not in _INST_CACHE:
        _INST_CACHE[iid] = _load(iid)
    return _INST_CACHE[iid]


def _reference() -> dict[str, dict]:
    with Path("data/reference_values.csv").open(encoding="utf-8") as fh:
        return {r["id"]: r for r in csv.DictReader(fh)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/full.yaml")
    ap.add_argument("--run", default="full")
    args = ap.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    instances = _instance_ids(config, None)
    methods = _method_ids(config, None)
    seeds = list(config["seeds"]["stochastic"])
    budget = float(config["budget"]["t_max_seconds"])
    ref = _reference()

    errors: list[str] = []
    warnings: list[str] = []

    # ---- expected set --------------------------------------------------------------
    expected: set[str] = set()
    for iid in instances:
        for method in methods:
            for seed in ([seeds[0]] if method in DETERMINISTIC else seeds):
                expected.add(result_path(iid, method, seed, args.run).name)
    present = {p.name for p in (Path("results/raw") / args.run).glob("*.json")}
    missing = sorted(expected - present)
    extra = sorted(present - expected)
    for m in missing:
        errors.append(f"missing result: {m}")
    for e in extra:
        errors.append(f"unexpected/extra file: {e}")

    # ---- per-record integrity + independent schedule re-verification ----------------
    n_checked = 0
    commits: Counter = Counter()
    hosts: Counter = Counter()
    versions_seen: set = set()
    affinity_cores: Counter = Counter()
    run_names: Counter = Counter()
    status_by_method: dict = {m: Counter() for m in methods}
    wall_stats: dict = {"min": float("inf"), "max": 0.0}
    budget_overrun = 0
    infeasible = 0
    crashed_jobs: list[str] = []

    for iid in instances:
        for method in methods:
            for seed in ([seeds[0]] if method in DETERMINISTIC else seeds):
                path = result_path(iid, method, seed, args.run)
                if not path.exists():
                    continue  # already recorded as missing
                try:
                    rec = json.loads(path.read_text(encoding="utf-8"))
                except Exception as exc:  # noqa: BLE001
                    errors.append(f"{path.name}: corrupted JSON ({exc!r})")
                    continue
                n_checked += 1

                if rec.get("crashed"):
                    crashed_jobs.append(path.name)
                    errors.append(f"{path.name}: crashed ({rec.get('error', '')[:80]})")
                    continue
                for f in REQUIRED:
                    if f not in rec:
                        errors.append(f"{path.name}: missing field {f}")

                # provenance
                commits[rec.get("git_commit", "MISSING")] += 1
                hosts[rec.get("hostname", "MISSING")] += 1
                versions_seen.add(json.dumps(rec.get("versions", {}), sort_keys=True))
                run_names[rec.get("run_name", "MISSING")] += 1
                aff = rec.get("affinity_core")
                if isinstance(aff, list):
                    for c in aff:
                        affinity_cores[c] += 1
                    if not set(aff) <= P_CORES:
                        errors.append(f"{path.name}: affinity {aff} not performance-core only")
                else:
                    errors.append(f"{path.name}: affinity_core missing/not a list")

                status_by_method[method][rec.get("status", "?")] += 1

                # timing
                wt = rec.get("wall_time")
                if isinstance(wt, (int, float)):
                    wall_stats["min"] = min(wall_stats["min"], wt)
                    wall_stats["max"] = max(wall_stats["max"], wt)
                    # a time-limited stochastic job that runs far past budget signals contention
                    if method not in DETERMINISTIC and wt > budget * 1.5:
                        budget_overrun += 1

                # feasibility + anti-fabrication re-verification
                if not rec.get("feasible_final"):
                    infeasible += 1
                    errors.append(f"{path.name}: not feasible_final")
                if rec.get("best_obj") is None:
                    errors.append(f"{path.name}: no best_obj")
                    continue
                if not rec.get("anytime"):
                    errors.append(f"{path.name}: empty anytime trace")
                sched_blob = rec.get("schedule")
                if not sched_blob:
                    errors.append(f"{path.name}: no stored schedule")
                    continue
                inst = _inst(iid)
                machine, start = sched_blob["machine"], sched_blob["start"]
                assignments = tuple(
                    Assignment(g, inst.operation(g).job, machine[g], start[g],
                               inst.operation(g).duration_on(machine[g]))
                    for g in range(inst.num_operations)
                )
                sched = Schedule(inst, assignments)
                if sched.makespan != rec["best_obj"]:
                    errors.append(
                        f"{path.name}: recomputed makespan {sched.makespan} != best_obj {rec['best_obj']}")
                if not check_feasibility(sched).feasible:
                    errors.append(f"{path.name}: stored schedule fails feasibility check")
                # decoder methods accrue repairs on FJSP and none on JSSP
                if method in DECODER_METHODS:
                    if inst.is_flexible and rec["n_repairs"] == 0:
                        warnings.append(f"{path.name}: FJSP decoder run recorded zero repairs")
                    if not inst.is_flexible and rec["n_repairs"] != 0:
                        errors.append(f"{path.name}: JSSP decoder run recorded repairs")

    # ---- calibration from stored values only ---------------------------------------
    calib: dict = {}
    for iid in instances:
        cp = result_path(iid, "cpsat", seeds[0], args.run)
        if cp.exists():
            rec = json.loads(cp.read_text(encoding="utf-8"))
            r = ref.get(iid, {})
            if r.get("proven_optimal") == "True" and rec.get("best_obj") is not None:
                bks = int(r["BKS"])
                proved = rec["status"] == "OPTIMAL"
                calib[iid] = {"cpsat": rec["best_obj"], "bks": bks, "status": rec["status"],
                              "proved_within_budget": proved}
                if rec["best_obj"] < bks:
                    errors.append(f"cpsat {iid}: incumbent {rec['best_obj']} below optimum {bks}")
                if proved and rec["best_obj"] != bks:
                    errors.append(f"cpsat {iid}: proved {rec['best_obj']} != optimum {bks}")
        # tabu must beat the dispatching bank (best over seeds vs the single deterministic run)
        tabu_files = [result_path(iid, "tabu", s, args.run) for s in seeds]
        disp_file = result_path(iid, "dispatching", seeds[0], args.run)
        if all(f.exists() for f in tabu_files) and disp_file.exists():
            tabu_best = min(json.loads(f.read_text())["best_obj"] for f in tabu_files)
            disp = json.loads(disp_file.read_text())["best_obj"]
            if tabu_best > disp:
                errors.append(f"tabu {iid} ({tabu_best}) did not beat dispatching ({disp})")

    report = {
        "run": args.run,
        "expected_jobs": len(expected),
        "present_files": len(present),
        "n_checked": n_checked,
        "n_missing": len(missing),
        "n_extra": len(extra),
        "missing_sample": missing[:20],
        "extra_sample": extra[:20],
        "n_crashed": len(crashed_jobs),
        "n_infeasible": infeasible,
        "provenance": {
            "git_commits": dict(commits),
            "hostnames": dict(hosts),
            "distinct_version_sets": len(versions_seen),
            "run_names": dict(run_names),
            "affinity_cores_used": dict(sorted(affinity_cores.items())),
        },
        "timing": {
            "wall_min": None if wall_stats["min"] == float("inf") else round(wall_stats["min"], 2),
            "wall_max": round(wall_stats["max"], 2),
            "budget_s": budget,
            "stochastic_jobs_over_1p5x_budget": budget_overrun,
        },
        "status_by_method": {m: dict(c) for m, c in status_by_method.items()},
        "calibration_instances": len(calib),
        "calibration": calib,
        "n_warnings": len(warnings),
        "warnings_sample": warnings[:20],
        "n_errors": len(errors),
        "errors_sample": errors[:50],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"checked {n_checked}/{len(expected)} expected; missing={len(missing)} "
          f"extra={len(extra)} crashed={len(crashed_jobs)} infeasible={infeasible} "
          f"errors={len(errors)} warnings={len(warnings)} -> {OUT}")
    for e in errors[:20]:
        print("  ERROR:", e)
    if errors:
        raise SystemExit(f"full verification FAILED with {len(errors)} error(s)")
    print("FULL RUN VERIFIED: complete, crash-free, schedules re-verified, "
          "performance-core affinity, calibration held")


if __name__ == "__main__":
    main()
