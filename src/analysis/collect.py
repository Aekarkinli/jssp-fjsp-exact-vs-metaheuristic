"""Collect every raw v2 result file into tidy derived tables.

Reads ``results/raw/<stage>/*.json`` for all stages of the study, joins the reference
values, extracts the objective reached at each wall-clock and each evaluation checkpoint
from the anytime trace, and writes tidy tables to ``results/derived/``. Every downstream
analysis reads only these tables, never the raw files.

    uv run python -m src.analysis.collect
"""
from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path("results/raw")
DERIVED = Path("results/derived")
TIME_CHECKPOINTS = (1, 10, 60, 300)
EVAL_CHECKPOINTS = (1_000, 10_000, 100_000)


# --------------------------------------------------------------------------- helpers
def reference_table() -> dict[str, dict]:
    with Path("data/reference_values.csv").open(encoding="utf-8") as fh:
        return {r["id"]: r for r in csv.DictReader(fh)}


def best_at_time(anytime, t: float):
    best = None
    for p in anytime:
        if p[1] is not None and p[0] <= t:
            best = p[1] if best is None else min(best, p[1])
    return best


def bound_at_time(anytime, t: float):
    best = None
    for p in anytime:
        if len(p) > 2 and p[2] is not None and p[0] <= t:
            best = p[2] if best is None else max(best, p[2])
    return best


def best_at_evals(anytime, e: int):
    """Best objective within e decoder evaluations, or None if the trace carries no index."""
    best, seen = None, False
    for p in anytime:
        if len(p) < 4 or p[3] is None:
            continue
        seen = True
        if p[1] is not None and p[3] <= e:
            best = p[1] if best is None else min(best, p[1])
    return best if seen else None


def time_to_reach(anytime, target: float):
    """First time at which the trace attains an objective at least as good as target."""
    for p in anytime:
        if p[1] is not None and p[1] <= target + 1e-9:
            return float(p[0])
    return None


# --------------------------------------------------------------------------- per file
def _row(path_str: str) -> tuple[dict, list]:
    r = json.loads(Path(path_str).read_text(encoding="utf-8"))
    anytime = r.get("anytime") or []
    settings = r.get("settings") or {}
    extra = r.get("extra") or {}
    row = {
        "stage": r.get("run_name") or Path(path_str).parent.name,
        "instance_id": r.get("instance_id"),
        "family": r.get("family"),
        "type": r.get("type"),
        "n_jobs": r.get("n_jobs"),
        "n_machines": r.get("n_machines"),
        "n_op": r.get("n_op"),
        "method": r.get("method"),
        "variant": r.get("variant"),
        "seed": r.get("seed"),
        "budget_s": r.get("budget_s"),
        "status": r.get("status"),
        "best_obj": r.get("best_obj"),
        "best_bound": r.get("best_bound"),
        "rel_gap": r.get("rel_gap"),
        "time_to_first": r.get("time_to_first"),
        "time_to_best": r.get("time_to_best"),
        "n_decoder_calls": r.get("n_decoder_calls"),
        "n_repairs": r.get("n_repairs"),
        "n_rejected": r.get("n_rejected"),
        "n_restarts": r.get("n_restarts"),
        "feasible_final": r.get("feasible_final"),
        "crashed": bool(r.get("crashed")),
        "wall_time": r.get("wall_time"),
        "peak_mem_bytes": r.get("peak_mem_bytes"),
        "mapping": settings.get("mapping"),
        "pop_policy": settings.get("pop_policy"),
        "pop_size": settings.get("pop_size"),
        "seeder": extra.get("seeder"),
        "seed_objective": extra.get("seed_objective"),
        "seed_time_s": extra.get("seed_time_s"),
        "best_rule": extra.get("best_rule"),
        "n_trace": len(anytime),
    }
    for cp in TIME_CHECKPOINTS:
        row[f"obj_at_{cp}s"] = best_at_time(anytime, cp)
        row[f"bound_at_{cp}s"] = bound_at_time(anytime, cp)
    for ec in EVAL_CHECKPOINTS:
        row[f"obj_at_{ec}e"] = best_at_evals(anytime, ec)
    trace = [
        (row["instance_id"], row["method"], row["seed"], row["budget_s"],
         float(p[0]), None if p[1] is None else float(p[1]),
         None if len(p) < 3 or p[2] is None else float(p[2]),
         None if len(p) < 4 or p[3] is None else int(p[3]))
        for p in anytime
    ]
    return row, trace


def collect_stage(stage: str, workers: int = 8) -> tuple[pd.DataFrame, pd.DataFrame]:
    files = sorted(str(p) for p in (RAW / stage).glob("*.json"))
    if not files:
        raise SystemExit(f"no result files under results/raw/{stage}")
    with mp.Pool(processes=workers) as pool:
        out = pool.map(_row, files, chunksize=64)
    rows = [o[0] for o in out]
    traces = [t for o in out for t in o[1]]
    df = pd.DataFrame(rows)
    tr = pd.DataFrame(traces, columns=["instance_id", "method", "seed", "budget_s",
                                       "t", "obj", "bound", "evals"])
    ref = reference_table()
    df["bks"] = df["instance_id"].map(lambda i: float(ref[i]["BKS"]) if i in ref else np.nan)
    df["lb"] = df["instance_id"].map(lambda i: float(ref[i]["LB"]) if i in ref else np.nan)
    df["bks_proven"] = df["instance_id"].map(
        lambda i: ref.get(i, {}).get("proven_optimal") == "True")
    df["rpd_bks"] = (df["best_obj"] - df["bks"]) / df["bks"] * 100
    df["gap_to_lb"] = (df["best_obj"] - df["lb"]) / df["lb"] * 100
    for cp in TIME_CHECKPOINTS:
        df[f"rpd_at_{cp}s"] = (df[f"obj_at_{cp}s"] - df["bks"]) / df["bks"] * 100
    for ec in EVAL_CHECKPOINTS:
        df[f"rpd_at_{ec}e"] = (df[f"obj_at_{ec}e"] - df["bks"]) / df["bks"] * 100
    df["evals_per_s"] = np.where(
        (df["wall_time"].fillna(0) > 0) & (df["n_decoder_calls"].fillna(0) > 0),
        df["n_decoder_calls"] / df["wall_time"], np.nan)
    return df, tr


# --------------------------------------------------------------------------- instances
def instance_features() -> pd.DataFrame:
    """Structural features of every instance, computed from the parsed instance files."""
    from src.io.instance_sources import FJSP_INSTANCES, JSSP_INSTANCES
    from src.io.loaders import load_fjsp_file, load_jssp_file

    data = Path("data/instances")
    ref = reference_table()
    rows = []
    for spec, loader, kind in (
        [(s, load_jssp_file, "jssp") for s in JSSP_INSTANCES]
        + [(s, load_fjsp_file, "fjsp") for s in FJSP_INSTANCES]
    ):
        inst = loader(data / spec.local, spec.id, spec.family)
        ops = inst.operations
        elig = np.array([len(o.modes) for o in ops], dtype=float)
        durs = np.array([d for o in ops for _, d in o.modes], dtype=float)
        meta = ref.get(spec.id, {})
        rows.append({
            "instance_id": spec.id,
            "family": spec.family,
            "type": inst.problem_type,
            "n_jobs": inst.num_jobs,
            "n_machines": inst.num_machines,
            "n_op": inst.num_operations,
            "mean_eligible": float(elig.mean()),
            "flex_ratio": float(elig.mean() / inst.num_machines),
            "frac_flexible_ops": float((elig > 1).mean()),
            "mean_duration": float(durs.mean()),
            "cv_duration": float(durs.std() / durs.mean()),
            "bks": float(meta["BKS"]) if meta else np.nan,
            "lb": float(meta["LB"]) if meta else np.nan,
            "bks_proven": meta.get("proven_optimal") == "True",
            "bks_source": meta.get("BKS_source", ""),
            "lb_source": meta.get("LB_source", ""),
            "kind": kind,
        })
    return pd.DataFrame(rows).sort_values(["type", "family", "n_op", "instance_id"])


# --------------------------------------------------------------------------- CEC stage
def collect_cec() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, errs = [], []
    for p in sorted((RAW / "cec").glob("*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        rows.append({k: d[k] for k in (
            "suite", "function", "function_official", "function_name", "dim", "method",
            "runs", "budget_evaluations", "pop_size", "f_optimum", "mean_error",
            "median_error", "std_error", "best_error", "worst_error", "wall_s")})
        for i, e in enumerate(d["errors"]):
            errs.append((d["function"], d["function_official"], d["dim"], d["method"], i, float(e)))
    return (pd.DataFrame(rows),
            pd.DataFrame(errs, columns=["function", "function_official", "dim",
                                        "method", "run", "error"]))


# --------------------------------------------------------------------------- driver
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--stages", default="full,hybrid,reference,ablation_legacy,"
                                        "sens_pop20,sens_pop100,sens_recommended")
    args = ap.parse_args()
    DERIVED.mkdir(parents=True, exist_ok=True)

    feats = instance_features()
    feats.to_parquet(DERIVED / "instances.parquet")
    feats.to_csv(DERIVED / "instances.csv", index=False)
    print(f"instances: {len(feats)} ({(feats.type == 'JSSP').sum()} JSSP, "
          f"{(feats.type == 'FJSP').sum()} FJSP)")

    for stage in args.stages.split(","):
        df, tr = collect_stage(stage, args.workers)
        df.to_parquet(DERIVED / f"{stage}_runs.parquet")
        tr.to_parquet(DERIVED / f"{stage}_traces.parquet")
        print(f"{stage}: {len(df)} runs, {len(tr)} trace points, "
              f"{df.method.nunique()} methods, {df.instance_id.nunique()} instances, "
              f"crashed={int(df.crashed.sum())}, infeasible="
              f"{int((~df.feasible_final.fillna(False)).sum())}")

    cec, cec_err = collect_cec()
    cec.to_parquet(DERIVED / "cec_summary.parquet")
    cec_err.to_parquet(DERIVED / "cec_errors.parquet")
    print(f"cec: {len(cec)} (function, method) cells, {cec.method.nunique()} methods, "
          f"{cec.function.nunique()} functions, {int(cec.runs.max())} runs each")


if __name__ == "__main__":
    main()
