"""Supporting analyses: hybrid accounting, proof model, flexibility, transfer, robustness.

Each function reads only the derived tables and writes a derived table of its own.

    uv run python -m src.analysis.extras
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

from src.analysis.panel import BUDGETS, DECODED, HYBRIDS, PANEL, THRESHOLDS
from src.analysis.stats import holm, load_runs, median_matrix, rank_biserial

DERIVED = Path("results/derived")


# ------------------------------------------------------------------ hybrid
def hybrid_analysis() -> pd.DataFrame:
    """Every hybrid variant against pure CP-SAT and against the tabu search at the same
    total budget, paired over instances. The seeding cost is inside the budget, so the
    comparison is between processes that consumed the same wall-clock time."""
    runs = load_runs("full")
    hyb = pd.read_parquet(DERIVED / "hybrid_runs.parquet")
    hyb = hyb[hyb["feasible_final"].fillna(False)]
    rows = []
    for B in BUDGETS:
        pure = median_matrix(runs, f"rpd_at_{B}s", ["cpsat", "tabu"])
        for v in HYBRIDS:
            sub = hyb[(hyb["method"] == v) & (hyb["budget_s"] == B)]
            med = sub.groupby("instance_id")["rpd_bks"].median()
            joined = pure.join(med.rename("hybrid"), how="inner").dropna()
            for ref in ("cpsat", "tabu"):
                d = joined["hybrid"].to_numpy(float) - joined[ref].to_numpy(float)
                nz = d[d != 0]
                p = float(wilcoxon(nz, alternative="two-sided")[1]) if nz.size else 1.0
                r, n_eff, n_tie = rank_biserial(d)
                rows.append({
                    "variant": v, "budget_s": B, "reference": ref,
                    "median_hybrid": float(joined["hybrid"].median()),
                    "median_reference": float(joined[ref].median()),
                    "median_difference": float(np.median(d)),
                    "n_better": int(np.sum(d < 0)), "n_worse": int(np.sum(d > 0)),
                    "n_equal": n_tie, "n_instances": int(len(d)),
                    "p_raw": p, "rank_biserial": r, "n_effective": n_eff,
                })
    out = pd.DataFrame(rows)
    out["p_holm"] = holm(out["p_raw"].to_numpy(float))
    out.to_csv(DERIVED / "hybrid_vs_pure.csv", index=False)

    # seeding cost of every variant, and the oracle premium over the oracle-free variants
    cost = (hyb.groupby(["method", "budget_s"])
            .agg(seed_time_median=("seed_time_s", "median"),
                 seed_time_max=("seed_time_s", "max"),
                 rpd_median=("rpd_bks", "median"),
                 seed_rpd=("seed_objective", "median"))
            .reset_index())
    cost.to_csv(DERIVED / "hybrid_cost.csv", index=False)
    return out


# ------------------------------------------------------------------ proof model
def proof_model() -> dict:
    """Logistic model of exact proof success at the final budget on instance structure.

    Proof is an instance-level property in this run: every instance is either proved by all
    twenty seeds or by none, so the model is fitted on one binary response per instance and
    the seed replicates are not treated as independent trials.
    """
    import statsmodels.formula.api as smf

    runs = pd.read_parquet(DERIVED / "full_runs.parquet")
    ex = runs[runs["method"] == "cpsat"].copy()
    ex["proved"] = (ex["status"] == "OPTIMAL") & (ex["wall_time"] <= 300)
    agg = ex.groupby("instance_id")["proved"].agg(["sum", "size"]).reset_index()
    inst = pd.read_parquet(DERIVED / "instances.parquet")
    df = agg.merge(inst, on="instance_id")
    df["log_n_op"] = np.log(df["n_op"])
    df["proof_rate"] = df["sum"] / df["size"]
    df["proved"] = (df["proof_rate"] >= 0.5).astype(int)
    model = smf.logit("proved ~ log_n_op + n_machines + flex_ratio + cv_duration",
                      data=df).fit(disp=False)
    out = {
        "params": {k: float(v) for k, v in model.params.items()},
        "odds_ratio": {k: float(np.exp(v)) for k, v in model.params.items()},
        "pvalues": {k: float(v) for k, v in model.pvalues.items()},
        "conf_int": {k: [float(a), float(b)]
                     for k, (a, b) in model.conf_int().T.to_dict("list").items()},
        "n_instances": int(len(df)),
        "n_proved": int(df["proved"].sum()),
        "pseudo_r2": float(model.prsquared),
        "llr_pvalue": float(model.llr_pvalue),
        "all_or_nothing": bool(((df["proof_rate"] == 0) | (df["proof_rate"] == 1)).all()),
    }
    # coefficient stability: refit on 2000 bootstrap resamples of the instances and report
    # the share of resamples in which each coefficient keeps the sign of the point estimate
    rng = np.random.default_rng(20260817)
    keep = {k: 0 for k in out["params"]}
    n_ok = 0
    for _ in range(2000):
        take = df.iloc[rng.integers(0, len(df), len(df))]
        if take["proved"].nunique() < 2:
            continue
        try:
            fit = smf.logit("proved ~ log_n_op + n_machines + flex_ratio + cv_duration",
                            data=take).fit(disp=False, maxiter=200)
        except Exception:  # noqa: BLE001 - a separated resample contributes nothing
            continue
        n_ok += 1
        for k in keep:
            if np.sign(fit.params.get(k, 0.0)) == np.sign(out["params"][k]):
                keep[k] += 1
    out["sign_stability"] = {k: v / max(1, n_ok) for k, v in keep.items()}
    out["bootstrap_fits"] = n_ok
    df["proof_rate"] = df["sum"] / df["size"]
    df[["instance_id", "type", "family", "n_op", "n_machines", "flex_ratio",
        "cv_duration", "proof_rate"]].to_csv(DERIVED / "proof_by_instance.csv", index=False)
    (DERIVED / "proof_model.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out


# ------------------------------------------------------------------ flexibility
def flexibility_analysis() -> pd.DataFrame:
    """Matched Hurink instances at three flexibility levels, same base problems."""
    runs = load_runs("full")
    inst = pd.read_parquet(DERIVED / "instances.parquet")
    hur = inst[inst["family"].str.startswith("hurink_")].copy()
    hur["base"] = hur["instance_id"].str.split("_").str[1]
    hur["level"] = hur["family"].str.replace("hurink_", "", regex=False)
    sub = runs[runs["instance_id"].isin(hur["instance_id"])]
    med = (sub.groupby(["instance_id", "method"])["rpd_at_300s"].median().reset_index()
           .merge(hur[["instance_id", "base", "level", "flex_ratio"]], on="instance_id"))
    ex = runs[(runs["method"] == "cpsat") & runs["instance_id"].isin(hur["instance_id"])].copy()
    ex["proved"] = (ex["status"] == "OPTIMAL") & (ex["wall_time"] <= 300)
    proof = (ex.groupby("instance_id")["proved"].mean().reset_index()
             .merge(hur[["instance_id", "base", "level", "flex_ratio"]], on="instance_id"))
    med.to_csv(DERIVED / "flexibility_rpd.csv", index=False)
    proof.to_csv(DERIVED / "flexibility_proof.csv", index=False)
    return med


# ------------------------------------------------------------------ transfer
def transfer_analysis() -> dict:
    """Continuous-benchmark ranking against scheduling ranking for the decoded optimisers."""
    cec = pd.read_parquet(DERIVED / "cec_summary.parquet")
    piv = cec.pivot(index="function_official", columns="method", values="median_error")
    piv = piv[[m for m in DECODED if m in piv.columns]]
    cec_rank = piv.rank(axis=1).mean().rename("cec_mean_rank")

    runs = load_runs("full")
    sched = median_matrix(runs, "rpd_at_300s", DECODED)
    sched_rank = sched.rank(axis=1).mean().rename("sched_mean_rank")
    sched_e = median_matrix(runs, "rpd_at_100000e", DECODED)
    sched_rank_e = sched_e.rank(axis=1).mean().rename("sched_eval_mean_rank")

    tbl = pd.concat([cec_rank, sched_rank, sched_rank_e], axis=1).reset_index()
    tbl = tbl.rename(columns={"index": "method"})
    tbl["cec_median_error_rank"] = tbl["cec_mean_rank"].rank()
    tbl["sched_rank_position"] = tbl["sched_mean_rank"].rank()
    tbl.to_csv(DERIVED / "transfer_ranks.csv", index=False)

    rho_t = spearmanr(tbl["cec_mean_rank"], tbl["sched_mean_rank"])
    rho_e = spearmanr(tbl["cec_mean_rank"], tbl["sched_eval_mean_rank"])
    n = len(tbl)
    def ci(r):
        z = np.arctanh(r)
        se = 1 / np.sqrt(n - 3)
        return [float(np.tanh(z - 1.96 * se)), float(np.tanh(z + 1.96 * se))]
    out = {
        "n_methods": int(n), "n_functions": int(piv.shape[0]),
        "rho_wallclock": float(rho_t.statistic), "p_wallclock": float(rho_t.pvalue),
        "ci_wallclock": ci(rho_t.statistic),
        "rho_equal_evals": float(rho_e.statistic), "p_equal_evals": float(rho_e.pvalue),
        "ci_equal_evals": ci(rho_e.statistic),
        "cec_best": tbl.sort_values("cec_mean_rank")["method"].tolist(),
        "sched_best": tbl.sort_values("sched_mean_rank")["method"].tolist(),
    }
    # the same question asked of the scheduling gap to the tabu search rather than of ranks
    (DERIVED / "transfer.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out


# ------------------------------------------------------------------ decoder ablation
def ablation_analysis() -> pd.DataFrame:
    """Corrected eligible-set decoding against the label-dependent mapping, matched design."""
    leg = pd.read_parquet(DERIVED / "ablation_legacy_runs.parquet")
    leg = leg[leg["feasible_final"].fillna(False)]
    runs = pd.read_parquet(DERIVED / "full_runs.parquet")
    runs = runs[runs["feasible_final"].fillna(False)]
    keys = leg[["instance_id", "method", "seed"]].drop_duplicates()
    cur = runs.merge(keys, on=["instance_id", "method", "seed"])
    inst = pd.read_parquet(DERIVED / "instances.parquet").set_index("instance_id")
    for d in (leg, cur):
        d["flex_ops"] = d["instance_id"].map(inst["n_op"] * inst["frac_flexible_ops"])
        d["repairs_norm"] = d["n_repairs"] / (d["n_decoder_calls"] * d["flex_ops"])
    a = leg.groupby(["instance_id", "method"])["rpd_bks"].median().rename("legacy")
    b = cur.groupby(["instance_id", "method"])["rpd_bks"].median().rename("eligible")
    rep_a = leg.groupby(["instance_id", "method"])["repairs_norm"].median().rename("legacy_repairs")
    rep_b = cur.groupby(["instance_id", "method"])["repairs_norm"].median().rename("eligible_repairs")
    tbl = pd.concat([a, b, rep_a, rep_b], axis=1).reset_index().dropna()
    tbl["difference"] = tbl["legacy"] - tbl["eligible"]
    d = tbl["difference"].to_numpy(float)
    r, n_eff, n_tie = rank_biserial(d)
    stat = {
        "n_cells": int(len(tbl)), "n_instances": int(tbl["instance_id"].nunique()),
        "n_methods": int(tbl["method"].nunique()),
        "median_legacy": float(tbl["legacy"].median()),
        "median_eligible": float(tbl["eligible"].median()),
        "median_difference": float(np.median(d)),
        "p_value": float(wilcoxon(d[d != 0], alternative="two-sided")[1]),
        "rank_biserial": r, "n_effective": n_eff,
        "n_legacy_worse": int(np.sum(d > 0)),
        "legacy_repairs_per_flexible_operation": float(tbl["legacy_repairs"].median()),
        "legacy_repairs_iqr": float(tbl["legacy_repairs"].quantile(0.75)
                                    - tbl["legacy_repairs"].quantile(0.25)),
        "eligible_repairs_per_flexible_operation": float(tbl["eligible_repairs"].median()),
    }
    tbl.to_csv(DERIVED / "ablation_decoder.csv", index=False)
    (DERIVED / "ablation_decoder.json").write_text(json.dumps(stat, indent=1), encoding="utf-8")
    return tbl


# ------------------------------------------------------------------ population sensitivity
def sensitivity_analysis() -> pd.DataFrame:
    """Population-size policy: 20, 50 (common setting), 100, and author-recommended rules."""
    frames = []
    for stage, label in (("sens_pop20", "20"), ("sens_pop100", "100"),
                         ("sens_recommended", "recommended")):
        df = pd.read_parquet(DERIVED / f"{stage}_runs.parquet")
        df = df[df["feasible_final"].fillna(False)]
        df["setting"] = label
        frames.append(df[["instance_id", "method", "seed", "setting", "rpd_bks", "pop_size"]])
    sens = pd.concat(frames, ignore_index=True)
    ref = pd.read_parquet(DERIVED / "full_runs.parquet")
    ref = ref[ref["feasible_final"].fillna(False)]
    ref = ref[ref["instance_id"].isin(sens["instance_id"].unique())
              & ref["method"].isin(sens["method"].unique())
              & ref["seed"].isin(sens["seed"].unique())]
    base = ref[["instance_id", "method", "seed", "rpd_at_60s"]].rename(
        columns={"rpd_at_60s": "rpd_bks"})
    base["setting"] = "50"
    base["pop_size"] = 50
    sens = pd.concat([sens, base], ignore_index=True)
    med = sens.groupby(["method", "setting", "instance_id"])["rpd_bks"].median().reset_index()
    piv = med.pivot_table(index=["method", "instance_id"], columns="setting", values="rpd_bks")
    rows = []
    for setting in ("20", "100", "recommended"):
        d = (piv[setting] - piv["50"]).dropna().to_numpy(float)
        nz = d[d != 0]
        rows.append({
            "setting": setting, "n_cells": int(d.size),
            "median_difference_vs_common": float(np.median(d)),
            "p_value": float(wilcoxon(nz, alternative="two-sided")[1]) if nz.size else 1.0,
            "rank_biserial": rank_biserial(d)[0],
            "n_better": int(np.sum(d < 0)), "n_worse": int(np.sum(d > 0)),
        })
    per_method = piv.groupby("method").median().reset_index()
    per_method.to_csv(DERIVED / "sensitivity_by_method.csv", index=False)
    out = pd.DataFrame(rows)
    out.to_csv(DERIVED / "sensitivity_population.csv", index=False)
    return out


# ------------------------------------------------------------------ anytime / profiles
def profiles() -> None:
    """Performance profile on the objective and target-attainment curves over time.

    The profile follows the original definition on a strictly positive performance measure,
    namely the makespan itself. Using a deviation instead would divide by zero on the
    instances where the best method reaches the best-known value exactly.
    """
    runs = load_runs("full")
    mat = median_matrix(runs, "obj_at_300s", PANEL)
    best = mat.min(axis=1)
    ratio = mat.div(best, axis=0)
    taus = np.unique(np.concatenate([[1.0], np.linspace(1.0, ratio.to_numpy().max(), 400)]))
    prof = pd.DataFrame({m: [(ratio[m] <= t).mean() for t in taus] for m in mat.columns},
                        index=taus)
    prof.index.name = "tau"
    prof.to_csv(DERIVED / "performance_profile.csv")

    # target attainment: fraction of instances at or below a deviation target over time
    rows = []
    for B in BUDGETS:
        m = median_matrix(runs, f"rpd_at_{B}s", PANEL)
        for target in (0.0, 1.0, 2.0, 5.0, 10.0):
            for meth in m.columns:
                rows.append({"budget_s": B, "target_rpd": target, "method": meth,
                             "fraction": float((m[meth] <= target + 1e-9).mean())})
    pd.DataFrame(rows).to_csv(DERIVED / "target_attainment.csv", index=False)


# ------------------------------------------------------------------ reference run
def anytime_curves(n_grid: int = 60) -> pd.DataFrame:
    """Median deviation of every method over a logarithmic wall-clock grid.

    The trace of every run is stepped onto a common grid, converted to a deviation from
    the best-known solution, and summarised across seeds and instances, so one curve per
    method shows how the panel behaves at every budget rather than at four checkpoints.
    """
    traces = pd.read_parquet(DERIVED / "full_traces.parquet")
    runs = pd.read_parquet(DERIVED / "full_runs.parquet")
    bks = runs.drop_duplicates("instance_id").set_index("instance_id")["bks"]
    grid = np.unique(np.concatenate([[0.01], np.logspace(-2, np.log10(300), n_grid)]))
    rows = []
    for (inst, meth, seed), g in traces.groupby(["instance_id", "method", "seed"], sort=False):
        g = g.sort_values("t")
        t, o = g["t"].to_numpy(float), g["obj"].to_numpy(float)
        keep = ~np.isnan(o)
        t, o = t[keep], np.minimum.accumulate(o[keep])
        if t.size == 0:
            continue
        pos = np.searchsorted(t, grid, side="right") - 1
        val = np.where(pos >= 0, o[np.clip(pos, 0, None)], np.nan)
        b = float(bks.get(inst, np.nan))
        rows.append(pd.DataFrame({"instance_id": inst, "method": meth, "seed": seed,
                                  "t": grid, "rpd": (val - b) / b * 100}))
    long = pd.concat(rows, ignore_index=True)
    per_inst = long.groupby(["method", "instance_id", "t"])["rpd"].median().reset_index()
    curve = (per_inst.groupby(["method", "t"])["rpd"]
             .agg(median="median", q1=lambda s: s.quantile(0.25),
                  q3=lambda s: s.quantile(0.75), coverage=lambda s: s.notna().mean())
             .reset_index())
    curve.to_csv(DERIVED / "anytime_curves.csv", index=False)
    return curve


def multithread_reference() -> pd.DataFrame:
    """The multi-thread exact reference, reported separately from the fair comparison."""
    ref = pd.read_parquet(DERIVED / "reference_runs.parquet")
    runs = load_runs("full")
    single = (runs[runs["method"] == "cpsat"].groupby("instance_id")
              .agg(single_rpd=("rpd_bks", "median"),
                   single_status=("status", lambda s: (s == "OPTIMAL").mean())).reset_index())
    best_heur = (runs[runs["method"] == "tabu"].groupby("instance_id")["rpd_bks"]
                 .median().rename("tabu_rpd").reset_index())
    tbl = (ref[["instance_id", "best_obj", "best_bound", "rel_gap", "status", "rpd_bks",
                "wall_time", "bks", "lb"]]
           .merge(single, on="instance_id").merge(best_heur, on="instance_id"))
    tbl.to_csv(DERIVED / "reference_multithread.csv", index=False)
    return tbl


def environment() -> dict:
    """Software and hardware description of the timed runs, read from the result files."""
    import json as _json
    import platform

    sample = _json.loads(next((Path("results/raw/full").glob("*.json"))).read_text(
        encoding="utf-8"))
    cfg = __import__("yaml").safe_load(Path("config/full.yaml").read_text(encoding="utf-8"))
    out = {
        "python": sample["python"],
        "versions": sample["versions"],
        "git_commit": sample["git_commit"],
        "cpu_id": sample["hw_cpu"],
        "cpu_model": cfg["execution"]["timed_affinity"]["detected_machine"],
        "workers": cfg["execution"]["timed_affinity"]["workers"],
        "os": platform.platform(),
    }
    (DERIVED / "environment.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out


def reference_provenance() -> pd.DataFrame:
    """Per-instance best-known value, lower bound, certified status and source."""
    import csv as _csv

    with Path("data/reference_values.csv").open(encoding="utf-8") as fh:
        ref = pd.DataFrame(list(_csv.DictReader(fh)))
    inst = pd.read_parquet(DERIVED / "instances.parquet")
    tbl = inst[["instance_id", "type", "n_jobs", "n_machines", "n_op"]].merge(
        ref.rename(columns={"id": "instance_id"}), on="instance_id", how="left")
    tbl.to_csv(DERIVED / "reference_provenance.csv", index=False)
    return tbl


def main() -> None:
    print("environment:", json.dumps(environment()["versions"]))
    print("reference provenance:", len(reference_provenance()))
    print("hybrid:", len(hybrid_analysis()))
    print("proof model:", json.dumps(proof_model()["params"], indent=1))
    print("flexibility:", len(flexibility_analysis()))
    print("transfer:", json.dumps(transfer_analysis(), indent=1))
    print("ablation:", len(ablation_analysis()))
    print("sensitivity:\n", sensitivity_analysis())
    profiles()
    print("anytime curves:", len(anytime_curves()))
    print("reference:", len(multithread_reference()))


if __name__ == "__main__":
    main()
