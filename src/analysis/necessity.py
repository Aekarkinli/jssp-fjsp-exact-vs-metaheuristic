"""Solver-aware necessity classification.

For every instance and every budget the role of heuristic search relative to an accessible
exact solver is decided by one deterministic procedure with a fixed priority order. The
thresholds were fixed in a versioned configuration before the run: an objective margin
delta, a time factor tau, a tolerance epsilon to the best-known solution, a level alpha and
an effect-size floor.

Two runs of two different stochastic algorithms that happen to carry the same seed label are
not matched observations, so the comparison of a heuristic distribution with the exact
solver's distribution uses an independent-samples rank test with a stochastic-dominance
effect size. The reference heuristic is fixed in advance rather than chosen as the best of
the panel on the same data, which removes the selection step from the inference. Because one
test is performed per instance within each budget, the p values are controlled by the
Benjamini and Hochberg procedure inside each budget and each test family.

    uv run python -m src.analysis.necessity
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

from src.analysis.panel import BUDGETS, DETERMINISTIC, HEURISTICS, THRESHOLDS

DERIVED = Path("results/derived")
EXACT = "cpsat"
REFERENCE_HEURISTIC = "tabu"      # fixed in advance, not selected on the data
REFERENCE_HYBRID = "hyb_tabu"     # rule-seeded variant fixed in advance
FDR_Q = 0.05


# --------------------------------------------------------------------- trace helpers
def trace_index(traces: pd.DataFrame) -> dict:
    """(instance, method, seed) -> (times, objectives), sorted by time."""
    out = {}
    for key, g in traces.groupby(["instance_id", "method", "seed"], sort=False):
        g = g.sort_values("t")
        out[key] = (g["t"].to_numpy(float), g["obj"].to_numpy(float))
    return out


def first_time_at_or_below(idx, key, target: float, cap: float) -> float:
    tr = idx.get(key)
    if tr is None:
        return np.inf
    t, o = tr
    hit = np.where((o <= target + 1e-9) & (t <= cap + 1e-9))[0]
    return float(t[hit[0]]) if hit.size else np.inf


# --------------------------------------------------------------------- inference
def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """Stochastic dominance of x over y for a minimised score.

    Returns (#(x<y) - #(x>y)) / (n_x n_y), so a positive value means x tends to be lower,
    that is better. Defined for independent samples, unlike a matched-pairs measure.
    """
    if x.size == 0 or y.size == 0:
        return float("nan")
    diff = np.sign(y[None, :] - x[:, None])
    return float(diff.sum() / (x.size * y.size))


def dominance_test(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """One-sided Mann-Whitney U test that x is stochastically smaller than y.

    Returns (p value, Cliff's delta). Deterministic methods contribute a single value, in
    which case no test is possible and the p value is returned as not-a-number.
    """
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if x.size < 2 or y.size < 2:
        return float("nan"), cliffs_delta(x, y)
    if np.all(x == x[0]) and np.all(y == y[0]):
        # two constant samples: the test is undefined, the dominance measure is exact
        return (0.0 if x[0] < y[0] else 1.0), cliffs_delta(x, y)
    p = float(mannwhitneyu(x, y, alternative="less", method="asymptotic").pvalue)
    return p, cliffs_delta(x, y)


def benjamini_hochberg(p: np.ndarray, q: float = FDR_Q) -> np.ndarray:
    """Return the rejection mask of the Benjamini and Hochberg procedure at level q."""
    p = np.asarray(p, dtype=float)
    ok = np.isfinite(p)
    reject = np.zeros(p.shape, dtype=bool)
    idx = np.where(ok)[0]
    if idx.size == 0:
        return reject
    order = idx[np.argsort(p[idx])]
    m = order.size
    thresh = q * (np.arange(1, m + 1) / m)
    passed = p[order] <= thresh
    if passed.any():
        k = np.max(np.where(passed)[0])
        reject[order[: k + 1]] = True
    return reject


# --------------------------------------------------------------------- classification
def _cell_statistics(runs: pd.DataFrame, traces_idx: dict, hybrid: pd.DataFrame,
                     budgets, reference: str, hybrid_variant: str | None) -> pd.DataFrame:
    """One row per instance and budget with every quantity the decision rule needs."""
    ok = runs[runs["feasible_final"].fillna(False)]
    meta = ok.drop_duplicates("instance_id").set_index("instance_id")[
        ["type", "family", "n_op", "n_jobs", "n_machines", "bks", "lb"]]
    rows = []
    for inst in sorted(ok["instance_id"].unique()):
        sub = ok[ok["instance_id"] == inst]
        info = meta.loc[inst]
        bks = float(info["bks"])
        ex = sub[sub["method"] == EXACT].sort_values("seed")
        hyb_i = hybrid[hybrid["instance_id"] == inst] if hybrid is not None else None
        for B in budgets:
            col = f"obj_at_{B}s"
            zE_seeds = np.where(np.isnan(ex[col].to_numpy(float)), np.inf,
                                ex[col].to_numpy(float))
            zE = float(np.median(zE_seeds))
            proof = ((ex["status"] == "OPTIMAL") & (ex["wall_time"] <= B)).to_numpy()
            proven = bool(proof.mean() >= 0.5)
            proof_time = float(ex.loc[proof, "wall_time"].median()) if proven else np.nan

            if reference == "panel":
                best_m, zH, zH_seeds = None, np.inf, None
                for m in HEURISTICS:
                    v = sub[sub["method"] == m][col].dropna().to_numpy(float)
                    if v.size and float(np.median(v)) < zH:
                        zH, best_m, zH_seeds = float(np.median(v)), m, v
            else:
                best_m = reference
                zH_seeds = sub[sub["method"] == reference][col].dropna().to_numpy(float)
                zH = float(np.median(zH_seeds)) if zH_seeds.size else np.inf

            zY, zY_seeds = np.inf, None
            if hyb_i is not None and hybrid_variant is not None:
                hv = hyb_i[(hyb_i["method"] == hybrid_variant) & (hyb_i["budget_s"] == B)]
                arr = hv["best_obj"].dropna().to_numpy(float)
                if arr.size:
                    zY, zY_seeds = float(np.median(arr)), arr

            p_h, d_h = (dominance_test(zH_seeds, zE_seeds)
                        if (zH_seeds is not None and np.isfinite(zE)) else (np.nan, np.nan))
            if zY_seeds is not None and np.isfinite(zE):
                p_ye, d_ye = dominance_test(zY_seeds, zE_seeds)
                p_yh, d_yh = dominance_test(zY_seeds, zH_seeds)
            else:
                p_ye = d_ye = p_yh = d_yh = np.nan

            # time to reach the heuristic's own quality, for the speed condition
            t_h = t_e = np.nan
            if best_m is not None and np.isfinite(zH):
                seeds_h = sub[sub["method"] == best_m]["seed"].to_numpy()
                t_h = float(np.median([
                    first_time_at_or_below(traces_idx, (inst, best_m, int(s)), zH, B)
                    for s in seeds_h]))
                t_e = float(np.median([
                    first_time_at_or_below(traces_idx, (inst, EXACT, int(s)), zH, B)
                    for s in ex["seed"].to_numpy()]))
            t_opt = np.nan
            if proven and best_m is not None:
                seeds_h = sub[sub["method"] == best_m]["seed"].to_numpy()
                t_opt = float(np.median([
                    first_time_at_or_below(traces_idx, (inst, best_m, int(s)), zE, B)
                    for s in seeds_h]))

            rows.append({
                "instance_id": inst, "budget_s": B, "type": info["type"],
                "family": info["family"], "n_op": int(info["n_op"]), "bks": bks,
                "z_exact": zE, "proof_rate": float(proof.mean()), "proven": proven,
                "proof_time": proof_time, "reference_heuristic": best_m,
                "z_heuristic": zH, "z_hybrid": zY,
                "p_heuristic": p_h, "delta_heuristic": d_h,
                "p_hybrid_exact": p_ye, "delta_hybrid_exact": d_ye,
                "p_hybrid_heuristic": p_yh, "delta_hybrid_heuristic": d_yh,
                "t_heuristic": t_h, "t_exact": t_e, "t_optimum": t_opt,
                "rpd_exact": (zE - bks) / bks * 100 if np.isfinite(zE) else np.nan,
                "rpd_heuristic": (zH - bks) / bks * 100 if np.isfinite(zH) else np.nan,
            })
    return pd.DataFrame(rows)


def _assign_classes(cells: pd.DataFrame, thr=THRESHOLDS, q: float = FDR_Q) -> pd.DataFrame:
    """Apply the decision rule after controlling the false discovery rate per budget."""
    delta, tau, eps = thr["delta"], thr["tau"], thr["epsilon"]
    floor = thr["rbc_floor"]
    cells = cells.copy()
    cells["sig_heuristic"] = False
    cells["sig_hybrid"] = False

    for B, block in cells.groupby("budget_s"):
        # only cells where the margin condition holds are tested, so the test family is the
        # set of candidate discoveries at that budget
        cand_h = block.index[(block["z_exact"] - block["z_heuristic"]
                              >= delta * block["z_exact"])
                             & np.isfinite(block["z_exact"])]
        rej = benjamini_hochberg(cells.loc[cand_h, "p_heuristic"].to_numpy(float), q)
        eff = cells.loc[cand_h, "delta_heuristic"].abs().to_numpy(float) >= floor
        cells.loc[cand_h, "sig_heuristic"] = rej & eff

        cand_y = block.index[(block["z_hybrid"] <= (1 - delta) * block[["z_exact",
                                                                       "z_heuristic"]].min(axis=1))
                             & np.isfinite(block["z_hybrid"])]
        rej_e = benjamini_hochberg(cells.loc[cand_y, "p_hybrid_exact"].to_numpy(float), q)
        rej_h = benjamini_hochberg(cells.loc[cand_y, "p_hybrid_heuristic"].to_numpy(float), q)
        eff_y = ((cells.loc[cand_y, "delta_hybrid_exact"].abs().to_numpy(float) >= floor)
                 & (cells.loc[cand_y, "delta_hybrid_heuristic"].abs().to_numpy(float) >= floor))
        cells.loc[cand_y, "sig_hybrid"] = rej_e & rej_h & eff_y

    labels, subs, reasons = [], [], []
    for _, r in cells.iterrows():
        label, sub, reason = "INCONCLUSIVE", "", ""
        if not np.isfinite(r["z_exact"]):
            label, sub, reason = ("HEURISTIC_NECESSARY", "3_no_exact_incumbent",
                                  "no_exact_incumbent")
        elif r["proven"]:
            label = "EXACT_SUFFICIENT"
            fast = (np.isfinite(r["t_optimum"]) and np.isfinite(r["proof_time"])
                    and r["t_optimum"] * tau <= r["proof_time"])
            sub = "1b_heuristic_faster" if fast else "1a_redundant"
        elif r["sig_hybrid"]:
            label, sub = "HYBRID_RECOMMENDED", "4_warm_start"
        elif (r["z_exact"] - r["z_heuristic"] >= r["z_exact"] * 0.01) and r["sig_heuristic"]:
            label = "HEURISTIC_NECESSARY"
            sub = ("3_strong_near_bks" if r["z_heuristic"] <= r["bks"] * (1 + eps)
                   else "3_advantage_only")
        elif abs(r["z_heuristic"] - r["z_exact"]) <= 0.01 * r["z_exact"]:
            if np.isfinite(r["t_heuristic"]) and r["t_heuristic"] * tau <= r["t_exact"]:
                label, sub = "HEURISTIC_USEFUL", "2_faster_to_same_quality"
            else:
                reason = "matched_but_not_faster"
        elif r["z_exact"] - r["z_heuristic"] >= r["z_exact"] * 0.01:
            reason = "advantage_not_significant"
        else:
            reason = "exact_incumbent_better"
        labels.append(label)
        subs.append(sub)
        reasons.append(reason)
    cells["class"] = labels
    cells["subclass"] = subs
    cells["reason"] = reasons
    return cells


def classify(runs, traces_idx, hybrid, budgets=BUDGETS, thr=THRESHOLDS,
             reference: str = REFERENCE_HEURISTIC,
             hybrid_variant: str | None = REFERENCE_HYBRID, q: float = FDR_Q) -> pd.DataFrame:
    cells = _cell_statistics(runs, traces_idx, hybrid, budgets, reference, hybrid_variant)
    return _assign_classes(cells, thr, q)


def _thr(**kw):
    return {**THRESHOLDS, **kw}


def run_all() -> dict:
    runs = pd.read_parquet(DERIVED / "full_runs.parquet")
    traces = pd.read_parquet(DERIVED / "full_traces.parquet")
    idx = trace_index(traces)
    hybrid = pd.read_parquet(DERIVED / "hybrid_runs.parquet")
    hybrid = hybrid[hybrid["feasible_final"].fillna(False)]

    main = classify(runs, idx, hybrid)
    main.to_csv(DERIVED / "necessity.csv", index=False)
    main.to_parquet(DERIVED / "necessity.parquet")

    variants = {
        "delta_strict": dict(thr=_thr(delta=THRESHOLDS["delta_strict"])),
        "tau_low": dict(thr=_thr(tau=THRESHOLDS["tau_low"])),
        "tau_high": dict(thr=_thr(tau=THRESHOLDS["tau_high"])),
        "best_of_panel": dict(reference="panel"),
        "cheap_hybrid": dict(hybrid_variant="hyb_cheap"),
        "no_hybrid": dict(hybrid_variant=None),
        "no_fdr": dict(q=1.0),
    }
    summary = {}
    for name, kw in variants.items():
        t = classify(runs, idx, hybrid, **kw)
        t.to_csv(DERIVED / f"necessity_{name}.csv", index=False)
        m = main.merge(t, on=["instance_id", "budget_s"], suffixes=("", "_v"))
        summary[name] = int((m["class"] != m["class_v"]).sum())
    (DERIVED / "necessity_sensitivity.json").write_text(json.dumps(summary, indent=1),
                                                        encoding="utf-8")
    return {"n_cells": len(main), "changed_under_variants": summary}


def main() -> None:
    argparse.ArgumentParser().parse_args()
    out = run_all()
    t = pd.read_csv(DERIVED / "necessity.csv")
    print(t.groupby(["budget_s", "class"]).size().unstack(fill_value=0))
    print()
    print(t[t["class"] == "INCONCLUSIVE"].groupby(["budget_s", "reason"]).size())
    print()
    print(json.dumps(out["changed_under_variants"], indent=1))


if __name__ == "__main__":
    main()
