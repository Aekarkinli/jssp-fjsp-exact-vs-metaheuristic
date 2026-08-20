"""Comparative statistics over the derived run tables.

The per-instance score of a method is the median over seeds of its relative percentage
deviation from the best-known solution at a stated budget, so every method contributes one
value per instance and all tests are paired over the 67 instances. The module produces the
descriptive summary, the Friedman omnibus with a critical-difference diagram, Holm-corrected
Wilcoxon signed-rank tests against the strongest problem-specific baseline with matched-pairs
rank-biserial effect sizes and paired superiority proportions, a Bayesian signed-rank
comparison with a region of practical equivalence, the same analyses within strata, the
equal-evaluation repeat, and a mixed model at the level of method families.

    uv run python -m src.analysis.stats
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import friedmanchisquare, norm, spearmanr, wilcoxon  # noqa: E402

from src.analysis.panel import (  # noqa: E402
    DECODED,
    DETERMINISTIC,
    FAMILY,
    HEURISTICS,
    PANEL,
    THRESHOLDS,
    display,
)

DERIVED = Path("results/derived")
FIGURES = Path("paper/figures")
BASELINE = "tabu"


# ------------------------------------------------------------------ matrices
def load_runs(stage: str = "full") -> pd.DataFrame:
    df = pd.read_parquet(DERIVED / f"{stage}_runs.parquet")
    return df[df["feasible_final"].fillna(False)]


def median_matrix(runs: pd.DataFrame, col: str, methods=None) -> pd.DataFrame:
    med = runs.groupby(["instance_id", "method"])[col].median().reset_index()
    mat = med.pivot(index="instance_id", columns="method", values=col)
    if methods is not None:
        mat = mat[[m for m in methods if m in mat.columns]]
    return mat.dropna(axis=0, how="any")


# ------------------------------------------------------------------ effect sizes
def rank_biserial(d: np.ndarray) -> tuple[float, int, int]:
    """Matched-pairs rank-biserial correlation of paired differences d = x - baseline.

    Returns (r, n_effective, n_ties). Positive r means x exceeds the baseline (worse, since
    the score is a deviation). Zero differences are dropped, as in the signed-rank test.
    """
    nz = d[d != 0]
    n_tie = int(d.size - nz.size)
    if nz.size == 0:
        return float("nan"), 0, n_tie
    ranks = pd.Series(np.abs(nz)).rank().to_numpy()
    r_pos, r_neg = ranks[nz > 0].sum(), ranks[nz < 0].sum()
    return float((r_pos - r_neg) / (r_pos + r_neg)), int(nz.size), n_tie


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return float(centre - half), float(centre + half)


def holm(p: np.ndarray) -> np.ndarray:
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for i, idx in enumerate(order):
        running = max(running, (m - i) * p[idx])
        adj[idx] = min(1.0, running)
    return adj


# ------------------------------------------------------------------ descriptive
def describe(runs: pd.DataFrame, col: str, methods=PANEL) -> pd.DataFrame:
    mat = median_matrix(runs, col, methods)
    ranks = mat.rank(axis=1, method="average")
    rows = []
    for m in mat.columns:
        v = mat[m].to_numpy(float)
        sub = runs[runs.method == m]
        rows.append({
            "method": m, "family": FAMILY.get(m, ""),
            "n_instances": int(len(v)),
            "median": float(np.median(v)), "iqr": float(np.percentile(v, 75) - np.percentile(v, 25)),
            "q1": float(np.percentile(v, 25)), "q3": float(np.percentile(v, 75)),
            "mean": float(np.mean(v)), "p90": float(np.percentile(v, 90)),
            "worst": float(np.max(v)),
            "n_at_bks": int(np.sum(v <= 1e-9)),
            "n_below_bks": int(np.sum(v < -1e-9)),
            "mean_rank": float(ranks[m].mean()),
            "median_time_to_best": float(sub["time_to_best"].median()),
            "median_evals_per_s": float(sub["evals_per_s"].median()),
            "seed_iqr_median": float(
                sub.groupby("instance_id")["rpd_bks"]
                .apply(lambda s: s.quantile(0.75) - s.quantile(0.25)).median()),
        })
    return pd.DataFrame(rows).sort_values("mean_rank").reset_index(drop=True)


# ------------------------------------------------------------------ omnibus + CD
def nemenyi_cd(k: int, n: int, alpha: float = 0.05) -> float:
    """Critical difference of the Nemenyi post-hoc test for k methods on n instances."""
    from scipy.stats import studentized_range
    q = float(studentized_range.ppf(1 - alpha, k, np.inf)) / np.sqrt(2.0)
    return float(q * np.sqrt(k * (k + 1) / (6.0 * n)))


def omnibus(mat: pd.DataFrame, alpha: float = 0.05) -> dict:
    """Friedman omnibus test with the Nemenyi critical difference and mean ranks."""
    stat, p = friedmanchisquare(*[mat[c].to_numpy(float) for c in mat.columns])
    ranks = mat.rank(axis=1, method="average")
    k, n = mat.shape[1], mat.shape[0]
    return {"friedman_stat": float(stat), "pvalue": float(p),
            "n_instances": int(n), "n_methods": int(k),
            "cd": nemenyi_cd(k, n, alpha),
            "mean_rank": {c: float(ranks[c].mean()) for c in mat.columns},
            "se_rank": {c: float(ranks[c].std(ddof=1) / np.sqrt(n)) for c in mat.columns}}


# ------------------------------------------------------------------ pairwise
def pairwise_vs(mat: pd.DataFrame, baseline: str = BASELINE,
                alpha: float = THRESHOLDS["alpha"]) -> pd.DataFrame:
    base = mat[baseline].to_numpy(float)
    rows = []
    for m in mat.columns:
        if m == baseline:
            continue
        x = mat[m].to_numpy(float)
        d = x - base
        nz = d[d != 0]
        if nz.size >= 1:
            stat, p = wilcoxon(nz, alternative="two-sided")
        else:
            stat, p = float("nan"), 1.0
        r, n_eff, n_tie = rank_biserial(d)
        wins = int(np.sum(d < 0))
        lo, hi = wilson(wins, int(d.size))
        rows.append({
            "method": m, "baseline": baseline,
            "median": float(np.median(x)), "baseline_median": float(np.median(base)),
            "median_difference": float(np.median(d)),
            "wilcoxon_W": float(stat), "p_raw": float(p),
            "rank_biserial": r, "n_effective": n_eff, "n_ties": n_tie,
            "n_instances": int(d.size), "n_wins": wins,
            "superiority": wins / d.size, "sup_lo": lo, "sup_hi": hi,
        })
    out = pd.DataFrame(rows)
    out["p_holm"] = holm(out["p_raw"].to_numpy(float))
    out["significant"] = (out["p_holm"] < alpha) & (out["rank_biserial"].abs() >= THRESHOLDS["rbc_floor"])
    return out.sort_values("median").reset_index(drop=True)


def all_pairs(mat: pd.DataFrame, alpha: float = THRESHOLDS["alpha"]) -> pd.DataFrame:
    cols = list(mat.columns)
    rows = []
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            d = mat[a].to_numpy(float) - mat[b].to_numpy(float)
            nz = d[d != 0]
            p = wilcoxon(nz, alternative="two-sided")[1] if nz.size else 1.0
            r, n_eff, n_tie = rank_biserial(d)
            rows.append({"a": a, "b": b, "p_raw": float(p), "rank_biserial": r,
                         "n_effective": n_eff, "median_difference": float(np.median(d))})
    out = pd.DataFrame(rows)
    out["p_holm"] = holm(out["p_raw"].to_numpy(float))
    out["significant"] = (out["p_holm"] < alpha) & (out["rank_biserial"].abs() >= THRESHOLDS["rbc_floor"])
    return out


# ------------------------------------------------------------------ Bayesian
def bayes_table(mat: pd.DataFrame, pairs, rope: float = 1.0, nsamples: int = 50_000) -> pd.DataFrame:
    import baycomp
    rows = []
    for a, b in pairs:
        if a not in mat.columns or b not in mat.columns:
            continue
        left, mid, right = baycomp.two_on_multiple(
            -mat[a].to_numpy(float), -mat[b].to_numpy(float), rope=rope, nsamples=nsamples)
        rows.append({"left": a, "right": b, "rope": rope,
                     "p_left": float(left), "p_rope": float(mid), "p_right": float(right)})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ family model
def family_model(mat: pd.DataFrame) -> dict:
    """Mixed model of per-instance deviation with method as a random effect nested in family.

    Pooling raw runs across methods of unequal count would let one family's size drive the
    comparison, so the fixed effect is the family and the method is a random intercept.
    """
    import statsmodels.formula.api as smf
    long = mat.reset_index().melt(id_vars="instance_id", var_name="method", value_name="rpd")
    long["family"] = long["method"].map(FAMILY)
    long = long.dropna(subset=["rpd"])
    md = smf.mixedlm("rpd ~ C(family, Treatment(reference='problem_specific'))",
                     long, groups=long["method"])
    fit = md.fit(reml=True, method="lbfgs")
    out = {"params": {k: float(v) for k, v in fit.params.items()},
           "pvalues": {k: float(v) for k, v in fit.pvalues.items()},
           "n_obs": int(fit.nobs), "n_groups": int(len(long["method"].unique()))}
    member = (long.groupby(["family", "method"])["rpd"].median().reset_index()
              .groupby("family")["rpd"].agg(["count", "median", "min", "max"]).reset_index())
    out["member_medians"] = member.to_dict("records")
    return out


# ------------------------------------------------------------------ driver
def run_all() -> dict:
    runs = load_runs("full")
    DERIVED.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    summary: dict = {}

    # descriptive summary at every wall-clock budget
    frames = []
    for b in (1, 10, 60, 300):
        d = describe(runs, f"rpd_at_{b}s")
        d["budget_s"] = b
        frames.append(d)
    desc = pd.concat(frames, ignore_index=True)
    desc.to_csv(DERIVED / "summary_by_budget.csv", index=False)

    mat = median_matrix(runs, "rpd_at_300s", PANEL)
    mat.to_csv(DERIVED / "matrix_rpd_300s.csv")

    # C1: omnibus + post hoc against the strongest problem-specific baseline
    summary["cd_panel"] = omnibus(mat)
    pv = pairwise_vs(mat, BASELINE)
    pv.to_csv(DERIVED / "pairwise_vs_tabu_300s.csv", index=False)
    all_pairs(mat).to_csv(DERIVED / "all_pairs_300s.csv", index=False)

    # C2: equal decoder-evaluation budget, decoded optimisers only
    mat_e = median_matrix(runs, "rpd_at_100000e", DECODED)
    mat_e.to_csv(DERIVED / "matrix_rpd_100ke.csv")
    summary["cd_decoded_evals"] = omnibus(mat_e)
    mat_t = median_matrix(runs, "rpd_at_300s", DECODED)
    summary["wallclock_vs_eval_rank_rho"] = float(spearmanr(
        mat_t.rank(axis=1).mean(), mat_e.rank(axis=1).mean()).statistic)
    pairwise_vs(mat_e, "cmaes").to_csv(DERIVED / "pairwise_decoded_evals.csv", index=False)

    # strata
    inst = pd.read_parquet(DERIVED / "instances.parquet").set_index("instance_id")
    cut = float(inst["n_op"].median())
    strata = {
        "jssp": runs[runs["type"] == "JSSP"],
        "fjsp": runs[runs["type"] == "FJSP"],
        "large": runs[runs["n_op"] > cut],
        "small": runs[runs["n_op"] <= cut],
    }
    # proof-status strata: instances where the exact solver proved optimality within 300 s
    proved = set(runs[(runs.method == "cpsat") & (runs.status == "OPTIMAL")
                      & (runs.wall_time <= 300)]["instance_id"].unique())
    strata["proved"] = runs[runs.instance_id.isin(proved)]
    strata["unproved"] = runs[~runs.instance_id.isin(proved)]
    strat_rows = []
    for name, sub in strata.items():
        sm = median_matrix(sub, "rpd_at_300s", PANEL)
        if sm.shape[0] < 4:
            continue
        summary[f"cd_{name}"] = omnibus(sm)
        p = pairwise_vs(sm, BASELINE)
        p["stratum"] = name
        p["n_stratum"] = sm.shape[0]
        strat_rows.append(p)
        d = describe(sub, "rpd_at_300s")
        d["stratum"] = name
        d.to_csv(DERIVED / f"summary_{name}.csv", index=False)
    pd.concat(strat_rows, ignore_index=True).to_csv(DERIVED / "pairwise_by_stratum.csv", index=False)
    summary["n_op_median"] = cut
    summary["n_proved_instances"] = len(proved)

    # Bayesian comparison on the contrasts that carry the argument
    pairs = [(BASELINE, m) for m in ("cpsat", "sa", "cmaes", "ga", "mde", "lshade", "imode",
                                     "gwo", "csa", "rime", "brkga")]
    bayes_table(mat, pairs).to_csv(DERIVED / "bayesian_vs_tabu.csv", index=False)

    # family-level mixed model
    fam = family_model(median_matrix(runs, "rpd_at_300s", HEURISTICS))
    (DERIVED / "family_model.json").write_text(json.dumps(fam, indent=1), encoding="utf-8")

    (DERIVED / "stats_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.parse_args()
    out = run_all()
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
