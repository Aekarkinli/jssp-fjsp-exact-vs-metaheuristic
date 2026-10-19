"""Supporting analyses of the main comparison and the targeted experiments.

Every function reads the derived tables of the completed runs and writes its own output to
``results/derived/rev1_*``. Nothing here changes a locked main result. Where an analysis
corrects an earlier computation, the correction and its effect are written out explicitly.

- ``charged_runs``: the tabu search and the annealing method build their initial schedule
  with the dispatching bank before their clock starts. The construction time is now charged
  to their trace, using the measured time of the dispatching bank on the same instance.
- ``sensitivity_grid``: the classification under every pair of objective margin and time
  factor, at every budget.
- ``noise_floor``: seed-to-seed dispersion of the reference methods against the objective
  margin.
- ``feasibility_coverage``: share of instances holding a feasible schedule as a function of
  time, per method.
- ``proof_refits``: the proof model with a problem-class indicator, within the flexible
  class alone, and with bias-reduced (Firth) estimation.
- ``variant_pairs``: base methods against their improved variants.

    .venv/Scripts/python.exe tools/ecore.py -m src.analysis.revision [--only NAME]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from src.analysis.necessity import (
    _assign_classes,
    _cell_statistics,
    trace_index,
)
from src.analysis.panel import BUDGETS, THRESHOLDS

DERIVED = Path("results/derived")
CHARGED = ("tabu", "sa")


def _write(name: str, obj) -> None:
    path = DERIVED / name
    if isinstance(obj, pd.DataFrame):
        obj.to_csv(path, index=False)
    else:
        path.write_text(json.dumps(obj, indent=1, default=float), encoding="utf-8")
    print("wrote", path)


# ------------------------------------------------------------------ charged accounting
def construction_time(runs: pd.DataFrame) -> pd.Series:
    """Measured wall-clock time of the dispatching bank per instance."""
    d = runs[runs["method"] == "dispatching"]
    return d.groupby("instance_id")["wall_time"].median()


def charged_runs(runs: pd.DataFrame, traces: pd.DataFrame,
                 budgets=BUDGETS) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Shift the traces of the methods that start from the dispatching bank by its cost."""
    ct = construction_time(runs)
    runs = runs.copy()
    traces = traces.copy()
    mask_t = traces["method"].isin(CHARGED)
    traces.loc[mask_t, "t"] = traces.loc[mask_t, "t"] + traces.loc[mask_t, "instance_id"].map(ct)
    sub = traces[mask_t].sort_values("t")
    for B in budgets:
        best = (sub[sub["t"] <= B].groupby(["instance_id", "method", "seed"])["obj"].min())
        mask_r = runs["method"].isin(CHARGED)
        keys = list(zip(runs.loc[mask_r, "instance_id"], runs.loc[mask_r, "method"],
                        runs.loc[mask_r, "seed"]))
        runs.loc[mask_r, f"obj_at_{B}s"] = [best.get(k, np.nan) for k in keys]
        runs.loc[mask_r, f"rpd_at_{B}s"] = ((runs.loc[mask_r, f"obj_at_{B}s"]
                                             - runs.loc[mask_r, "bks"])
                                            / runs.loc[mask_r, "bks"] * 100)
    mr = runs["method"].isin(CHARGED)
    runs.loc[mr, "time_to_first"] = runs.loc[mr, "instance_id"].map(ct)
    return runs, traces


def _load_main(charged: bool = True):
    runs = pd.read_parquet(DERIVED / "full_runs.parquet")
    traces = pd.read_parquet(DERIVED / "full_traces.parquet")
    if charged:
        runs, traces = charged_runs(runs, traces)
    hybrid = pd.read_parquet(DERIVED / "hybrid_runs.parquet")
    hybrid = hybrid[hybrid["feasible_final"].fillna(False)]
    return runs, traces, hybrid


def charged_effect() -> dict:
    """Classification and 1-second medians with and without charging the construction."""
    out = {}
    raw_runs, raw_tr, hybrid = _load_main(charged=False)
    ch_runs, ch_tr, _ = _load_main(charged=True)
    a = _assign_classes(_cell_statistics(raw_runs, trace_index(raw_tr), hybrid, BUDGETS,
                                         "tabu", "hyb_tabu"))
    b = _assign_classes(_cell_statistics(ch_runs, trace_index(ch_tr), hybrid, BUDGETS,
                                         "tabu", "hyb_tabu"))
    m = a.merge(b, on=["instance_id", "budget_s"], suffixes=("", "_c"))
    changed = m[m["class"] != m["class_c"]]
    out["changed_cells"] = int(len(changed))
    out["changed_by_budget"] = {int(k): int(v) for k, v in
                                changed.groupby("budget_s").size().items()}
    out["changed_list"] = changed[["instance_id", "budget_s", "class", "class_c"]].to_dict(
        "records")
    ct = construction_time(raw_runs)
    out["construction_median_s"] = float(ct.median())
    out["construction_max_s"] = float(ct.max())
    out["construction_max_instance"] = str(ct.idxmax())
    for B in BUDGETS:
        col = f"rpd_at_{B}s"
        for name, r in (("raw", raw_runs), ("charged", ch_runs)):
            med = r[r["method"] == "tabu"].groupby("instance_id")[col].median().median()
            out[f"tabu_median_rpd_{B}s_{name}"] = float(med)
    b.to_csv(DERIVED / "rev1_necessity_charged.csv", index=False)
    _write("rev1_charged_effect.json", out)
    return out


# ------------------------------------------------------------------ sensitivity grid
DELTAS = (0.005, 0.01, 0.02, 0.05)
TAUS = (2.0, 5.0, 10.0, 20.0)
EPSILONS = (0.01, 0.02, 0.05)


def sensitivity_grid() -> pd.DataFrame:
    runs, traces, hybrid = _load_main()
    cells = _cell_statistics(runs, trace_index(traces), hybrid, BUDGETS, "tabu", "hyb_tabu")
    base = _assign_classes(cells, THRESHOLDS)
    rows = []
    for d in DELTAS:
        for t in TAUS:
            thr = {**THRESHOLDS, "delta": d, "tau": t}
            c = _assign_classes(cells, thr)
            for B in BUDGETS:
                cb, bb = c[c["budget_s"] == B], base[base["budget_s"] == B]
                counts = cb["class"].value_counts()
                rows.append({
                    "delta": d, "tau": t, "budget_s": B,
                    "changed": int((cb["class"].to_numpy() != bb["class"].to_numpy()).sum()),
                    **{f"n_{k}": int(counts.get(k, 0)) for k in
                       ("EXACT_SUFFICIENT", "HEURISTIC_USEFUL", "HEURISTIC_NECESSARY",
                        "HYBRID_RECOMMENDED", "INCONCLUSIVE")},
                })
    grid = pd.DataFrame(rows)
    # epsilon only splits the heuristic-necessary class into its two sub-classes
    eps_rows = []
    for e in EPSILONS:
        c = _assign_classes(cells, {**THRESHOLDS, "epsilon": e})
        for B in BUDGETS:
            cb = c[c["budget_s"] == B]
            eps_rows.append({"epsilon": e, "budget_s": B,
                             "n_strong": int((cb["subclass"] == "3_strong_near_bks").sum()),
                             "n_necessary": int((cb["class"] == "HEURISTIC_NECESSARY").sum())})
    _write("rev1_sensitivity_grid.csv", grid)
    _write("rev1_sensitivity_epsilon.csv", pd.DataFrame(eps_rows))
    # robustness summary: which conclusions hold in every cell of the grid
    summ = {}
    for B in BUDGETS:
        g = grid[grid["budget_s"] == B]
        summ[str(B)] = {k: [int(g[k].min()), int(g[k].max())] for k in
                        ("changed", "n_EXACT_SUFFICIENT", "n_HEURISTIC_NECESSARY",
                         "n_HYBRID_RECOMMENDED", "n_INCONCLUSIVE", "n_HEURISTIC_USEFUL")}
    # stability of each instance's class at 300 s across the whole grid
    stable = {}
    for B in BUDGETS:
        labels = []
        for d in DELTAS:
            for t in TAUS:
                c = _assign_classes(cells, {**THRESHOLDS, "delta": d, "tau": t})
                labels.append(c[c["budget_s"] == B].set_index("instance_id")["class"])
        lab = pd.concat(labels, axis=1)
        stable[str(B)] = int((lab.nunique(axis=1) == 1).sum())
    summ["instances_with_one_class_across_grid"] = stable
    # main effect of each threshold on the number of changed cells, averaged over the other
    summ["mean_changed_by_delta"] = grid.groupby("delta")["changed"].mean().to_dict()
    summ["mean_changed_by_tau"] = grid.groupby("tau")["changed"].mean().to_dict()
    summ["mean_changed_by_budget"] = grid.groupby("budget_s")["changed"].mean().to_dict()
    _write("rev1_sensitivity_summary.json", summ)
    return grid


# ------------------------------------------------------------------ noise floor
def noise_floor() -> dict:
    """Seed-to-seed dispersion of the exact solver and the tabu search, in percent.

    For every instance the interquartile range of the twenty final makespans is expressed
    as a percentage of their median. The objective margin of the classification should lie
    above the typical run-to-run noise of the two reference methods, otherwise a one-percent
    difference could arise from the seed alone.
    """
    runs, _, _ = _load_main()
    out = {}
    for B in BUDGETS:
        col = f"obj_at_{B}s"
        for m in ("cpsat", "tabu"):
            g = runs[runs["method"] == m].groupby("instance_id")[col]
            rel = (g.quantile(0.75) - g.quantile(0.25)) / g.median() * 100
            half = rel / 2
            out[f"{m}_{B}s"] = {
                "median_iqr_pct": float(rel.median()),
                "p90_iqr_pct": float(rel.quantile(0.9)),
                "median_half_iqr_pct": float(half.median()),
                "share_iqr_below_delta": float((rel < 100 * THRESHOLDS["delta"]).mean()),
            }
    # absolute makespan represented by one percent on the instances of the panel
    inst = pd.read_parquet(DERIVED / "instances.parquet")
    one_pct = inst["bks"] * 0.01
    out["one_percent_of_bks_median"] = float(one_pct.median())
    out["one_percent_of_bks_range"] = [float(one_pct.min()), float(one_pct.max())]
    mean_dur = inst["mean_duration"]
    out["one_percent_in_mean_processing_times_median"] = float((one_pct / mean_dur).median())
    _write("rev1_noise_floor.json", out)
    return out


# ------------------------------------------------------------------ feasibility coverage
def feasibility_coverage(n_grid: int = 90) -> pd.DataFrame:
    """Share of instances on which a method holds a feasible schedule, against time.

    An instance counts as covered at time t when at least half of the method's runs on it
    hold a feasible schedule by t, which is the same median-seed convention as every other
    per-instance statistic. The share of individual runs is reported alongside.
    """
    runs, _, _ = _load_main()
    grid = np.logspace(-4, np.log10(300), n_grid)
    rows = []
    for m, g in runs.groupby("method"):
        tf = g["time_to_first"].astype(float).fillna(np.inf)
        tf = tf.where(g["feasible_final"].fillna(False) | (tf < np.inf), np.inf)
        for t in grid:
            ok = tf <= t
            per_inst = ok.groupby(g["instance_id"]).mean()
            rows.append({"method": m, "t": float(t),
                         "share_instances": float((per_inst >= 0.5).mean()),
                         "share_runs": float(ok.mean())})
    cov = pd.DataFrame(rows)
    _write("rev1_feasibility_coverage.csv", cov)
    # the exact solver by size band, since its first schedule waits for model construction
    bands = pd.cut(runs["n_op"], [0, 100, 300, 10_000], labels=["<=100", "101-300", ">300"])
    brows = []
    for (m, band), g in runs.assign(band=bands).groupby(["method", "band"], observed=True):
        if m not in ("cpsat", "tabu"):
            continue
        tf = g["time_to_first"].astype(float).fillna(np.inf)
        for t in grid:
            per_inst = (tf <= t).groupby(g["instance_id"]).mean()
            brows.append({"method": m, "band": str(band), "n_instances": int(per_inst.size),
                          "t": float(t), "share_instances": float((per_inst >= 0.5).mean())})
    _write("rev1_feasibility_coverage_bands.csv", pd.DataFrame(brows))
    first = (runs.groupby(["method", "instance_id"])["time_to_first"].median()
             .groupby("method").agg(["median", "max"]).reset_index())
    first["n_instances_no_feasible_at_1s"] = [
        int((runs[(runs.method == m)].groupby("instance_id")["obj_at_1s"]
             .apply(lambda s: s.isna().mean() > 0.5)).sum()) for m in first["method"]]
    _write("rev1_time_to_first.csv", first)
    return cov


# ------------------------------------------------------------------ proof model refits
def _firth(X: np.ndarray, y: np.ndarray, fixed: int | None = None, iters: int = 300,
           tol: float = 1e-9):
    """Bias-reduced logistic regression (Firth, 1993) by modified Newton iterations.

    Returns (coefficients, standard errors, penalised log-likelihood). Finite estimates exist
    even under complete separation, which a binary proof outcome on few instances can show.
    With ``fixed`` the coefficient of that column is held at zero while the penalty is still
    computed from the full design, which is the profile penalised likelihood used for the
    likelihood-ratio test of a single term (as in the logistf implementation).
    """
    n, k = X.shape
    beta = np.zeros(k)
    free = np.array([j for j in range(k) if j != fixed])

    def pll(b):
        p = np.clip(1 / (1 + np.exp(-X @ b)), 1e-12, 1 - 1e-12)
        W = p * (1 - p)
        _, logdet = np.linalg.slogdet(X.T @ (X * W[:, None]))
        return float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)) + 0.5 * logdet)

    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ beta))
        W = p * (1 - p)
        info = X.T @ (X * W[:, None])
        inv = np.linalg.inv(info)
        h = np.einsum("ij,jk,ik->i", X * W[:, None], inv, X)
        U = X.T @ (y - p + h * (0.5 - p))
        step = np.zeros(k)
        step[free] = np.linalg.solve(info[np.ix_(free, free)], U[free])
        cur = pll(beta)
        s = 1.0
        while s > 1e-6 and pll(beta + s * step) < cur - 1e-12:
            s /= 2
        beta = beta + s * step
        if np.max(np.abs(s * step)) < tol:
            break
    p = 1 / (1 + np.exp(-X @ beta))
    W = p * (1 - p)
    se = np.sqrt(np.diag(np.linalg.inv(X.T @ (X * W[:, None]))))
    return beta, se, pll(beta)


def _firth_table(df: pd.DataFrame, terms: list[str]) -> dict:
    """Firth fit with penalised likelihood-ratio p values for every term."""
    X = np.column_stack([np.ones(len(df))] + [df[t].to_numpy(float) for t in terms])
    y = df["proved"].to_numpy(float)
    beta, se, ll = _firth(X, y)
    names = ["Intercept"] + terms
    res = {"n": int(len(df)), "n_proved": int(y.sum()), "terms": {}}
    for j, name in enumerate(names):
        p = np.nan
        if j > 0:
            _, _, llr = _firth(X, y, fixed=j)
            p = float(stats.chi2.sf(max(0.0, 2 * (ll - llr)), 1))
        res["terms"][name] = {"coef": float(beta[j]), "se": float(se[j]),
                              "ci95": [float(beta[j] - 1.96 * se[j]),
                                       float(beta[j] + 1.96 * se[j])],
                              "p_plr": p}
    return res


def proof_refits() -> dict:
    runs = pd.read_parquet(DERIVED / "full_runs.parquet")
    ex = runs[runs["method"] == "cpsat"].copy()
    ex["proved"] = (ex["status"] == "OPTIMAL") & (ex["wall_time"] <= 300)
    agg = ex.groupby("instance_id")["proved"].mean().reset_index()
    inst = pd.read_parquet(DERIVED / "instances.parquet")
    df = agg.merge(inst, on="instance_id")
    df["proved"] = (df["proved"] >= 0.5).astype(int)
    df["log_n_op"] = np.log(df["n_op"])
    df["fjsp"] = (df["type"] == "FJSP").astype(int)
    df["flex_excess"] = df["flex_ratio"] - 1.0
    out = {
        "original_firth": _firth_table(df, ["log_n_op", "n_machines", "flex_ratio",
                                            "cv_duration"]),
        "with_class": _firth_table(df, ["log_n_op", "n_machines", "fjsp", "flex_ratio",
                                        "cv_duration"]),
        "with_class_only": _firth_table(df, ["log_n_op", "fjsp"]),
        "fjsp_only": _firth_table(df[df["fjsp"] == 1], ["log_n_op", "flex_ratio"]),
        "fjsp_only_full": _firth_table(df[df["fjsp"] == 1],
                                       ["log_n_op", "n_machines", "flex_ratio",
                                        "cv_duration"]),
        # the same models with flexibility measured as the mean number of eligible
        # machines per operation, the measure of the benchmark table, which equals one for
        # every job-shop instance
        "with_class_elig": _firth_table(df, ["log_n_op", "n_machines", "fjsp",
                                             "mean_eligible", "cv_duration"]),
        "fjsp_only_elig": _firth_table(df[df["fjsp"] == 1], ["log_n_op", "mean_eligible"]),
    }
    # correlation between flexibility and problem class, the confound to be separated
    out["corr_flex_class"] = float(np.corrcoef(df["flex_ratio"], df["fjsp"])[0, 1])
    out["corr_elig_class"] = float(np.corrcoef(df["mean_eligible"], df["fjsp"])[0, 1])
    out["jssp_flex_ratio_range"] = [float(df.loc[df.fjsp == 0, "flex_ratio"].min()),
                                    float(df.loc[df.fjsp == 0, "flex_ratio"].max())]
    _write("rev1_proof_refits.json", out)
    return out


# ------------------------------------------------------------------ variant pairs
PAIRS_MAIN = [("de", "lshade"), ("de", "imode"), ("de", "mde"), ("ga", "brkga"),
              ("sa", "tabu")]
PAIRS_REVISION = [("de", "jade"), ("jade", "shade"), ("shade", "lshade"), ("de", "shade"),
                  ("pso", "clpso"), ("gwo", "igwo")]
MEMETIC_PAIRS = [("ga", "ga+ls"), ("brkga", "brkga+ls"), ("lshade", "lshade+ls"),
                 ("cmaes", "cmaes+ls")]


def _rank_biserial(diff: np.ndarray) -> float:
    d = diff[diff != 0]
    if d.size == 0:
        return 0.0
    r = stats.rankdata(np.abs(d))
    return float((r[d > 0].sum() - r[d < 0].sum()) / r.sum())


def _pair_rows(scores: pd.DataFrame, pairs, budget_col: str, label: str) -> list[dict]:
    rows = []
    for base, imp in pairs:
        if base not in scores.columns or imp not in scores.columns:
            continue
        s = scores[[base, imp]].dropna()
        diff = (s[imp] - s[base]).to_numpy(float)  # negative: the improved variant is better
        nz = diff[diff != 0]
        p = float(stats.wilcoxon(nz).pvalue) if nz.size >= 5 else float("nan")
        rows.append({"set": label, "budget": budget_col, "base": base, "variant": imp,
                     "n": int(len(s)), "median_base": float(s[base].median()),
                     "median_variant": float(s[imp].median()),
                     "median_paired_diff": float(np.median(diff)),
                     "variant_better": int((diff < 0).sum()),
                     "variant_worse": int((diff > 0).sum()),
                     "rbc": _rank_biserial(diff), "p": p})
    return rows


def variant_pairs() -> pd.DataFrame:
    """Base methods against improved variants on per-instance median deviation.

    Main-panel pairs use all 67 instances and 20 seeds. Pairs that involve a revision run
    use the revision subset with the first ten seeds for both members, so the two are
    summarised over identical runs.
    """
    from src.run.revision_sets import REVISION_SEEDS, revision_subset

    runs, _, _ = _load_main()
    rows = []
    for B in BUDGETS:
        col = f"rpd_at_{B}s"
        sc = runs.groupby(["instance_id", "method"])[col].median().unstack()
        rows += _pair_rows(sc, PAIRS_MAIN, col, "main")
    sub = revision_subset()
    seeds = sorted(runs[runs["method"] == "tabu"]["seed"].unique())[:REVISION_SEEDS]
    base = runs[runs["instance_id"].isin(sub) & runs["seed"].isin(seeds)]
    extra = []
    for stage, rename in (("rev_variants", None), ("rev_memetic", "+ls")):
        p = DERIVED / f"{stage}_runs.parquet"
        if p.exists():
            r = pd.read_parquet(p)
            if rename:
                r = r.assign(method=r["method"] + rename)
            extra.append(r)
    if extra:
        allr = pd.concat([base] + extra, ignore_index=True)
        for B in BUDGETS:
            col = f"rpd_at_{B}s"
            sc = allr.groupby(["instance_id", "method"])[col].median().unstack()
            rows += _pair_rows(sc, PAIRS_MAIN + PAIRS_REVISION, col, "subset")
            rows += _pair_rows(sc, MEMETIC_PAIRS, col, "memetic")
    out = pd.DataFrame(rows)
    _write("rev1_variant_pairs.csv", out)
    return out


def code_versions() -> dict:
    """Code versions recorded by the result files of the further experiments, per stage."""
    import glob

    out = {}
    for stage in ("rev_hurink", "rev_memetic", "rev_variants", "rev_opperm", "rev_large",
                  "cec"):
        seen: dict[str, int] = {}
        for f in glob.glob(f"results/raw/{stage}/*.json"):
            try:
                c = json.loads(Path(f).read_text(encoding="utf-8")).get("git_commit", "")
            except Exception:  # noqa: BLE001
                continue
            if stage == "cec" and not any(f.endswith(f"__{m}.json")
                                          for m in ("jade", "shade", "clpso", "igwo")):
                continue
            seen[str(c)[:10]] = seen.get(str(c)[:10], 0) + 1
        if seen:
            out[stage] = seen
    _write("rev1_code_versions.json", out)
    return out



# ------------------------------------------------------------------ robustness checks
def _base_problem(instance_id: str, family: str) -> str:
    """Hurink variants share their base problem with the classical instance of that name."""
    return instance_id.split("_", 1)[1] if family.startswith("hurink_") else instance_id


def robustness(n_boot: int = 2000, n_perm: int = 200_000) -> dict:
    """Checks on the main panel that use the existing records only.

    - ``best_heuristic``: the cells in which the tabu search has the lowest median makespan
      among the heuristics, counted alone and with ties, since the selection of a single
      best method breaks ties by panel order.
    - ``hn_final``: the heuristic-necessary instances at the final budget with their size.
    - ``transfer``: the rank correlation between CEC2017 and scheduling standing with a
      percentile bootstrap interval that resamples instances and functions, and an exact-size
      permutation test over the twelve optimisers.
    - ``clusters``: the paired comparisons of the tabu search with every other heuristic and
      the transfer correlation when the Hurink variants and the classical instance of one
      base problem are averaged into one unit.
    """
    from statsmodels.stats.multitest import multipletests

    from src.analysis.panel import DECODED, HEURISTICS

    runs, _, _ = _load_main()
    ok = runs[runs["feasible_final"].fillna(False)]
    out = {}

    cells = alone = tied = 0
    for B in BUDGETS:
        m = (ok[ok.method.isin(HEURISTICS)].groupby(["instance_id", "method"])[f"obj_at_{B}s"]
             .median().unstack())
        isb = m.sub(m.min(axis=1), axis=0).abs() <= 1e-9
        k = isb.sum(axis=1)
        cells += len(m)
        alone += int((isb["tabu"] & (k == 1)).sum())
        tied += int((isb["tabu"] & (k > 1)).sum())
    out["best_heuristic"] = {"cells": cells, "tabu_alone": alone, "tabu_tied": tied,
                             "tabu_alone_or_tied": alone + tied}

    nec = pd.read_csv(DERIVED / "necessity.csv")
    fin = nec[(nec.budget_s == 300) & (nec["class"] == "HEURISTIC_NECESSARY")]
    out["hn_final"] = [[str(i), int(n)] for i, n in
                       fin.sort_values("n_op")[["instance_id", "n_op"]].to_numpy()]

    rng = np.random.default_rng(20261018)
    cec = pd.read_parquet(DERIVED / "cec_summary.parquet")
    piv = cec.pivot(index="function_official", columns="method", values="median_error")[DECODED]
    sched = (ok[ok.method.isin(DECODED)].groupby(["instance_id", "method"])["rpd_at_300s"]
             .median().unstack()[DECODED])

    def rho_of(c, s):
        return float(stats.spearmanr(c.rank(axis=1).mean(), s.rank(axis=1).mean()).statistic)

    rho = rho_of(piv, sched)
    boot = []
    for _ in range(n_boot):
        ci = rng.integers(0, len(piv), len(piv))
        si = rng.integers(0, len(sched), len(sched))
        boot.append(rho_of(piv.iloc[ci], sched.iloc[si]))
    cr, sr = piv.rank(axis=1).mean().to_numpy(), sched.rank(axis=1).mean().to_numpy()
    perm = np.array([stats.spearmanr(cr, rng.permutation(sr)).statistic for _ in range(n_perm)])
    out["transfer"] = {"rho": rho, "boot_ci95": [float(np.percentile(boot, 2.5)),
                                                 float(np.percentile(boot, 97.5))],
                       "n_boot": n_boot, "perm_p_two_sided":
                       float((np.abs(perm) >= abs(rho) - 1e-12).mean()), "n_perm": n_perm}

    fam = ok.drop_duplicates("instance_id").set_index("instance_id")["family"]
    base = {i: _base_problem(i, f) for i, f in fam.items()}
    heur = (ok[ok.method.isin(HEURISTICS)].groupby(["instance_id", "method"])["rpd_at_300s"]
            .median().unstack())
    clus = heur.groupby(heur.index.map(base)).mean()
    rows = []
    for m in [h for h in HEURISTICS if h != "tabu"]:
        d = (clus[m] - clus["tabu"]).dropna().to_numpy(float)
        nz = d[d != 0]
        rows.append({"method": m, "tabu_better": int((d > 0).sum()), "n": int(len(d)),
                     "p": float(stats.wilcoxon(nz).pvalue)})
    ct = pd.DataFrame(rows)
    ct["p_holm"] = multipletests(ct.p, method="holm")[1]
    _write("rev1_cluster_pairs.csv", ct)
    sched_c = sched.groupby(sched.index.map(base)).mean()
    out["clusters"] = {"n_units": int(len(clus)), "n_instances": int(len(heur)),
                       "n_significant_holm": int((ct.p_holm < 0.05).sum()),
                       "n_comparisons": int(len(ct)), "max_p_holm": float(ct.p_holm.max()),
                       "transfer_rho": rho_of(piv, sched_c)}
    _write("rev1_robustness.json", out)
    return out


ANALYSES = {
    "versions": code_versions,
    "charged": charged_effect,
    "grid": sensitivity_grid,
    "noise": noise_floor,
    "coverage": feasibility_coverage,
    "proof": proof_refits,
    "pairs": variant_pairs,
    "robustness": robustness,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help=",".join(ANALYSES))
    args = ap.parse_args()
    names = args.only.split(",") if args.only else list(ANALYSES)
    for n in names:
        print(f"==> {n}")
        res = ANALYSES[n]()
        if isinstance(res, dict):
            print(json.dumps(res, indent=1, default=float)[:3000])


if __name__ == "__main__":
    main()
