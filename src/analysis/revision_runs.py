"""Analysis of the five targeted experiments.

Reads ``results/derived/rev_*_runs.parquet`` (written by ``src.analysis.collect --stages``)
and the main-panel tables, and writes ``results/derived/rev1_*``. Each experiment is compared
with the main study on the same instances and the same seeds.

- ``hurink``: matched flexibility study, proof and gap within base problems
- ``representation``: the decoded optimisers under the operation-based permutation
- ``memetic``: the decoded optimisers with critical-block descent in the decoder
- ``variants``: improved variants against their base methods, on scheduling and on CEC2017
- ``large``: the complete panel on the large-scale extension

    .venv/Scripts/python.exe tools/ecore.py -m src.analysis.revision_runs [--only NAME]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from src.analysis.revision import _pair_rows, _write, charged_runs

DERIVED = Path("results/derived")
BUDGETS = (1, 10, 60, 300)
LEVELS = ["sdata", "edata", "rdata", "vdata"]


def _runs(stage: str) -> pd.DataFrame:
    return pd.read_parquet(DERIVED / f"{stage}_runs.parquet")


def _main_subset(instances, seeds) -> pd.DataFrame:
    runs = pd.read_parquet(DERIVED / "full_runs.parquet")
    traces = pd.read_parquet(DERIVED / "full_traces.parquet")
    runs, _ = charged_runs(runs, traces)
    return runs[runs.instance_id.isin(instances) & runs.seed.isin(seeds)]


def _scores(runs: pd.DataFrame, B: int = 300) -> pd.DataFrame:
    return runs.groupby(["instance_id", "method"])[f"rpd_at_{B}s"].median().unstack()


def _holm(p) -> np.ndarray:
    """Holm-adjusted p values over one family of tests; missing values stay missing."""
    from statsmodels.stats.multitest import multipletests

    p = np.asarray(p, dtype=float)
    out = np.full(p.shape, np.nan)
    ok = np.isfinite(p)
    if ok.any():
        out[ok] = multipletests(p[ok], method="holm")[1]
    return out


def _mean_ranks(scores: pd.DataFrame) -> pd.Series:
    return scores.rank(axis=1, method="average").mean()


def _page_test(mat: np.ndarray) -> tuple[float, float]:
    """Page's L test for an increasing trend across ordered columns, blocks in rows.

    Ranks are taken within each row with ties averaged, and the normal approximation of
    Page (1963) gives a one-sided p value for the alternative that the response grows with
    the column order.
    """
    n, k = mat.shape
    ranks = np.apply_along_axis(stats.rankdata, 1, mat)
    L = float((ranks.sum(axis=0) * np.arange(1, k + 1)).sum())
    mean = n * k * (k + 1) ** 2 / 4
    var = n * k ** 2 * (k + 1) * (k ** 2 - 1) / 144
    z = (L - mean) / np.sqrt(var)
    return z, float(stats.norm.sf(z))


# ------------------------------------------------------------------ E5 Hurink
def hurink() -> dict:
    from src.run.runner import _load

    from src.run.revision_sets import HURINK_EXCLUDED

    runs = _runs("rev_hurink")
    runs = runs.assign(level=runs.instance_id.str.split("_").str[1],
                       base=runs.instance_id.str.split("_", n=2).str[2])
    runs = runs[~runs.base.isin(HURINK_EXCLUDED)]
    rows = []
    for iid, g in runs.groupby("instance_id"):
        ex = g[g.method == "cpsat"]
        tb = g[g.method == "tabu"]
        inst = _load(iid)
        proved = ((ex.status == "OPTIMAL") & (ex.wall_time <= 300)).mean() >= 0.5
        tproof = (float(ex.loc[ex.status == "OPTIMAL", "wall_time"].median())
                  if proved else np.inf)
        inc, bnd = ex.best_obj.median(), ex.best_bound.median()
        rows.append({
            "instance_id": iid, "base": g.base.iloc[0], "level": g.level.iloc[0],
            "n_op": inst.num_operations,
            "mean_eligible": float(np.mean([len(o.modes) for o in inst.operations])),
            "proved": int(proved), "time_to_proof": tproof,
            "gap": float((inc - bnd) / inc * 100) if np.isfinite(bnd) else np.nan,
            "exact_rpd": float(ex.rpd_bks.median()), "tabu_rpd": float(tb.rpd_bks.median()),
            "tabu_minus_exact": float(tb.best_obj.median() - inc) / inc * 100,
        })
    df = pd.DataFrame(rows)
    df["level_idx"] = df.level.map({lv: i for i, lv in enumerate(LEVELS)})
    df.to_csv(DERIVED / "rev1_hurink_instances.csv", index=False)

    out = {"n_bases": int(df.base.nunique()), "n_instances": int(len(df)),
           "excluded_bases": sorted(HURINK_EXCLUDED)}
    by = df.groupby("level")
    out["proved_by_level"] = {lv: int(by.get_group(lv).proved.sum()) for lv in LEVELS}
    out["mean_eligible_by_level"] = {lv: float(by.get_group(lv).mean_eligible.mean())
                                     for lv in LEVELS}
    out["median_gap_by_level"] = {lv: float(by.get_group(lv).gap.median()) for lv in LEVELS}
    out["median_tabu_rpd_by_level"] = {lv: float(by.get_group(lv).tabu_rpd.median())
                                       for lv in LEVELS}
    out["median_exact_rpd_by_level"] = {lv: float(by.get_group(lv).exact_rpd.median())
                                        for lv in LEVELS}
    out["median_tabu_minus_exact_by_level"] = {
        lv: float(by.get_group(lv).tabu_minus_exact.median()) for lv in LEVELS}

    # at the highest flexibility: how often each method reaches the best-known value, and how
    # often the source certifies that value as optimal
    ref = pd.read_csv("data/reference_values_rev1.csv").set_index("id")
    v = df[df.level == "vdata"]
    out["vdata_exact_at_bks"] = int((v.exact_rpd <= 1e-9).sum())
    out["vdata_tabu_at_bks"] = int((v.tabu_rpd <= 1e-9).sum())
    out["vdata_certified"] = int((ref.loc[v.instance_id, "proven_optimal"].astype(str)
                                  == "True").sum())
    out["vdata_n"] = int(len(v))

    # within-base comparison of proof between the classical and the most flexible variant
    piv = df.pivot(index="base", columns="level", values="proved")[LEVELS]
    out["bases_proved_s_not_v"] = int(((piv.sdata == 1) & (piv.vdata == 0)).sum())
    out["bases_proved_v_not_s"] = int(((piv.sdata == 0) & (piv.vdata == 1)).sum())
    b, c = out["bases_proved_s_not_v"], out["bases_proved_v_not_s"]
    out["mcnemar_exact_p"] = float(stats.binomtest(b, b + c, 0.5).pvalue) if b + c else 1.0
    out["bases_monotone_nonincreasing"] = int((piv.diff(axis=1).iloc[:, 1:] <= 0)
                                              .all(axis=1).sum())

    # conditional logistic regression, stratified by base problem
    try:
        from statsmodels.discrete.conditional_models import ConditionalLogit

        for name, x in (("level", df.level_idx.to_numpy(float)),
                        ("log_eligible", np.log(df.mean_eligible.to_numpy(float)))):
            m = ConditionalLogit(df.proved.to_numpy(float), x[:, None],
                                 groups=df.base.to_numpy()).fit(disp=False)
            ci = m.conf_int()[0]
            out[f"clogit_{name}"] = {"coef": float(m.params[0]), "se": float(m.bse[0]),
                                     "ci95": [float(ci[0]), float(ci[1])],
                                     "p": float(m.pvalues[0]),
                                     "odds_ratio": float(np.exp(m.params[0]))}
        out["clogit_informative_bases"] = int((piv.nunique(axis=1) > 1).sum())
        # the same model with one indicator per flexible dataset, sdata as reference, so
        # that the ordered coding of the levels is not assumed to be linear
        dummies = np.column_stack([(df.level == lv).to_numpy(float) for lv in LEVELS[1:]])
        m = ConditionalLogit(df.proved.to_numpy(float), dummies,
                             groups=df.base.to_numpy()).fit(disp=False)
        ci = m.conf_int()
        out["clogit_categorical"] = {
            lv: {"coef": float(m.params[i]), "odds_ratio": float(np.exp(m.params[i])),
                 "ci95_or": [float(np.exp(ci[i][0])), float(np.exp(ci[i][1]))],
                 "p": float(m.pvalues[i])} for i, lv in enumerate(LEVELS[1:])}
    except Exception as exc:  # noqa: BLE001
        out["clogit_error"] = repr(exc)

    # Page trend tests, base problems as blocks, levels in order of flexibility
    for col, label in (("time_to_proof", "time_to_proof"), ("gap", "gap"),
                       ("tabu_rpd", "tabu_rpd"), ("tabu_minus_exact", "tabu_minus_exact")):
        mat = df.pivot(index="base", columns="level", values=col)[LEVELS]
        mat = mat.replace(np.inf, 1e9).dropna().to_numpy(float)
        z, p = _page_test(mat)
        out[f"page_{label}"] = {"z": z, "p_increasing": p, "n_blocks": int(mat.shape[0])}
    # one family: the two conditional models, the McNemar test and the four trend tests
    fam = [("clogit_level", "p"), ("clogit_log_eligible", "p"), ("mcnemar", None),
           ("page_time_to_proof", "p_increasing"), ("page_gap", "p_increasing"),
           ("page_tabu_rpd", "p_increasing"), ("page_tabu_minus_exact", "p_increasing")]
    raw = [out["mcnemar_exact_p"] if k == "mcnemar" else out.get(k, {}).get(f, np.nan)
           for k, f in fam]
    out["holm"] = {k: float(v) for (k, _), v in zip(fam, _holm(raw))}
    out["holm_family_size"] = int(np.isfinite(raw).sum())
    _write("rev1_hurink.json", out)
    return out


# ------------------------------------------------------------------ E1 representation
def representation() -> dict:
    from src.run.revision_sets import DECODED_METHODS, REVISION_SEEDS, revision_subset

    sub = revision_subset()
    rev = _runs("rev_opperm")
    seeds = sorted(rev.seed.unique())[:REVISION_SEEDS]
    base = _main_subset(sub, seeds)
    out = {"n_instances": len(sub), "seeds": len(seeds)}
    rows = []
    for B in BUDGETS:
        a = _scores(base[base.method.isin(DECODED_METHODS)], B)
        b = _scores(rev, B)
        for m in DECODED_METHODS:
            s = pd.concat([a[m].rename("rk"), b[m].rename("op")], axis=1).dropna()
            diff = (s.op - s.rk).to_numpy(float)
            nz = diff[diff != 0]
            rows.append({"budget": B, "method": m, "median_randomkey": float(s.rk.median()),
                         "median_opperm": float(s.op.median()),
                         "median_paired_diff": float(np.median(diff)),
                         "opperm_better": int((diff < 0).sum()), "n": int(len(s)),
                         "p": float(stats.wilcoxon(nz).pvalue) if nz.size >= 5 else np.nan})
    tab = pd.DataFrame(rows)
    # family: the twelve optimisers at one budget
    tab["p_holm"] = tab.groupby("budget").p.transform(lambda s: _holm(s.to_numpy()))
    tab.to_csv(DERIVED / "rev1_representation.csv", index=False)
    ref = _scores(base[base.method.isin(["tabu", "cpsat", "sa"])], 300)
    a, b = _scores(base[base.method.isin(DECODED_METHODS)], 300), _scores(rev, 300)
    # every summary is taken over the instances present under both representations
    common = a.index.intersection(b.index)
    a, b, ref = a.loc[common], b.loc[common], ref.loc[common]
    out["n_instances_compared"] = int(len(common))
    ra, rb = _mean_ranks(a[DECODED_METHODS]), _mean_ranks(b[DECODED_METHODS])
    rho, p = stats.spearmanr(ra, rb)
    out["rank_rho_between_representations"] = float(rho)
    out["rank_p_between_representations"] = float(p)
    out["tabu_median"] = float(ref["tabu"].median())
    out["sa_median"] = float(ref["sa"].median())
    out["cpsat_median"] = float(ref["cpsat"].median())
    best_rk = a[DECODED_METHODS].median().idxmin()
    best_op = b[DECODED_METHODS].median().idxmin()
    out["best_randomkey"] = [best_rk, float(a[best_rk].median())]
    out["best_opperm"] = [best_op, float(b[best_op].median())]
    out["worst_opperm"] = float(b[DECODED_METHODS].median().max())
    out["n_methods_improved"] = int((tab[tab.budget == 300].median_paired_diff < 0).sum())
    out["median_of_method_differences"] = float(tab[tab.budget == 300].median_paired_diff
                                                .median())
    # does any decoded optimiser under either representation beat the tabu search
    both = pd.concat([a[DECODED_METHODS].add_suffix("_rk"), b[DECODED_METHODS].add_suffix("_op")],
                     axis=1)
    out["instances_any_decoded_beats_tabu"] = int(
        (both.min(axis=1) < ref["tabu"].reindex(both.index)).sum())
    out["best_opperm_vs_tabu_ratio"] = float(b[best_op].median() / max(ref["tabu"].median(),
                                                                       1e-9))
    # the best optimiser under the second representation against the two graph methods
    for refm in ("tabu", "sa"):
        d = (b[best_op] - ref[refm]).to_numpy(float)
        nz = d[d != 0]
        out[f"best_opperm_vs_{refm}"] = {
            "median_paired_diff": float(np.median(d)), "better": int((d < 0).sum()),
            "worse": int((d > 0).sum()),
            "p": float(stats.wilcoxon(nz).pvalue) if nz.size >= 5 else float("nan")}
    # by problem class and budget: the tabu search against the best optimiser under each
    # representation, chosen by its median over the whole subset at the final budget
    typ = base.drop_duplicates("instance_id").set_index("instance_id")["type"]
    by_class = {}
    for B in BUDGETS:
        aB = _scores(base[base.method.isin(DECODED_METHODS)], B).loc[common]
        bB = _scores(rev, B).loc[common]
        rB = _scores(base[base.method.isin(["tabu", "sa"])], B).loc[common]
        for ty in ("JSSP", "FJSP"):
            ids = [i for i in common if typ.get(i) == ty]
            d = (bB.loc[ids, best_op] - rB.loc[ids, "tabu"]).to_numpy(float)
            nz = d[d != 0]
            by_class[f"{ty}_{B}"] = {
                "n": len(ids), "tabu": float(rB.loc[ids, "tabu"].median()),
                "best_opperm": float(bB.loc[ids, best_op].median()),
                "best_randomkey": float(aB.loc[ids, best_rk].median()),
                "best_opperm_better": int((d < 0).sum()),
                "best_opperm_worse": int((d > 0).sum()),
                "p": float(stats.wilcoxon(nz).pvalue) if nz.size >= 5 else float("nan")}
    # family: both classes at every budget; and the two comparisons with the graph methods
    keys = list(by_class)
    for k, v in zip(keys, _holm([by_class[k]["p"] for k in keys])):
        by_class[k]["p_holm"] = float(v)
    out["by_class"] = by_class
    adj = _holm([out["best_opperm_vs_tabu"]["p"], out["best_opperm_vs_sa"]["p"]])
    out["best_opperm_vs_tabu"]["p_holm"] = float(adj[0])
    out["best_opperm_vs_sa"]["p_holm"] = float(adj[1])
    out["n_opperm_below_sa_median"] = int((b[DECODED_METHODS].median()
                                           < ref["sa"].median()).sum())
    out["n_randomkey_below_sa_median"] = int((a[DECODED_METHODS].median()
                                              < ref["sa"].median()).sum())
    # transfer under the second representation
    tr = pd.read_csv(DERIVED / "transfer_ranks.csv").set_index("method")
    common = [m for m in DECODED_METHODS if m in tr.index]
    rho2, p2 = stats.spearmanr(tr.loc[common, "cec_mean_rank"], rb[common])
    out["transfer_rho_opperm"] = float(rho2)
    out["transfer_p_opperm"] = float(p2)
    rho3, p3 = stats.spearmanr(tr.loc[common, "cec_mean_rank"], ra[common])
    out["transfer_rho_randomkey_subset"] = float(rho3)
    out["transfer_p_randomkey_subset"] = float(p3)
    pd.DataFrame({"randomkey": ra, "opperm": rb}).to_csv(DERIVED / "rev1_representation_ranks.csv")
    _write("rev1_representation.json", out)
    return out


# ------------------------------------------------------------------ E2 memetic
def memetic() -> dict:
    from src.run.revision_sets import MEMETIC_METHODS, REVISION_SEEDS, revision_subset

    sub = revision_subset()
    mem = _runs("rev_memetic")
    seeds = sorted(mem.seed.unique())[:REVISION_SEEDS]
    base = _main_subset(sub, seeds)
    out = {}
    rows = []
    for B in BUDGETS:
        a = _scores(base, B)
        b = _scores(mem, B)
        for m in MEMETIC_METHODS:
            s = pd.concat([a[m].rename("plain"), b[m].rename("ls"), a["tabu"].rename("tabu")],
                          axis=1).dropna()
            diff = (s.ls - s.plain).to_numpy(float)
            nz = diff[diff != 0]
            gap_plain = (s.plain - s.tabu).median()
            gap_ls = (s.ls - s.tabu).median()
            rows.append({"budget": B, "method": m, "median_plain": float(s.plain.median()),
                         "median_memetic": float(s.ls.median()),
                         "median_tabu": float(s.tabu.median()),
                         "median_paired_diff": float(np.median(diff)),
                         "memetic_better": int((diff < 0).sum()), "n": int(len(s)),
                         "p": float(stats.wilcoxon(nz).pvalue) if nz.size >= 5 else np.nan,
                         "gap_to_tabu_plain": float(gap_plain),
                         "gap_to_tabu_memetic": float(gap_ls),
                         "share_gap_closed": float(1 - gap_ls / gap_plain) if gap_plain else np.nan,
                         "memetic_beats_tabu": int((s.ls < s.tabu).sum()),
                         "memetic_ties_tabu": int((s.ls == s.tabu).sum())})
    tab = pd.DataFrame(rows)
    # family: the four optimisers at one budget
    tab["p_holm"] = tab.groupby("budget").p.transform(lambda s: _holm(s.to_numpy()))
    tab.to_csv(DERIVED / "rev1_memetic.csv", index=False)
    fin = tab[tab.budget == 300]
    out["share_gap_closed_min"] = float(fin.share_gap_closed.min())
    out["share_gap_closed_max"] = float(fin.share_gap_closed.max())
    out["best_memetic"] = [fin.loc[fin.median_memetic.idxmin(), "method"],
                           float(fin.median_memetic.min())]
    out["tabu_median"] = float(fin.median_tabu.iloc[0])
    evals = mem.groupby("method").n_decoder_calls.median()
    plain_evals = base[base.method.isin(MEMETIC_METHODS)].groupby("method").n_decoder_calls.median()
    out["evaluation_ratio"] = {m: float(evals[m] / plain_evals[m]) for m in MEMETIC_METHODS}
    _write("rev1_memetic.json", out)
    return out


# ------------------------------------------------------------------ E3 variants
VARIANT_BASE = {"jade": "de", "shade": "de", "clpso": "pso", "igwo": "gwo"}


def variants() -> dict:
    from src.analysis.revision import variant_pairs

    pairs = variant_pairs()  # scheduling side, on the revision subset
    cec = pd.read_parquet(DERIVED / "cec_summary.parquet")
    med = cec.pivot(index="function", columns="method", values="median_error")
    out = {}
    lineage = [("de", "jade"), ("jade", "shade"), ("shade", "lshade"), ("de", "lshade"),
               ("pso", "clpso"), ("gwo", "igwo"), ("de", "imode"), ("de", "mde"),
               ("ga", "brkga")]
    rows = []
    for base, var in lineage:
        if base not in med or var not in med:
            continue
        d = (med[var] - med[base]).to_numpy(float)
        sc = pairs[(pairs["set"] == "subset") & (pairs.base == base) & (pairs.variant == var)
                   & (pairs.budget == "rpd_at_300s")]
        rows.append({"base": base, "variant": var,
                     "cec_variant_better": int((d < 0).sum()),
                     "cec_variant_worse": int((d > 0).sum()), "cec_functions": int(len(d)),
                     "cec_p": float(stats.wilcoxon(d[d != 0]).pvalue) if (d != 0).sum() >= 5
                     else np.nan,
                     "sched_variant_better": int(sc.variant_better.iloc[0]) if len(sc) else -1,
                     "sched_variant_worse": int(sc.variant_worse.iloc[0]) if len(sc) else -1,
                     "sched_n": int(sc.n.iloc[0]) if len(sc) else -1,
                     "sched_median_diff": float(sc.median_paired_diff.iloc[0]) if len(sc)
                     else np.nan,
                     "sched_rbc": float(sc.rbc.iloc[0]) if len(sc) else np.nan,
                     "sched_p": float(sc.p.iloc[0]) if len(sc) else np.nan})
    tab = pd.DataFrame(rows)
    # families: the nine lineage pairs on scheduling, and the same pairs on CEC2017
    tab["sched_p_holm"] = _holm(tab.sched_p.to_numpy())
    tab["cec_p_holm"] = _holm(tab.cec_p.to_numpy())
    tab.to_csv(DERIVED / "rev1_variant_lineage.csv", index=False)
    # agreement in sign: an improvement on CEC that is also an improvement on scheduling
    tab["cec_improves"] = tab.cec_variant_better > tab.cec_variant_worse
    tab["sched_improves"] = tab.sched_median_diff < 0
    out["n_pairs"] = int(len(tab))
    out["n_cec_improves"] = int(tab.cec_improves.sum())
    out["n_sched_improves"] = int(tab.sched_improves.sum())
    out["n_agree"] = int((tab.cec_improves == tab.sched_improves).sum())
    out["n_sched_sig_holm"] = int((tab.sched_p_holm < 0.05).sum())
    out["n_cec_sig_holm"] = int((tab.cec_p_holm < 0.05).sum())

    # one second: each variant against its base, one family of eight pairs, with the
    # evaluation throughput and the comparison at a fixed number of evaluations
    from src.run.revision_sets import REVISION_SEEDS, revision_subset

    one_pairs = [("de", "jade"), ("de", "shade"), ("de", "lshade"), ("pso", "clpso"),
                 ("gwo", "igwo"), ("de", "imode"), ("de", "mde"), ("ga", "brkga")]
    one = pairs[(pairs["set"] == "subset") & (pairs.budget == "rpd_at_1s")]
    one = one[[(b_, v_) in set(one_pairs) for b_, v_ in zip(one.base, one.variant)]].copy()
    one["p_holm"] = _holm(one.p.to_numpy())
    one.to_csv(DERIVED / "rev1_variants_one_second.csv", index=False)
    out["one_second"] = {"n_pairs": int(len(one)),
                         "n_variant_worse_median": int((one.median_paired_diff > 0).sum()),
                         "n_variant_worse_holm": int(((one.median_paired_diff > 0)
                                                      & (one.p_holm < 0.05)).sum())}
    sub = revision_subset()
    var = _runs("rev_variants")
    seeds = sorted(var.seed.unique())[:REVISION_SEEDS]
    allr = pd.concat([var, _main_subset(sub, seeds)])
    eps = allr.groupby("method").evals_per_s.median()
    ratios = [float(eps[v_] / eps[b_]) for b_, v_ in one_pairs]
    fixed = allr.groupby(["instance_id", "method"]).rpd_at_1000e.median().unstack()
    worse_fixed = sum(int(np.median((fixed[v_] - fixed[b_]).dropna()) > 0)
                      for b_, v_ in one_pairs)
    out["one_second"].update({"throughput_ratio_min": min(ratios),
                              "throughput_ratio_max": max(ratios),
                              "n_variant_worse_at_1000_evaluations": worse_fixed})
    _write("rev1_variants.json", out)
    return out


# ------------------------------------------------------------------ E4 large-scale
def large() -> dict:
    from src.analysis.necessity import _assign_classes, _cell_statistics, trace_index
    from src.run.revision_sets import large_set

    runs = _runs("rev_large")
    traces = pd.read_parquet(DERIVED / "rev_large_traces.parquet")
    runs, traces = charged_runs(runs, traces)
    out = {"n_instances": int(runs.instance_id.nunique())}
    cells = _cell_statistics(runs, trace_index(traces), None, BUDGETS, "tabu", None)
    cls = _assign_classes(cells)
    cls.to_csv(DERIVED / "rev1_large_necessity.csv", index=False)
    out["classes"] = {str(B): cls[cls.budget_s == B]["class"].value_counts().to_dict()
                      for B in BUDGETS}
    sc = _scores(runs, 300)
    ranks = _mean_ranks(sc)
    out["median_rpd"] = sc.median().to_dict()
    out["mean_rank"] = ranks.to_dict()
    ex = runs[runs.method == "cpsat"]
    proved = ex.groupby("instance_id").apply(
        lambda g: ((g.status == "OPTIMAL") & (g.wall_time <= 300)).mean() >= 0.5)
    out["n_proved"] = int(proved.sum())
    gap = ex.assign(g=(ex.best_obj - ex.best_bound) / ex.best_obj * 100).groupby(
        "instance_id").g.median()
    out["exact_gap_median"] = float(gap.median())
    # Friedman and paired tests against the tabu search
    mat = sc.dropna(axis=1)
    out["friedman"] = list(map(float, stats.friedmanchisquare(*[mat[c] for c in mat.columns])))
    rows = []
    for m in mat.columns:
        if m == "tabu":
            continue
        diff = (mat[m] - mat["tabu"]).to_numpy(float)
        nz = diff[diff != 0]
        rows.append({"method": m, "median": float(mat[m].median()),
                     "median_paired_diff": float(np.median(diff)),
                     "better_than_tabu": int((diff < 0).sum()),
                     "p": float(stats.wilcoxon(nz).pvalue) if nz.size >= 5 else np.nan})
    pt = pd.DataFrame(rows).sort_values("median")
    from statsmodels.stats.multitest import multipletests

    ok = pt.p.notna()
    if ok.any():
        pt.loc[ok, "p_holm"] = multipletests(pt.loc[ok, "p"], method="holm")[1]
    pt.to_csv(DERIVED / "rev1_large_pairwise.csv", index=False)
    by_type = {}
    for t, g in runs.groupby("type"):
        s2 = _scores(g, 300)
        by_type[t] = s2.median().to_dict()
    out["median_rpd_by_type"] = by_type
    # per budget: exact against tabu, best decoded optimiser, per problem class
    from src.run.revision_sets import DECODED_METHODS

    budgets = {}
    for B in BUDGETS:
        entry = {}
        for t, g in [("all", runs)] + list(runs.groupby("type")):
            s = _scores(g, B)
            dec = [m for m in DECODED_METHODS if m in s.columns]
            d = (s["tabu"] - s["cpsat"]).to_numpy(float)
            med = s[dec].median()
            heur = [m for m in s.columns if m != "cpsat"]
            entry[t] = {"n": int(len(s)), "tabu": float(s["tabu"].median()),
                        "cpsat": float(s["cpsat"].median()),
                        "sa": float(s["sa"].median()) if "sa" in s else float("nan"),
                        "tabu_better_than_cpsat": int((d < 0).sum()),
                        "cpsat_better_than_tabu": int((d > 0).sum()),
                        "best_decoded": [str(med.idxmin()), float(med.min())],
                        "tabu_best_heuristic": int((s[heur].idxmin(axis=1) == "tabu").sum())}
        budgets[str(B)] = entry
    out["budgets"] = budgets
    # annealing runs that never leave their dispatching start, here and in the main study
    sa = runs[runs.method == "sa"].groupby(["type", "instance_id"]).n_trace.median()
    out["sa_at_start"] = {t: int((v <= 1).sum()) for t, v in sa.groupby(level=0)}
    main = pd.read_parquet(DERIVED / "full_runs.parquet")
    med = main.groupby(["instance_id", "method"]).rpd_at_300s.median().unstack()
    typ = main.groupby("instance_id").type.first()
    at_start = (med["sa"] - med["dispatching"]).abs() < 1e-9
    opt_start = at_start & (med["dispatching"].abs() < 1e-9)
    out["main_sa_at_start"] = {t: int(at_start[typ == t].sum()) for t in ("JSSP", "FJSP")}
    out["main_sa_at_optimal_start"] = {t: int(opt_start[typ == t].sum())
                                       for t in ("JSSP", "FJSP")}
    out["main_n_by_type"] = {t: int((typ == t).sum()) for t in ("JSSP", "FJSP")}
    ex1 = runs[runs.method == "cpsat"].groupby("instance_id").rpd_at_1s.median()
    out["cpsat_no_incumbent_1s"] = int(ex1.isna().sum())
    ex10 = runs[runs.method == "cpsat"].groupby("instance_id").rpd_at_10s.median()
    out["cpsat_no_incumbent_10s"] = int(ex10.isna().sum())
    if "p_holm" in pt:
        dec_rows = pt[pt.method.isin(DECODED_METHODS)]
        out["n_decoded_worse_holm"] = int(((dec_rows.p_holm < 0.05)
                                           & (dec_rows.median_paired_diff > 0)).sum())
        out["n_decoded"] = int(len(dec_rows))
    per = runs.groupby(["instance_id", "method"]).rpd_at_300s.median().unstack()
    per.to_csv(DERIVED / "rev1_large_per_instance.csv")
    _write("rev1_large.json", out)
    return out


ANALYSES = {"hurink": hurink, "representation": representation, "memetic": memetic,
            "variants": variants, "large": large}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    args = ap.parse_args()
    for n in (args.only.split(",") if args.only else ANALYSES):
        print(f"==> {n}")
        try:
            res = ANALYSES[n]()
            print(json.dumps(res, indent=1, default=float)[:4000])
        except FileNotFoundError as exc:
            print(f"skipped {n}: {exc}")


if __name__ == "__main__":
    main()
