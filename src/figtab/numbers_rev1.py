"""Generate paper/generated/numbers_rev1.tex, the macros of the targeted experiments.

Same rules as ``numbers.py``: every number the revised text quotes is one of these macros,
generated from ``results/derived/rev1_*``. A block whose input does not exist yet is skipped,
so the file can be rebuilt as the revision experiments complete.

    .venv/Scripts/python.exe -m src.figtab.numbers_rev1
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.figtab.numbers import Macros

DERIVED = Path("results/derived")
GEN = Path("paper/generated")


def _json(name: str) -> dict:
    return json.loads((DERIVED / name).read_text(encoding="utf-8"))


def _p(x: float) -> str:
    r"""p value with its relation, for use inside math as ``$p\Macro$``."""
    if x is None or not np.isfinite(x):
        return r"=\text{--}"
    return "<10^{-4}" if x < 1e-4 else f"={x:.4f}"


def _pct(x: float) -> float:
    return 100.0 * float(x)


def block_charged(M: Macros) -> None:
    c = _json("rev1_charged_effect.json")
    M.add("RevChargedChanged", int(c["changed_cells"]))
    M.add("RevConstructionMedianMs", 1000 * c["construction_median_s"], 1)
    M.add("RevConstructionMax", c["construction_max_s"], 2)
    M.add("RevConstructionMaxInstance", c["construction_max_instance"])


def block_grid(M: Macros) -> None:
    g = pd.read_csv(DERIVED / "rev1_sensitivity_grid.csv")
    s = _json("rev1_sensitivity_summary.json")
    M.add("RevGridCells", int(len(g) // g.budget_s.nunique()))
    M.add("RevGridDeltaMin", _pct(g.delta.min()), 1)
    M.add("RevGridDeltaMax", _pct(g.delta.max()), 0)
    M.add("RevGridTauMin", int(g.tau.min()))
    M.add("RevGridTauMax", int(g.tau.max()))

    def _list(vals):
        vals = [f"{v:g}" for v in vals]
        return ", ".join(vals[:-1]) + " and " + vals[-1]

    M.add("RevGridDeltaList", _list(sorted(100 * g.delta.unique())))
    M.add("RevGridTauList", _list(sorted(g.tau.unique())))
    for B, tag in ((1, "One"), (10, "Ten"), (60, "Sixty"), (300, "Final")):
        sub = g[g.budget_s == B]
        M.add(f"RevGridMaxChanged{tag}", int(sub.changed.max()))
        # within the moderate band of margins, 0.5 to 2 percent, across every time factor
        mod = sub[sub.delta <= 0.02]
        M.add(f"RevGridModerateMaxChanged{tag}", int(mod.changed.max()))
        M.add(f"RevGridNecMin{tag}", int(sub.n_HEURISTIC_NECESSARY.min()))
        M.add(f"RevGridNecMax{tag}", int(sub.n_HEURISTIC_NECESSARY.max()))
        M.add(f"RevGridModerateNecMin{tag}", int(mod.n_HEURISTIC_NECESSARY.min()))
        M.add(f"RevGridModerateNecMax{tag}", int(mod.n_HEURISTIC_NECESSARY.max()))
        M.add(f"RevGridExactMin{tag}", int(sub.n_EXACT_SUFFICIENT.min()))
        M.add(f"RevGridExactMax{tag}", int(sub.n_EXACT_SUFFICIENT.max()))
        M.add(f"RevGridStable{tag}", int(s["instances_with_one_class_across_grid"][str(B)]))
        tau_only = sub[sub.delta == 0.01]
        M.add(f"RevGridTauOnlyMax{tag}", int(tau_only.changed.max()))
    md = s["mean_changed_by_delta"]
    mt = s["mean_changed_by_tau"]
    M.add("RevGridMeanChangedDeltaFive", float(md["0.05"]), 1)
    M.add("RevGridMeanChangedDeltaHalf", float(md["0.005"]), 1)
    M.add("RevGridMeanChangedTauRange", float(max(mt.values()) - min(mt.values())), 1)
    e = pd.read_csv(DERIVED / "rev1_sensitivity_epsilon.csv")
    fin = e[e.budget_s == 300].set_index("epsilon")
    M.add("RevEpsStrongOne", int(fin.loc[0.01, "n_strong"]))
    M.add("RevEpsStrongFive", int(fin.loc[0.05, "n_strong"]))


def block_noise(M: Macros) -> None:
    n = _json("rev1_noise_floor.json")
    for B, tag in ((1, "One"), (300, "Final")):
        M.add(f"RevNoiseTabuMedianIqr{tag}", n[f"tabu_{B}s"]["median_iqr_pct"], 2)
        M.add(f"RevNoiseTabuPNinetyIqr{tag}", n[f"tabu_{B}s"]["p90_iqr_pct"], 2)
        M.add(f"RevNoiseTabuShareBelow{tag}", _pct(n[f"tabu_{B}s"]["share_iqr_below_delta"]), 0)
        M.add(f"RevNoiseExactMedianIqr{tag}", n[f"cpsat_{B}s"]["median_iqr_pct"], 2)
        M.add(f"RevNoiseExactPNinetyIqr{tag}", n[f"cpsat_{B}s"]["p90_iqr_pct"], 2)
        M.add(f"RevNoiseExactShareBelow{tag}",
              _pct(n[f"cpsat_{B}s"]["share_iqr_below_delta"]), 0)
    M.add("RevOnePercentMedian", n["one_percent_of_bks_median"], 1)
    M.add("RevOnePercentMin", n["one_percent_of_bks_range"][0], 2)
    M.add("RevOnePercentMax", n["one_percent_of_bks_range"][1], 1)
    M.add("RevOnePercentInOps", n["one_percent_in_mean_processing_times_median"], 2)


def block_coverage(M: Macros) -> None:
    f = pd.read_csv(DERIVED / "rev1_time_to_first.csv").set_index("method")
    from src.analysis.panel import DECODED

    M.add("RevFirstExactMedianMs", 1000 * float(f.loc["cpsat", "median"]), 0)
    M.add("RevFirstExactMax", float(f.loc["cpsat", "max"]), 2)
    M.add("RevFirstDecodedMedianMs", 1000 * float(f.loc[list(DECODED), "median"].median()), 1)
    M.add("RevFirstDecodedMax", float(f.loc[list(DECODED), "max"].max()), 2)
    M.add("RevFirstDispatchMedianMs", 1000 * float(f.loc["dispatching", "median"]), 1)
    M.add("RevFirstDispatchMax", float(f.loc["dispatching", "max"]), 2)
    cov = pd.read_csv(DERIVED / "rev1_feasibility_coverage.csv")

    def t_full(m):
        s = cov[(cov.method == m) & (cov.share_instances >= 1.0)].t
        return float(s.min()) if len(s) else float("nan")

    M.add("RevFullCoverageExact", t_full("cpsat"), 2)
    dec = max(t_full(m) for m in DECODED)
    M.add("RevFullCoverageDecoded", dec, 2)
    b = pd.read_csv(DERIVED / "rev1_feasibility_coverage_bands.csv")
    M.add("RevBandSmallMax", 100)   # band edges used by the coverage analysis
    M.add("RevBandLargeMin", 300)
    for band, tag in (("<=100", "Small"), ("101-300", "Medium"), (">300", "Large")):
        s = b[(b.method == "cpsat") & (b.band == band)]
        half = s[s.share_instances >= 0.5].t
        M.add(f"RevExactHalfCoverage{tag}", float(half.min()) if len(half) else float("nan"),
              3)
        M.add(f"RevBandN{tag}", int(s.n_instances.iloc[0]))


def block_proof(M: Macros) -> None:
    p = _json("rev1_proof_refits.json")
    M.add("RevCorrFlexClass", p["corr_flex_class"], 2)
    M.add("RevCorrEligClass", p["corr_elig_class"], 2)
    M.add("RevJsspFlexRatioMin", p["jssp_flex_ratio_range"][0], 2)
    M.add("RevJsspFlexRatioMax", p["jssp_flex_ratio_range"][1], 2)
    for key, tag in (("original_firth", "Orig"), ("with_class", "Class"),
                     ("fjsp_only", "Fjsp"), ("fjsp_only_full", "FjspFull"),
                     ("with_class_only", "ClassOnly"), ("with_class_elig", "ClassElig"),
                     ("fjsp_only_elig", "FjspElig")):
        terms = p[key]["terms"]
        for term, ttag in (("flex_ratio", "Flex"), ("mean_eligible", "Flex"),
                           ("log_n_op", "Size"), ("fjsp", "Cls")):
            if term in terms:
                r = terms[term]
                M.add(f"RevProof{tag}{ttag}Coef", r["coef"], 2)
                M.add(f"RevProof{tag}{ttag}Lo", r["ci95"][0], 2)
                M.add(f"RevProof{tag}{ttag}Hi", r["ci95"][1], 2)
                M.add(f"RevProof{tag}{ttag}P", _p(r["p_plr"]))
        M.add(f"RevProof{tag}N", int(p[key]["n"]))
        M.add(f"RevProof{tag}NProved", int(p[key]["n_proved"]))


def block_pairs(M: Macros) -> None:
    t = pd.read_csv(DERIVED / "rev1_variant_pairs.csv")
    main = t[t["set"] == "main"]
    for (base, var), tag in ((("de", "lshade"), "DeLshade"), (("de", "imode"), "DeImode"),
                             (("de", "mde"), "DeMde"), (("ga", "brkga"), "GaBrkga"),
                             (("sa", "tabu"), "SaTabu")):
        for B, btag in ((1, "One"), (300, "Final")):
            r = main[(main.base == base) & (main.variant == var)
                     & (main.budget == f"rpd_at_{B}s")]
            if r.empty:
                continue
            r = r.iloc[0]
            M.add(f"RevPair{tag}{btag}Diff", float(r.median_paired_diff), 2)
            M.add(f"RevPair{tag}{btag}Rbc", float(r.rbc), 2)
            M.add(f"RevPair{tag}{btag}Better", int(r.variant_better))
            M.add(f"RevPair{tag}{btag}P", _p(float(r.p)))


def block_design(M: Macros) -> None:
    """Design constants of the revision experiments, read from their definitions."""
    from src.io.instance_sources import HURINK_BASES, HURINK_VARIANTS, LARGE_FJSP_IDS
    from src.run.revision_sets import (
        DECODED_METHODS,
        HURINK_EXCLUDED,
        HURINK_SEEDS,
        MEMETIC_METHODS,
        PER_CLASS,
        REVISION_SEEDS,
        VARIANT_METHODS,
        large_set,
        revision_subset,
    )
    from src.run.runner import _load

    M.add("RevSubsetN", len(revision_subset()))
    M.add("RevSubsetPerClass", PER_CLASS)
    M.add("RevSeeds", REVISION_SEEDS)
    M.add("RevDecodedN", len(DECODED_METHODS))
    M.add("RevMemeticN", len(MEMETIC_METHODS))
    M.add("RevVariantsN", len(VARIANT_METHODS))
    M.add("RevHurinkBasesAll", len(HURINK_BASES))
    M.add("RevHurinkBases", len(HURINK_BASES) - len(HURINK_EXCLUDED))
    M.add("RevHurinkLevels", len(HURINK_VARIANTS))
    M.add("RevHurinkInstances", (len(HURINK_BASES) - len(HURINK_EXCLUDED))
          * len(HURINK_VARIANTS))
    M.add("RevHurinkSeeds", HURINK_SEEDS)
    large = large_set()
    insts = [_load(i) for i in large]
    M.add("RevLargeN", len(large))
    M.add("RevLargeJsspN", len(large) - len(LARGE_FJSP_IDS))
    M.add("RevLargeFjspN", len(LARGE_FJSP_IDS))
    jssp = [i for i in insts if not i.is_flexible]
    fjsp = [i for i in insts if i.is_flexible]
    M.add("RevLargeJsspMinOps", min(i.num_operations for i in jssp))
    M.add("RevLargeJsspMaxOps", max(i.num_operations for i in jssp))
    M.add("RevLargeFjspMinOps", min(i.num_operations for i in fjsp))
    M.add("RevLargeFjspMaxOps", max(i.num_operations for i in fjsp))
    flex = [sum(len(o.modes) for o in i.operations) / i.num_operations for i in fjsp]
    M.add("RevLargeFjspMinFlex", min(flex), 1)
    M.add("RevLargeFjspMaxFlex", max(flex), 1)
    M.add("RevLargeMaxMachines", max(i.num_machines for i in fjsp))
    M.add("RevLargeMaxJobs", max(i.num_jobs for i in insts))


def _pz(x: float) -> str:
    return "<10^{-6}" if x < 1e-6 else _p(x)


def block_hurink(M: Macros) -> None:
    h = _json("rev1_hurink.json")
    tags = {"sdata": "S", "edata": "E", "rdata": "R", "vdata": "V"}
    for lv, tg in tags.items():
        M.add(f"RevHurProved{tg}", int(h["proved_by_level"][lv]))
        M.add(f"RevHurElig{tg}", h["mean_eligible_by_level"][lv], 2)
        M.add(f"RevHurGap{tg}", h["median_gap_by_level"][lv], 1)
        M.add(f"RevHurTabuRpd{tg}", h["median_tabu_rpd_by_level"][lv], 2)
        M.add(f"RevHurTabuMinusExact{tg}", h["median_tabu_minus_exact_by_level"][lv], 2)
    M.add("RevHurNBases", int(h["n_bases"]))
    M.add("RevHurVExactAtBks", int(h["vdata_exact_at_bks"]))
    M.add("RevHurVTabuAtBks", int(h["vdata_tabu_at_bks"]))
    M.add("RevHurVCertified", int(h["vdata_certified"]))
    M.add("RevHurNInstances", int(h["n_instances"]))
    M.add("RevHurSNotV", int(h["bases_proved_s_not_v"]))
    M.add("RevHurVNotS", int(h["bases_proved_v_not_s"]))
    M.add("RevHurMcNemarP", _pz(h["mcnemar_exact_p"]))
    M.add("RevHurMonotone", int(h["bases_monotone_nonincreasing"]))
    M.add("RevHurInformative", int(h["clogit_informative_bases"]))
    for key, tg in (("clogit_level", "Level"), ("clogit_log_eligible", "Elig")):
        c = h[key]
        M.add(f"RevHurClogit{tg}Coef", c["coef"], 2)
        M.add(f"RevHurClogit{tg}Lo", c["ci95"][0], 2)
        M.add(f"RevHurClogit{tg}Hi", c["ci95"][1], 2)
        M.add(f"RevHurClogit{tg}Or", c["odds_ratio"], 3)
        M.add(f"RevHurClogit{tg}P", _pz(c["p"]))
    for key, tg in (("page_time_to_proof", "Time"), ("page_gap", "Gap"),
                    ("page_tabu_rpd", "TabuRpd"), ("page_tabu_minus_exact", "TabuMinus")):
        M.add(f"RevHurPage{tg}Z", h[key]["z"], 2)
        M.add(f"RevHurPage{tg}P", _pz(h[key]["p_increasing"]))
    M.add("RevHurHolmFamily", int(h["holm_family_size"]))
    M.add("RevHurHolmMaxP", _p(max(h["holm"].values())))
    for key, tg in (("clogit_level", "Level"), ("clogit_log_eligible", "Elig"),
                    ("mcnemar", "McNemar"), ("page_time_to_proof", "PageTime"),
                    ("page_gap", "PageGap"), ("page_tabu_rpd", "PageTabuRpd"),
                    ("page_tabu_minus_exact", "PageTabuMinus")):
        M.add(f"RevHurHolm{tg}", _pz(h["holm"][key]))
    for lv, tg in (("edata", "E"), ("rdata", "R"), ("vdata", "V")):
        c = h["clogit_categorical"][lv]
        M.add(f"RevHurCat{tg}Or", c["odds_ratio"], 3)
        M.add(f"RevHurCat{tg}Lo", c["ci95_or"][0], 3)
        M.add(f"RevHurCat{tg}Hi", c["ci95_or"][1], 3)
        M.add(f"RevHurCat{tg}P", _pz(c["p"]))


def block_memetic(M: Macros) -> None:
    t = pd.read_csv(DERIVED / "rev1_memetic.csv")
    j = _json("rev1_memetic.json")
    names = {"ga": "Ga", "brkga": "Brkga", "lshade": "Lshade", "cmaes": "Cmaes"}
    for B, btag in ((1, "One"), (300, "Final")):
        s = t[t.budget == B].set_index("method")
        M.add(f"RevMemTabu{btag}", float(s.median_tabu.iloc[0]), 2)
        for m, tag in names.items():
            r = s.loc[m]
            M.add(f"RevMem{tag}Plain{btag}", float(r.median_plain), 2)
            M.add(f"RevMem{tag}Ls{btag}", float(r.median_memetic), 2)
            M.add(f"RevMem{tag}Diff{btag}", float(r.median_paired_diff), 2)
            M.add(f"RevMem{tag}Better{btag}", int(r.memetic_better))
            M.add(f"RevMem{tag}P{btag}", _p(float(r.p)))
            M.add(f"RevMem{tag}PHolm{btag}", _p(float(r.p_holm)))
            M.add(f"RevMem{tag}GapPlain{btag}", float(r.gap_to_tabu_plain), 2)
            M.add(f"RevMem{tag}GapLs{btag}", float(r.gap_to_tabu_memetic), 2)
            M.add(f"RevMem{tag}Closed{btag}", 100 * float(r.share_gap_closed), 0)
    fin = t[t.budget == 300]
    M.add("RevMemClosedMin", 100 * float(fin.share_gap_closed.min()), 0)
    M.add("RevMemClosedMax", 100 * float(fin.share_gap_closed.max()), 0)
    M.add("RevMemBestName", {"ga": "GA", "brkga": "BRKGA", "lshade": "L-SHADE",
                             "cmaes": "CMA-ES"}[j["best_memetic"][0]])
    M.add("RevMemBestMedian", float(j["best_memetic"][1]), 2)
    M.add("RevMemBestRatioTabu", float(j["best_memetic"][1]) / float(j["tabu_median"]), 1)
    er = j["evaluation_ratio"].values()
    M.add("RevMemEvalShareMin", 100 * min(er), 0)
    M.add("RevMemEvalShareMax", 100 * max(er), 0)
    M.add("RevMemNBetterThanTabuMax", int(fin.memetic_beats_tabu.max()))
    M.add("RevMemNInstances", int(fin.n.max()))
    one = t[t.budget == 1]
    M.add("RevMemNWorseAtOne", int((one.median_memetic > one.median_plain).sum()))
    M.add("RevMemNPairedWorseAtOne", int((one.median_paired_diff > 0).sum()))
    M.add("RevMemNSigHolmAtOne", int((one.p_holm < 0.05).sum()))
    M.add("RevMemNSigHolmFinal", int((fin.p_holm < 0.05).sum()))
    M.add("RevMemNPairedBetterFinal", int((fin.median_paired_diff < 0).sum()))


def block_variants(M: Macros) -> None:
    t = pd.read_csv(DERIVED / "rev1_variant_lineage.csv")
    j = _json("rev1_variants.json")
    tag = {"de": "De", "jade": "Jade", "shade": "Shade", "lshade": "Lshade", "pso": "Pso",
           "clpso": "Clpso", "gwo": "Gwo", "igwo": "Igwo", "imode": "Imode", "mde": "Mde",
           "ga": "Ga", "brkga": "Brkga"}
    for _, r in t.iterrows():
        k = f"RevVar{tag[r.base]}{tag[r.variant]}"
        M.add(f"{k}CecBetter", int(r.cec_variant_better))
        M.add(f"{k}CecWorse", int(r.cec_variant_worse))
        M.add(f"{k}SchedBetter", int(r.sched_variant_better))
        M.add(f"{k}SchedWorse", int(r.sched_variant_worse))
        M.add(f"{k}SchedDiff", float(r.sched_median_diff), 2)
        M.add(f"{k}SchedP", _p(float(r.sched_p)))
        M.add(f"{k}SchedPHolm", _p(float(r.sched_p_holm)))
        M.add(f"{k}CecPHolm", _pz(float(r.cec_p_holm)))
    M.add("RevVarNPairs", int(j["n_pairs"]))
    M.add("RevVarNAgree", int(j["n_agree"]))
    M.add("RevVarNSchedSigHolm", int(j["n_sched_sig_holm"]))
    M.add("RevVarNCecSigHolm", int(j["n_cec_sig_holm"]))
    o = j["one_second"]
    M.add("RevVarWorseAtOneHolm", int(o["n_variant_worse_holm"]))
    M.add("RevVarThroughputMin", float(o["throughput_ratio_min"]), 2)
    M.add("RevVarThroughputMax", float(o["throughput_ratio_max"]), 2)
    M.add("RevVarWorseAtThousandEvals", int(o["n_variant_worse_at_1000_evaluations"]))
    M.add("RevVarCecFunctions", int(t.cec_functions.max()))
    M.add("RevVarSchedN", int(t.sched_n.max()))
    # base methods against variants at one second, on the evaluation subset
    p = pd.read_csv(DERIVED / "rev1_variant_pairs.csv")
    one = p[(p["set"] == "subset") & (p.budget == "rpd_at_1s")]
    pairs = {("de", "jade"), ("de", "shade"), ("de", "lshade"), ("pso", "clpso"),
             ("gwo", "igwo"), ("de", "imode"), ("de", "mde"), ("ga", "brkga")}
    sel = one[[(b, v) in pairs for b, v in zip(one.base, one.variant)]]
    M.add("RevVarWorseAtOne", int((sel.median_paired_diff > 0).sum()))
    M.add("RevVarPairsAtOne", int(len(sel)))


def block_representation(M: Macros) -> None:
    j = _json("rev1_representation.json")
    t = pd.read_csv(DERIVED / "rev1_representation.csv")
    fin = t[t.budget == 300]
    names = {"ga": "GA", "brkga": "BRKGA", "de": "DE", "pso": "PSO", "abc": "ABC",
             "gwo": "GWO", "lshade": "L-SHADE", "imode": "IMODE", "cmaes": "CMA-ES",
             "rime": "RIME", "mde": "MDE", "csa": "CSA"}
    M.add("RevRepNImproved", int(j["n_methods_improved"]))
    M.add("RevRepNMethods", int(len(fin)))
    M.add("RevRepMedianDiff", float(j["median_of_method_differences"]), 1)
    M.add("RevRepDiffMin", float(fin.median_paired_diff.max()), 1)
    M.add("RevRepDiffMax", float(fin.median_paired_diff.min()), 1)
    M.add("RevRepRankRho", float(j["rank_rho_between_representations"]), 2)
    M.add("RevRepRankP", _pz(float(j["rank_p_between_representations"])))
    M.add("RevRepTransferRho", float(j["transfer_rho_opperm"]), 2)
    M.add("RevRepTransferP", _p(float(j["transfer_p_opperm"])))
    M.add("RevRepTransferRhoRk", float(j["transfer_rho_randomkey_subset"]), 2)
    M.add("RevRepBestRkName", names[j["best_randomkey"][0]])
    M.add("RevRepBestRk", float(j["best_randomkey"][1]), 2)
    M.add("RevRepBestOpName", names[j["best_opperm"][0]])
    M.add("RevRepBestOp", float(j["best_opperm"][1]), 2)
    M.add("RevRepWorstOp", float(j["worst_opperm"]), 2)
    M.add("RevRepTabu", float(j["tabu_median"]), 2)
    M.add("RevRepSa", float(j["sa_median"]), 2)
    M.add("RevRepBestOpRatio", float(j["best_opperm_vs_tabu_ratio"]), 1)
    M.add("RevRepBestRkRatio", float(j["best_randomkey"][1]) / float(j["tabu_median"]), 1)
    vt, vs = j["best_opperm_vs_tabu"], j["best_opperm_vs_sa"]
    M.add("RevRepVsTabuDiff", vt["median_paired_diff"], 2)
    M.add("RevRepVsTabuBetter", int(vt["better"]))
    M.add("RevRepVsTabuWorse", int(vt["worse"]))
    M.add("RevRepVsTabuP", _p(vt["p"]))
    M.add("RevRepVsSaDiff", vs["median_paired_diff"], 2)
    M.add("RevRepVsSaBetter", int(vs["better"]))
    M.add("RevRepVsSaP", _p(vs["p"]))
    M.add("RevRepVsSaPHolm", _p(j["best_opperm_vs_sa"]["p_holm"]))
    M.add("RevRepVsTabuPHolm", _p(j["best_opperm_vs_tabu"]["p_holm"]))
    M.add("RevRepMaxPHolmFinal", _pz(float(t[t.budget == 300].p_holm.max())))
    M.add("RevRepMaxPHolmOne", _pz(float(t[t.budget == 1].p_holm.max())))
    M.add("RevRepNBelowSa", int(j["n_opperm_below_sa_median"]))
    M.add("RevRepNBelowSaRk", int(j["n_randomkey_below_sa_median"]))
    for key, v in j["by_class"].items():
        ty, B = key.split("_")
        tg = ("J" if ty == "JSSP" else "F") + {"1": "One", "10": "Ten", "60": "Sixty",
                                              "300": "Final"}[B]
        M.add(f"RevRep{tg}N", int(v["n"]))
        M.add(f"RevRep{tg}Tabu", v["tabu"], 2)
        M.add(f"RevRep{tg}BestOp", v["best_opperm"], 2)
        M.add(f"RevRep{tg}BestRk", v["best_randomkey"], 2)
        M.add(f"RevRep{tg}Better", int(v["best_opperm_better"]))
        M.add(f"RevRep{tg}Worse", int(v["best_opperm_worse"]))
        M.add(f"RevRep{tg}P", _p(v["p"]))
        M.add(f"RevRep{tg}PHolm", _p(v["p_holm"]))
    one = t[t.budget == 1]
    M.add("RevRepOneDiffMin", float(one.median_paired_diff.max()), 1)
    M.add("RevRepOneDiffMax", float(one.median_paired_diff.min()), 1)


def block_large(M: Macros) -> None:
    j = _json("rev1_large.json")
    names = {"ga": "GA", "brkga": "BRKGA", "de": "DE", "pso": "PSO", "abc": "ABC",
             "gwo": "GWO", "lshade": "L-SHADE", "imode": "IMODE", "cmaes": "CMA-ES",
             "rime": "RIME", "mde": "MDE", "csa": "CSA"}
    M.add("RevLargeDone", int(j["n_instances"]))
    M.add("RevLargeProved", int(j["n_proved"]))
    M.add("RevLargeExactGap", float(j["exact_gap_median"]), 1)
    M.add("RevLargeFriedmanP", _pz(float(j["friedman"][1])))
    if "n_decoded_worse_holm" in j:
        M.add("RevLargeNDecodedWorse", int(j["n_decoded_worse_holm"]))
        M.add("RevLargeNDecoded", int(j["n_decoded"]))
    cls = {"EXACT_SUFFICIENT": "Sufficient", "HEURISTIC_USEFUL": "Useful",
           "HEURISTIC_NECESSARY": "Necessary", "HYBRID_RECOMMENDED": "Hybrid",
           "INCONCLUSIVE": "Inconclusive"}
    M.add("RevLargeDispatchingFinal", float(j["median_rpd"]["dispatching"]), 2)
    M.add("RevLargeCpsatNoIncOne", int(j["cpsat_no_incumbent_1s"]))
    M.add("RevLargeCpsatNoIncTen", int(j["cpsat_no_incumbent_10s"]))
    M.add("RevLargeSaAtStartF", int(j["sa_at_start"].get("FJSP", 0)))
    M.add("RevLargeSaAtStartJ", int(j["sa_at_start"].get("JSSP", 0)))
    M.add("RevMainSaAtStartF", int(j["main_sa_at_start"]["FJSP"]))
    M.add("RevMainSaAtStartJ", int(j["main_sa_at_start"]["JSSP"]))
    M.add("RevMainSaAtOptStartF", int(j["main_sa_at_optimal_start"]["FJSP"]))
    M.add("RevMainNFjsp", int(j["main_n_by_type"]["FJSP"]))
    btag = {"1": "One", "10": "Ten", "60": "Sixty", "300": "Final"}
    for B, tb in btag.items():
        for c, tc in cls.items():
            M.add(f"RevLarge{tc}{tb}", int(j["classes"][B].get(c, 0)))
        for t, tt in (("all", ""), ("JSSP", "J"), ("FJSP", "F")):
            v = j["budgets"][B].get(t)
            if v is None:
                continue
            p = f"RevLarge{tt}{tb}"
            M.add(f"{p}Tabu", v["tabu"], 2)
            M.add(f"{p}Cpsat", v["cpsat"], 2)
            M.add(f"{p}Sa", v["sa"], 2)
            M.add(f"{p}N", v["n"])
            M.add(f"{p}TabuBeatsCpsat", v["tabu_better_than_cpsat"])
            M.add(f"{p}CpsatBeatsTabu", v["cpsat_better_than_tabu"])
            M.add(f"{p}TabuBestHeur", v["tabu_best_heuristic"])
            M.add(f"{p}BestDecName", names[v["best_decoded"][0]])
            M.add(f"{p}BestDec", v["best_decoded"][1], 2)
            M.add(f"{p}BestDecRatio", v["best_decoded"][1] / v["tabu"], 1)


def block_compute(M: Macros) -> None:
    """Runs and core-hours of the further experiments, defined as in ``numbers.py``."""
    total, n = 0.0, 0
    for stage in ("rev_hurink", "rev_memetic", "rev_variants", "rev_opperm", "rev_large"):
        d = pd.read_parquet(DERIVED / f"{stage}_runs.parquet")
        total += float(d["wall_time"].fillna(0).sum())
        n += len(d)
    M.add("RevNRuns", n)
    M.add("RevComputeHours", total / 3600, 0)
    from src.run.revision_sets import VARIANT_METHODS

    cec = pd.read_parquet(DERIVED / "cec_summary.parquet")
    M.add("RevCecVariantRuns", int(cec[cec.method.isin(VARIANT_METHODS)].runs.sum()))


def block_robustness(M: Macros) -> None:
    r = _json("rev1_robustness.json")
    b = r["best_heuristic"]
    M.add("RevBestHeurCells", int(b["cells"]))
    M.add("RevTabuBestAlone", int(b["tabu_alone"]))
    M.add("RevTabuBestTied", int(b["tabu_tied"]))
    M.add("RevTabuBestAloneOrTied", int(b["tabu_alone_or_tied"]))
    M.add("RevTabuBestAloneShare", 100 * b["tabu_alone"] / b["cells"], 0)
    M.add("RevTabuBestAloneOrTiedShare", 100 * b["tabu_alone_or_tied"] / b["cells"], 0)
    hn = r["hn_final"]
    small = [i for i, n in hn if n <= 500]
    M.add("RevHnFinalUpToFiveHundred", len(small))
    M.add("RevHnFinalAboveFiveHundred", len([i for i, n in hn if n > 500]))
    words = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
    M.add("RevHnFinalAboveFiveHundredWord", words[len([i for i, n in hn if n > 500])])
    M.add("RevHnFinalSmallNames", ", ".join(small[:-1]) + " and " + small[-1]
          if len(small) > 1 else (small[0] if small else "none"))
    M.add("RevHnFinalSmallestOps", min(n for _, n in hn))
    tr = r["transfer"]
    M.add("RevTransferBootLo", tr["boot_ci95"][0], 2)
    M.add("RevTransferBootHi", tr["boot_ci95"][1], 2)
    M.add("RevTransferPermP", _p(tr["perm_p_two_sided"]))
    M.add("RevTransferNBoot", int(tr["n_boot"]))
    c = r["clusters"]
    M.add("RevClusterUnits", int(c["n_units"]))
    M.add("RevClusterNSig", int(c["n_significant_holm"]))
    M.add("RevClusterNComp", int(c["n_comparisons"]))
    M.add("RevClusterMaxP", _pz(c["max_p_holm"]))
    M.add("RevClusterTransferRho", c["transfer_rho"], 2)
    g = pd.read_csv(DERIVED / "rev1_sensitivity_grid.csv")
    mod = g[(g.budget_s == 300) & (g.delta <= 0.02)]
    M.add("RevGridNecModerateMinFinal", int(mod.n_HEURISTIC_NECESSARY.min()))
    M.add("RevGridNecModerateMaxFinal", int(mod.n_HEURISTIC_NECESSARY.max()))


BLOCKS = [block_design, block_hurink, block_memetic, block_variants, block_representation, block_charged, block_grid, block_noise, block_coverage, block_proof, block_pairs, block_large, block_compute, block_robustness]


def build() -> Macros:
    M = Macros()
    for blk in BLOCKS:
        try:
            blk(M)
        except FileNotFoundError as exc:
            print(f"skipped {blk.__name__}: {exc}")
    return M


def main() -> None:
    M = build()
    path = GEN / "numbers_rev1.tex"
    lines = ["% Generated by src/figtab/numbers_rev1.py. Do not edit by hand.",
             "% Macros of the further experiments."]
    for k, v in sorted(M.items.items()):
        lines.append(f"\\newcommand{{\\{k}}}{{{v}}}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {path} with {len(M.items)} macros")


if __name__ == "__main__":
    main()
