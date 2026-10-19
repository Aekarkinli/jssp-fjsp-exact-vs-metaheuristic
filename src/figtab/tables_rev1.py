"""Appendix tables of the further experiments, generated from ``results/derived/rev1_*``.

- ``tab_rev_instances``: the evaluation subset and the larger instances with their
  dimensions, routing flexibility and reference values.
- ``tab_rev_representation``: the decoded optimisers under the two decoding configurations.
- ``tab_rev_memetic``: the four optimisers with and without the critical-block descent.
- ``tab_rev_variants``: base methods against their improved variants on both domains.
- ``tab_rev_hurink``: the matched flexibility study by dataset level.

Every p value is given unadjusted and Holm-adjusted within the family named in the caption.

    .venv/Scripts/python.exe -m src.figtab.tables_rev1
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis.panel import display
from src.figtab.tables import _num, _p, _write

DERIVED = Path("results/derived")
LEVELS = ["sdata", "edata", "rdata", "vdata"]


def _id(x: str) -> str:
    return str(x).replace("_", r"\_")


def _src(x: str) -> str:
    s = str(x)
    for key, short in (("JSPLIB", "JSPLIB"), ("Weise", "Weise collection"),
                       ("ScheduleOpt", "ScheduleOpt")):
        if key in s:
            return short
    return s.split("(")[0].strip()[:24]


def tab_rev_instances() -> None:
    from src.run.revision_sets import large_set, revision_subset
    from src.run.runner import _load

    ref = pd.concat([pd.read_csv("data/reference_values.csv"),
                     pd.read_csv("data/reference_values_rev1.csv")]).drop_duplicates("id")
    ref = ref.set_index("id")
    head = r"Instance & Set & Class & $n\times m$ & Ops & Eligible & BKS & LB & Source \\"
    lines = [
        r"{\footnotesize\setlength{\tabcolsep}{4pt}",
        r"\begin{longtable}{lllrrrrrl}",
        r"\caption{Instances of the targeted experiments. The evaluation subset (E) holds ten "
        r"instances of each problem class at evenly spaced ranks of operation count and is "
        r"used for the representation, local-search and variant experiments. The larger "
        r"instances (L) are used for the extension to larger and more flexible instances. "
        r"Eligible is the mean number of eligible machines per operation. BKS and LB are the "
        r"best-known value and lower bound of the named source.}\label{tab:rev-instances}\\",
        r"\toprule", head, r"\midrule", r"\endfirsthead",
        r"\toprule", head, r"\midrule", r"\endhead",
        r"\bottomrule", r"\endfoot",
    ]
    for tag, ids in (("E", revision_subset()), ("L", large_set())):
        for i in ids:
            inst = _load(i)
            elig = np.mean([len(o.modes) for o in inst.operations])
            r = ref.loc[i]
            lines.append(
                f"{_id(i)} & {tag} & {'FJSP' if inst.is_flexible else 'JSSP'} & "
                f"{inst.num_jobs}$\\times${inst.num_machines} & {inst.num_operations} & "
                f"{elig:.2f} & {float(r.BKS):.0f} & {float(r.LB):.0f} & {_src(r.BKS_source)} \\\\")
        if tag == "E":
            lines.append(r"\midrule")
    lines.append(r"\end{longtable}}")
    _write("tab_rev_instances.tex", "\n".join(lines))


def tab_rev_representation() -> None:
    t = pd.read_csv(DERIVED / "rev1_representation.csv")
    lines = [
        r"\begin{table}[htbp]", r"\centering", r"\small",
        r"\caption{The twelve decoded optimisers under the random-key active-schedule "
        r"configuration (RK) and the operation-based configuration (OB) on the evaluation "
        r"subset, ten seeds. Medians are over instances of the per-instance median deviation "
        r"from the best-known value. The paired difference is OB minus RK, Better counts the "
        r"instances on which OB is lower, and $p$ is the Wilcoxon signed-rank test, "
        r"unadjusted and Holm-adjusted over the twelve optimisers at the same budget.}",
        r"\label{tab:rev-representation}",
        r"\begin{tabular}{lrrrrrrrrrr}", r"\toprule",
        r" & \multicolumn{5}{c}{1 s} & \multicolumn{5}{c}{300 s} \\",
        r"\cmidrule(lr){2-6}\cmidrule(lr){7-11}",
        r"Optimiser & RK & OB & Diff. & $p$ & $p_{\mathrm{Holm}}$ & RK & OB & Diff. & "
        r"Better & $p_{\mathrm{Holm}}$ \\",
        r"\midrule",
    ]
    one = t[t.budget == 1].set_index("method")
    fin = t[t.budget == 300].set_index("method")
    for m in fin.sort_values("median_opperm").index:
        a, b = one.loc[m], fin.loc[m]
        lines.append(
            f"{display(m)} & {_num(a.median_randomkey)} & {_num(a.median_opperm)} & "
            f"{_num(a.median_paired_diff)} & {_p(a.p)} & {_p(a.p_holm)} & "
            f"{_num(b.median_randomkey)} & {_num(b.median_opperm)} & "
            f"{_num(b.median_paired_diff)} & {int(b.opperm_better)}/{int(b.n)} & "
            f"{_p(b.p_holm)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    _write("tab_rev_representation.tex", "\n".join(lines))


def tab_rev_memetic() -> None:
    t = pd.read_csv(DERIVED / "rev1_memetic.csv")
    lines = [
        r"\begin{table}[htbp]", r"\centering", r"\small",
        r"\caption{The four optimisers with (LS) and without the critical-block descent in "
        r"the decoder on the evaluation subset, ten seeds, at the four budgets. Better counts "
        r"the instances on which the descent lowers the median deviation, Closed is the share "
        r"of the median paired distance to the tabu search that the descent removes, and $p$ "
        r"is the Wilcoxon signed-rank test, unadjusted and Holm-adjusted over the four "
        r"optimisers at the same budget.}",
        r"\label{tab:rev-memetic}",
        r"\begin{tabular}{rlrrrrrrr}", r"\toprule",
        r"Budget (s) & Optimiser & Plain & LS & Diff. & Better & Closed (\%) & $p$ & "
        r"$p_{\mathrm{Holm}}$ \\",
        r"\midrule",
    ]
    for B, g in t.groupby("budget"):
        for _, r in g.iterrows():
            lines.append(
                f"{B} & {display(r.method)} & {_num(r.median_plain)} & "
                f"{_num(r.median_memetic)} & {_num(r.median_paired_diff)} & "
                f"{int(r.memetic_better)}/{int(r.n)} & {_num(100 * r.share_gap_closed, 0)} & "
                f"{_p(r.p)} & {_p(r.p_holm)} \\\\")
        if B != t.budget.max():
            lines.append(r"\addlinespace")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    _write("tab_rev_memetic.tex", "\n".join(lines))


def tab_rev_variants() -> None:
    t = pd.read_csv(DERIVED / "rev1_variant_lineage.csv")
    one = pd.read_csv(DERIVED / "rev1_variants_one_second.csv")
    lines = [
        r"\begin{table}[htbp]", r"\centering", r"\small",
        r"\caption{Base methods against their improved variants. On CEC2017 the counts are "
        r"the functions on which the variant has the lower or higher median error over 51 "
        r"runs. On scheduling they are the instances of the evaluation subset on which the "
        r"variant has the lower or higher median deviation at 300 s, with the median paired "
        r"difference, variant minus base. Every $p$ is a Wilcoxon signed-rank test, "
        r"Holm-adjusted over the nine pairs within each domain, and the unadjusted value is "
        r"given for scheduling.}",
        r"\label{tab:rev-variants}",
        r"\begin{tabular}{llrrrrrrr}", r"\toprule",
        r" & & \multicolumn{3}{c}{CEC2017} & \multicolumn{4}{c}{Scheduling, 300 s} \\",
        r"\cmidrule(lr){3-5}\cmidrule(lr){6-9}",
        r"Base & Variant & Better/worse & $p_{\mathrm{Holm}}$ & & Better/worse & Diff. & "
        r"$p$ & $p_{\mathrm{Holm}}$ \\",
        r"\midrule",
    ]
    for _, r in t.iterrows():
        lines.append(
            f"{display(r.base)} & {display(r.variant)} & "
            f"{int(r.cec_variant_better)}/{int(r.cec_variant_worse)} & {_p(r.cec_p_holm)} & & "
            f"{int(r.sched_variant_better)}/{int(r.sched_variant_worse)} & "
            f"{_num(r.sched_median_diff)} & {_p(r.sched_p)} & {_p(r.sched_p_holm)} \\\\")
    lines += [r"\midrule",
              r"\multicolumn{9}{l}{One second, scheduling, Holm over the eight pairs below} \\",
              r"\midrule"]
    for _, r in one.iterrows():
        lines.append(
            f"{display(r.base)} & {display(r.variant)} & & & & "
            f"{int(r.variant_better)}/{int(r.variant_worse)} & {_num(r.median_paired_diff)} & "
            f"{_p(r.p)} & {_p(r.p_holm)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    _write("tab_rev_variants.tex", "\n".join(lines))


def tab_rev_hurink() -> None:
    d = pd.read_csv(DERIVED / "rev1_hurink_instances.csv")
    h = json.loads((DERIVED / "rev1_hurink.json").read_text(encoding="utf-8"))
    lines = [
        r"\begin{table}[htbp]", r"\centering", r"\small",
        r"\caption{Matched flexibility study by dataset level, on the base problems of the "
        r"Hurink collection with three seeds each. An instance counts as proved when at "
        r"least two of its three exact runs prove optimality within 300 s, and the duality "
        r"gap and the deviations are medians over the three runs. The odds ratio of a proof "
        r"against sdata comes from a conditional logistic regression with one indicator per "
        r"dataset, stratified by base problem. Parentheses contain the unadjusted $p$ values of "
        r"these dataset indicators, a sensitivity check reported separately from the "
        r"seven-test family of Section~\ref{sec:revision-experiments}.}",
        r"\label{tab:rev-hurink}",
        r"\begin{tabular}{lrrrrrrr}", r"\toprule",
        r"Dataset & Instances & Eligible & Proved & Gap (\%) & Exact at BKS & "
        r"Tabu RPD (\%) & Odds ratio of proof \\",
        r"\midrule",
    ]
    for lv in LEVELS:
        g = d[d.level == lv]
        orr = ("1 (reference)" if lv == "sdata" else
               f"{h['clogit_categorical'][lv]['odds_ratio']:.3f} "
               f"({_p(h['clogit_categorical'][lv]['p'])})")
        lines.append(
            f"{lv} & {len(g)} & {g.mean_eligible.mean():.2f} & {int(g.proved.sum())} & "
            f"{g.gap.median():.1f} & {int((g.exact_rpd <= 1e-9).sum())} & "
            f"{g.tabu_rpd.median():.2f} & {orr} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    _write("tab_rev_hurink.tex", "\n".join(lines))


def main() -> None:
    tab_rev_instances()
    tab_rev_representation()
    tab_rev_memetic()
    tab_rev_variants()
    tab_rev_hurink()


if __name__ == "__main__":
    main()
