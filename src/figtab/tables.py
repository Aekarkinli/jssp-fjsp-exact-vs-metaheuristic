"""Every table of the manuscript, generated as a LaTeX fragment from the derived tables.

    uv run python -m src.figtab.tables
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis.panel import (
    BUDGETS,
    CLASS_LABEL,
    CLASS_ORDER,
    DECODED,
    FAMILY,
    FAMILY_LABEL,
    PANEL,
    display,
)

DERIVED = Path("results/derived")
GEN = Path("paper/generated")


FIT = (r"\resizebox{\ifdim\width>\linewidth\linewidth\else\width\fi}{!}{%")


def _write(name: str, body: str) -> None:
    """Write a table fragment, shrinking it to the text width only when it overflows."""
    GEN.mkdir(parents=True, exist_ok=True)
    if r"\begin{tabular}" in body:
        body = body.replace(r"\small", r"\footnotesize" + "\n" + r"\setlength{\tabcolsep}{4pt}")
        body = body.replace(r"\begin{tabular}", FIT + "\n" + r"\begin{tabular}", 1)
        body = body.replace(r"\end{tabular}", r"\end{tabular}}", 1)
    (GEN / name).write_text(body.rstrip() + "\n", encoding="utf-8")
    print("wrote", name)


def _num(x, d=2):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "--"
    return f"{x:.{d}f}"


def _p(x):
    if x is None or not np.isfinite(x):
        return "--"
    if x < 1e-4:
        return "$<10^{-4}$"
    return f"{x:.4f}".rstrip("0").rstrip(".")


# ------------------------------------------------------------------ 1. instances
FAMILY_CITE = {
    "fisher_thompson": "fisher1963probabilistic",
    "lawrence": "lawrence1984resource",
    "adams_balas_zawack": "adams1988shifting",
    "applegate_cook_orb": "applegate1991computational",
    "storer_wu_vaccari": "storer1992new",
    "taillard": "taillard1993benchmarks",
    "brandimarte": "brandimarte1993routing",
    "hurink_edata": "hurink1994tabu",
    "hurink_rdata": "hurink1994tabu",
    "hurink_vdata": "hurink1994tabu",
    "fattahi": "fattahi2007mathematical",
}
FAMILY_NAME = {
    "fisher_thompson": "Fisher and Thompson",
    "lawrence": "Lawrence",
    "adams_balas_zawack": "Adams, Balas and Zawack",
    "applegate_cook_orb": "Applegate and Cook",
    "storer_wu_vaccari": "Storer, Wu and Vaccari",
    "taillard": "Taillard",
    "brandimarte": "Brandimarte",
    "hurink_edata": "Hurink edata",
    "hurink_rdata": "Hurink rdata",
    "hurink_vdata": "Hurink vdata",
    "fattahi": "Fattahi et al.",
}


def tab_instances() -> None:
    inst = pd.read_parquet(DERIVED / "instances.parquet")
    rows = []
    for (typ, fam), g in inst.groupby(["type", "family"], sort=False):
        rows.append((typ, fam, len(g),
                     f"{g.n_jobs.min()}--{g.n_jobs.max()}",
                     f"{g.n_machines.min()}--{g.n_machines.max()}",
                     f"{g.n_op.min()}--{g.n_op.max()}",
                     g.mean_eligible.mean()))
    rows.sort(key=lambda r: (r[0] != "JSSP", r[1]))
    lines = [
        r"\begin{table}[htbp]", r"\centering", r"\small",
        r"\caption{Benchmark composition. Flexibility is the mean number of eligible "
        r"machines per operation, which equals one for every job-shop instance.}",
        r"\label{tab:instances}",
        r"\begin{tabular}{llrccccl}", r"\toprule",
        r"Problem & Family & $n$ & Jobs & Machines & Operations & Flexibility & Source \\",
        r"\midrule",
    ]
    for typ, fam, n, j, m, o, flex in rows:
        lines.append(f"{typ} & {FAMILY_NAME[fam]} & {n} & {j} & {m} & {o} & "
                     f"{flex:.2f} & \\cite{{{FAMILY_CITE[fam]}}} \\\\")
    tot = len(inst)
    lines += [r"\midrule",
              rf"\multicolumn{{2}}{{l}}{{Total}} & {tot} & "
              rf"{inst.n_jobs.min()}--{inst.n_jobs.max()} & "
              rf"{inst.n_machines.min()}--{inst.n_machines.max()} & "
              rf"{inst.n_op.min()}--{inst.n_op.max()} & & \\",
              r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    _write("tab_instances.tex", "\n".join(lines))


# ------------------------------------------------------------------ 2. method panel
METHOD_ROWS = [
    ("cpsat", "Lazy clause generation over an integer constraint model, one search worker",
     "perron2023cpsat,lan2025pyjobshop"),
    ("dispatching", "Priority rules, shortest processing time, most work remaining and "
     "most operations remaining", "panwalkar1977survey"),
    ("greedy", "Active-schedule construction", "giffler1960algorithms"),
    ("tabu", "Critical-block neighbourhood, randomised tenure, aspiration, restarts",
     "glover1986future,nowicki1996fast"),
    ("sa", "Disjunctive-graph annealing, instance-calibrated cooling", "vanlaarhoven1992job"),
    ("brkga", "Biased random-key genetic algorithm, elite-biased uniform crossover",
     "bean1994genetic,goncalves2011biased"),
    ("ga", "Real-coded genetic algorithm", "holland1975adaptation"),
    ("de", "Differential evolution, rand/1/bin", "storn1997differential"),
    ("pso", "Particle swarm optimisation with inertia weight", "kennedy1995particle"),
    ("abc", "Artificial bee colony, employed, onlooker and scout phases",
     "karaboga2007powerful"),
    ("lshade", "Success-history adaptive DE with linear population reduction",
     "tanabe2014improving"),
    ("imode", "Improved multi-operator DE, CEC 2020 competition entry",
     "sallam2020improved"),
    ("cmaes", "Covariance matrix adaptation evolution strategy, separable mode",
     "hansen2001completely"),
    ("gwo", "Grey wolf optimiser", "mirjalili2014grey"),
    ("rime", "RIME optimisation algorithm", "su2023rime"),
    ("mde", "Multi-population differential evolution", "karkinli2023detection"),
    ("csa", "Colony-based search algorithm", "civicioglu2024colony"),
]


def tab_methods() -> None:
    lines = [
        r"\begin{table}[htbp]", r"\centering", r"\small",
        r"\caption{Method panel. Every population-based method uses the same population "
        r"size and the same wall-clock budget, and no parameter is tuned per instance.}",
        r"\label{tab:methods}",
        r"\begin{tabular}{llp{6.6cm}l}", r"\toprule",
        r"Method & Role & Configuration & Source \\", r"\midrule",
    ]
    last = None
    for m, cfg, cite in METHOD_ROWS:
        fam = FAMILY_LABEL[FAMILY[m]]
        shown = "" if fam == last else fam
        last = fam
        lines.append(f"{display(m)} & {shown} & {cfg} & \\cite{{{cite}}} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    _write("tab_methods.tex", "\n".join(lines))


# ------------------------------------------------------------------ 3. objective summary
CONFIG_ROWS = [
    ("cpsat", "OR-Tools CP-SAT via PyJobShop", "one search worker",
     "library values", "not applicable"),
    ("dispatching", "this study", "three priority rules", "source", "not applicable"),
    ("greedy", "this study", "no parameters", "source", "not applicable"),
    ("tabu", "this study", r"tenure $\sim U[8,16]$, restart after $\max(200,2N)$",
     "moves from source, tenure set here", "not applicable"),
    ("sa", "this study", r"$\delta_c=0.1$, $\chi_0=0.9$, chain $\max(100,N)$",
     "source", "moves keep feasibility"),
    ("brkga", "this study", r"elite $0.20$, mutant $0.15$, $\rho_e=0.70$",
     "source intervals", "keys drawn inside the box"),
    ("ga", "mealpy", r"$p_c=0.95$, $p_m=0.025$", "library", "clipped to the box"),
    ("de", "mealpy", r"rand/1/bin, $F=0.1$, $CR=0.9$", "library", "clipped to the box"),
    ("pso", "mealpy", r"$w=0.4$, $c_1=c_2=2.05$", "library", "clipped to the box"),
    ("abc", "mealpy", "abandonment limit 25", "library", "clipped to the box"),
    ("lshade", "mealpy", r"$\mu_F=\mu_{CR}=0.5$, linear reduction", "library values",
     "clipped to the box"),
    ("imode", "mealpy", "memory 5, archive 20", "library", "clipped to the box"),
    ("cmaes", "cma", r"$\sigma_0=0.3$, diagonal covariance", "library", "library transform, then clipping"),
    ("gwo", "mealpy", "no free parameters", "library", "clipped to the box"),
    ("rime", "mealpy", "soft-rime parameter 5", "library", "clipped to the box"),
    ("mde", "port of the author code", "three subpopulations", "author code", "resample towards the bound"),
    ("csa", "port of the author code", "clan drawn from the colony", "author code", "resample towards the bound"),
]


def tab_configuration() -> None:
    env = json.loads((DERIVED / "environment.json").read_text(encoding="utf-8"))
    v = env["versions"]
    impl_version = {"mealpy": f"mealpy {v['mealpy']}", "cma": f"cma {v['cma']}",
                    "OR-Tools CP-SAT via PyJobShop":
                        f"OR-Tools {v['ortools']}, PyJobShop {v['pyjobshop']}"}
    lines = [
        r"\begin{table}[htbp]", r"\centering", r"\small",
        r"\caption{Implementation and parameter provenance. Every population-based method "
        r"uses the same population of 50, which is a comparability choice of this study and "
        r"is examined separately in Table~\ref{tab:robustness}. The last column states how "
        r"each optimiser treats a candidate that leaves the unit box, which differs between "
        r"implementations and is therefore not controlled by the shared decoder.}",
        r"\label{tab:configuration}",
        r"\begin{tabular}{lllll}", r"\toprule",
        r"Method & Implementation & Parameters & Values from & Box handling \\",
        r"\midrule",
    ]
    for m, impl, params, src, bound in CONFIG_ROWS:
        shown = impl_version.get(impl, impl)
        lines.append(f"{display(m)} & {shown} & {params} & {src} & {bound} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    _write("tab_configuration.tex", "\n".join(lines))


def tab_environment() -> None:
    env = json.loads((DERIVED / "environment.json").read_text(encoding="utf-8"))
    v = env["versions"]
    rows = [
        ("Processor", env["cpu_model"]),
        ("Timed workers", f"{env['workers']} performance cores, one job per core, "
                          "no simultaneous-multithreading sibling in use"),
        ("Operating system", env["os"].replace("_", " ")),
        ("Python", env["python"]),
        ("Exact solver", f"OR-Tools CP-SAT {v['ortools']} through PyJobShop {v['pyjobshop']}"),
        ("Metaheuristic library", f"mealpy {v['mealpy']}"),
        ("Evolution strategy library", f"cma {v['cma']}"),
        ("Benchmark function library", f"opfunu {v['opfunu']}"),
        ("Numerical libraries", f"numpy {v['numpy']}, scipy {v['scipy']}, "
                                f"statsmodels {v['statsmodels']}"),
        ("Code revision", env["git_commit"][:10]),
    ]
    lines = [
        r"\begin{table}[htbp]", r"\centering", r"\small",
        r"\caption{Execution environment. Every timed result in this study refers to this "
        r"machine and these versions, which is part of the definition of the necessity "
        r"classification rather than an incidental detail.}",
        r"\label{tab:environment}",
        r"\begin{tabular}{ll}", r"\toprule", r"Item & Value \\", r"\midrule",
    ]
    for k, val in rows:
        lines.append(f"{k} & {val} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    _write("tab_environment.tex", "\n".join(lines))


def tab_reference_provenance() -> None:
    t = pd.read_csv(DERIVED / "reference_provenance.csv")
    short = {"JSPLIB instances.json (tamy0612/JSPLIB@eea2b60)": "JSPLIB",
             "Weise instances_with_bks (thomasWeise/jsspInstancesAndResults@29a50db)":
                 "Weise collection"}

    def src(x: str) -> str:
        for k, v in short.items():
            if str(x).startswith(k[:20]):
                return v
        return str(x).split("(")[0].strip()[:28]

    head = (r"Instance & $n\times m$ & Ops & BKS & LB & Certified & Source \\")
    lines = [
        r"{\footnotesize\setlength{\tabcolsep}{5pt}",
        r"\begin{longtable}{lrrrrcl}",
        r"\caption{Reference values and their provenance. BKS is the best-known makespan "
        r"used as the denominator of every deviation reported in this study, LB is the best "
        r"lower bound recorded by the same source, and Certified states whether the source "
        r"records the value as a proven optimum. Collections were accessed at the pinned "
        r"revisions given in the repository.}\label{tab:reference-provenance}\\",
        r"\toprule", head, r"\midrule", r"\endfirsthead",
        r"\toprule", head, r"\midrule", r"\endhead",
        r"\bottomrule", r"\endfoot",
    ]
    for _, r in t.iterrows():
        lines.append(
            f"{str(r.instance_id).replace('_', chr(92) + '_')} & "
            f"{r.n_jobs}$\\times${r.n_machines} & {r.n_op} & {float(r.BKS):.0f} & "
            f"{float(r.LB):.0f} & {'yes' if str(r.proven_optimal) == 'True' else 'no'} & "
            f"{src(r.BKS_source)} \\\\")
    lines.append(r"\end{longtable}}")
    _write("tab_reference_provenance.tex", "\n".join(lines))


def tab_summary() -> None:
    s = pd.read_csv(DERIVED / "summary_by_budget.csv")
    fin = s[s.budget_s == 300].sort_values("mean_rank")
    piv = s.pivot(index="method", columns="budget_s", values="median")
    lines = [
        r"\begin{table}[htbp]", r"\centering", r"\small",
        r"\caption{Objective quality over the 67 instances. Each entry summarises the "
        r"per-instance median over seeds of the relative deviation from the best-known "
        r"solution. Columns 1--300\,s give that median across instances at four budgets, "
        r"and the remaining columns describe the 300\,s distribution. The 1\,s column "
        r"covers 66 instances because the exact solver holds no feasible schedule on one "
        r"instance at that budget. $n_{\mathrm{bks}}$ counts instances whose median matches "
        r"the best-known solution, and Rank is the mean rank across instances.}",
        r"\label{tab:summary}",
        r"\begin{tabular}{llrrrrrrrrr}", r"\toprule",
        r"& & \multicolumn{4}{c}{Median deviation (\%) at} & \multicolumn{5}{c}{At 300 s} \\",
        r"\cmidrule(lr){3-6}\cmidrule(lr){7-11}",
        r"Method & Role & 1\,s & 10\,s & 60\,s & 300\,s & IQR & 90th & Worst & "
        r"$n_{\mathrm{bks}}$ & Rank \\",
        r"\midrule",
    ]
    last = None
    for _, r in fin.iterrows():
        fam = FAMILY_LABEL[FAMILY[r.method]]
        shown = "" if fam == last else fam
        last = fam
        lines.append(
            f"{display(r.method)} & {shown} & " +
            " & ".join(_num(piv.loc[r.method, b], 2) for b in BUDGETS) +
            f" & {_num(r.iqr)} & {_num(r.p90)} & {_num(r.worst)} & "
            f"{int(r.n_at_bks)} & {_num(r.mean_rank)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    _write("tab_summary.tex", "\n".join(lines))


# ------------------------------------------------------------------ 4. pairwise
def tab_pairwise() -> None:
    p = pd.read_csv(DERIVED / "pairwise_vs_tabu_300s.csv")
    cd = json.loads((DERIVED / "stats_summary.json").read_text(encoding="utf-8"))["cd_panel"]
    lines = [
        r"\begin{table}[htbp]", r"\centering", r"\small",
        rf"\caption{{Paired comparison of every method with the tabu search at the 300\,s "
        rf"budget over the 67 instances. The omnibus Friedman test rejects equality "
        rf"($\chi^2={cd['friedman_stat']:.1f}$, $p<10^{{-4}}$). $p$ values are "
        rf"Holm-corrected within the family of 16 comparisons. A positive rank-biserial "
        rf"correlation means the method deviates more than the tabu search. The median "
        rf"paired difference is the median over instances of the per-instance difference, "
        rf"which is not the difference of the two medians. Superiority is "
        rf"the share of instances on which the method is strictly better, with a 95\,\% "
        rf"Wilson interval.}}",
        r"\label{tab:pairwise}",
        r"\begin{tabular}{lrrrrrl}", r"\toprule",
        r"Method & Median (\%) & Median paired & $W$ & $p_{\text{Holm}}$ & $r_{rb}$ "
        r"($n_{\text{eff}}$) & Superiority [95\,\% CI] \\",
        r" & & difference & & & & \\", r"\midrule",
    ]
    for _, r in p.iterrows():
        lines.append(
            f"{display(r.method)} & {_num(r['median'])} & {_num(r.median_difference)} & "
            f"{_num(r.wilcoxon_W, 0)} & {_p(r.p_holm)} & "
            f"{_num(r.rank_biserial)} ({int(r.n_effective)}) & "
            f"{r.superiority:.2f} [{r.sup_lo:.2f}, {r.sup_hi:.2f}] \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    _write("tab_pairwise.tex", "\n".join(lines))


# ------------------------------------------------------------------ 5. necessity
def tab_necessity() -> None:
    t = pd.read_csv(DERIVED / "necessity.csv")
    counts = (t.groupby(["budget_s", "class"]).size().unstack(fill_value=0)
              .reindex(columns=CLASS_ORDER, fill_value=0))
    sens = json.loads((DERIVED / "necessity_sensitivity.json").read_text(encoding="utf-8"))
    lines = [
        r"\begin{table}[htbp]", r"\centering", r"\small",
        r"\caption{Necessity classification of the 67 instances at four budgets, and its "
        r"dependence on the decision rule. The main panel uses the pre-specified thresholds "
        r"$\delta=0.01$, $\tau=10$ and $\varepsilon=0.02$, the tabu search as the reference "
        r"heuristic, the tabu-seeded warm start as the reference hybrid, and false-discovery "
        r"control across the instances tested within each budget. The lower panel counts the "
        r"cells whose class changes when one element of that rule is replaced.}",
        r"\label{tab:necessity}",
        r"\begin{tabular}{lrrrrr}", r"\toprule",
        r"Budget & " + " & ".join(CLASS_LABEL[c] for c in CLASS_ORDER) + r" \\", r"\midrule",
    ]
    for b in BUDGETS:
        lines.append(f"{b}\\,s & " + " & ".join(str(int(counts.loc[b, c])) for c in CLASS_ORDER)
                     + r" \\")
    lines += [r"\midrule",
              r"\multicolumn{6}{l}{\emph{Cells whose class changes under an alternative "
              r"decision rule}} \\",
              r"Variation & 1\,s & 10\,s & 60\,s & 300\,s & Total \\", r"\midrule"]
    names = {"delta_strict": r"$\delta=0.02$", "tau_low": r"$\tau=5$",
             "tau_high": r"$\tau=20$",
             "best_of_panel": "best heuristic of the panel as reference",
             "cheap_hybrid": "rule-seeded cheap hybrid as reference",
             "no_hybrid": "hybrid excluded",
             "no_fdr": "no false-discovery control"}
    for key, label in names.items():
        v = pd.read_csv(DERIVED / f"necessity_{key}.csv")
        m = t.merge(v, on=["instance_id", "budget_s"], suffixes=("", "_v"))
        diff = m[m["class"] != m["class_v"]]
        per = diff.groupby("budget_s").size()
        lines.append(f"{label} & " + " & ".join(str(int(per.get(b, 0))) for b in BUDGETS)
                     + f" & {len(diff)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    del sens
    _write("tab_necessity.tex", "\n".join(lines))


# ------------------------------------------------------------------ 6. proof model
def tab_proof() -> None:
    m = json.loads((DERIVED / "proof_model.json").read_text(encoding="utf-8"))
    labels = {"Intercept": "Intercept", "log_n_op": "log operations",
              "n_machines": "Machines", "flex_ratio": "Flexibility ratio",
              "cv_duration": "Duration dispersion"}
    lines = [
        r"\begin{table}[htbp]", r"\centering", r"\small",
        rf"\caption{{Logistic model of proof of optimality within 300\,s, fitted on one "
        rf"binary response per instance ($n={m['n_instances']}$, "
        rf"{m['n_proved']} proved). McFadden $R^2={m['pseudo_r2']:.2f}$, likelihood-ratio "
        rf"$p<10^{{-4}}$.}}",
        r"\label{tab:proof}",
        r"\begin{tabular}{lrrr}", r"\toprule",
        r"Term & Coefficient & 95\,\% CI & $p$ \\", r"\midrule",
    ]
    for k, lab in labels.items():
        lo, hi = m["conf_int"][k]
        lines.append(f"{lab} & {_num(m['params'][k])} & [{_num(lo)}, {_num(hi)}] & "
                     f"{_p(m['pvalues'][k])} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    _write("tab_proof.tex", "\n".join(lines))


# ------------------------------------------------------------------ 7. hybrid
def tab_hybrid() -> None:
    h = pd.read_csv(DERIVED / "hybrid_vs_pure.csv")
    c = pd.read_csv(DERIVED / "hybrid_cost.csv")
    lines = [
        r"\begin{table}[htbp]", r"\centering", r"\small",
        r"\caption{Warm-started exact search under one end-to-end budget, so the seeding "
        r"phase is charged to the same clock. Each variant is compared with pure exact "
        r"search and with the tabu search at the same total budget, paired over instances "
        r"and Holm-corrected. The oracle variant is seeded with the best heuristic schedule "
        r"found anywhere in the study and is an unattainable bound rather than a usable "
        r"method.}",
        r"\label{tab:hybrid}",
        r"\begin{tabular}{llrrrrrl}", r"\toprule",
        r"Variant & Budget & Seeding (s) & Median (\%) & Reference & Median paired & "
        r"$r_{rb}$ & $p_{\text{Holm}}$ \\",
        r" & & & & & difference & & \\", r"\midrule",
    ]
    for v in ("hyb_cheap", "hyb_tabu", "hyb_oracle"):
        for b in BUDGETS:
            sub = h[(h.variant == v) & (h.budget_s == b)]
            seed_t = c[(c.method == v) & (c.budget_s == b)]["seed_time_median"]
            st = float(seed_t.iloc[0]) if len(seed_t) else float("nan")
            for i, (_, r) in enumerate(sub.iterrows()):
                first = i == 0
                lines.append(
                    f"{display(v) if first and b == 1 else ''} & "
                    f"{str(b) + chr(92) + ',s' if first else ''} & "
                    f"{_num(st, 3) if first else ''} & "
                    f"{_num(r.median_hybrid) if first else ''} & "
                    f"{display(r.reference)} & {_num(r.median_difference)} & "
                    f"{_num(r.rank_biserial)} & {_p(r.p_holm)} \\\\")
        lines.append(r"\addlinespace")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    _write("tab_hybrid.tex", "\n".join(lines))


# ------------------------------------------------------------------ 8. transfer
def tab_transfer() -> None:
    tr = pd.read_csv(DERIVED / "transfer_ranks.csv")
    info = json.loads((DERIVED / "transfer.json").read_text(encoding="utf-8"))
    s = pd.read_csv(DERIVED / "summary_by_budget.csv")
    s = s[s.budget_s == 300].set_index("method")
    tr = tr.sort_values("cec_mean_rank")
    lines = [
        r"\begin{table}[htbp]", r"\centering", r"\small",
        rf"\caption{{Continuous-benchmark standing against scheduling standing for the "
        rf"twelve optimisers that search through the shared decoder. Continuous ranks are "
        rf"mean ranks of the median error over {info['n_functions']} CEC2017 functions at "
        rf"$D=30$, each run 51 times to a budget of $10^4 D$ evaluations, and scheduling "
        rf"ranks are mean ranks over the 67 instances. "
        rf"Spearman correlation between the two mean-rank vectors is "
        rf"$\rho={info['rho_wallclock']:.2f}$ "
        rf"(95\,\% CI [{info['ci_wallclock'][0]:.2f}, {info['ci_wallclock'][1]:.2f}], "
        rf"$p={info['p_wallclock']:.3f}$).}}",
        r"\label{tab:transfer}",
        r"\begin{tabular}{lrrrr}", r"\toprule",
        r"Method & CEC2017 mean rank & Scheduling mean rank & "
        r"Equal-effort mean rank & Median deviation (\%) \\", r"\midrule",
    ]
    for _, r in tr.iterrows():
        lines.append(f"{display(r.method)} & {_num(r.cec_mean_rank)} & "
                     f"{_num(r.sched_mean_rank)} & {_num(r.sched_eval_mean_rank)} & "
                     f"{_num(s.loc[r.method, 'median'])} \\\\")
    lines += [r"\midrule",
              rf"Tabu search & -- & -- & -- & {_num(s.loc['tabu', 'median'])} \\",
              rf"Exact solver & -- & -- & -- & {_num(s.loc['cpsat', 'median'])} \\",
              r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    _write("tab_transfer.tex", "\n".join(lines))


# ------------------------------------------------------------------ 9. Bayesian
def tab_bayes() -> None:
    b = pd.read_csv(DERIVED / "bayesian_vs_tabu.csv")
    lines = [
        r"\begin{table}[htbp]", r"\centering", r"\small",
        r"\caption{Bayesian signed-rank comparison against the tabu search on per-instance "
        r"median deviation, with a region of practical equivalence of one deviation point. "
        r"Entries are posterior probabilities that the tabu search is practically better, "
        r"that the two are practically equivalent, and that the other method is practically "
        r"better.}",
        r"\label{tab:bayes}",
        r"\begin{tabular}{lrrr}", r"\toprule",
        r"Comparison & P(tabu better) & P(equivalent) & P(other better) \\", r"\midrule",
    ]
    for _, r in b.iterrows():
        lines.append(f"Tabu vs {display(r.right)} & {r.p_left:.3f} & {r.p_rope:.3f} & "
                     f"{r.p_right:.3f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    _write("tab_bayes.tex", "\n".join(lines))


# ------------------------------------------------------------------ 10. robustness
def tab_robustness() -> None:
    ab = json.loads((DERIVED / "ablation_decoder.json").read_text(encoding="utf-8"))
    se = pd.read_csv(DERIVED / "sensitivity_population.csv")
    lines = [
        r"\begin{table}[htbp]", r"\centering", r"\small",
        r"\caption{Robustness of the evaluation choices. The upper panel compares the "
        r"eligible-set machine mapping with a machine-index mapping repaired to the nearest "
        r"eligible machine, on 10 flexible instances and 8 decoded optimisers with 10 seeds "
        r"each. The lower panel compares three alternative population policies with the "
        r"common setting of 50, on 10 instances and 12 optimisers at a 60\,s budget.}",
        r"\label{tab:robustness}",
        r"\begin{tabular}{lrrrr}", r"\toprule",
        r"Comparison & Median (\%) & Median paired difference & $r_{rb}$ & $p$ \\",
        r"\midrule",
        rf"Index mapping & {_num(ab['median_legacy'])} & {_num(ab['median_difference'])} & "
        rf"{_num(ab['rank_biserial'])} & {_p(ab['p_value'])} \\",
        rf"Eligible-set mapping & {_num(ab['median_eligible'])} & -- & -- & -- \\",
        rf"\multicolumn{{5}}{{l}}{{Repairs per flexible operation and evaluation: "
        rf"{ab['legacy_repairs_per_flexible_operation']:.2f} under index mapping, "
        rf"{ab['eligible_repairs_per_flexible_operation']:.2f} under eligible-set mapping}} \\",
        r"\addlinespace",
    ]
    for _, r in se.iterrows():
        label = ("author-recommended rule" if r.setting == "recommended"
                 else f"population {r.setting}")
        lines.append(f"{label} vs population 50 & -- & "
                     f"{_num(r.median_difference_vs_common)} & {_num(r.rank_biserial)} & "
                     f"{_p(r.p_value)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    _write("tab_robustness.tex", "\n".join(lines))


# ------------------------------------------------------------------ 11. strata
def tab_strata() -> None:
    frames = {}
    for name in ("jssp", "fjsp", "small", "large", "proved", "unproved"):
        d = pd.read_csv(DERIVED / f"summary_{name}.csv").set_index("method")
        frames[name] = d["median"]
    tbl = pd.DataFrame(frames)
    order = pd.read_csv(DERIVED / "summary_by_budget.csv")
    order = order[order.budget_s == 300].sort_values("mean_rank")["method"].tolist()
    heads = {"jssp": "Job shop", "fjsp": "Flexible", "small": "Small", "large": "Large",
             "proved": "Proved", "unproved": "Unproved"}
    counts = {name: len(pd.read_csv(DERIVED / f"matrix_rpd_300s.csv")) for name in frames}
    lines = [
        r"\begin{table}[htbp]", r"\centering", r"\small",
        r"\caption{Median deviation at 300\,s within strata. Small and large split the "
        r"panel at the median operation count, and the last two columns split it by whether "
        r"the exact solver proved optimality within the budget.}",
        r"\label{tab:strata}",
        r"\begin{tabular}{l" + "r" * len(frames) + "}", r"\toprule",
        r"Method & " + " & ".join(heads[k] for k in frames) + r" \\", r"\midrule",
    ]
    for m in order:
        if m not in tbl.index:
            continue
        lines.append(f"{display(m)} & " + " & ".join(_num(tbl.loc[m, k]) for k in frames)
                     + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    _write("tab_strata.tex", "\n".join(lines))
    del counts


# ------------------------------------------------------------------ 12. per-instance appendix
def tab_instance_detail() -> None:
    inst = pd.read_parquet(DERIVED / "instances.parquet").set_index("instance_id")
    runs = pd.read_parquet(DERIVED / "full_runs.parquet")
    runs = runs[runs["feasible_final"].fillna(False)]
    nec = pd.read_csv(DERIVED / "necessity.csv")
    nec300 = nec[nec.budget_s == 300].set_index("instance_id")
    ex = runs[runs.method == "cpsat"]
    exs = ex.groupby("instance_id").agg(obj=("best_obj", "median"),
                                        proved=("status", lambda s: (s == "OPTIMAL").mean()),
                                        gap=("rel_gap", "median"))
    tabu = runs[runs.method == "tabu"].groupby("instance_id")["best_obj"].median()
    dec = (runs[runs.method.isin(DECODED)].groupby(["instance_id", "method"])["best_obj"]
           .median().reset_index())
    best_dec = dec.loc[dec.groupby("instance_id")["best_obj"].idxmin()].set_index("instance_id")

    head = (r"Instance & T & $n\times m$ & Ops & BKS & Exact & Pr & TS & "
            r"Decoded & Class \\")
    lines = [
        r"{\footnotesize\setlength{\tabcolsep}{4pt}",
        r"\begin{longtable}{llrrrrcrrl}",
        r"\caption{Per-instance results at the 300\,s budget. Exact, tabu (TS) and decoded "
        r"entries are medians over 20 seeds, and the decoded column gives the best of the "
        r"twelve decoded optimisers with the method attaining it. T is the problem class, J "
        r"for job shop and F for flexible, and Pr states whether the exact solver proved "
        r"optimality. Classes are ES for exact-sufficient, HN for heuristic-necessary, HU "
        r"for heuristic-useful, HY for hybrid-recommended and IN for inconclusive.}"
        r"\label{tab:instance-detail}\\",
        r"\toprule", head, r"\midrule", r"\endfirsthead",
        r"\toprule", head, r"\midrule", r"\endhead",
        r"\bottomrule", r"\endfoot",
    ]
    short = {"EXACT_SUFFICIENT": "ES", "HEURISTIC_NECESSARY": "HN",
             "HEURISTIC_USEFUL": "HU", "HYBRID_RECOMMENDED": "HY", "INCONCLUSIVE": "IN"}
    for iid in inst.index:
        if iid not in exs.index:
            continue
        r = inst.loc[iid]
        e = exs.loc[iid]
        cls = short.get(nec300.loc[iid, "class"], "--") if iid in nec300.index else "--"
        bd = best_dec.loc[iid] if iid in best_dec.index else None
        lines.append(
            f"{iid.replace('_', chr(92) + '_')} & {r['type'][0]} & "
            f"{r.n_jobs}$\\times${r.n_machines} & "
            f"{r.n_op} & {r.bks:.0f} & {e.obj:.0f} & "
            f"{'y' if e.proved >= 0.999 else ('p' if e.proved > 0 else 'n')} & "
            f"{tabu.get(iid, float('nan')):.0f} & "
            f"{bd.best_obj:.0f}\\,{display(bd.method)} & {cls} \\\\")
    lines.append(r"\end{longtable}}")
    _write("tab_instance_detail.tex", "\n".join(lines))


# ------------------------------------------------------------------ 13. reference run
def tab_reference() -> None:
    t = pd.read_csv(DERIVED / "reference_multithread.csv")
    lines = [
        r"\begin{table}[htbp]", r"\centering", r"\small",
        r"\caption{Multi-thread exact reference on the hardest subset, eight search workers "
        r"and a 900\,s budget, run alone. This run is a reference for the attainable "
        r"quality of the exact solver on this hardware and is not part of the timed "
        r"comparison.}",
        r"\label{tab:reference}",
        r"\begin{tabular}{lrrrrrr}", r"\toprule",
        r"Instance & Objective & Bound & Gap (\%) & Deviation (\%) & "
        r"Single thread (\%) & Tabu (\%) \\", r"\midrule",
    ]
    for _, r in t.sort_values("instance_id").iterrows():
        gap = r.rel_gap * 100 if np.isfinite(r.rel_gap) else float("nan")
        lines.append(f"{r.instance_id.replace('_', chr(92) + '_')} & {r.best_obj:.0f} & "
                     f"{r.best_bound:.0f} & {_num(gap)} & {_num(r.rpd_bks)} & "
                     f"{_num(r.single_rpd)} & {_num(r.tabu_rpd)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    _write("tab_reference.tex", "\n".join(lines))


def main() -> None:
    tab_instances()
    tab_methods()
    tab_configuration()
    tab_environment()
    tab_reference_provenance()
    tab_summary()
    tab_pairwise()
    tab_necessity()
    tab_proof()
    tab_hybrid()
    tab_transfer()
    tab_bayes()
    tab_robustness()
    tab_strata()
    tab_instance_detail()
    tab_reference()


if __name__ == "__main__":
    main()
