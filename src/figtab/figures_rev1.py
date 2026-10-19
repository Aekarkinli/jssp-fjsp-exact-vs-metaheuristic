"""Figures of the further experiments, in the visual grammar of ``figures.py``.

Each function states the question, the statistic, the encoding and the source file, as in
the main figure module. No value is typed into this file.

    .venv/Scripts/python.exe -m src.figtab.figures_rev1
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.analysis.panel import BUDGETS, DECODED, LINESTYLE, color, display  # noqa: E402
from src.figtab.style import FAINT, GREY, INK, MUTED, WIDTH, panel_label, save, set_style  # noqa: E402

DERIVED = Path("results/derived")
FIG = Path("paper/figures")


# --------------------------------------------------------------- feasibility coverage
def fig_coverage() -> None:
    """Question: how quickly does each method hold a feasible schedule. Statistic: share of
    instances on which at least half of a method's runs hold a feasible schedule by time t.
    Encoding: (a) one curve per method family on a log time axis, the twelve decoded
    optimisers as an envelope with their median; (b) the exact solver by instance size band. Uncertainty: none plotted. Source: rev1_feasibility_coverage*.csv."""
    cov = pd.read_csv(DERIVED / "rev1_feasibility_coverage.csv")
    bands = pd.read_csv(DERIVED / "rev1_feasibility_coverage_bands.csv")
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.7), sharey=True)
    fig.subplots_adjust(wspace=0.08)

    ax = axes[0]
    dec = cov[cov.method.isin(DECODED)].pivot(index="t", columns="method",
                                              values="share_instances")
    ax.fill_between(dec.index, dec.min(axis=1), dec.max(axis=1), color=color("ga"),
                    alpha=0.18, linewidth=0, label="decoded optimisers (range)")
    ax.plot(dec.index, dec.median(axis=1), color=color("ga"), linewidth=1.3,
            label="decoded optimisers (median)")
    for m in ("cpsat", "tabu", "dispatching", "greedy"):
        s = cov[cov.method == m].sort_values("t")
        lab = display(m) if m != "tabu" else "TS, SA (start from PDR)"
        ax.step(s.t, s.share_instances, where="post", color=color(m), linewidth=1.3,
                linestyle=LINESTYLE.get(m, "-"), label=lab)
    ax.set_xscale("log")
    ax.set_xlim(1e-4, 300)
    ax.set_ylim(0, 1.03)
    ax.set_xlabel("wall-clock time (s)")
    ax.set_ylabel("share of instances with a feasible schedule")
    ax.legend(fontsize=6.2, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2)
    panel_label(ax, "(a)", " ")

    ax = axes[1]
    styles = {"<=100": "-", "101-300": "--", ">300": ":"}
    for band in ("<=100", "101-300", ">300"):
        s = bands[(bands.method == "cpsat") & (bands.band == band)].sort_values("t")
        n = int(s.n_instances.iloc[0])
        label = f"{band.replace('<=', '≤')} ops ({n})"
        ax.step(s.t, s.share_instances, where="post", color=color("cpsat"), linewidth=1.3,
                linestyle=styles[band], label=label)
    for b in BUDGETS:
        ax.axvline(b, color=FAINT, linewidth=0.6, zorder=0)
    ax.set_xscale("log")
    ax.set_xlim(1e-4, 300)
    ax.set_xlabel("wall-clock time (s)")
    ax.legend(fontsize=6.2, loc="lower right",
              title="CP-SAT by instance size",
              title_fontsize=6.4)
    panel_label(ax, "(b)", " ")
    save(fig, FIG / "fig_coverage.pdf")


# --------------------------------------------------------------- sensitivity heatmap
def fig_sensitivity_grid() -> None:
    """Question: how sensitive is the classification to its two main thresholds. Statistic:
    number of instances whose class differs from the pre-specified rule (delta 1 percent,
    tau 10), with the number of heuristic-necessary instances in brackets. Encoding: one
    heatmap per budget, objective margin on the rows and time factor on the columns, the
    pre-specified cell outlined. Uncertainty: none. Source: rev1_sensitivity_grid.csv."""
    g = pd.read_csv(DERIVED / "rev1_sensitivity_grid.csv")
    deltas = sorted(g.delta.unique())
    taus = sorted(g.tau.unique())
    vmax = max(1, int(g.changed.max()))
    fig, axes = plt.subplots(1, 4, figsize=(WIDTH, 2.25), sharey=True)
    fig.subplots_adjust(wspace=0.08)
    for k, (ax, B) in enumerate(zip(axes, BUDGETS)):
        s = g[g.budget_s == B]
        mat = s.pivot(index="delta", columns="tau", values="changed").loc[deltas, taus]
        nec = s.pivot(index="delta", columns="tau",
                      values="n_HEURISTIC_NECESSARY").loc[deltas, taus]
        ax.imshow(mat.to_numpy(float), cmap="Blues", vmin=0, vmax=vmax * 1.15,
                  aspect="auto")
        for i in range(len(deltas)):
            for j in range(len(taus)):
                v = int(mat.iloc[i, j])
                ax.text(j, i, f"{v}\n({int(nec.iloc[i, j])})", ha="center", va="center",
                        fontsize=6.0, color="white" if v > vmax * 0.6 else INK,
                        linespacing=0.95)
        i0, j0 = deltas.index(0.01), taus.index(10.0)
        ax.add_patch(plt.Rectangle((j0 - 0.5, i0 - 0.5), 1, 1, fill=False, edgecolor=INK,
                                   linewidth=1.2))
        ax.set_xticks(range(len(taus)))
        ax.set_xticklabels([f"{t:g}" for t in taus])
        ax.set_yticks(range(len(deltas)))
        ax.set_yticklabels([f"{100 * d:g}%" for d in deltas])
        ax.set_xlabel(r"time factor $\tau$")
        ax.grid(False)
        ax.set_title(f"({'abcd'[k]}) {B} s", fontsize=8, loc="left")
    axes[0].set_ylabel(r"objective margin $\delta$")
    save(fig, FIG / "fig_sensitivity_grid.pdf")


# --------------------------------------------------------------- matched flexibility
LEVELS = ["sdata", "edata", "rdata", "vdata"]
LEVEL_LABEL = {"sdata": "job shop\n(sdata)", "edata": "low\n(edata)",
               "rdata": "medium\n(rdata)", "vdata": "high\n(vdata)"}


def fig_hurink() -> None:
    """Question: does routing freedom itself reduce proof success. Statistic: per level of
    the matched Hurink sets, (a) the number of base problems proved within the budget,
    (b) the final duality gap of the exact solver and (c) the difference between the tabu
    search and the exact incumbent, with one thin line per base problem. Encoding: ordered
    categorical axis of flexibility, thin within-base trajectories under a median line.
    Uncertainty: the within-base spread is shown directly. Source: rev1_hurink_instances.csv."""
    df = pd.read_csv(DERIVED / "rev1_hurink_instances.csv")
    x = np.arange(len(LEVELS))
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 2.5))
    fig.subplots_adjust(wspace=0.42)

    ax = axes[0]
    counts = [int(df[df.level == lv].proved.sum()) for lv in LEVELS]
    totals = [int((df.level == lv).sum()) for lv in LEVELS]
    ax.bar(x, totals, color=FAINT, width=0.62, zorder=1)
    ax.bar(x, counts, color=color("cpsat"), width=0.62, zorder=2)
    for xi, c, n in zip(x, counts, totals):
        ax.text(xi, c + max(totals) * 0.02, f"{c}/{n}", ha="center", fontsize=6.2)
    ax.set_ylabel("base problems proved in 300 s")
    panel_label(ax, "(a)", " ")

    for ax, col, ylabel, letter, mcol in (
            (axes[1], "gap", "exact duality gap (%)", "(b)", "cpsat"),
            (axes[2], "tabu_minus_exact", "tabu minus exact incumbent (%)", "(c)", "tabu")):
        piv = df.pivot(index="base", columns="level", values=col)[LEVELS]
        for _, row in piv.iterrows():
            ax.plot(x, row.to_numpy(float), color=GREY, linewidth=0.5, alpha=0.6, zorder=1)
        ax.plot(x, piv.median().to_numpy(float), color=color(mcol), linewidth=1.8,
                marker="o", markersize=3.5, zorder=3, label="median over base problems")
        ax.set_ylabel(ylabel)
        ax.axhline(0, color=MUTED, linewidth=0.6, zorder=0)
        panel_label(ax, letter, " ")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels([LEVEL_LABEL[lv] for lv in LEVELS], fontsize=6.0)
        ax.grid(axis="x", visible=False)
    save(fig, FIG / "fig_hurink.pdf")


# --------------------------------------------------------------- representation
def fig_representation() -> None:
    """Question: do the decoded optimisers and their ordering depend on the representation.
    Statistic: median deviation at 300 s on the revision subset under the random-key
    active-schedule decoder and under the operation-based permutation, per optimiser, with
    the tabu search and the exact solver as references. Encoding: dumbbell per optimiser.
    Uncertainty: none plotted, the paired tests are in the text. Source:
    rev1_representation.csv, rev1_representation.json."""
    tab = pd.read_csv(DERIVED / "rev1_representation.csv")
    info = json.loads((DERIVED / "rev1_representation.json").read_text(encoding="utf-8"))
    t = tab[tab.budget == 300].sort_values("median_randomkey", ascending=False)
    y = np.arange(len(t))
    fig, ax = plt.subplots(figsize=(WIDTH * 0.7, 3.0))
    for yi, (_, r) in zip(y, t.iterrows()):
        ax.plot([r.median_randomkey, r.median_opperm], [yi, yi], color=GREY, linewidth=1.0,
                zorder=1)
        ax.plot(r.median_randomkey, yi, "o", color=color(r.method), markersize=4.5, zorder=3)
        ax.plot(r.median_opperm, yi, "D", markerfacecolor="white",
                markeredgecolor=color(r.method), markersize=4.5, zorder=3)
    # the exact solver sits at zero on this subset, so only the tabu search is drawn as a
    # line and the exact solver is named in the caption
    ax.axvline(info["tabu_median"], color=color("tabu"), linewidth=1.0, zorder=0)
    ax.text(info["tabu_median"] + 0.25, len(t) - 0.55, "TS", fontsize=6.6,
            color=color("tabu"))
    ax.set_yticks(y)
    ax.set_yticklabels([display(m) for m in t.method])
    ax.set_xlabel("median deviation at 300 s on the evaluation subset (%)")
    ax.set_xlim(left=0)
    ax.grid(axis="y", visible=False)
    ax.plot([], [], "o", color=MUTED, label="random keys, active-schedule decoder")
    ax.plot([], [], "D", markerfacecolor="white", markeredgecolor=MUTED,
            label="operation-based permutation")
    ax.legend(fontsize=6.4, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2)
    save(fig, FIG / "fig_representation.pdf")


# --------------------------------------------------------------- memetic
def fig_memetic() -> None:
    """Question: how much of the distance to the tabu search does the problem-specific move
    set close. Statistic: median deviation at four budgets on the revision subset for each
    of four optimisers without and with critical-block descent, and for the tabu search.
    Encoding: small multiples, one per optimiser, budgets on a log axis. Uncertainty: none
    plotted. Source: rev1_memetic.csv."""
    tab = pd.read_csv(DERIVED / "rev1_memetic.csv")
    methods = list(dict.fromkeys(tab.method))
    fig, axes = plt.subplots(1, len(methods), figsize=(WIDTH, 2.3), sharey=True)
    fig.subplots_adjust(wspace=0.08)
    for k, (ax, m) in enumerate(zip(axes, methods)):
        s = tab[tab.method == m].sort_values("budget")
        ax.plot(s.budget, s.median_plain, color=color(m), linewidth=1.3, marker="o",
                markersize=3, label="decoder only")
        ax.plot(s.budget, s.median_memetic, color=color(m), linewidth=1.3, linestyle="--",
                marker="D", markersize=3, markerfacecolor="white", label="with descent")
        ax.plot(s.budget, s.median_tabu, color=color("tabu"), linewidth=1.1, label="TS")
        ax.set_xscale("log")
        ax.set_xticks(list(BUDGETS))
        ax.set_xticklabels([str(b) for b in BUDGETS])
        ax.set_xlabel("budget (s)")
        ax.set_title(f"({'abcd'[k]}) {display(m)}", fontsize=8, loc="left")
        if k == 0:
            ax.set_ylabel("median deviation (%)")
            ax.legend(fontsize=6.0, loc="upper right")
    save(fig, FIG / "fig_memetic.pdf")


# --------------------------------------------------------------- variants
# vertical label offsets for points that sit close together
VARIANT_LABEL_DY = {"DE→L-SHADE": 6, "DE→IMODE": 2, "DE→JADE": -6,
                    "SHADE→L-SHADE": 6, "JADE→SHADE": -6}
def fig_variants() -> None:
    """Question: does an improvement on the continuous suite carry over to scheduling.
    Statistic: for each base-variant pair, the share of CEC2017 functions on which the
    variant has the lower median error against the share of revision instances on which it
    has the lower median deviation at 300 s. Encoding: labelled scatter with the two
    one-half lines. Uncertainty: none plotted. Source: rev1_variant_lineage.csv."""
    t = pd.read_csv(DERIVED / "rev1_variant_lineage.csv")
    fig, ax = plt.subplots(figsize=(WIDTH * 0.55, 2.9))
    xs = t.cec_variant_better / t.cec_functions
    ys = t.sched_variant_better / t.sched_n
    ax.axvline(0.5, color=GREY, linewidth=0.8, linestyle="--", zorder=0)
    ax.axhline(0.5, color=GREY, linewidth=0.8, linestyle="--", zorder=0)
    # pairs that land on the same point share one label
    groups: dict[tuple[float, float], list[str]] = {}
    for xv, yv, (_, r) in zip(xs, ys, t.iterrows()):
        ax.scatter(xv, yv, s=30, color=color(r.variant), edgecolor="white", linewidth=0.6,
                   zorder=3)
        groups.setdefault((round(xv, 3), round(yv, 3)), []).append(
            f"{display(r.base)}→{display(r.variant)}")
    for (xv, yv), names in groups.items():
        dx, ha = (-5, "right") if xv > 0.8 else (5, "left")
        dy = VARIANT_LABEL_DY.get(names[0], 3)
        ax.annotate("\n".join(names), (xv, yv), textcoords="offset points",
                    xytext=(dx, dy), fontsize=5.8, ha=ha, va="center")
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.03)
    ax.set_aspect("equal")
    ax.set_xlabel("share of CEC2017 functions improved")
    ax.set_ylabel("share of scheduling instances improved")
    save(fig, FIG / "fig_variants.pdf")


# --------------------------------------------------------------- large scale
def fig_large() -> None:
    """Question: does the picture change on larger and more flexible instances. Statistic:
    per-instance median deviation at 300 s on the large-scale extension, per method,
    ordered by median. Encoding: horizontal box plot with one point per instance, coloured by
    role, job shop as circles and flexible job shop as triangles. Uncertainty: the spread
    over instances is shown directly. Source: rev1_large_per_instance.csv, instances."""
    per = pd.read_csv(DERIVED / "rev1_large_per_instance.csv", index_col=0)
    order = per.median().sort_values(ascending=False).index.tolist()
    fjsp = per.index.str.startswith(("behnke", "dp"))
    fig, ax = plt.subplots(figsize=(WIDTH, 3.9))
    data = [per[m].dropna().to_numpy(float) for m in order]
    bp = ax.boxplot(data, vert=False, widths=0.55, showfliers=False, patch_artist=True)
    for patch, m in zip(bp["boxes"], order):
        patch.set_facecolor("white")
        patch.set_edgecolor(color(m))
    for med in bp["medians"]:
        med.set_color(INK)
    for i, m in enumerate(order, start=1):
        v = per[m]
        ax.scatter(v[~fjsp], np.full((~fjsp).sum(), i) + RNG_JITTER((~fjsp).sum()), s=9,
                   marker="o", color=color(m), alpha=0.8, zorder=3)
        ax.scatter(v[fjsp], np.full(fjsp.sum(), i) + RNG_JITTER(fjsp.sum()), s=11,
                   marker="^", color=color(m), alpha=0.8, zorder=3)
    ax.set_yticks(range(1, len(order) + 1))
    ax.set_yticklabels([display(m) for m in order])
    ax.set_xlabel("deviation from the best-known value at 300 s (%)")
    ax.scatter([], [], marker="o", color=MUTED, s=9, label="job shop")
    ax.scatter([], [], marker="^", color=MUTED, s=11, label="flexible job shop")
    ax.legend(fontsize=6.4, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2,
              frameon=False)
    ax.grid(axis="y", visible=False)
    save(fig, FIG / "fig_large.pdf")


_RNG = np.random.default_rng(20261013)


def RNG_JITTER(n: int) -> np.ndarray:  # noqa: N802 - small helper used as a function
    return _RNG.uniform(-0.18, 0.18, n)


FIGURES = [fig_coverage, fig_sensitivity_grid, fig_hurink, fig_representation, fig_memetic,
           fig_variants, fig_large]


def main() -> None:
    set_style()
    FIG.mkdir(parents=True, exist_ok=True)
    for fn in FIGURES:
        try:
            fn()
            print("wrote", fn.__name__)
        except FileNotFoundError as exc:
            print("skipped", fn.__name__, "-", exc)


if __name__ == "__main__":
    main()
