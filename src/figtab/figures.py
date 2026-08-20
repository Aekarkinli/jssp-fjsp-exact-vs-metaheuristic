"""Every manuscript figure, regenerated from the derived tables by the build pipeline.

Each function states, in its own docstring, the scientific question, the statistic plotted,
the visual encoding, the uncertainty shown and the source file, so that a change of
appearance can never quietly change a scientific meaning. No value is typed into this file.

    uv run python -m src.figtab.figures
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

from src.analysis.panel import (  # noqa: E402
    BUDGETS,
    CLASS_COLOR,
    CLASS_LABEL,
    CLASS_ORDER,
    DECODED,
    FAMILY,
    FAMILY_COLOR,
    FAMILY_LABEL,
    LINESTYLE,
    MARKER,
    PANEL,
    color,
    display,
)
from src.figtab.style import FAINT, GREY, INK, MUTED, WIDTH, panel_label, save, set_style  # noqa: E402

DERIVED = Path("results/derived")
FIG = Path("paper/figures")
RNG = np.random.default_rng(20260817)


def _matrix(name: str = "matrix_rpd_300s.csv") -> pd.DataFrame:
    return pd.read_csv(DERIVED / name, index_col=0)


def _summary(budget: int = 300) -> pd.DataFrame:
    s = pd.read_csv(DERIVED / "summary_by_budget.csv")
    return s[s.budget_s == budget].set_index("method")


def _order() -> list[str]:
    """Presentation order used by every figure that lists methods: overall mean rank."""
    return _summary().sort_values("mean_rank").index.tolist()


def _family_handles(families):
    return [mpatches.Patch(facecolor=FAMILY_COLOR[f], label=FAMILY_LABEL[f],
                           edgecolor="white", linewidth=0.5) for f in families]


# --------------------------------------------------------------------- 1. design
def fig_overview() -> None:
    """Question: how is the comparison organised. Statistic: none, a design schematic.
    Encoding: lanes and connectors. Uncertainty: none. Source: the study design."""
    fig, ax = plt.subplots(figsize=(WIDTH, 2.05))
    ax.set_axis_off()
    ax.set_xlim(0, 100)
    ax.set_ylim(-2, 42)

    def box(x, y, w, h, text, edge, fs=7.0):
        ax.add_patch(mpatches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.5,rounding_size=1.2",
            linewidth=0.7, edgecolor=edge, facecolor="white"))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=INK)

    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", linewidth=0.6, color=MUTED,
                                    shrinkA=2, shrinkB=2))

    box(0, 15, 14, 12, "67 instances\n37 job shop\n30 flexible", GREY)
    box(20, 30, 24, 8.5, "Exact solver", FAMILY_COLOR["exact"])
    box(20, 17.5, 24, 8.5, "Problem-specific search", FAMILY_COLOR["problem_specific"])
    box(20, 3, 24, 11, "Continuous optimisers\nthrough one shared decoder",
        FAMILY_COLOR["recent"])
    box(50, 15, 20, 12, "Anytime traces\nwall clock and\nevaluation count", GREY)
    box(77, 24, 23, 9, "Comparison at\n1, 10, 60, 300 s", GREY)
    box(77, 9, 23, 9, "Necessity class per\ninstance and budget", GREY)

    for y in (34, 21.5, 8.5):
        arrow(14, 21, 20, y)
        arrow(44, y, 50, 21)
    arrow(70, 22, 77, 28)
    arrow(70, 20, 77, 14)
    ax.text(32, -1.5, "20 seeds per stochastic method, one performance core per run",
            ha="center", fontsize=6.6, color=MUTED)
    save(fig, FIG / "fig_overview.pdf")


# --------------------------------------------------------------------- 2. proof
def fig_proof() -> None:
    """Question: where does the exact solver prove optimality. Statistic: per-instance proof
    outcome and time to proof. Encoding: jittered outcomes on a log size axis with the fitted
    logistic curve, and a cumulative proof curve. Uncertainty: none plotted, the model is a
    point estimate. Source: proof_by_instance.csv, proof_model.json, full_runs.parquet."""
    p = pd.read_csv(DERIVED / "proof_by_instance.csv")
    runs = pd.read_parquet(DERIVED / "full_runs.parquet")
    ex = runs[runs.method == "cpsat"]
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.35))
    fig.subplots_adjust(wspace=0.30)

    ax = axes[0]
    m = json.loads((DERIVED / "proof_model.json").read_text(encoding="utf-8"))["params"]
    xs = np.logspace(np.log10(p.n_op.min()), np.log10(p.n_op.max()), 200)
    lin = (m["Intercept"] + m["log_n_op"] * np.log(xs)
           + m["n_machines"] * p.n_machines.median()
           + m["flex_ratio"] * p.flex_ratio.median()
           + m["cv_duration"] * p.cv_duration.median())
    ax.plot(xs, 1 / (1 + np.exp(-lin)), color=MUTED, linewidth=1.0, linestyle="--", zorder=1)
    for t, mk in (("JSSP", "o"), ("FJSP", "s")):
        s = p[p.type == t]
        jitter = RNG.uniform(-0.05, 0.05, len(s))
        face = FAMILY_COLOR["problem_specific"] if t == "JSSP" else "white"
        edge = FAMILY_COLOR["problem_specific"] if t == "JSSP" else INK
        ax.scatter(s.n_op, s.proof_rate + jitter + (0.07 if t == "JSSP" else -0.07), s=16,
                   marker=mk, facecolor=face, edgecolor=edge, linewidth=0.7, alpha=0.9,
                   label=t, zorder=3)
    ax.set_xscale("log")
    ax.set_xlabel("operations per instance")
    ax.set_ylabel("optimality proved within 300 s")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["no", "yes"])
    ax.set_ylim(-0.32, 1.32)
    ax.legend(loc="center left", fontsize=6.8)
    ax.text(0.98, 0.60, "fitted model", transform=ax.transAxes, fontsize=6.6,
            color=MUTED, ha="right")
    panel_label(ax, "(a)")

    ax = axes[1]
    proved = ex[(ex.status == "OPTIMAL") & (ex.wall_time <= 300)]
    t = proved.groupby("instance_id")["wall_time"].median().sort_values()
    ax.step(t.to_numpy(), np.arange(1, len(t) + 1) / 67 * 100, where="post",
            color=FAMILY_COLOR["exact"], linewidth=1.4)
    ax.set_xscale("log")
    ax.set_xlabel("time to proof (s)")
    ax.set_ylabel("instances proved (%)")
    ax.set_ylim(0, 64)
    for b in BUDGETS:
        ax.axvline(b, color=GREY, linewidth=0.5, linestyle=":", zorder=0)
        ax.text(b, 61, f"{b} s", fontsize=6.4, ha="center", color=MUTED)
    panel_label(ax, "(b)")
    save(fig, FIG / "fig_proof.pdf")


# --------------------------------------------------------------------- 3. flexibility
def fig_flexibility() -> None:
    """Question: does routing freedom change proof success and approximation quality.
    Statistic: proof outcome and median deviation on five matched base instances at three
    flexibility levels. Encoding: a binary tile matrix and matched trajectories, so the
    matched design stays visible. Uncertainty: none, the five matched instances are shown.
    Source: flexibility_proof.csv, flexibility_rpd.csv."""
    proof = pd.read_csv(DERIVED / "flexibility_proof.csv")
    rpd = pd.read_csv(DERIVED / "flexibility_rpd.csv")
    levels = ["edata", "rdata", "vdata"]
    labels = ["low\n(edata)", "medium\n(rdata)", "high\n(vdata)"]
    bases = sorted(proof.base.unique())
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH * 0.88, 2.4))
    fig.subplots_adjust(wspace=0.42)

    ax = axes[0]
    grid = np.array([[proof[(proof.level == lv) & (proof.base == b)]["proved"].iloc[0]
                      for lv in levels] for b in bases])
    ax.imshow(grid, cmap=ListedColormap(["white", FAMILY_COLOR["exact"]]), vmin=0, vmax=1,
              aspect="auto")
    for i in range(len(bases)):
        for j in range(3):
            ax.add_patch(mpatches.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                            edgecolor=MUTED, linewidth=0.5))
    ax.set_xticks(range(3))
    ax.set_xticklabels(labels)
    ax.set_yticks(range(len(bases)))
    ax.set_yticklabels(bases)
    ax.set_ylabel("matched base instance")
    ax.grid(False)
    for j in range(3):
        ax.text(j, -0.9, f"{int(grid[:, j].sum())}/{len(bases)}", ha="center", fontsize=7,
                color=INK)
    ax.set_ylim(len(bases) - 0.5, -1.25)
    panel_label(ax, "(a)")

    ax = axes[1]
    best_decoded = [m for m in _order() if m in DECODED][0]
    for m in ("tabu", best_decoded):
        for b in bases:
            v = [rpd[(rpd.level == lv) & (rpd.method == m)
                     & (rpd.instance_id.str.endswith(b))]["rpd_at_300s"].median()
                 for lv in levels]
            ax.plot(range(3), v, color=color(m), linewidth=0.5, alpha=0.35, zorder=1)
        med = [rpd[(rpd.level == lv) & (rpd.method == m)]["rpd_at_300s"].median()
               for lv in levels]
        ax.plot(range(3), med, color=color(m), linewidth=1.6, marker=MARKER.get(m, "o"),
                markersize=4.2, markeredgecolor="white", markeredgewidth=0.5,
                linestyle=LINESTYLE.get(m, "-"), label=display(m), zorder=3)
    ax.set_xticks(range(3))
    ax.set_xticklabels(labels)
    ax.set_ylabel("median deviation at 300 s (%)")
    ax.set_xlim(-0.25, 2.25)
    ax.legend(fontsize=6.8, loc="upper left")
    ax.text(0.98, 0.04, f"thin lines: the {len(bases)} matched instances",
            transform=ax.transAxes, fontsize=6.4, color=MUTED, ha="right")
    panel_label(ax, "(b)")
    for a in axes:
        a.set_xlabel("machine flexibility")
    save(fig, FIG / "fig_flexibility.pdf")


# --------------------------------------------------------------------- 4. necessity
def fig_necessity() -> None:
    """Question: what role does heuristic search play per instance and budget. Statistic:
    the necessity class of every instance at four budgets. Encoding: a categorical matrix of
    all 67 instances against the four budgets, split into two columns for legibility and
    ordered by operation count, with the marginal class shares beside it. Uncertainty: the
    classes already embed the tests. Source: necessity.csv."""
    t = pd.read_csv(DERIVED / "necessity.csv")
    order = t[t.budget_s == 300].sort_values(["n_op", "instance_id"])["instance_id"].tolist()
    meta = t[t.budget_s == 300].set_index("instance_id")
    idx = {c: i for i, c in enumerate(CLASS_ORDER)}
    half = (len(order) + 1) // 2
    blocks = [order[:half], order[half:]]

    fig = plt.figure(figsize=(WIDTH, 4.3))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.15], wspace=0.75)
    cmap = ListedColormap([CLASS_COLOR[c] for c in CLASS_ORDER])
    for k, block in enumerate(blocks):
        ax = fig.add_subplot(gs[0, k])
        grid = np.full((len(block), len(BUDGETS)), np.nan)
        for i, inst in enumerate(block):
            for j, b in enumerate(BUDGETS):
                row = t[(t.instance_id == inst) & (t.budget_s == b)]
                if len(row):
                    grid[i, j] = idx[row["class"].iloc[0]]
        ax.imshow(grid, cmap=cmap, vmin=-0.5, vmax=len(CLASS_ORDER) - 0.5, aspect="auto",
                  interpolation="nearest")
        ax.set_xticks(range(len(BUDGETS)))
        ax.set_xticklabels([f"{b}" for b in BUDGETS])
        ax.set_yticks(range(len(block)))
        ax.set_yticklabels([f"{i}  {int(meta.loc[i, 'n_op'])}" for i in block], fontsize=6.0)
        for tick, inst in zip(ax.get_yticklabels(), block):
            tick.set_color(INK if meta.loc[inst, "type"] == "JSSP" else MUTED)
        ax.set_xlabel("budget (s)")
        ax.grid(False)
        for j in range(1, len(BUDGETS)):
            ax.axvline(j - 0.5, color="white", linewidth=0.7)
        panel_label(ax, "(%s)" % "ab"[k])
    fig.text(-0.055, 0.5, "instance and operation count, ordered by size", rotation=90,
             va="center", ha="center", fontsize=8, color=INK)

    ax = fig.add_subplot(gs[0, 2])
    counts = (t.groupby(["budget_s", "class"]).size().unstack(fill_value=0)
              .reindex(columns=CLASS_ORDER, fill_value=0))
    share = counts.div(counts.sum(axis=1), axis=0) * 100
    y = np.arange(len(share))[::-1]
    left = np.zeros(len(share))
    for cls in CLASS_ORDER:
        ax.barh(y, share[cls], left=left, color=CLASS_COLOR[cls], height=0.5,
                edgecolor="white", linewidth=0.7)
        for i, (v, l0, n) in enumerate(zip(share[cls], left, counts[cls])):
            if v >= 9:
                ax.text(l0 + v / 2, y[i], f"{int(n)}", ha="center", va="center",
                        fontsize=6.4, color="white" if cls != "INCONCLUSIVE" else INK)
        left += share[cls].to_numpy()
    ax.set_yticks(y)
    ax.set_yticklabels([f"{int(b)} s" for b in share.index])
    ax.set_xlabel("share of instances (%)")
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.7, len(share) - 0.3)
    ax.grid(axis="y", visible=False)
    panel_label(ax, "(c)")

    handles = [mpatches.Patch(color=CLASS_COLOR[c], label=CLASS_LABEL[c]) for c in CLASS_ORDER]
    handles += [plt.Line2D([], [], linestyle="none", marker="s", color=INK, markersize=5,
                           label="job shop (label)"),
                plt.Line2D([], [], linestyle="none", marker="s", color=MUTED, markersize=5,
                           label="flexible job shop (label)")]
    fig.legend(handles=handles, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.07),
               fontsize=6.8)
    save(fig, FIG / "fig_necessity.pdf")


# --------------------------------------------------------------------- 5. map
def fig_necessity_map() -> None:
    """Question: where in the size and advantage plane do the classes fall. Statistic:
    percentage advantage of the fixed reference heuristic over the exact incumbent.
    Encoding: log size against advantage, colour by class and marker by proof status.
    Uncertainty: the classification carries it. Source: necessity.csv."""
    t = pd.read_csv(DERIVED / "necessity.csv")
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.9), sharey=True, sharex=True)
    fig.subplots_adjust(wspace=0.06)
    for k, (ax, B) in enumerate(zip(axes, (10, 300))):
        s = t[t.budget_s == B].copy()
        s["adv"] = (s.z_exact - s.z_heuristic) / s.z_exact * 100
        s.loc[~np.isfinite(s.adv), "adv"] = 26.0
        ax.axhspan(-14, 0, color=FAINT, zorder=0, linewidth=0)
        ax.axhline(0, color=MUTED, linewidth=0.7, zorder=1)
        ax.axhline(1, color=MUTED, linewidth=0.6, linestyle="--", zorder=1)
        for cls in CLASS_ORDER:
            for proven, marker in ((True, "s"), (False, "o")):
                g = s[(s["class"] == cls) & (s["proven"] == proven)]
                if g.empty:
                    continue
                ax.scatter(g.n_op, g.adv, s=26, marker=marker,
                           facecolor=CLASS_COLOR[cls] if proven else "white",
                           edgecolor=CLASS_COLOR[cls], linewidth=0.9, zorder=3)
        if B == 300:
            nec = s[s["class"] == "HEURISTIC_NECESSARY"].sort_values("n_op")
            for i, (_, r) in enumerate(nec.iterrows()):
                ax.annotate(r.instance_id, (r.n_op, r.adv), textcoords="offset points",
                            xytext=(7, 4 if i % 2 == 0 else -10), fontsize=6.0, color=INK,
                            arrowprops=dict(arrowstyle="-", linewidth=0.4, color=GREY,
                                            shrinkA=0, shrinkB=2))
        ax.set_xscale("log")
        ax.set_xlabel("operations per instance")
        ax.set_ylim(-14, 31)
        panel_label(ax, "(a)" if k == 0 else "(b)", f"{B} s budget")
    axes[0].set_ylabel("advantage of tabu search over\nthe exact incumbent (%)")
    axes[0].text(0.02, 0.03, "exact incumbent better below 0", transform=axes[0].transAxes,
                 fontsize=6.4, color=MUTED)
    axes[1].text(0.98, 0.05, "dashed line: 1 % threshold", transform=axes[1].transAxes,
                 fontsize=6.4, color=MUTED, ha="right")
    handles = [mpatches.Patch(color=CLASS_COLOR[c], label=CLASS_LABEL[c]) for c in CLASS_ORDER]
    handles += [plt.Line2D([], [], marker="s", linestyle="none", color=MUTED,
                           label="optimality proved"),
                plt.Line2D([], [], marker="o", linestyle="none", markerfacecolor="white",
                           markeredgecolor=MUTED, color=MUTED, label="no proof")]
    fig.legend(handles=handles, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.13),
               fontsize=6.8)
    save(fig, FIG / "fig_necessity_map.pdf")


# --------------------------------------------------------------------- 6. distribution
def fig_deviation() -> None:
    """Question: how do the methods compare in quality and spread. Statistic: per-instance
    median deviation from the best-known solution at 300 s. Encoding: horizontal box plot
    with every instance shown as a jittered point. Uncertainty: the distribution itself.
    Source: matrix_rpd_300s.csv."""
    mat = _matrix()
    order = [m for m in _order() if m in mat.columns][::-1]
    fig, ax = plt.subplots(figsize=(WIDTH, 4.1))
    for i, m in enumerate(order):
        v = mat[m].to_numpy(float)
        ax.boxplot(v, positions=[i], vert=False, widths=0.55, showfliers=False,
                   patch_artist=True,
                   boxprops=dict(facecolor=color(m), alpha=0.30, edgecolor=color(m),
                                 linewidth=0.8),
                   medianprops=dict(color=color(m), linewidth=1.7),
                   whiskerprops=dict(linewidth=0.7, color=MUTED),
                   capprops=dict(linewidth=0.7, color=MUTED))
        ax.scatter(v, i + RNG.uniform(-0.17, 0.17, v.size), s=5, color=color(m), alpha=0.55,
                   linewidth=0, zorder=3)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([display(m) for m in order])
    for tick, m in zip(ax.get_yticklabels(), order):
        if m in ("cpsat", "tabu"):
            tick.set_fontweight("bold")
    ax.set_xlabel("deviation from the best-known solution at 300 s (%)")
    xmax = 100
    clipped = int((mat.to_numpy() > xmax).sum())
    ax.set_xlim(-3, xmax)
    ax.set_ylim(-0.8, len(order) - 0.2)
    ax.grid(axis="y", visible=False)
    fams = [f for f in FAMILY_LABEL if f in {FAMILY[m] for m in order}]
    ax.legend(handles=_family_handles(fams), ncol=2, loc="upper right", fontsize=6.8)
    ax.text(0.995, 0.845, "one point per instance, box shows the quartiles, "
            f"{clipped} values lie beyond the axis", transform=ax.transAxes, fontsize=6.3,
            color=MUTED, ha="right")
    save(fig, FIG / "fig_deviation.pdf")


# --------------------------------------------------------------------- 7. mean ranks
def _rank_panel(ax, mat: pd.DataFrame, info: dict, title: str = "",
                order: list[str] | None = None) -> None:
    ranks = mat.rank(axis=1, method="average")
    mean = ranks.mean()
    rng = np.random.default_rng(7)
    arr = ranks.to_numpy()
    boot = np.array([arr[rng.integers(0, arr.shape[0], arr.shape[0])].mean(axis=0)
                     for _ in range(2000)])
    lo = pd.Series(np.percentile(boot, 2.5, axis=0), index=mean.index)
    hi = pd.Series(np.percentile(boot, 97.5, axis=0), index=mean.index)
    seq = [m for m in (order if order is not None else mean.sort_values().index) if m in mean.index]
    cd = info["cd"]
    ax.axvspan(mean.min(), mean.min() + cd, color=FAINT, zorder=0, linewidth=0)
    y = np.arange(len(seq))[::-1]
    for yi, m in zip(y, seq):
        emph = m in ("cpsat", "tabu")
        ax.plot([lo[m], hi[m]], [yi, yi], color=color(m), linewidth=1.9 if emph else 1.1,
                alpha=0.95 if emph else 0.7, solid_capstyle="butt", zorder=2)
        ax.plot(mean[m], yi, marker=MARKER.get(m, "o"), color=color(m),
                markersize=5.2 if emph else 4.0, markeredgecolor="white",
                markeredgewidth=0.6, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels([display(m) for m in seq])
    for tick, m in zip(ax.get_yticklabels(), seq):
        if m in ("cpsat", "tabu"):
            tick.set_fontweight("bold")
    ax.grid(axis="y", visible=False)
    hi_lim = float(mean.max() + 2.4)
    ax.set_xlim(0.2, hi_lim)
    ax.set_ylim(-0.8, len(seq) - 0.2)
    for yi, m in zip(y, seq):
        ax.text(hi_lim - 0.1, yi, f"{mean[m]:.1f}", fontsize=6.2, color=MUTED,
                va="center", ha="right")
    ax.text(mean.min() + cd, len(seq) - 0.55, f" critical difference {cd:.2f}",
            fontsize=6.3, color=MUTED, va="center")
    if title:
        ax.set_title(title, fontsize=8, loc="left")


def fig_rank() -> None:
    """Question: how do the methods rank overall. Statistic: mean rank of the per-instance
    median deviation at 300 s. Encoding: dot with a 95 percent bootstrap interval of the mean
    rank, and a neutral band of one Nemenyi critical difference from the leader. Uncertainty:
    percentile bootstrap over instances, 2000 resamples. Source: matrix_rpd_300s.csv."""
    info = json.loads((DERIVED / "stats_summary.json").read_text(encoding="utf-8"))["cd_panel"]
    fig, ax = plt.subplots(figsize=(WIDTH * 0.76, 3.6))
    _rank_panel(ax, _matrix(), info)
    ax.set_xlabel("mean rank over 67 instances (lower is better)")
    save(fig, FIG / "fig_rank.pdf")


def fig_rank_strata() -> None:
    """Question: does the ranking hold inside each problem class. Statistic and encoding as
    in fig_rank, with the pooled order kept in both panels so displacement is readable.
    Source: full_runs.parquet through median_matrix."""
    from src.analysis.stats import median_matrix
    summary = json.loads((DERIVED / "stats_summary.json").read_text(encoding="utf-8"))
    runs = pd.read_parquet(DERIVED / "full_runs.parquet")
    runs = runs[runs["feasible_final"].fillna(False)]
    pooled = _order()
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 3.4), sharex=True)
    fig.subplots_adjust(wspace=0.42)
    for k, (ax, key, label) in enumerate(zip(axes, ("jssp", "fjsp"),
                                             ("Job shop", "Flexible job shop"))):
        sub = runs[runs["type"] == ("JSSP" if key == "jssp" else "FJSP")]
        info = summary[f"cd_{key}"]
        _rank_panel(ax, median_matrix(sub, "rpd_at_300s", PANEL), info,
                    "", order=pooled)
        ax.set_xlabel("mean rank")
        panel_label(ax, "(a)" if k == 0 else "(b)",
                    f"{label} (n = {info['n_instances']})")
    save(fig, FIG / "fig_rank_strata.pdf")


# --------------------------------------------------------------------- 8. anytime
def fig_anytime() -> None:
    """Question: how does quality develop with time. Statistic: median over instances of the
    per-instance median deviation. Encoding: four small multiples on identical scales, family
    hue with member line style, the exact solver and the tabu search repeated in every panel
    as common references. Uncertainty: none plotted. Source: anytime_curves.csv."""
    c = pd.read_csv(DERIVED / "anytime_curves.csv")
    c = c[c["coverage"] >= 0.95]
    groups = [("problem_specific", "Exact and problem-specific"),
              ("general", "Classical general-purpose and random-key"),
              ("adaptive", "Competition-grade adaptive"),
              ("recent", "Recent metaphor-based")]
    fig, axes = plt.subplots(2, 2, figsize=(WIDTH, 3.9), sharex=True, sharey=True)
    fig.subplots_adjust(hspace=0.32, wspace=0.10)
    for k, (ax, (fam, title)) in enumerate(zip(axes.ravel(), groups)):
        for b in BUDGETS:
            ax.axvline(b, color=FAINT, linewidth=0.6, zorder=0)
        members = [m for m in PANEL if FAMILY[m] == fam]
        if fam == "problem_specific":
            members = ["cpsat"] + members
        else:
            if fam == "general":
                members = members + ["brkga"]
            for ref, dy in (("cpsat", 2.4), ("tabu", -4.6)):
                s = c[c.method == ref].sort_values("t")
                ax.plot(s.t, s["median"], color=GREY, linewidth=1.0, zorder=1)
                ax.text(s.t.iloc[0] * 1.3, float(s["median"].iloc[0]) + dy, display(ref),
                        fontsize=6.2, color=MUTED, ha="left")
        for m in members:
            s = c[c.method == m].sort_values("t")
            ax.plot(s.t, s["median"], color=color(m), linewidth=1.3,
                    linestyle=LINESTYLE.get(m, "-"), label=display(m), zorder=3)
        ax.set_xscale("log")
        ax.set_xlim(0.01, 320)
        ax.set_ylim(0, 55)
        ax.legend(ncol=2, fontsize=6.5, loc="upper right")
        panel_label(ax, "(%s)" % "abcd"[k], title)
    exact = c[c.method == "cpsat"].sort_values("t")[["t", "median"]]
    tabu = c[c.method == "tabu"].sort_values("t")[["t", "median"]]
    merged = pd.merge(exact, tabu, on="t", suffixes=("_e", "_t"))
    cross = merged[merged["median_e"] <= merged["median_t"]]
    if len(cross):
        tc, yc = float(cross["t"].iloc[0]), float(cross["median_e"].iloc[0])
        axes[0, 0].annotate("exact overtakes\nthe tabu search", xy=(tc, yc),
                            xytext=(0.02, 26), fontsize=6.4, color=INK,
                            arrowprops=dict(arrowstyle="->", linewidth=0.6, color=MUTED))
    for ax in axes[1]:
        ax.set_xlabel("wall-clock time (s)")
    for ax in axes[:, 0]:
        ax.set_ylabel("median deviation (%)")
    save(fig, FIG / "fig_anytime.pdf")


# --------------------------------------------------------------------- 9. profile
def fig_profile() -> None:
    """Question: how often is a method within a factor of the best. Statistic: Dolan and More
    performance profile on the per-instance median makespan at 300 s. Encoding: step curves,
    highlighted methods in family hue and the rest in light grey, with direct end labels.
    Uncertainty: none. Source: performance_profile.csv."""
    prof = pd.read_csv(DERIVED / "performance_profile.csv", index_col=0)
    order = _order()
    show = order[:4] + [m for m in order if m in DECODED][:2]
    fig, ax = plt.subplots(figsize=(WIDTH * 0.66, 2.7))
    for m in prof.columns:
        if m not in show:
            ax.step(prof.index, prof[m], where="post", color=GREY, linewidth=0.7, zorder=1)
    xmax = 1.30
    for m in show:
        if m not in prof.columns:
            continue
        ax.step(prof.index, prof[m], where="post", color=color(m), linewidth=1.35,
                linestyle=LINESTYLE.get(m, "-"), zorder=3)
        y = float(prof[m][prof.index <= xmax].iloc[-1])
        ax.text(xmax * 1.004, y, display(m), fontsize=6.6, color=color(m), va="center")
    ax.set_xlim(1, xmax)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel(r"performance ratio $\tau$ on the median makespan")
    ax.set_ylabel(r"share of instances within $\tau$")
    ax.text(0.02, 0.97, "grey: the remaining methods", transform=ax.transAxes, fontsize=6.4,
            color=MUTED, va="top")
    save(fig, FIG / "fig_profile.pdf")


# --------------------------------------------------------------------- 10. transfer
def fig_transfer() -> None:
    """Question: does continuous-benchmark standing carry over. Statistic: mean ranks on
    CEC2017 and on scheduling, and median scheduling deviation. Encoding: an equal-scale
    scatter with the identity diagonal, and a lollipop chart with the two reference methods.
    Uncertainty: the Spearman coefficient and its p value are printed. Source:
    transfer_ranks.csv, transfer.json, summary_by_budget.csv."""
    tr = pd.read_csv(DERIVED / "transfer_ranks.csv")
    info = json.loads((DERIVED / "transfer.json").read_text(encoding="utf-8"))
    s = _summary()
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 3.0))
    fig.subplots_adjust(wspace=0.38)

    ax = axes[0]
    lim = [0.5, 12.5]
    ax.plot(lim, lim, color=GREY, linewidth=0.9, linestyle="--", zorder=1)
    for _, r in tr.iterrows():
        ax.scatter(r.cec_mean_rank, r.sched_mean_rank, s=32, color=color(r.method),
                   marker=MARKER.get(r.method, "o"), edgecolor="white", linewidth=0.6,
                   zorder=3)
        up = r.sched_mean_rank >= r.cec_mean_rank
        right = r.cec_mean_rank < 9
        dx = 5 if right else -5
        ax.annotate(display(r.method), (r.cec_mean_rank, r.sched_mean_rank),
                    textcoords="offset points", xytext=(dx, 4) if up else (dx, -9),
                    fontsize=6.2, color=INK, ha="left" if right else "right")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect("equal")
    ax.set_xlabel("mean rank on CEC2017")
    ax.set_ylabel("mean rank on scheduling")
    ax.text(0.03, 0.97, rf"Spearman $\rho={info['rho_wallclock']:.2f}$, "
                        rf"$p={info['p_wallclock']:.3f}$",
            transform=ax.transAxes, fontsize=7, va="top")
    panel_label(ax, "(a)")

    ax = axes[1]
    seq = tr.sort_values("sched_mean_rank", ascending=False)["method"].tolist()
    y = np.arange(len(seq))
    for yi, m in zip(y, seq):
        v = float(s.loc[m, "median"])
        ax.plot([0, v], [yi, yi], color=color(m), linewidth=1.0, alpha=0.5, zorder=2)
        ax.plot(v, yi, marker=MARKER.get(m, "o"), color=color(m), markersize=4.8,
                markeredgecolor="white", markeredgewidth=0.5, zorder=3)
    for ref, style, dy in (("tabu", "-", 0.7), ("cpsat", "--", 2.0)):
        v = float(s.loc[ref, "median"])
        ax.axvline(v, color=INK, linewidth=0.9, linestyle=style, zorder=1)
        ax.text(v + 0.35, len(seq) - dy, display(ref), fontsize=6.8, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels([display(m) for m in seq])
    ax.set_xlabel("median deviation at 300 s (%)")
    ax.set_xlim(0, None)
    ax.set_ylim(-0.8, len(seq) - 0.2)
    ax.grid(axis="y", visible=False)
    panel_label(ax, "(b)")
    save(fig, FIG / "fig_transfer.pdf")


# --------------------------------------------------------------------- 11. equal effort
def fig_equal_effort() -> None:
    """Question: does implementation throughput explain the ranking. Statistic: mean rank at
    a fixed wall-clock budget against mean rank at a fixed evaluation count. Encoding: square
    panel, identity diagonal, equal aspect. Uncertainty: the Spearman coefficient is printed.
    Source: transfer_ranks.csv, stats_summary.json."""
    tr = pd.read_csv(DERIVED / "transfer_ranks.csv")
    rho = json.loads((DERIVED / "stats_summary.json").read_text(
        encoding="utf-8"))["wallclock_vs_eval_rank_rho"]
    fig, ax = plt.subplots(figsize=(WIDTH * 0.54, 2.8))
    lim = [0.5, 12.5]
    ax.plot(lim, lim, color=GREY, linewidth=0.9, linestyle="--", zorder=1)
    for _, r in tr.iterrows():
        ax.scatter(r.sched_mean_rank, r.sched_eval_mean_rank, s=32, color=color(r.method),
                   marker=MARKER.get(r.method, "o"), edgecolor="white", linewidth=0.6,
                   zorder=3)
        up = r.sched_eval_mean_rank >= r.sched_mean_rank
        ax.annotate(display(r.method), (r.sched_mean_rank, r.sched_eval_mean_rank),
                    textcoords="offset points", xytext=(5, 4) if up else (5, -9),
                    fontsize=6.4, color=INK)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect("equal")
    ax.set_xlabel("mean rank at 300 s of wall clock")
    ax.set_ylabel(r"mean rank at $10^5$ evaluations")
    ax.text(0.04, 0.96, rf"Spearman $\rho={rho:.2f}$", transform=ax.transAxes, fontsize=7,
            va="top")
    save(fig, FIG / "fig_equal_effort.pdf")


# --------------------------------------------------------------------- 12. hybrid
def fig_hybrid() -> None:
    """Question: does warm starting the exact solver pay once seeding is charged. Statistic:
    median deviation at four total budgets. Encoding: points at the observed budgets joined
    only to guide the eye, references in charcoal and blue, the oracle variant in grey.
    Uncertainty: none plotted, the tests are in the table. Source: hybrid_vs_pure.csv,
    summary_by_budget.csv."""
    h = pd.read_csv(DERIVED / "hybrid_vs_pure.csv")
    s = pd.read_csv(DERIVED / "summary_by_budget.csv")
    fig, ax = plt.subplots(figsize=(WIDTH * 0.62, 2.8))
    for m, c, mk in (("cpsat", FAMILY_COLOR["exact"], "o"),
                     ("tabu", FAMILY_COLOR["problem_specific"], "s")):
        v = s[s.method == m].sort_values("budget_s")
        ax.plot(v.budget_s, v["median"], color=c, marker=mk, markersize=4.2,
                markeredgecolor="white", markeredgewidth=0.5, linewidth=1.6,
                label=display(m))
    hyb = {"hyb_cheap": (FAMILY_COLOR["general"], "-", "^", display("hyb_cheap")),
           "hyb_tabu": (FAMILY_COLOR["adaptive"], "-", "v", display("hyb_tabu")),
           "hyb_oracle": (MUTED, "--", "D", "CP-SAT+oracle, unattainable")}
    for v, (c, ls, mk, lab) in hyb.items():
        sub = h[(h.variant == v) & (h.reference == "cpsat")].sort_values("budget_s")
        ax.plot(sub.budget_s, sub.median_hybrid, color=c, linestyle=ls, marker=mk,
                markersize=3.8, markeredgecolor="white", markeredgewidth=0.4, linewidth=1.1,
                label=lab)
    one_e = float(s[(s.method == "cpsat") & (s.budget_s == 1)]["median"].iloc[0])
    one_h = float(h[(h.variant == "hyb_tabu") & (h.budget_s == 1)
                    & (h.reference == "cpsat")]["median_hybrid"].iloc[0])
    ax.annotate("", xy=(1, one_h), xytext=(1, one_e),
                arrowprops=dict(arrowstyle="<->", linewidth=0.7, color=MUTED))
    ax.text(1.15, (one_e + one_h) / 2, "warm-start gain,\nshortest budget only",
            fontsize=6.4, color=MUTED, va="center")
    ax.set_xscale("log")
    ax.set_xticks(list(BUDGETS))
    ax.set_xticklabels([str(b) for b in BUDGETS])
    ax.set_xlabel("total budget (s)")
    ax.set_ylabel("median deviation (%)")
    ax.legend(fontsize=6.6)
    save(fig, FIG / "fig_hybrid.pdf")


# --------------------------------------------------------------------- 13. robustness
def fig_robustness() -> None:
    """Question: do two evaluation choices drive the comparison. Statistics: paired deviation
    under two machine mappings, and the change in deviation when the shared population is
    replaced. Encoding: paired scatter against the identity line, and three zero-centred
    difference panels on a common axis. Uncertainty: the reported test statistics are
    annotated. Source: ablation_decoder.csv, ablation_decoder.json,
    sensitivity_by_method.csv."""
    ab = pd.read_csv(DERIVED / "ablation_decoder.csv")
    abj = json.loads((DERIVED / "ablation_decoder.json").read_text(encoding="utf-8"))
    se = pd.read_csv(DERIVED / "sensitivity_by_method.csv").set_index("method")
    fig = plt.figure(figsize=(WIDTH, 3.1))
    gs = fig.add_gridspec(1, 5, width_ratios=[1.25, 0.22, 0.85, 0.85, 0.85], wspace=0.16)

    ax = fig.add_subplot(gs[0, 0])
    lim = [0, max(ab.legacy.max(), ab.eligible.max()) * 1.06]
    ax.plot(lim, lim, color=GREY, linewidth=0.9, linestyle="--", zorder=1)
    for m, g in ab.groupby("method"):
        ax.scatter(g.eligible, g.legacy, s=18, color=color(m), marker=MARKER.get(m, "o"),
                   edgecolor="white", linewidth=0.4, alpha=0.85, zorder=3)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect("equal")
    ax.set_xlabel("eligible-set decoding (%)")
    ax.set_ylabel("index decoding (%)")
    ax.text(0.05, 0.95, f"{abj['n_legacy_worse']} of {abj['n_cells']} cells above"
            " the line", transform=ax.transAxes, fontsize=6.3, va="top", color=INK)
    panel_label(ax, "(a)", "Decoder mapping")

    order = [m for m in _order() if m in se.index][::-1]
    axes = []
    for k, (setting, label) in enumerate((("20", "population 20"), ("100", "population 100"),
                                          ("recommended", "author rule"))):
        ax = fig.add_subplot(gs[0, k + 2])
        axes.append(ax)
        d = (se[setting] - se["50"]).reindex(order)
        ax.axvline(0, color=MUTED, linewidth=0.8, zorder=1)
        ax.axvline(float(d.median()), color=INK, linewidth=0.9, linestyle=":", zorder=2)
        for yi, m in enumerate(order):
            ax.plot(d[m], yi, marker=MARKER.get(m, "o"), color=color(m), markersize=4,
                    markeredgecolor="white", markeredgewidth=0.4, zorder=3)
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels([display(m) for m in order] if k == 0 else [], fontsize=6.4)
        ax.grid(axis="y", visible=False)
        ax.set_xlim(-7, 7)
        ax.set_xticks([-5, 0, 5])
        ax.set_ylim(-0.8, len(order) - 0.2)
        panel_label(ax, "(%s)" % "abcd"[k + 1], label)
    fig.text(0.70, -0.02, "change in median deviation against a population of 50 (points)",
             ha="center", fontsize=8, color=INK)
    save(fig, FIG / "fig_robustness.pdf")


def main() -> None:
    set_style()
    FIG.mkdir(parents=True, exist_ok=True)
    for fn in (fig_overview, fig_proof, fig_flexibility, fig_necessity, fig_necessity_map,
               fig_deviation, fig_rank, fig_rank_strata, fig_anytime, fig_profile,
               fig_transfer, fig_equal_effort, fig_hybrid, fig_robustness):
        fn()
        print("wrote", fn.__name__)


if __name__ == "__main__":
    main()
