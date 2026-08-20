"""Panel metadata: display names, families and ordering used by every analysis.

Implementation identifiers live here and in the code only. Figures, tables and the
manuscript use the display names below.
"""
from __future__ import annotations

EXACT = "cpsat"

# ordered panel of the fair single-thread comparison
PANEL = ["cpsat", "dispatching", "greedy", "tabu", "sa", "brkga", "ga", "de", "pso", "abc",
         "lshade", "imode", "cmaes", "gwo", "rime", "mde", "csa"]

HEURISTICS = [m for m in PANEL if m != EXACT]

# optimisers that search through the shared decoder (the only ones with an evaluation index)
DECODED = ["ga", "brkga", "de", "pso", "abc", "lshade", "imode", "cmaes",
           "gwo", "rime", "mde", "csa"]

DETERMINISTIC = ["dispatching", "greedy"]

HYBRIDS = ["hyb_cheap", "hyb_tabu", "hyb_oracle"]

FAMILY = {
    "cpsat": "exact",
    "dispatching": "problem_specific", "greedy": "problem_specific",
    "tabu": "problem_specific", "sa": "problem_specific",
    "brkga": "random_key",
    "ga": "general", "de": "general", "pso": "general", "abc": "general",
    "lshade": "adaptive", "imode": "adaptive", "cmaes": "adaptive",
    "gwo": "recent", "rime": "recent", "mde": "recent", "csa": "recent",
}

FAMILY_LABEL = {
    "exact": "Exact",
    "problem_specific": "Problem-specific",
    "random_key": "Random-key",
    "general": "Classical general-purpose",
    "adaptive": "Competition-grade adaptive",
    "recent": "Recent metaphor-based",
}

DISPLAY = {
    "cpsat": "CP-SAT", "cpsat_mt": "CP-SAT (8 threads)",
    "dispatching": "PDR", "greedy": "GT", "tabu": "TS", "sa": "SA",
    "brkga": "BRKGA", "ga": "GA", "de": "DE", "pso": "PSO", "abc": "ABC",
    "lshade": "L-SHADE", "imode": "IMODE", "cmaes": "CMA-ES",
    "gwo": "GWO", "rime": "RIME", "mde": "MDE", "csa": "CSA",
    "hyb_cheap": "CP-SAT+PDR", "hyb_tabu": "CP-SAT+TS", "hyb_oracle": "CP-SAT+oracle",
}

# One hue per family, taken in the order that clears the colour-vision separation checks.
# The exact solver is drawn in reference ink rather than a categorical hue, because it is
# the baseline every other series is read against.
FAMILY_COLOR = {
    "exact": "#0B0B0B",
    "problem_specific": "#2A78D6",
    "random_key": "#EB6834",
    "general": "#1BAF7A",
    "adaptive": "#EDA100",
    "recent": "#E87BA4",
}

# Secondary encoding inside a family, so identity never rests on hue alone.
LINESTYLE = {
    "cpsat": "-", "tabu": "-", "sa": "--", "dispatching": ":", "greedy": "-.",
    "brkga": "-",
    "ga": "-", "de": "--", "pso": ":", "abc": "-.",
    "lshade": "-", "imode": "--", "cmaes": ":",
    "gwo": "-", "rime": "--", "mde": ":", "csa": "-.",
}
MARKER = {
    "cpsat": "o", "tabu": "s", "sa": "^", "dispatching": "v", "greedy": "D",
    "brkga": "P",
    "ga": "o", "de": "s", "pso": "^", "abc": "v",
    "lshade": "o", "imode": "s", "cmaes": "^",
    "gwo": "o", "rime": "s", "mde": "^", "csa": "v",
}

CLASS_ORDER = ["EXACT_SUFFICIENT", "HEURISTIC_USEFUL", "HEURISTIC_NECESSARY",
               "HYBRID_RECOMMENDED", "INCONCLUSIVE"]

CLASS_LABEL = {
    "EXACT_SUFFICIENT": "Exact-sufficient",
    "HEURISTIC_USEFUL": "Heuristic-useful",
    "HEURISTIC_NECESSARY": "Heuristic-necessary",
    "HYBRID_RECOMMENDED": "Hybrid-recommended",
    "INCONCLUSIVE": "Inconclusive",
}

CLASS_COLOR = {
    "EXACT_SUFFICIENT": "#2A78D6", "HEURISTIC_USEFUL": "#1BAF7A",
    "HEURISTIC_NECESSARY": "#EB6834", "HYBRID_RECOMMENDED": "#E87BA4",
    "INCONCLUSIVE": "#C4C3BE",
}

THRESHOLDS = {
    "delta": 0.01, "delta_strict": 0.02, "tau": 10.0, "tau_low": 5.0, "tau_high": 20.0,
    "alpha": 0.05, "rbc_floor": 0.30, "epsilon": 0.02,
}

BUDGETS = (1, 10, 60, 300)
EVAL_BUDGETS = (1_000, 10_000, 100_000)


def display(m: str) -> str:
    return DISPLAY.get(m, m)


def color(m: str) -> str:
    return FAMILY_COLOR.get(FAMILY.get(m, "general"), "#0072B2")
