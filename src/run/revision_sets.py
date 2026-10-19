"""Instance subsets and run definitions of the targeted experiments.

The revision subset is a deterministic function of the locked main panel, fixed before any
revision run: within each problem class the instances are ordered by operation count (ties
by identifier) and ten are taken at evenly spaced ranks, from the smallest to the largest.
The subset therefore spans the whole size range of both classes without any reference to
results.

    uv run python -m src.run.revision_sets     # print the subsets
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from src.io.instance_sources import HURINK_STUDY, LARGE_FJSP_IDS, LARGE_JSSP_IDS

MANIFEST = Path("data/instance_manifest.csv")
PER_CLASS = 10

MEMETIC_METHODS = ["ga", "brkga", "lshade", "cmaes"]
DECODED_METHODS = ["ga", "brkga", "de", "pso", "abc", "gwo", "lshade", "imode", "cmaes",
                   "rime", "mde", "csa"]
VARIANT_METHODS = ["jade", "shade", "clpso", "igwo"]
FULL_PANEL = ["cpsat", "dispatching", "greedy", "tabu", "sa"] + DECODED_METHODS
HURINK_METHODS = ["cpsat", "tabu"]
REVISION_SEEDS = 10
HURINK_SEEDS = 3
# Base problems that cannot enter the matched study. orb7 contains an operation with a
# processing time of zero, which the instance model rejects (every processing time must be
# positive), so all four of its variants fail to load and the base is left out.
HURINK_EXCLUDED = {"orb7": "an operation with zero processing time"}


def _manifest() -> list[dict]:
    with MANIFEST.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def revision_subset() -> list[str]:
    rows = _manifest()
    out: list[str] = []
    for kind in ("JSSP", "FJSP"):
        cls = sorted((r for r in rows if r["type"].upper() == kind),
                     key=lambda r: (int(r["n_op"]), r["id"]))
        idx = np.unique(np.round(np.linspace(0, len(cls) - 1, PER_CLASS)).astype(int))
        out += [cls[i]["id"] for i in idx]
    return out


def large_set() -> list[str]:
    return list(LARGE_JSSP_IDS) + list(LARGE_FJSP_IDS)


def hurink_set() -> list[str]:
    return [s.id for s in HURINK_STUDY]


if __name__ == "__main__":
    print("revision subset:", revision_subset())
    print("large set:", large_set())
    print("hurink study:", len(hurink_set()), "instances")
