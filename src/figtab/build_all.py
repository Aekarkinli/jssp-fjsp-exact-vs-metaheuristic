"""Rebuild every derived table, figure, LaTeX fragment and macro from the raw results.

    uv run python -m src.figtab.build_all            # analysis + figures + tables + macros
    uv run python -m src.figtab.build_all --figures  # skip the analysis stage

The manuscript is then built with ``latexmk -pdf main.tex`` inside ``paper/``.
"""
from __future__ import annotations

import argparse
import subprocess
import sys

STAGES = [
    ("collect raw results", ["-m", "src.analysis.collect"]),
    ("comparative statistics", ["-m", "src.analysis.stats"]),
    ("necessity classification", ["-m", "src.analysis.necessity"]),
    ("supporting analyses", ["-m", "src.analysis.extras"]),
]
OUTPUTS = [
    ("figures", ["-m", "src.figtab.figures"]),
    ("tables", ["-m", "src.figtab.tables"]),
    ("macros", ["-m", "src.figtab.numbers"]),
]


def run(label: str, args: list[str]) -> None:
    print(f"==> {label}")
    subprocess.run([sys.executable, *args], check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--figures", action="store_true",
                    help="rebuild only the figures, tables and macros")
    args = ap.parse_args()
    if not args.figures:
        for label, cmd in STAGES:
            run(label, cmd)
    for label, cmd in OUTPUTS:
        run(label, cmd)
    print("\nall artifacts rebuilt: results/derived, paper/figures, paper/generated")


if __name__ == "__main__":
    main()
