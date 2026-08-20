"""Check that every macro used in the manuscript is defined in the generated macro file,
and report defined macros that no section uses.

    uv run python tools/check_macros.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

GEN = Path("paper/generated/numbers.tex")
SECTIONS = sorted(Path("paper/sections").glob("*.tex")) + [Path("paper/main.tex")]

LATEX_BUILTIN = set("""
section subsection subsubsection label ref cite citep citet input begin end textbf textit
emph caption includegraphics centering small footnotesize table tabular toprule midrule
bottomrule addlinespace multicolumn cmidrule figure item itemize enumerate documentclass
usepackage newcommand renewcommand bibliographystyle bibliography appendix frac sum max min
mathrm mathcal text times leq geq neq approx alpha beta gamma delta epsilon varepsilon tau
rho sigma mu chi lambda pi theta omega Delta Omega Sigma Pi hline url href textsc
title author ead cortext affiliation abstract keyword sep frontmatter journal FloatBarrier
linewidth textwidth quad qquad noindent par left right big Big log exp ln argmin argmax
in cdot dots ldots cdots infty forall exists subseteq cup cap setminus prime hat bar tilde
displaystyle nonumber notag equation align gather split cases matrix pmatrix bmatrix
Gamma Lambda Theta Phi Psi Upsilon Xi Rightarrow Leftarrow vee wedge mathcal mathbf mathit resizebox linewidth
paragraph appendix cite eqref footnotesize scriptsize normalsize
""".split())


def main() -> int:
    defined = set(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}", GEN.read_text(encoding="utf-8")))
    used: dict[str, list[str]] = {}
    for path in SECTIONS:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        text = re.sub(r"(?<!\\)%.*", "", text)
        for m in re.findall(r"\\([A-Za-z]+)", text):
            used.setdefault(m, []).append(path.name)
    missing = {m: sorted(set(v)) for m, v in used.items()
               if m not in defined and m not in LATEX_BUILTIN and m[0].isupper()}
    unused = sorted(defined - set(used))
    for m, files in sorted(missing.items()):
        print(f"UNDEFINED \\{m}  used in {', '.join(files)}")
    print(f"\n{len(defined)} macros defined, {len(missing)} undefined in use, "
          f"{len(unused)} defined but unused")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
