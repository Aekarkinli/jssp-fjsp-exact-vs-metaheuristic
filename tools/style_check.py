"""Academic-style linter for the manuscript.

Strips LaTeX commands and math, then scans the body text and reports every violation with
its location and surrounding context. Rules:

1. No semicolons joining clauses, and no colons appending an explanation in running text
   (a colon is allowed only when it introduces a displayed list). Narrow exceptions: numeric
   ranges and clock times.
2. No contrastive "not X but/rather Y" frame.
3. No parenthesised in-paragraph enumeration, e.g. (1) ... (2) ... or (a) ... (b).
4. No paragraph opening with a summarising connective (finally, overall, in conclusion,
   in summary, to summarise, ...).
5. Stock machine-generated phrases (delve, leverage, underscore, pivotal, crucial role,
   rich tapestry, realm, ... ).
6. Overuse of transitional adverbs (moreover, furthermore, additionally).

Exit code is non-zero if any violation is found, unless a line carries a justification
comment "% style-ok: <reason>".

    uv run python tools/style_check.py paper/sections/*.tex
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

STOCK_PHRASES = [
    r"\bdelve\b", r"\bleverag(e|ing|es|ed)\b", r"\bunderscor(e|es|ing|ed)\b", r"\bpivotal\b",
    r"\bcrucial role\b", r"\bplays? a key role\b", r"\brich tapestry\b", r"\brealm\b",
    r"\blandscape\b", r"\brobust\b", r"\bcomprehensive\b", r"\bnotably\b", r"\bimportantly\b",
    r"\bit is worth noting\b", r"\bit should be noted\b", r"\ba testament to\b",
    r"\bnavigat(e|ing) the complexit(y|ies)\b", r"\bshed(s|ding)? light\b",
    r"\bparadigm shift\b", r"\bgame changer\b", r"\bcutting edge\b", r"\bseamless(ly)?\b",
    r"\bholistic\b", r"\bin today's world\b", r"\bever-(evolving|changing)\b",
]
SUMMARY_OPENERS = [r"finally", r"overall", r"in conclusion", r"in summary",
                   r"to summarise", r"to summarize", r"to sum up", r"in short"]
TRANSITIONS = [r"\bmoreover\b", r"\bfurthermore\b", r"\badditionally\b"]
CONTRASTIVE = re.compile(r"\bnot\b[^.;:]{1,60}\b(but|rather)\b", re.IGNORECASE)
EMDASH = re.compile(r"---|—|–")
PAREN_ENUM = re.compile(r"\((?:[1-9]|[ivx]+|[a-d])\)")
RANGE_OR_TIME = re.compile(r"\d\s*[:;]\s*\d")  # 1:10, time/range; exempt from clause rules


_MATH_ENVS = r"equation|align|gather|multline|displaymath|eqnarray"


def _blank(match: re.Match) -> str:
    """Replace a span with spaces but keep its newlines, so line numbers are preserved."""
    return re.sub(r"[^\n]", " ", match.group(0))


def mask_math(text: str) -> str:
    """Blank out display- and inline-math regions (line structure preserved)."""
    text = re.sub(rf"\\begin\{{({_MATH_ENVS})\*?\}}.*?\\end\{{\1\*?\}}", _blank, text, flags=re.DOTALL)
    text = re.sub(r"\\\[.*?\\\]", _blank, text, flags=re.DOTALL)
    text = re.sub(r"\$\$.*?\$\$", _blank, text, flags=re.DOTALL)
    text = re.sub(r"\$[^$\n]*\$", _blank, text)
    return text


def strip_latex(text: str) -> str:
    text = re.sub(r"(?<!\\)%.*", "", text)            # comments
    text = re.sub(r"\\begin\{.*?\}.*?\\end\{.*?\}", " ", text, flags=re.DOTALL)  # environments
    text = re.sub(r"\\[a-zA-Z@]+\*?(\[[^\]]*\])?(\{[^}]*\})?", " ", text)  # commands + args
    text = re.sub(r"\\[^a-zA-Z\s]", " ", text)         # non-letter commands (\; \, \! \[ \] ..)
    text = re.sub(r"[{}]", " ", text)
    return text


def _paragraphs(text: str):
    para, start_line = [], 1
    line_no = 1
    buf, buf_start = [], 1
    for raw in text.splitlines():
        if raw.strip() == "":
            if buf:
                para.append((buf_start, "\n".join(buf)))
                buf = []
        else:
            if not buf:
                buf_start = line_no
            buf.append(raw)
        line_no += 1
    if buf:
        para.append((buf_start, "\n".join(buf)))
    return para


def lint_text(raw: str) -> list[tuple[int, str, str]]:
    """Return (line, rule, context) violations. Line numbers refer to the raw file."""
    violations = []
    masked = mask_math(raw)
    raw_lines = raw.splitlines()
    masked_lines = masked.splitlines()
    in_caption = 0
    for i, (rline, mline) in enumerate(zip(raw_lines, masked_lines), start=1):
        if "% style-ok" in rline:
            continue
        # Panel references such as (a) and (b) are the standard way to point at a subfigure,
        # so the in-paragraph enumeration rule does not apply inside a caption.
        was_caption = in_caption > 0 or "\caption{" in rline
        if "\caption{" in rline:
            in_caption = rline.count("{") - rline.count("}")
        elif in_caption > 0:
            in_caption += rline.count("{") - rline.count("}")
            in_caption = max(in_caption, 0)
        line = rline
        body = strip_latex(mline)
        low = body.lower()
        # clause-joining punctuation (exempt numeric ranges / times)
        masked = RANGE_OR_TIME.sub("  ", body)
        if ";" in masked:
            violations.append((i, "semicolon", line.strip()[:90]))
        # a colon mid-sentence that is not introducing a displayed list
        if ":" in masked and not re.search(r":\s*\\?$", body.strip()):
            if re.search(r"[a-z]:\s+[a-z]", masked):
                violations.append((i, "colon-explanation", line.strip()[:90]))
        if CONTRASTIVE.search(body):
            violations.append((i, "contrastive-not-but", line.strip()[:90]))
        if PAREN_ENUM.search(body) and not was_caption:
            violations.append((i, "paren-enumeration", line.strip()[:90]))
        if EMDASH.search(rline):
            violations.append((i, "em-dash", line.strip()[:90]))
        for pat in STOCK_PHRASES:
            m = re.search(pat, low)
            if m:
                violations.append((i, f"stock-phrase:{m.group(0)}", line.strip()[:90]))
    # paragraph-opening summarising connective
    for start, para in _paragraphs(strip_latex(masked)):
        first = para.lstrip().lower()
        for opener in SUMMARY_OPENERS:
            if re.match(rf"{opener}\b", first):
                violations.append((start, f"summary-opener:{opener}", para.strip()[:90]))
    # transition overuse (whole document)
    body_all = strip_latex(masked).lower()
    for pat in TRANSITIONS:
        n = len(re.findall(pat, body_all))
        if n > 2:
            word = pat.strip("\\b")
            violations.append((0, f"transition-overuse:{word}({n})", ""))
    return violations


def lint_file(path: Path) -> list[tuple[int, str, str]]:
    return lint_text(path.read_text(encoding="utf-8"))


def main() -> None:
    args = sys.argv[1:]
    if not args:
        args = [str(p) for p in Path("paper/sections").glob("*.tex")]
    total = 0
    for arg in args:
        for path in sorted(Path().glob(arg)) if any(c in arg for c in "*?[") else [Path(arg)]:
            if not path.exists():
                continue
            vs = lint_file(path)
            for line, rule, ctx in vs:
                print(f"{path}:{line}: {rule}  | {ctx}")
            total += len(vs)
    print(f"\nstyle check: {total} violation(s)")
    if total:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
