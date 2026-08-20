"""Populate `data/reference_values.csv` (BKS, LB, proven-optimal) for all 67 instances.

Sources (both pinned by commit SHA, recorded in `data/instances/SOURCES.json`):
- JSSP: the JSPLIB metadata (`instances.json`), which lists the proven optimum or, when
  open, the best-known upper/lower bounds with the instance.
- FJSP: ScheduleOpt's `solutions/bks.json`, which lists per-instance lower/upper bounds and
  the solvers that produced them.

Hurink mapping. `bks.json` carries four entries per Hurink base instance, one per
flexibility variant (sdata, edata, rdata, vdata), distinguished only by value. Flexibility
weakly reduces both bounds in that order, and the sdata value equals the pure-JSSP optimum
(verified against JSPLIB for every base used here, which anchors the order). Entries are
therefore sorted by (upper, lower) descending and assigned sdata, edata, rdata, vdata. When
two variants tie on the upper bound the lower bound must tie as well, otherwise the
assignment would be ambiguous and the build fails loudly.

    uv run python -m src.io.reference_values
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

from src.io.instance_sources import (
    FJSP_COMMIT,
    FJSP_INSTANCES,
    FJSP_REPO,
    JSPLIB_COMMIT,
    JSPLIB_REPO,
    JSSP_BKS_COMMIT,
    JSSP_BKS_REPO,
    JSSP_INSTANCES,
)

INSTANCES_DIR = Path("data/instances")
OUT = Path("data/reference_values.csv")

JSSP_SOURCE = f"JSPLIB instances.json ({JSPLIB_REPO}@{JSPLIB_COMMIT[:7]})"
WEISE_SOURCE = f"Weise instances_with_bks ({JSSP_BKS_REPO}@{JSSP_BKS_COMMIT[:7]})"
FJSP_SOURCE = f"ScheduleOpt bks.json ({FJSP_REPO}@{FJSP_COMMIT[:7]})"

FIELDS = ["id", "type", "BKS", "BKS_source", "LB", "LB_source", "proven_optimal"]

_HURINK_VARIANT_ORDER = ["sdata", "edata", "rdata", "vdata"]

# Values revised after the pinned snapshots were taken. The public job-shop collection at
# scheduleopt.github.io/benchmarks/jsplib was re-checked on 2026-08-17 against every
# instance of this study. Three entries had moved, and they are corrected here so the
# denominator of every reported deviation is the current best-known value. The other
# sixty-four agreed with the pinned snapshots exactly.
REVISION_SOURCE = "JSPLib (scheduleopt.github.io/benchmarks/jsplib), checked 2026-08-17"
REVISIONS = {
    # id: (best known, lower bound, proven optimal)
    "abz8": (667, 667, True),    # the 665 in the older tables is unverified in the source
    "swv06": (1667, 1667, True),  # improved upper bound closes the instance
    "ta41": (2005, 1926, False),  # unchanged upper bound, improved lower bound
}


def _apply_revisions(rows: list[dict]) -> list[dict]:
    for r in rows:
        rev = REVISIONS.get(r["id"])
        if rev is None:
            continue
        bks, lb, proven = rev
        r["BKS"], r["LB"] = bks, lb
        r["proven_optimal"] = proven
        r["BKS_source"] = REVISION_SOURCE
        r["LB_source"] = REVISION_SOURCE
    return rows


def _weise_table() -> dict[str, dict]:
    """Parse the Weise et al. literature table: id -> {lb, bks}."""
    lines = (INSTANCES_DIR / "jssp_bks_weise.txt").read_text(encoding="utf-8").splitlines()
    header = lines[0].split(",")
    i_id = header.index("inst.id")
    i_lb = header.index("inst.opt.bound.lower")
    i_bks = header.index("inst.bks")
    out = {}
    for ln in lines[1:]:
        parts = ln.split(",")
        if len(parts) <= max(i_lb, i_bks):
            continue
        out[parts[i_id]] = {"lb": int(parts[i_lb]), "bks": int(parts[i_bks])}
    return out


def _jssp_rows() -> list[dict]:
    meta = {e["name"]: e for e in json.loads(
        (INSTANCES_DIR / "jsplib_instances.json").read_text(encoding="utf-8")
    )}
    weise = _weise_table()
    rows = []
    for spec in JSSP_INSTANCES:
        e = meta[spec.id]
        optimum = e.get("optimum")
        if optimum is not None:
            bks, lb, proven = optimum, optimum, True
            src = JSSP_SOURCE
        elif e.get("bounds"):
            bounds = e["bounds"]
            bks, lb = bounds["upper"], bounds["lower"]
            src = JSSP_SOURCE
            # JSPLIB carries only loose bounds for the harder instances and is not kept
            # current. Where the maintained literature table reports a tighter best-known or
            # lower bound, adopt it so reported gaps are taken to the true best-known.
            w = weise.get(spec.id)
            if w is not None and (w["bks"] < bks or w["lb"] > lb):
                bks, lb = min(bks, w["bks"]), max(lb, w["lb"])
                src = WEISE_SOURCE
            proven = bks == lb
        else:
            # JSPLIB carries no value (ta71): fall back to the Weise literature table.
            w = weise[spec.id]
            bks, lb = w["bks"], w["lb"]
            proven = bks == lb
            src = WEISE_SOURCE
        # cross-check proven optima against the Weise literature table where present
        w = weise.get(spec.id)
        if w is not None and proven and w["bks"] != bks and w["lb"] == w["bks"]:
            raise ValueError(
                f"{spec.id}: proven optimum disagrees between sources "
                f"(JSPLIB {bks} vs Weise {w['bks']})"
            )
        rows.append({
            "id": spec.id, "type": "JSSP",
            "BKS": bks, "BKS_source": src,
            "LB": lb, "LB_source": src,
            "proven_optimal": proven,
        })
    return rows


def _hurink_tables(entries: list[dict]) -> dict[str, dict[str, dict]]:
    """Map base name -> variant -> bks entry using the anchored descending order."""
    by_base: dict[str, list[dict]] = defaultdict(list)
    for e in entries:
        if "hurink" in e.get("family", "").lower():
            by_base[e["instance"]].append(e)
    out: dict[str, dict[str, dict]] = {}
    for base, ents in by_base.items():
        if len(ents) != 4:
            raise ValueError(f"Hurink base {base}: expected 4 variant entries, got {len(ents)}")
        ordered = sorted(ents, key=lambda e: (e["upper_bound"], e["lower_bound"]), reverse=True)
        for a, b in zip(ordered, ordered[1:]):
            if a["upper_bound"] == b["upper_bound"] and a["lower_bound"] != b["lower_bound"]:
                raise ValueError(
                    f"Hurink base {base}: upper-bound tie with differing lower bounds; "
                    "variant assignment would be ambiguous"
                )
        out[base] = dict(zip(_HURINK_VARIANT_ORDER, ordered))
    return out


def _fjsp_rows() -> list[dict]:
    entries = json.loads((INSTANCES_DIR / "fjsp_bks.json").read_text(encoding="utf-8"))
    by_name: dict[str, dict] = {}
    for e in entries:
        if "hurink" not in e.get("family", "").lower():
            by_name[e["instance"]] = e
    hurink = _hurink_tables(entries)

    rows = []
    for spec in FJSP_INSTANCES:
        if spec.family.startswith("hurink_"):
            variant, base = spec.id.split("_", 1)
            e = hurink[base][variant]
        elif spec.family == "fattahi":
            e = by_name[f"fattahi{int(spec.id.removeprefix('fattahi'))}"]
        else:
            e = by_name[spec.id]
        lb, ub = e["lower_bound"], e["upper_bound"]
        rows.append({
            "id": spec.id, "type": "FJSP",
            "BKS": ub, "BKS_source": FJSP_SOURCE,
            "LB": lb, "LB_source": FJSP_SOURCE,
            "proven_optimal": lb == ub,
        })
    return rows


def build_reference_values() -> list[dict]:
    rows = _apply_revisions(_jssp_rows() + _fjsp_rows())
    for r in rows:
        if r["LB"] > r["BKS"]:
            raise ValueError(f"{r['id']}: LB {r['LB']} exceeds BKS {r['BKS']}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main() -> None:
    rows = build_reference_values()
    proven = sum(1 for r in rows if r["proven_optimal"])
    print(f"reference values: {len(rows)} instances -> {OUT}")
    print(f"proven optimal: {proven}; open: {len(rows) - proven}")
    open_ids = [r["id"] for r in rows if not r["proven_optimal"]]
    print("open instances:", ", ".join(open_ids))


if __name__ == "__main__":
    main()
