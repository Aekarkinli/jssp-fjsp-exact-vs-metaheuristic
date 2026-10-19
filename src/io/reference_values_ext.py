"""Reference values for the revision-round instance sets (large-scale and Hurink study).

Written to ``data/reference_values_rev1.csv``, apart from the locked main table. The sources
and the Hurink variant mapping are those of ``src.io.reference_values``. The Hurink
collection lists the four variants of a base problem without labels, and the main table maps
them by the anchored descending order of their bounds. For a base whose order is ambiguous
(an upper-bound tie with differing lower bounds) the four rows are written with
``mapping_ok`` false and empty values instead of failing, because the Hurink study rests on
the solver's own proof and bound, and the best-known value only enters a secondary measure.

    uv run python -m src.io.reference_values_ext
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

from src.io.instance_sources import HURINK_STUDY, LARGE_FJSP, LARGE_JSSP
from src.io.reference_values import (
    FJSP_SOURCE,
    JSSP_SOURCE,
    WEISE_SOURCE,
    _HURINK_VARIANT_ORDER,
    _weise_table,
)

INSTANCES_DIR = Path("data/instances")
OUT = Path("data/reference_values_rev1.csv")
FIELDS = ["id", "set", "type", "BKS", "BKS_source", "LB", "LB_source", "proven_optimal",
          "mapping_ok"]


def _jssp_rows() -> list[dict]:
    meta = {e["name"]: e for e in json.loads(
        (INSTANCES_DIR / "jsplib_instances.json").read_text(encoding="utf-8"))}
    weise = _weise_table()
    rows = []
    for spec in LARGE_JSSP:
        e = meta.get(spec.id, {})
        w = weise.get(spec.id)
        if e.get("optimum") is not None:
            bks = lb = e["optimum"]
            src = JSSP_SOURCE
        elif e.get("bounds"):
            bks, lb = e["bounds"]["upper"], e["bounds"]["lower"]
            src = JSSP_SOURCE
            if w is not None and (w["bks"] < bks or w["lb"] > lb):
                bks, lb = min(bks, w["bks"]), max(lb, w["lb"])
                src = WEISE_SOURCE
        else:
            bks, lb, src = w["bks"], w["lb"], WEISE_SOURCE
        rows.append({"id": spec.id, "set": "large", "type": "JSSP", "BKS": bks,
                     "BKS_source": src, "LB": lb, "LB_source": src,
                     "proven_optimal": bks == lb, "mapping_ok": True})
    return rows


def _fjsp_rows() -> list[dict]:
    entries = json.loads((INSTANCES_DIR / "fjsp_bks.json").read_text(encoding="utf-8"))
    by_key = {}
    hurink: dict[str, list[dict]] = defaultdict(list)
    for e in entries:
        fam = e.get("family", "").lower()
        if "hurink" in fam:
            hurink[e["instance"]].append(e)
        elif "dauz" in fam:
            by_key["dp" + e["instance"]] = e
        else:
            by_key[e["instance"]] = e

    rows = []
    for spec in LARGE_FJSP:
        e = by_key[spec.id]
        lb, ub = e["lower_bound"], e["upper_bound"]
        rows.append({"id": spec.id, "set": "large", "type": "FJSP", "BKS": ub,
                     "BKS_source": FJSP_SOURCE, "LB": lb, "LB_source": FJSP_SOURCE,
                     "proven_optimal": lb == ub, "mapping_ok": True})

    mapped: dict[str, dict[str, dict] | None] = {}
    for base, ents in hurink.items():
        ordered = sorted(ents, key=lambda e: (e["upper_bound"], e["lower_bound"]),
                         reverse=True)
        ok = len(ents) == 4 and not any(
            a["upper_bound"] == b["upper_bound"] and a["lower_bound"] != b["lower_bound"]
            for a, b in zip(ordered, ordered[1:]))
        mapped[base] = dict(zip(_HURINK_VARIANT_ORDER, ordered)) if ok else None

    for spec in HURINK_STUDY:
        _, variant, base = spec.id.split("_", 2)
        table = mapped.get(base)
        if table is None:
            rows.append({"id": spec.id, "set": "hurink", "type": "FJSP", "BKS": "",
                         "BKS_source": "", "LB": "", "LB_source": "",
                         "proven_optimal": "", "mapping_ok": False})
            continue
        e = table[variant]
        lb, ub = e["lower_bound"], e["upper_bound"]
        rows.append({"id": spec.id, "set": "hurink", "type": "FJSP", "BKS": ub,
                     "BKS_source": FJSP_SOURCE, "LB": lb, "LB_source": FJSP_SOURCE,
                     "proven_optimal": lb == ub, "mapping_ok": True})
    return rows


REVISION_SOURCE = "ScheduleOpt benchmarks (scheduleopt.github.io/benchmarks), checked 2026-10-13"
REVISIONS = {
    # id: (best known, lower bound); values that moved after the pinned snapshot
    "ta42": (1937, 1900), "ta46": (2002, 1976),
    "behnke11": (228, 223), "behnke16": (412, 391), "behnke31": (226, 226),
    "behnke36": (388, 381), "behnke51": (218, 215), "behnke56": (390, 384),
}


def _apply_revisions(rows: list[dict]) -> list[dict]:
    for r in rows:
        rev = REVISIONS.get(r["id"])
        if rev is None:
            continue
        r["BKS"], r["LB"] = rev
        r["proven_optimal"] = rev[0] == rev[1]
        r["BKS_source"] = r["LB_source"] = REVISION_SOURCE
    return rows


def build() -> list[dict]:
    rows = _apply_revisions(_jssp_rows() + _fjsp_rows())
    for r in rows:
        if r["mapping_ok"] and r["LB"] > r["BKS"]:
            raise ValueError(f"{r['id']}: LB {r['LB']} exceeds BKS {r['BKS']}")
    with OUT.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    return rows


def main() -> None:
    rows = build()
    bad = [r["id"] for r in rows if not r["mapping_ok"]]
    print(f"{len(rows)} rows -> {OUT}; ambiguous Hurink mappings: {len(bad)}")
    if bad:
        print("  ", ", ".join(bad))


if __name__ == "__main__":
    main()
