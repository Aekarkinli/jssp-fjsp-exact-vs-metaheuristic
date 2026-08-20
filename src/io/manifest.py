"""Build `data/instance_manifest.csv` and validate every parsed instance.

Validation compares the parsed (jobs, machines, operations) against an independent
reference wherever one exists:
- JSSP: the JSPLIB `instances.json` metadata (jobs, machines); operations = jobs x machines.
- FJSP Brandimarte: the canonical dimensions from Brandimarte (1993).
- FJSP Hurink: the dimensions of the underlying JSSP base instance (same job/machine/
  operation structure, with added machine flexibility), taken from JSPLIB metadata.
- FJSP Fattahi: the file-defined structure with internal-consistency checks (the loader
  already validates machine ranges and positive durations); no separate literature table.

    uv run python -m src.io.manifest
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

from src.core.instance import Instance
from src.io.instance_sources import FJSP_INSTANCES, JSSP_INSTANCES
from src.io.loaders import load_fjsp_file, load_jssp_file

INSTANCES_DIR = Path("data/instances")
MANIFEST = Path("data/instance_manifest.csv")

# Canonical Brandimarte (1993) dimensions: (jobs, machines, operations).
BRANDIMARTE_DIMS = {
    "mk01": (10, 6, 55), "mk02": (10, 6, 58), "mk03": (15, 8, 150), "mk04": (15, 8, 90),
    "mk05": (15, 4, 106), "mk06": (10, 15, 150), "mk07": (20, 5, 100), "mk08": (20, 10, 225),
    "mk09": (20, 10, 240), "mk10": (20, 15, 240),
}

# Fattahi, Saidi-Mehrabad & Jolai (2007) selected instances: (jobs, machines, operations).
# fattahi1..10 are the small SFJS set, fattahi11..20 the medium MFJS set.
FATTAHI_DIMS = {
    "fattahi01": (2, 2, 4), "fattahi05": (3, 2, 6), "fattahi10": (4, 5, 12),
    "fattahi15": (7, 7, 21), "fattahi20": (12, 8, 48),
}

FIELDS = [
    "id", "family", "type", "jobs", "machines", "n_op", "flexible",
    "exp_jobs", "exp_machines", "exp_n_op", "dim_check", "dim_source", "parser_ok", "source",
]


def _jsplib_meta() -> dict[str, dict]:
    obj = json.loads((INSTANCES_DIR / "jsplib_instances.json").read_text(encoding="utf-8"))
    return {e["name"]: e for e in obj}


def _row(inst: Instance, ptype: str, spec, dim_check, dim_source, expected) -> dict:
    ej, em, en = expected
    return {
        "id": inst.name,
        "family": inst.family,
        "type": ptype,
        "jobs": inst.num_jobs,
        "machines": inst.num_machines,
        "n_op": inst.num_operations,
        "flexible": inst.is_flexible,
        "exp_jobs": ej if ej is not None else "",
        "exp_machines": em if em is not None else "",
        "exp_n_op": en if en is not None else "",
        "dim_check": {True: "ok", False: "MISMATCH", None: "file-defined"}[dim_check],
        "dim_source": dim_source,
        "parser_ok": bool(dim_check is not False),
        "source": spec.url,
    }


def _validate_fjsp(spec, inst: Instance, meta: dict):
    if spec.family == "brandimarte":
        exp = BRANDIMARTE_DIMS[spec.id]
        ok = (inst.num_jobs, inst.num_machines, inst.num_operations) == exp
        return ok, "Brandimarte (1993)", exp
    if spec.family.startswith("hurink_"):
        base = spec.id.split("_", 1)[1]
        m = meta.get(base, {})
        ej, em = m.get("jobs"), m.get("machines")
        if ej is None:
            return None, f"JSPLIB base {base} (not found)", (None, None, None)
        en = ej * em
        ok = (inst.num_jobs, inst.num_machines, inst.num_operations) == (ej, em, en)
        return ok, f"JSPLIB base {base}", (ej, em, en)
    if spec.family == "fattahi" and spec.id in FATTAHI_DIMS:
        exp = FATTAHI_DIMS[spec.id]
        ok = (inst.num_jobs, inst.num_machines, inst.num_operations) == exp
        return ok, "Fattahi et al. (2007)", exp
    # any other: file-defined, internal consistency only (loader-validated)
    return None, "repo file-defined", (None, None, None)


def build_manifest() -> list[dict]:
    meta = _jsplib_meta()
    rows: list[dict] = []

    for spec in JSSP_INSTANCES:
        inst = load_jssp_file(INSTANCES_DIR / spec.local, spec.id, spec.family, source=spec.url)
        m = meta.get(spec.id, {})
        ej, em = m.get("jobs"), m.get("machines")
        if ej is None:
            dim_check, expected = None, (None, None, None)
        else:
            en = ej * em
            expected = (ej, em, en)
            dim_check = (inst.num_jobs, inst.num_machines, inst.num_operations) == expected
        rows.append(_row(inst, "JSSP", spec, dim_check, "JSPLIB instances.json", expected))

    for spec in FJSP_INSTANCES:
        inst = load_fjsp_file(INSTANCES_DIR / spec.local, spec.id, spec.family, source=spec.url)
        dim_check, dim_source, expected = _validate_fjsp(spec, inst, meta)
        rows.append(_row(inst, "FJSP", spec, dim_check, dim_source, expected))

    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with MANIFEST.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main() -> None:
    rows = build_manifest()
    n = len(rows)
    jssp = sum(1 for r in rows if r["type"] == "JSSP")
    fjsp = n - jssp
    mismatches = [r for r in rows if r["dim_check"] == "MISMATCH"]
    checked = sum(1 for r in rows if r["dim_check"] == "ok")
    print(f"manifest: {n} instances ({jssp} JSSP, {fjsp} FJSP) -> {MANIFEST}")
    print(f"dimension cross-check: {checked} ok, {len(mismatches)} mismatch, "
          f"{n - checked - len(mismatches)} file-defined")
    for r in mismatches:
        print(f"  MISMATCH {r['id']}: parsed ({r['jobs']},{r['machines']},{r['n_op']}) "
              f"vs expected ({r['exp_jobs']},{r['exp_machines']},{r['exp_n_op']})")
    if mismatches:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
