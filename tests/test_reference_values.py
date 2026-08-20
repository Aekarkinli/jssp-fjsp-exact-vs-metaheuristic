"""Reference-value table tests: coverage, consistency, and spot checks vs the literature."""
from __future__ import annotations

from src.io.reference_values import build_reference_values


def _rows():
    return {r["id"]: r for r in build_reference_values()}


def test_full_coverage_and_consistency():
    rows = _rows()
    assert len(rows) == 67
    for r in rows.values():
        assert int(r["LB"]) <= int(r["BKS"]), r
        assert isinstance(r["proven_optimal"], bool)
        assert r["BKS_source"] and r["LB_source"]
        # proven optimal must mean LB == BKS
        assert r["proven_optimal"] == (int(r["LB"]) == int(r["BKS"])), r


def test_known_jssp_optima():
    rows = _rows()
    for inst, opt in [("ft06", 55), ("ft10", 930), ("ft20", 1165),
                      ("la01", 666), ("abz5", 1234), ("orb01", 1059)]:
        r = rows[inst]
        assert int(r["BKS"]) == opt and r["proven_optimal"], (inst, r)


def test_ta71_filled_from_weise_fallback():
    r = _rows()["ta71"]
    assert int(r["BKS"]) == 5464 == int(r["LB"])
    assert r["proven_optimal"]
    assert "Weise" in r["BKS_source"]


def test_known_fjsp_values():
    rows = _rows()
    assert int(rows["mk01"]["BKS"]) == 40 and rows["mk01"]["proven_optimal"]
    assert int(rows["mk06"]["BKS"]) == 57 and rows["mk06"]["proven_optimal"]
    # mk10 is open: LB 189 < UB 193
    assert (int(rows["mk10"]["LB"]), int(rows["mk10"]["BKS"])) == (189, 193)
    assert not rows["mk10"]["proven_optimal"]
    # Hurink variant mapping anchored by the JSSP optimum (sdata excluded from the study,
    # edata value must be below the pure-JSSP optimum 666 for la01)
    assert int(rows["edata_la01"]["BKS"]) == 609
    assert int(rows["vdata_la01"]["BKS"]) == 570


def test_open_instances_after_literature_update():
    # For the harder job-shop instances JSPLIB carries only loose bounds, so where the maintained
    # Weise literature table reports a tighter best-known the generator adopts it. That update
    # closes abz9, swv11, ta11 and ta21. The public collection was re-checked on 2026-08-17,
    # which closed abz8 and swv06 as well, leaving these four still open.
    rows = _rows()
    open_ids = {r["id"] for r in rows.values() if not r["proven_optimal"]}
    assert open_ids == {"ta41", "mk10", "rdata_la21", "rdata_la26"}


def test_revised_entries_carry_the_revision_source():
    # Three entries were revised against the public collection after the experiments were run.
    # Their bounds and their source string must both reflect that check.
    rows = _rows()
    for inst, bks, lb, proven in [("abz8", 667, 667, True), ("swv06", 1667, 1667, True),
                                  ("ta41", 2005, 1926, False)]:
        r = rows[inst]
        assert int(r["BKS"]) == bks and int(r["LB"]) == lb, (inst, r)
        assert bool(r["proven_optimal"]) is proven, (inst, r)
        assert "2026-08-17" in r["BKS_source"], (inst, r)


def test_taillard_best_known_updated_from_weise():
    # Verified against the maintained Taillard results (LB == UB, proven optimal). The stale JSPLIB
    # upper bounds (ta11 1361, ta21 1644) must not reappear, as they produced a negative gap.
    rows = _rows()
    for inst, opt in [("ta11", 1357), ("ta21", 1642)]:
        r = rows[inst]
        assert int(r["BKS"]) == opt == int(r["LB"]) and r["proven_optimal"], (inst, r)
        assert "Weise" in r["BKS_source"]
    assert int(rows["ta41"]["BKS"]) == 2005 and not rows["ta41"]["proven_optimal"]
