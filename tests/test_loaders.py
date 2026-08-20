"""Parser and manifest validation tests."""
from __future__ import annotations

from pathlib import Path

import pytest

from src.core.instance import build_instance
from src.io.instance_sources import FJSP_INSTANCES, JSSP_INSTANCES
from src.io.loaders import load_fjsp_file, load_jssp_file, parse_jsplib
from src.io.manifest import build_manifest

DATA = Path("data/instances")


def test_ft06_dimensions():
    inst = load_jssp_file(DATA / "jssp/ft06.txt", "ft06", "fisher_thompson")
    assert (inst.num_jobs, inst.num_machines, inst.num_operations) == (6, 6, 36)
    assert inst.problem_type == "JSSP"
    assert not inst.is_flexible
    assert all(len(op.modes) == 1 for op in inst.operations)


def test_mk06_dimensions_and_flexibility():
    inst = load_fjsp_file(DATA / "fjsp/mk06.txt", "mk06", "brandimarte")
    assert (inst.num_jobs, inst.num_machines, inst.num_operations) == (10, 15, 150)
    assert inst.is_flexible
    assert inst.num_flexible_operations > 0


def test_parse_jsplib_small_structure():
    text = "# comment line\n2 2\n0 3 1 2\n1 4 0 1\n"
    inst = parse_jsplib(text, "toy", "test")
    assert (inst.num_jobs, inst.num_machines, inst.num_operations) == (2, 2, 4)
    assert inst.jobs[0][0].modes == ((0, 3),)
    assert inst.jobs[0][1].modes == ((1, 2),)
    assert inst.jobs[1][0].modes == ((1, 4),)
    assert inst.jobs[1][1].modes == ((0, 1),)


def test_manifest_every_instance_cross_checked():
    rows = build_manifest()
    assert len(rows) == len(JSSP_INSTANCES) + len(FJSP_INSTANCES) == 67
    assert all(r["parser_ok"] for r in rows)
    # every instance matches an independent dimension reference (no mismatch, none unchecked)
    assert all(r["dim_check"] == "ok" for r in rows), [
        r["id"] for r in rows if r["dim_check"] != "ok"
    ]


def test_build_instance_rejects_malformed():
    with pytest.raises(ValueError):  # machine index out of range
        build_instance("bad", "t", "JSSP", 2, [[[(5, 3)]]])
    with pytest.raises(ValueError):  # JSSP operation must have exactly one mode
        build_instance("bad", "t", "JSSP", 3, [[[(0, 3), (1, 2)]]])
    with pytest.raises(ValueError):  # non-positive duration
        build_instance("bad", "t", "FJSP", 2, [[[(0, 0)]]])
    with pytest.raises(ValueError):  # duplicate machine in one operation
        build_instance("bad", "t", "FJSP", 2, [[[(0, 3), (0, 4)]]])
