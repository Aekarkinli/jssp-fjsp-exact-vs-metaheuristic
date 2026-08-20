"""Decoder correctness, determinism, repair, and feasibility tests."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.core.decoder import LEGACY, decode, decode_vector, split_vector, vector_length
from src.core.feasibility import check_feasibility
from src.core.instance import build_instance
from src.io.loaders import load_fjsp_file, load_jssp_file

DATA = Path("data/instances")


def _two_by_two():
    # Job0: M0 dur 3, then M1 dur 2 ;  Job1: M1 dur 4, then M0 dur 1
    return build_instance(
        "2x2", "test", "JSSP", 2,
        [[[(0, 3)], [(1, 2)]], [[(1, 4)], [(0, 1)]]],
    )


def test_decode_hand_traced_makespan_six():
    """Priorities make Job1's first operation win the machine-1 conflict (optimum 6)."""
    inst = _two_by_two()
    res = decode(inst, [0.0, 0.9, 0.1, 0.5])  # g2 < g1 -> Job1.op0 scheduled first on M1
    sched = res.schedule
    assert check_feasibility(sched).feasible
    assert sched.makespan == 6
    got = [(a.machine, a.start, a.end) for a in sched.assignments]
    assert got == [(0, 0, 3), (1, 4, 6), (1, 0, 4), (0, 4, 5)]


def test_decode_priority_changes_outcome():
    inst = _two_by_two()
    res = decode(inst, [0.0, 0.1, 0.9, 0.5])  # g1 < g2 -> Job0.op1 first; worse, makespan 10
    assert res.schedule.makespan == 10
    assert check_feasibility(res.schedule).feasible


def test_vector_length_and_split():
    j = load_jssp_file(DATA / "jssp/ft06.txt", "ft06", "f")
    assert vector_length(j) == j.num_operations
    pri, mk = split_vector(j, np.zeros(vector_length(j)))
    assert mk is None and pri.size == j.num_operations

    f = load_fjsp_file(DATA / "fjsp/mk01.txt", "mk01", "b")
    assert vector_length(f) == 2 * f.num_operations
    pri, mk = split_vector(f, np.zeros(vector_length(f)))
    assert mk is not None and pri.size == f.num_operations == mk.size


@pytest.mark.parametrize(
    "rel,name,kind",
    [
        ("jssp/ft06.txt", "ft06", "jssp"),
        ("jssp/la01.txt", "la01", "jssp"),
        ("jssp/abz7.txt", "abz7", "jssp"),
        ("fjsp/mk01.txt", "mk01", "fjsp"),
        ("fjsp/mk06.txt", "mk06", "fjsp"),
        ("fjsp/edata_la01.txt", "edata_la01", "fjsp"),
    ],
)
def test_decode_random_keys_feasible(rel, name, kind):
    path = DATA / rel
    inst = load_jssp_file(path, name, "x") if kind == "jssp" else load_fjsp_file(path, name, "x")
    rng = np.random.default_rng(2024)
    res = decode_vector(inst, rng.random(vector_length(inst)))
    fr = check_feasibility(res.schedule)
    assert fr.feasible, fr.violations
    assert res.schedule.makespan > 0


def test_decode_is_deterministic():
    inst = load_fjsp_file(DATA / "fjsp/mk01.txt", "mk01", "b")
    v = np.random.default_rng(7).random(vector_length(inst))
    a, b = decode_vector(inst, v), decode_vector(inst, v)
    assert a.schedule.makespan == b.schedule.makespan
    assert a.n_repairs == b.n_repairs
    assert [x.machine for x in a.schedule.assignments] == [x.machine for x in b.schedule.assignments]
    assert [x.start for x in a.schedule.assignments] == [x.start for x in b.schedule.assignments]


def test_machine_key_indexes_the_eligible_set():
    # op0 eligible on machines {0, 2}; op1 only on machine 1; three machines.
    inst = build_instance("flex", "test", "FJSP", 3, [[[(0, 5), (2, 4)], [(1, 3)]]])
    assert inst.is_flexible
    # Keys below 0.5 pick the first eligible machine, keys at or above it the second.
    lower = decode(inst, priorities=[0.0, 0.0], machine_keys=[0.4, 0.0])
    upper = decode(inst, priorities=[0.0, 0.0], machine_keys=[0.8, 0.0])
    assert lower.schedule.assignments[0].machine == 0
    assert upper.schedule.assignments[0].machine == 2
    assert lower.n_repairs == 0 and upper.n_repairs == 0
    assert check_feasibility(lower.schedule).feasible


def test_primary_mapping_never_repairs_on_any_instance():
    for rel, name in (("fjsp/mk01.txt", "mk01"), ("fjsp/mk06.txt", "mk06"),
                      ("fjsp/edata_la01.txt", "edata_la01")):
        inst = load_fjsp_file(DATA / rel, name, "x")
        rng = np.random.default_rng(11)
        for _ in range(20):
            res = decode_vector(inst, rng.random(vector_length(inst)))
            assert res.n_repairs == 0
            assert check_feasibility(res.schedule).feasible


def test_primary_mapping_is_uniform_over_eligible_machines():
    """A uniform key must give each eligible machine the same share.

    The mapping used in the first version sent the key to a global machine index and then
    moved an ineligible choice to the nearest index, which made the share depend on how the
    machines happened to be numbered. Here the two eligible machines of the first operation
    are far apart in index, so any index-distance effect would show up immediately.
    """
    inst = build_instance("flex", "test", "FJSP", 5, [[[(0, 5), (4, 5)], [(1, 3)]]])
    rng = np.random.default_rng(3)
    counts = {0: 0, 4: 0}
    n = 4000
    for _ in range(n):
        res = decode(inst, priorities=[0.0, 0.0], machine_keys=[rng.random(), 0.0])
        counts[res.schedule.assignments[0].machine] += 1
    assert abs(counts[0] / n - 0.5) < 0.03
    assert abs(counts[4] / n - 0.5) < 0.03


def test_primary_mapping_is_invariant_to_machine_relabelling():
    """Relabelling the machines must not change which eligible machine a key selects."""
    a = build_instance("a", "test", "FJSP", 5, [[[(0, 7), (4, 9)], [(2, 3)]]])
    b = build_instance("b", "test", "FJSP", 5, [[[(1, 7), (3, 9)], [(2, 3)]]])
    rng = np.random.default_rng(5)
    for _ in range(200):
        key = rng.random()
        ra = decode(a, priorities=[0.0, 0.0], machine_keys=[key, 0.0])
        rb = decode(b, priorities=[0.0, 0.0], machine_keys=[key, 0.0])
        # same rank inside the eligible set, hence the same processing time
        assert ra.schedule.assignments[0].duration == rb.schedule.assignments[0].duration


def test_legacy_mapping_reproduces_the_biased_behaviour():
    """The ablation mapping must still behave exactly as the first version's did."""
    inst = build_instance("flex", "test", "FJSP", 3, [[[(0, 5), (2, 4)], [(1, 3)]]])
    # key 0.4 -> global index 1, ineligible for op0 -> moved to the nearest eligible index
    res = decode(inst, priorities=[0.0, 0.0], machine_keys=[0.4, 0.0], mapping=LEGACY)
    assert res.n_repairs == 1
    assert res.schedule.assignments[0].machine == 0
    assert check_feasibility(res.schedule).feasible

    biased = build_instance("wide", "test", "FJSP", 5, [[[(0, 5), (4, 5)], [(1, 3)]]])
    rng = np.random.default_rng(3)
    first = 0
    n = 2000
    for _ in range(n):
        r = decode(biased, priorities=[0.0, 0.0], machine_keys=[rng.random(), 0.0],
                   mapping=LEGACY)
        first += int(r.schedule.assignments[0].machine == 0)
    # machines {0,4} of five: indices 0,1,2 fall to machine 0 and 3,4 to machine 4
    assert first / n > 0.55
