"""Parsers that turn benchmark instance files into the unified `Instance` model.

- **JSSP (JSPLIB "standard specification")**: comment lines start with `#`; the first
  data line is `num_jobs num_machines`; each subsequent line lists, for one job,
  `machine duration` pairs in processing order. Machines are 0-indexed. Every job visits
  every machine exactly once, so a job has `num_machines` operations.

- **FJSP (classic 1-indexed text format)**: parsed by the audited `fjsplib` reader, which
  returns `jobs` as `list[list[list[(machine, duration)]]]` with machines converted to
  0-indexed, then wrapped into the unified `Instance`. The number of machines comes from
  the file header (so declared-but-idle machines are preserved, as in Brandimarte mk06).
"""
from __future__ import annotations

from pathlib import Path

import fjsplib

from src.core.instance import Instance, build_instance


def parse_jsplib(text: str, name: str, family: str, source: str = "") -> Instance:
    """Parse a JSSP instance in JSPLIB standard-specification text format."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
    if not lines:
        raise ValueError(f"{name}: empty JSSP instance")
    header = lines[0].split()
    num_jobs, num_machines = int(header[0]), int(header[1])
    body = lines[1:]
    if len(body) < num_jobs:
        raise ValueError(f"{name}: expected {num_jobs} job lines, found {len(body)}")

    raw_jobs: list[list[list[tuple[int, int]]]] = []
    for j in range(num_jobs):
        nums = list(map(int, body[j].split()))
        if len(nums) != 2 * num_machines:
            raise ValueError(
                f"{name}: job {j} has {len(nums)} values, expected {2 * num_machines} "
                f"(machine, duration pairs for {num_machines} machines)"
            )
        ops = [[(nums[2 * k], nums[2 * k + 1])] for k in range(num_machines)]
        raw_jobs.append(ops)

    return build_instance(name, family, "JSSP", num_machines, raw_jobs, source)


def from_fjsplib(fi: "fjsplib.Instance", name: str, family: str, source: str = "") -> Instance:
    """Wrap an `fjsplib.Instance` into the unified `Instance` model."""
    raw_jobs = [[list(op) for op in job] for job in fi.jobs]
    return build_instance(name, family, "FJSP", fi.num_machines, raw_jobs, source)


def load_jssp_file(path: str | Path, name: str, family: str, source: str = "") -> Instance:
    return parse_jsplib(Path(path).read_text(encoding="utf-8"), name, family, source)


def load_fjsp_file(path: str | Path, name: str, family: str, source: str = "") -> Instance:
    return from_fjsplib(fjsplib.read(Path(path)), name, family, source)
