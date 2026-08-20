"""Detect CPU core topology on Windows (performance vs efficiency cores).

The timed fair comparison must run on a single homogeneous core type, one job per
physical performance core, with no simultaneous-multithreading (SMT) oversubscription.
On hybrid Intel CPUs (e.g. i9-13900HX: 8 P-cores + 16 E-cores) the operating system
exposes a per-physical-core EfficiencyClass. Performance cores carry the highest
EfficiencyClass value.

This module queries the Win32 API ``GetLogicalProcessorInformationEx`` (relationship
``RelationProcessorCore``) and returns, for each physical core, its EfficiencyClass,
whether it is SMT-capable, and the affinity mask of its logical processors. From that
it derives the recommended affinity for the timed comparison: one representative
logical processor per performance physical core (the lowest-indexed logical processor
of each P-core, so SMT siblings are never both used).

Standard library only (ctypes); runs on any Python. Importable by the runner and
runnable as a script that prints a JSON summary.
"""
from __future__ import annotations

import ctypes
import json
import os
import struct
import sys
from dataclasses import dataclass, asdict

RelationProcessorCore = 0
ERROR_INSUFFICIENT_BUFFER = 122
LTP_PC_SMT = 0x1


@dataclass
class PhysicalCore:
    efficiency_class: int
    smt: bool
    group: int
    mask: int  # KAFFINITY bitmask of logical processors belonging to this core
    logical_processors: list[int]


@dataclass
class Topology:
    hybrid: bool
    physical_core_count: int
    logical_processor_count: int
    perf_efficiency_class: int
    perf_physical_cores: int
    eff_physical_cores: int
    # one logical processor per performance physical core (no SMT sibling), for affinity
    timed_affinity_logical: list[int]
    timed_affinity_mask: int
    recommended_timed_workers: int
    cores: list[dict]


def _bits(mask: int) -> list[int]:
    out = []
    i = 0
    while mask:
        if mask & 1:
            out.append(i)
        mask >>= 1
        i += 1
    return out


def detect() -> Topology:
    if not sys.platform.startswith("win"):
        raise RuntimeError("cpu_topology.detect() currently supports Windows only")

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    GetLPIEx = kernel32.GetLogicalProcessorInformationEx
    GetLPIEx.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong)]
    GetLPIEx.restype = ctypes.c_int

    length = ctypes.c_ulong(0)
    GetLPIEx(RelationProcessorCore, None, ctypes.byref(length))
    if ctypes.get_last_error() not in (ERROR_INSUFFICIENT_BUFFER, 0):
        raise ctypes.WinError(ctypes.get_last_error())

    buf = (ctypes.c_byte * length.value)()
    if not GetLPIEx(RelationProcessorCore, buf, ctypes.byref(length)):
        raise ctypes.WinError(ctypes.get_last_error())

    raw = bytes(buf)
    cores: list[PhysicalCore] = []
    offset = 0
    n = len(raw)
    while offset < n:
        # SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX header: Relationship(DWORD) Size(DWORD)
        relationship, size = struct.unpack_from("<II", raw, offset)
        if size == 0:
            break
        if relationship == RelationProcessorCore:
            # PROCESSOR_RELATIONSHIP: Flags(BYTE) EfficiencyClass(BYTE) Reserved[20] GroupCount(WORD) GroupMask[]
            base = offset + 8
            flags = raw[base]
            eff = raw[base + 1]
            group_count = struct.unpack_from("<H", raw, base + 22)[0]
            gm = base + 24
            group = 0
            mask = 0
            logical: list[int] = []
            for g in range(max(group_count, 1)):
                # GROUP_AFFINITY: KAFFINITY Mask(ULONG_PTR=8 on x64) Group(WORD) Reserved[3](WORD)
                m = struct.unpack_from("<Q", raw, gm)[0]
                grp = struct.unpack_from("<H", raw, gm + 8)[0]
                gm += 16
                group = grp
                mask |= m
                logical.extend(_bits(m))
            cores.append(
                PhysicalCore(
                    efficiency_class=eff,
                    smt=bool(flags & LTP_PC_SMT),
                    group=group,
                    mask=mask,
                    logical_processors=logical,
                )
            )
        offset += size

    if not cores:
        raise RuntimeError("no processor-core information returned")

    eff_classes = sorted({c.efficiency_class for c in cores})
    perf_class = eff_classes[-1]
    hybrid = len(eff_classes) > 1
    perf_cores = [c for c in cores if c.efficiency_class == perf_class]
    eff_cores = [c for c in cores if c.efficiency_class != perf_class]

    # one logical processor per performance physical core (lowest index → avoid SMT sibling)
    timed_logical = sorted(min(c.logical_processors) for c in perf_cores)
    timed_mask = 0
    for lp in timed_logical:
        timed_mask |= (1 << lp)

    logical_count = sum(len(c.logical_processors) for c in cores)

    return Topology(
        hybrid=hybrid,
        physical_core_count=len(cores),
        logical_processor_count=logical_count,
        perf_efficiency_class=perf_class,
        perf_physical_cores=len(perf_cores),
        eff_physical_cores=len(eff_cores),
        timed_affinity_logical=timed_logical,
        timed_affinity_mask=timed_mask,
        recommended_timed_workers=len(perf_cores),
        cores=[asdict(c) for c in cores],
    )


def main() -> None:
    topo = detect()
    summary = asdict(topo)
    # keep the per-core dump compact in the printed summary
    summary["cores"] = [
        {"efficiency_class": c["efficiency_class"], "smt": c["smt"],
         "logical_processors": c["logical_processors"]}
        for c in summary["cores"]
    ]
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
