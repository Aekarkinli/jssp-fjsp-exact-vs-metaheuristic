"""Commit identity for result files, resolved in a way that survives the run account.

Every result file records the commit of the code that produced it, which is what lets a
reader tie a number in the manuscript to an exact version of the source. The experiment
queue runs under a service account that has no version-control client on its path, so asking
the shell for the commit there returns nothing. The queue driver therefore stamps the commit
into a small file whenever it can reach the client, and the workers read that file when the
direct query fails.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

STAMP = Path("results/run_state/git_commit.txt")
CANDIDATES = (
    "git",
    r"C:\Program Files\Git\cmd\git.exe",
    r"C:\Program Files (x86)\Git\cmd\git.exe",
)


def _query() -> str | None:
    for candidate in CANDIDATES:
        exe = shutil.which(candidate) if candidate == "git" else candidate
        if not exe or not Path(exe).exists():
            continue
        try:
            out = subprocess.check_output(
                [exe, "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL, timeout=15
            ).strip()
            if out:
                return out
        except Exception:
            continue
    return None


def stamp_commit() -> str:
    """Record the current commit, if it can be determined, and return it."""
    commit = _query()
    if commit:
        STAMP.parent.mkdir(parents=True, exist_ok=True)
        STAMP.write_text(commit, encoding="utf-8")
        return commit
    return git_commit()


def git_commit() -> str:
    commit = _query()
    if commit:
        return commit
    if STAMP.exists():
        # the stamp may be written by a shell that prefixes a byte-order mark
        recorded = STAMP.read_text(encoding="utf-8-sig").strip().lstrip("﻿")
        if recorded:
            return recorded
    return "unknown"
