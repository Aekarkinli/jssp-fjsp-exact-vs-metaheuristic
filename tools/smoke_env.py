"""Import smoke test + environment version record.

Confirms the pinned scientific stack imports (including the CP-SAT model object and the
PyJobShop / mealpy entry points) and writes a machine-readable version record to
results/environment.json so resolved versions are echoed into results metadata, as
required by the study. Exit code is non-zero if any required import fails.
"""
from __future__ import annotations

import importlib
import importlib.metadata as md
import json
import platform
import subprocess
import sys
from pathlib import Path

REQUIRED = [
    "numpy", "scipy", "pandas", "pyarrow", "matplotlib",
    "mealpy", "ortools", "pyjobshop", "autorank", "baycomp",
    "joblib", "tqdm", "yaml", "fitz", "psutil",
]
# Useful transitive deps (not hard requirements, but recorded if present).
OPTIONAL = ["opfunu", "fjsplib", "psplib", "statsmodels"]

# import name -> installed distribution name (when they differ)
DIST_NAME = {"yaml": "PyYAML", "fitz": "PyMuPDF"}


def _version(name: str, mod) -> str:
    for attr in ("__version__", "version", "VERSION"):
        v = getattr(mod, attr, None)
        if v is not None and not callable(v):
            return str(v)
    try:
        return md.version(DIST_NAME.get(name, name))
    except Exception:
        return "unknown"


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "uncommitted"


def main() -> None:
    versions: dict[str, str] = {}
    failures: list[str] = []

    for name in REQUIRED:
        try:
            versions[name] = _version(name, importlib.import_module(name))
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{name}: {exc!r}")

    for name in OPTIONAL:
        try:
            versions[name] = _version(name, importlib.import_module(name))
        except Exception:
            pass

    # Confirm the CP-SAT model object is actually constructible.
    try:
        from ortools.sat.python import cp_model
        cp_model.CpModel()
        versions["ortools.cp_model"] = "constructible"
    except Exception as exc:  # noqa: BLE001
        failures.append(f"ortools.cp_model: {exc!r}")

    record = {
        "python": sys.version.split()[0],
        "python_impl": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "git_commit": _git_commit(),
        "packages": dict(sorted(versions.items())),
    }

    out = Path("results/environment.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2), encoding="utf-8")

    print(json.dumps(record, indent=2))
    if failures:
        print("\nFAILED IMPORTS:", file=sys.stderr)
        for f in failures:
            print("  " + f, file=sys.stderr)
        raise SystemExit(1)
    print(f"\nOK: all {len(REQUIRED)} required imports succeeded; wrote {out}")


if __name__ == "__main__":
    main()
