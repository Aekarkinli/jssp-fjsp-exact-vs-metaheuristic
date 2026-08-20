"""Extract the per-stage archives of raw run records into results/raw/.

The raw records are stored as one compressed archive per experimental stage, because the
uncompressed collection holds more than forty thousand small files. Unpacking is only needed
to rebuild the derived tables from scratch. The derived tables themselves are stored
uncompressed and can be read directly.

    python tools/unpack_results.py            # all stages
    python tools/unpack_results.py full cec   # selected stages
"""
from __future__ import annotations

import sys
import tarfile
from pathlib import Path

RAW = Path("results/raw")


def main() -> None:
    archives = sorted(RAW.glob("*.tar.gz"))
    if not archives:
        raise SystemExit(f"no archives found in {RAW}")
    wanted = set(sys.argv[1:])
    for archive in archives:
        stage = archive.name.replace(".tar.gz", "")
        if wanted and stage not in wanted:
            continue
        target = RAW / stage
        if target.exists():
            print(f"{stage}: already extracted, skipping")
            continue
        with tarfile.open(archive) as tar:
            tar.extractall(RAW)
        n = sum(1 for _ in target.glob("*.json"))
        print(f"{stage}: {n} records extracted")


if __name__ == "__main__":
    main()
