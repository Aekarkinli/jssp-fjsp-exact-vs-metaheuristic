"""Download the curated benchmark set from the pinned sources, with provenance.

Writes JSSP text files to `data/instances/jssp/`, FJSP JSON files to
`data/instances/fjsp/`, the JSPLIB metadata to `data/instances/jsplib_instances.json`,
and a `data/instances/SOURCES.json` recording every file's source URL, SHA-256, and size,
plus the pinned commit SHAs. Idempotent: existing files are reused (checksummed in place)
unless `--force` is given.

    uv run python -m src.io.fetch_instances [--force]
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
import urllib.request
from pathlib import Path

from src.io.instance_sources import (
    FJSP_BKS_URL,
    FJSP_INSTANCES,
    JSPLIB_METADATA_URL,
    JSSP_BKS_URL,
    JSSP_INSTANCES,
    SOURCES,
)

INSTANCES_DIR = Path("data/instances")


def _download(url: str, timeout: int = 60) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "solver-aware-study/0.1"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 (pinned https URLs)
        return resp.read()


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fetch(force: bool = False) -> dict:
    (INSTANCES_DIR / "jssp").mkdir(parents=True, exist_ok=True)
    (INSTANCES_DIR / "fjsp").mkdir(parents=True, exist_ok=True)

    items: list[tuple[str, str]] = [
        ("jsplib_instances.json", JSPLIB_METADATA_URL),
        ("fjsp_bks.json", FJSP_BKS_URL),
        ("jssp_bks_weise.txt", JSSP_BKS_URL),
    ]
    items += [(s.local, s.url) for s in JSSP_INSTANCES]
    items += [(s.local, s.url) for s in FJSP_INSTANCES]

    provenance: dict = {
        "sources": SOURCES,
        "retrieved_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "files": {},
    }
    n_downloaded = 0
    for local, url in items:
        dest = INSTANCES_DIR / local
        if dest.exists() and not force:
            data = dest.read_bytes()
        else:
            data = _download(url)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
            n_downloaded += 1
        provenance["files"][local] = {"url": url, "sha256": _sha256(data), "bytes": len(data)}

    (INSTANCES_DIR / "SOURCES.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8"
    )
    provenance["_n_downloaded"] = n_downloaded
    return provenance


def main() -> None:
    force = "--force" in sys.argv
    prov = fetch(force=force)
    print(
        f"{len(prov['files'])} files recorded "
        f"({prov['_n_downloaded']} newly downloaded) -> {INSTANCES_DIR}"
    )


if __name__ == "__main__":
    main()
