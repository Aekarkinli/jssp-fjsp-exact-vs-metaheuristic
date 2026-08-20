"""Curated benchmark set and pinned download sources (single source of truth).

Two public repositories are pinned to immutable commit SHAs so the exact instance bytes
are reproducible. Original benchmark families are cited from their origin papers in the
manuscript and `references/verification_log.md`; these repositories are mirrors used only
to obtain the standard files.

- JSSP: tamy0612/JSPLIB (standard-specification text + `instances.json` with optima/bounds).
- FJSP: ScheduleOpt/benchmarks, `text_fjsp_old_format` (the classic 1-indexed FJSP format
  with explicit job/machine headers), parsed via the audited `fjsplib` reader.

The FJSP source was chosen over an alternative mirror after dimension cross-checking
revealed that the alternative had compacted Brandimarte mk06 to 10 machines, whereas the
canonical instance has 15 machines (five idle). ScheduleOpt carries the canonical headers.

Instance IDs are unique across the whole study; FJSP Hurink instances reuse Lawrence names
(la01, ...) so they are prefixed by their variant (edata/rdata/vdata).
"""
from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import quote

# --- pinned sources -----------------------------------------------------------
JSPLIB_REPO = "tamy0612/JSPLIB"
JSPLIB_COMMIT = "eea2b60dd7e2f5c907ff7302662c61812eb7efdf"

FJSP_REPO = "ScheduleOpt/benchmarks"
FJSP_COMMIT = "fff94e1d1735c61c65b231a8ca273431a9be3826"
_FJSP_BASE = "flexible jobshop/instances/text_fjsp_old_format"

_RAW = "https://raw.githubusercontent.com/{repo}/{commit}/{path}"


@dataclass(frozen=True)
class JsspSpec:
    id: str
    family: str

    @property
    def url(self) -> str:
        return _RAW.format(repo=JSPLIB_REPO, commit=JSPLIB_COMMIT, path=f"instances/{self.id}")

    @property
    def local(self) -> str:
        return f"jssp/{self.id}.txt"


@dataclass(frozen=True)
class FjspSpec:
    id: str
    family: str
    repo_path: str  # path within the FJSP repo (may contain spaces)

    @property
    def url(self) -> str:
        return _RAW.format(repo=FJSP_REPO, commit=FJSP_COMMIT, path=quote(self.repo_path, safe="/"))

    @property
    def local(self) -> str:
        return f"fjsp/{self.id}.txt"


JSPLIB_METADATA_URL = _RAW.format(repo=JSPLIB_REPO, commit=JSPLIB_COMMIT, path="instances.json")
FJSP_BKS_URL = _RAW.format(
    repo=FJSP_REPO, commit=FJSP_COMMIT, path=quote("flexible jobshop/solutions/bks.json", safe="/")
)

# Curated literature dataset of JSSP best-known solutions with bibliographic references
# (Weise et al.). Used as the fallback where JSPLIB metadata has no value (ta71) and as a
# cross-check on proven optima.
JSSP_BKS_REPO = "thomasWeise/jsspInstancesAndResults"
JSSP_BKS_COMMIT = "29a50db4b0e9b83469f1d262fe6a1d27f2991135"
JSSP_BKS_URL = _RAW.format(
    repo=JSSP_BKS_REPO, commit=JSSP_BKS_COMMIT,
    path="data-raw/instances/instances_with_bks.txt",
)


def _jssp(names: list[str], family: str) -> list[JsspSpec]:
    return [JsspSpec(id=n, family=family) for n in names]


# --- JSSP selection (37) ------------------------------------------------------
JSSP_INSTANCES: list[JsspSpec] = (
    _jssp(["ft06", "ft10", "ft20"], "fisher_thompson")
    + _jssp(
        ["la01", "la06", "la11", "la16", "la19", "la21", "la24", "la26", "la29",
         "la31", "la36", "la38", "la40"],
        "lawrence",
    )
    + _jssp(["abz5", "abz6", "abz7", "abz8", "abz9"], "adams_balas_zawack")
    + _jssp(["orb01", "orb02", "orb03", "orb04", "orb05"], "applegate_cook_orb")
    + _jssp(["swv01", "swv06", "swv11"], "storer_wu_vaccari")
    + _jssp(["ta01", "ta11", "ta21", "ta31", "ta41", "ta51", "ta61", "ta71"], "taillard")
)


# --- FJSP selection (30) ------------------------------------------------------
_HURINK_NAMES = ["la01", "la06", "la16", "la21", "la26"]
_FATTAHI_IDS = [1, 5, 10, 15, 20]  # span SFJS (small, 1-10) to MFJS (medium, 11-20)


def _brandimarte() -> list[FjspSpec]:
    return [
        FjspSpec(id=f"mk{i:02d}", family="brandimarte",
                 repo_path=f"{_FJSP_BASE}/Brandimarte1993/mk{i:02d}.txt")
        for i in range(1, 11)  # mk01..mk10 = classic Brandimarte (1993) set
    ]


def _hurink(variant: str, names: list[str]) -> list[FjspSpec]:
    return [
        FjspSpec(id=f"{variant}_{n}", family=f"hurink_{variant}",
                 repo_path=f"{_FJSP_BASE}/HurinkJurischThole1994/{variant}/{n}.txt")
        for n in names
    ]


def _fattahi() -> list[FjspSpec]:
    return [
        FjspSpec(id=f"fattahi{k:02d}", family="fattahi",
                 repo_path=f"{_FJSP_BASE}/FattahiMehrabadJolai2007/fattahi{k}.txt")
        for k in _FATTAHI_IDS
    ]


FJSP_INSTANCES: list[FjspSpec] = (
    _brandimarte()
    + _hurink("edata", _HURINK_NAMES)
    + _hurink("rdata", _HURINK_NAMES)
    + _hurink("vdata", _HURINK_NAMES)
    + _fattahi()
)


def all_specs() -> list:
    return list(JSSP_INSTANCES) + list(FJSP_INSTANCES)


SOURCES = {
    "jssp": {"repo": JSPLIB_REPO, "commit": JSPLIB_COMMIT, "metadata_url": JSPLIB_METADATA_URL},
    "jssp_bks": {"repo": JSSP_BKS_REPO, "commit": JSSP_BKS_COMMIT, "url": JSSP_BKS_URL},
    "fjsp": {
        "repo": FJSP_REPO,
        "commit": FJSP_COMMIT,
        "format": "text_fjsp_old_format (fjsplib)",
        "bks_url": FJSP_BKS_URL,
    },
}
