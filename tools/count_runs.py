"""Count recorded runs and core-hours per experimental stage from the raw records.

A timed record is one run, counted at its wall time, as in the manuscript's compute total
(the multi-thread exact reference is also counted at its wall time). A CEC2017 record holds
the runs of one function and one optimiser with their summed wall time, and the CEC2017 stage
is reported separately because it runs on an evaluation budget.

    .venv/Scripts/python.exe tools/count_runs.py
"""
from __future__ import annotations

import json
from pathlib import Path

RAW = Path(__file__).resolve().parents[1] / "results" / "raw"
REVISION = {"rev_hurink", "rev_memetic", "rev_variants", "rev_opperm", "rev_large"}
VARIANTS = ("jade", "shade", "clpso", "igwo")


def main() -> None:
    rows = {}
    for stage in sorted(p for p in RAW.iterdir() if p.is_dir()):
        runs, secs, runs_v, secs_v = 0, 0.0, 0, 0.0
        for f in stage.glob("*.json"):
            d = json.loads(f.read_text(encoding="utf-8"))
            if stage.name == "cec":
                n, s = int(d["runs"]), float(d["wall_s"])
                if f.stem.split("__")[-1] in VARIANTS:
                    runs_v += n
                    secs_v += s
                    continue
            else:
                n = 1
                s = float(d.get("wall_time") or 0.0)
            runs += n
            secs += s
        rows[stage.name] = (runs, secs / 3600)
        if stage.name == "cec":
            rows["cec_variants"] = (runs_v, secs_v / 3600)
    groups = {"main": [k for k in rows if k not in REVISION and not k.startswith("cec")],
              "further": sorted(REVISION & rows.keys()),
              "cec_main": ["cec"], "cec_further": ["cec_variants"]}
    for k, (n, h) in rows.items():
        print(f"{k:18s} {n:7d} runs {h:9.1f} core-h")
    out = {"stages": {k: {"runs": n, "core_hours": round(h, 1)} for k, (n, h) in rows.items()}}
    for g, ks in groups.items():
        n = sum(rows[k][0] for k in ks)
        h = sum(rows[k][1] for k in ks)
        out[g] = {"runs": n, "core_hours": round(h, 1)}
        print(f"{g:18s} {n:7d} runs {h:9.1f} core-h")
    (RAW.parent / "derived" / "run_counts.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
