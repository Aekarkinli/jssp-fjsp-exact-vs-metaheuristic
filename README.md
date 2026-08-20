# Exact and metaheuristic search in job-shop and flexible job-shop scheduling under fixed time budgets

Data and code for the study of the same name. The repository contains the complete
experimental software, the benchmark instances and reference values, every run record, the
derived analysis tables, and the scripts that turn those tables into the figures and tables
of the manuscript.

The study compares an open-source exact solver, four problem-specific scheduling methods and
twelve population-based optimisers that reach schedules through one shared random-key
decoder. Seventeen methods are run on 67 job-shop and flexible job-shop instances with 20
seeds per stochastic method under a single-thread budget of 300 seconds, and the values at 1,
10 and 60 seconds are read from the same anytime traces. A deterministic decision rule then
assigns every instance and budget to one of five roles for heuristic search.

## Repository layout

```
config/            locked experimental configuration: panel, seeds, budgets, thresholds
data/
  instances/       benchmark instance files with their source manifest
  instance_manifest.csv   parsed dimensions of every instance, checked against the sources
  reference_values.csv    best-known value, lower bound, certified status and source
src/
  core/            instance model, shared decoder, feasibility checker, schedule object
  methods/         one module per method behind a common solve interface
  run/             runners for the timed comparison, the hybrids, the reference run, CEC2017
  analysis/        raw records to derived tables, statistics, classification, extras
  figtab/          figures, LaTeX tables and the macro file the manuscript reads
results/
  raw/             one compressed archive per experimental stage, one record per run inside
  derived/         tidy tables produced from the raw records
paper/
  generated/       LaTeX tables and the macro file, as used in the manuscript
  figures/         figure files, as used in the manuscript
tools/             environment check, result unpacking, manuscript checks
tests/             unit tests for the decoder, the feasibility checker, the methods and the
                   reference values
```

## Environment

Python 3.11. Pinned versions are in `pyproject.toml`, `uv.lock` and `requirements.txt`. The
versions actually used for every run are recorded inside each raw record and are collected in
`results/derived/environment.json`.

```bash
uv sync                      # or: python -m venv .venv && pip install -r requirements.txt
python tools/smoke_env.py    # verifies that every required package imports
```

## Reproducing the analysis

The derived tables are included, so the figures, the LaTeX tables and every number quoted in
the manuscript can be regenerated without touching the raw records:

```bash
python -m src.figtab.build_all --figures
```

To rebuild the derived tables from the raw records as well, unpack the archives first:

```bash
python tools/unpack_results.py
python -m src.figtab.build_all
```

The second command runs the full chain, which collects the raw records, computes the
comparative statistics, the necessity classification and the supporting analyses, and then
writes the figures, tables and macros. Every number quoted in the manuscript comes from
`paper/generated/numbers.tex`, which this chain writes, and no value is entered by hand.
Rebuilding from the raw records reproduces all 293 macros exactly, with one exception. The
Bayesian comparison estimates its posterior probabilities by Monte Carlo sampling, so those
three values can move in the third decimal from one run to the next.

The figure files under `paper/figures/` are the versions used in the manuscript. Running the
figure step overwrites them with freshly rendered files, which are identical in content and
can differ in small details of layout.

`tools/style_check.py` and `tools/check_macros.py` are the two checks that were applied to
the manuscript sources during preparation. The first enforces the writing rules used
throughout the text and the second verifies that every macro appearing in the text is defined
by the generated macro file. The manuscript sources themselves are not part of this
repository.

## Repeating the experiments

The runners are resumable at the level of a single result file, so an interrupted run
continues where it stopped. Every job is single-threaded and pinned to one performance core,
and the number of concurrent workers never exceeds the number of performance cores.

```bash
python -m src.run.runner --config config/full.yaml --workers 8   # timed comparison
python -m src.run.run_hybrid --variant all --workers 8           # warm-started hybrids
python -m src.run.run_reference                                  # multi-thread reference
python -m src.run.run_cec                                        # continuous benchmark stage
```

The complete study consumed about 1986 core-hours over 40745 recorded runs. Wall-clock
results depend on the machine, which is described in `results/derived/environment.json` and
in the manuscript.

## Raw records

Each record under `results/raw/<stage>/` holds the instance identity and dimensions, the
method and seed, the status, the best objective and dual bound, the times to first and best
solution, the evaluation count, the repair and rejection counts, the final feasibility flag,
the peak memory, the wall time, the complete anytime trace stamped with both wall-clock time
and evaluation count, the final schedule, the library versions, the code revision and the
hardware identifier.

Stages:

| Stage | Content |
|---|---|
| `full` | timed comparison, 17 methods on 67 instances |
| `hybrid` | warm-started exact search, three variants at four total budgets |
| `reference` | multi-thread exact reference on the hardest subset |
| `cec` | continuous benchmark stage, 12 optimisers on 29 functions |
| `ablation_legacy` | decoder-mapping ablation on flexible instances |
| `sens_pop20`, `sens_pop100`, `sens_recommended` | population-size sensitivity |

## Benchmark instances and reference values

The instance files are redistributed from public collections, and `data/instances/SOURCES.json` records
the origin and pinned revision of each collection. `data/reference_values.csv` gives, for
every instance, the best-known value used as the denominator of all reported deviations, the
matching lower bound, whether the source records the value as a proven optimum, and the
collection the value came from. The collection was re-checked instance by instance on
17 August 2026. Three entries had moved since the pinned snapshots, and the current values
are used throughout.

## Tests

```bash
python -m pytest -q
```

The suite covers the instance parsers against published dimensions, the decoder and its
feasibility guarantees, the feasibility checker against deliberately corrupted schedules, each
method on a small instance, and the reference table against its sources.

## Licence

The code in `src/`, `tools/` and `tests/` is released under the MIT Licence, see `LICENSE`.
The benchmark instance files under `data/instances/` are redistributed from their original
public collections and remain under the terms of those collections.

## Citation

If you use this material, please cite the article. `CITATION.cff` holds the machine-readable
entry, which will be updated with the volume and page numbers once they are assigned.
