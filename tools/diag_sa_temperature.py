"""Diagnostic: initial temperature of the annealing method against the starting makespan.

Runs the annealing method for 30 s on the given instances and prints the calibrated initial
temperature, the start and best makespan, the number of evaluations and of improvements.
Used to explain why annealing stays at its dispatching start on flexible instances.

    .venv/Scripts/python.exe tools/diag_sa_temperature.py behnke11 dp15a mk10 ta42 la21
"""
import sys, time
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from src.run.runner import _load
from src.methods import anneal
from src.methods.disjunctive import Graph

calls = {"n": 0}
orig = Graph.evaluate
def counted(self, *a, **k):
    calls["n"] += 1
    return orig(self, *a, **k)
Graph.evaluate = counted
for iid in sys.argv[1:]:
    inst = _load(iid)
    calls["n"] = 0
    r = anneal.solve_anneal(inst, 30.0, 11)
    n = inst.num_operations
    print(f"{iid}: n_op={n} chain={max(100,n)} evals_30s={calls['n']} "
          f"chains~{calls['n']/max(100,n)/1.0:.1f} t0={r.extra['t0']:.1f} "
          f"start={r.anytime[0].obj:.0f} best={r.best_obj:.0f} improvements={len(r.anytime)-1}")
