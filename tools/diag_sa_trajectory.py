"""Temperature trajectory of the annealing method, from an instrumented copy (tools/_anneal_trace.py).

    .venv/Scripts/python.exe tools/diag_sa_trajectory.py tools dp15a behnke11 la21
"""
import sys
sys.path.insert(0, "."); sys.path.insert(0, sys.argv[1])
import _anneal_trace as A
from src.run.runner import _load
for iid in sys.argv[2:]:
    A.TRACE.clear()
    r = A.solve_anneal(_load(iid), 60.0, 11)
    tr = A.TRACE
    print(iid, "t0", round(r.extra["t0"], 1), "chains", len(tr))
    for k in [0, len(tr)//10, len(tr)//4, len(tr)//2, 3*len(tr)//4, len(tr)-1]:
        print("   t=%s T=%s cur=%s best=%s sigma=%s acc=%s" % tr[k])
