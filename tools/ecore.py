"""Run a Python module on the efficiency cores only.

Analysis work done while timed experiments occupy the performance cores is confined to the
efficiency cores, so it cannot compete with a timed job for a core. Child processes inherit
the affinity.

    .venv/Scripts/python.exe tools/ecore.py -m src.analysis.necessity
"""
import os
import runpy
import sys

import psutil

# drop the script directory: tools/queue.py would shadow the standard-library queue module
sys.path[:] = [p for p in sys.path if os.path.abspath(p) != os.path.dirname(os.path.abspath(__file__))]
sys.path.insert(0, os.getcwd())
E_CORES = list(range(16, psutil.cpu_count(logical=True)))
psutil.Process().cpu_affinity(E_CORES)
if len(sys.argv) >= 3 and sys.argv[1] == "-m":
    mod = sys.argv[2]
    sys.argv = [mod] + sys.argv[3:]
    runpy.run_module(mod, run_name="__main__", alter_sys=True)
else:
    raise SystemExit("usage: tools/ecore.py -m module [args]")
