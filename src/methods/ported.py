"""Python ports of the two provided methods, MDE and CSA.

Faithful transcriptions of the author MATLAB (`methods/reference_code/MDE/algo_MDE.m`,
`methods/reference_code/CSA/CSA.m`), following the code where it differs from the paper
(recorded in each method's NOTES.md). The random stream is NumPy's, not MATLAB's, so runs
are not bit-identical to the original; the port-validation gate checks that each reproduces
the qualitative convergence reported in its paper on the CEC functions before either is
attached to the scheduling decoder.

The objective is evaluated on a whole population matrix at once (rows are individuals),
matching the MATLAB `feval(fnc, P, mydata)` convention, so the same optimiser serves both
the continuous validation (CEC functions) and the scheduling use (decode each row).
"""
from __future__ import annotations

import time

import numpy as np

from src.core.decoder import DEFAULT_MAPPING, decode_vector, vector_length
from src.core.instance import Instance
from src.methods.base import MethodResult, RunRecorder

BatchObjective = "callable: (M, dim) array -> (M,) fitness"


# --------------------------------------------------------------------------- MDE
def _mde_border(x: np.ndarray, low: np.ndarray, up: np.ndarray, rng) -> np.ndarray:
    x = x.copy()
    below, above = x < low, x > up
    if below.any():
        cand = np.minimum(rng.random(x.shape) * low, low)
        x = np.where(below, cand, x)
    if above.any():
        cand = np.maximum(rng.random(x.shape) * up, up)
        x = np.where(above, cand, x)
    return x


def mde_optimize(obj, dim, lb, ub, pop_size, seed, max_cycles=10**9, stop=None):
    """Multi-population Based Differential Evolution (Karkinli 2023). Minimiser."""
    rng = np.random.default_rng(seed)
    n, t = pop_size, 3
    low = np.full(dim, lb, dtype=float)
    up = np.full(dim, ub, dtype=float)

    A = rng.random((t * n, dim)) * (up - low) + low
    fitA = np.asarray(obj(A), dtype=float)
    best_i = int(np.argmin(fitA))
    best = A[best_i].copy()
    best_val, best_sol = float(fitA[best_i]), best.copy()
    noise = np.zeros((n, dim))

    cycle = 0
    while cycle < max_cycles:
        if stop is not None and stop():
            break
        cycle += 1
        j0 = rng.permutation(t * n)[:n]
        B = A[j0].copy()
        fitB = fitA[j0].copy()

        # mutation: per-individual scale, then per-element dx/dy blending
        a = rng.integers(0, 2, size=n)
        e1 = rng.integers(1, 11, size=n)
        rn = rng.standard_normal(n)
        e2 = rng.integers(1, 6, size=n)
        scale = np.abs(a - rng.random(n) ** e1) * (rn ** e2)
        temp = np.empty((n, dim))
        for i in range(n):
            while True:
                r = rng.permutation(n)[:2]
                if r[0] != i and r[1] != i:
                    break
            dx = np.where(rng.random(dim) < 0.5, B[r[0]], best)
            dy = np.where(rng.random(dim) < 0.5, B[i], B[r[1]])
            temp[i] = B[r[1]] + scale[i] * (dx - dy)

        # parameter-free crossover mask
        c = 1 if rng.random() ** rng.integers(1, 6) < 0.5 else dim
        a_map = rng.integers(0, 2, size=(n, c))
        e_map = rng.integers(1, 6, size=(n, c))
        mask = np.abs(a_map - rng.random((n, dim)) ** e_map) < 0.5

        trial = B + mask * (temp + noise - B)
        trial = _mde_border(trial, low, up, rng)

        fit_trial = np.asarray(obj(trial), dtype=float)
        improved = fit_trial < fitB
        fitB[improved] = fit_trial[improved]
        B[improved] = trial[improved]
        A[j0] = B
        fitA[j0] = fitB

        best_i = int(np.argmin(fitA))
        best = A[best_i].copy()
        if fitA[best_i] < best_val:
            best_val, best_sol = float(fitA[best_i]), best.copy()

        noise = B * 10.0 ** rng.integers(-12, -8, size=(n, dim)) * (rng.random((n, 1)) - 0.5)

    return best_val, best_sol


# --------------------------------------------------------------------------- CSA
def _levy(alpha: np.ndarray, beta: np.ndarray, rng) -> np.ndarray:
    z = rng.random() + 1.0
    w = rng.gamma(alpha, float(rng.integers(2, 6)))
    return (beta * z / w ** (1.0 / alpha)).reshape(-1, 1)


def _csa_border(px: np.ndarray, low: np.ndarray, up: np.ndarray, rng) -> np.ndarray:
    px = px.copy()
    below, above = px < low, px > up
    if below.any():
        cand = low + rng.random(px.shape) ** rng.integers(1, 6, size=px.shape) * (up - low)
        px = np.where(below, cand, px)
    if above.any():
        cand = up + rng.random(px.shape) ** rng.integers(1, 6, size=px.shape) * (low - up)
        px = np.where(above, cand, px)
    return px


def csa_optimize(obj, dim, lb, ub, pop_size, seed, max_cycles=10**9, stop=None):
    """Colony-Based Search Algorithm (Civicioglu & Besdok 2024). Minimiser."""
    rng = np.random.default_rng(seed)
    n, T = pop_size, 2
    low = np.full(dim, lb, dtype=float)
    up = np.full(dim, ub, dtype=float)

    p0 = rng.random((T * n, dim)) * (up - low) + low
    fitp0 = np.asarray(obj(p0), dtype=float)
    moment = np.zeros((n, dim))
    initindex = np.arange(n)
    best_i = int(np.argmin(fitp0))
    best_val, best_sol = float(fitp0[best_i]), p0[best_i].copy()

    cycle = 0
    while cycle < max_cycles:
        if stop is not None and stop():
            break
        cycle += 1
        # clan selection: positional mismatch with the previous selection
        while True:
            index = rng.permutation(T * n)[:n]
            if not np.any(index == initindex):
                initindex = index
                break
        p = p0[index].copy()
        fitp = fitp0[index].copy()

        # direction scale: Cauchy-like ratio or sign-flipped Levy flight
        c = 1 if rng.random() < rng.random() else dim
        if rng.random() < rng.random():
            scale = (rng.random((n, c)) - 0.5) / (rng.random((n, c)) - 0.5)
        else:
            alpha = rng.integers(2, 6, size=n)
            beta = rng.integers(1, 11, size=n).astype(float) ** float(rng.choice([-1, 1]))
            scale = np.sign(rng.random((n, 1)) - 0.5) * _levy(alpha, beta, rng)

        # mutation control matrix
        m = np.zeros((n, dim))
        for j in range(n):
            ind = rng.permutation(dim)
            k = abs(rng.integers(0, 2) - rng.random() ** rng.integers(2, 11))
            m[j, ind[: int(np.ceil(k * dim))]] = 1.0

        # evolutionary direction (one of three interaction models)
        while True:
            v1, v2 = rng.permutation(n), rng.permutation(n)
            ident = np.arange(n)
            if not np.any(v1 == ident) and not np.any(v2 == v1) and not np.any(v2 == ident):
                break
        index0 = np.argsort(fitp)  # ascending: best first
        v = rng.integers(1, 4)
        if v == 1:
            dx = p[v2] - p[v1]
        elif v == 2:
            dx = p[v1] - p
        else:
            dx = p[index0[rng.integers(0, int(np.ceil(n / 5)))]] - p

        s = (rng.random((n, 1)) - 0.5) * rng.random((n, 1)) ** rng.integers(2, 11)
        px = p + scale * m * dx + s * moment
        px = _csa_border(px, low, up, rng)

        fitpx = np.asarray(obj(px), dtype=float)
        improved = fitpx < fitp
        p[improved] = px[improved]
        fitp[improved] = fitpx[improved]
        p0[index] = p
        fitp0[index] = fitp

        best_i = int(np.argmin(fitp0))
        if fitp0[best_i] < best_val:
            best_val, best_sol = float(fitp0[best_i]), p0[best_i].copy()

        moment = (np.abs(rng.integers(0, 2, size=(n, 1))) - m) * dx

    return best_val, best_sol


_PORTS = {"mde": mde_optimize, "csa": csa_optimize}


def solve_ported(
    instance: Instance,
    method: str,
    time_limit: float,
    seed: int,
    pop_size: int = 50,
    mapping: str = DEFAULT_MAPPING,
) -> MethodResult:
    """Run a ported optimiser through the shared decoder under a wall-clock budget."""
    if method not in _PORTS:
        raise ValueError(f"unknown ported method {method!r}; options: {sorted(_PORTS)}")
    recorder = RunRecorder()
    dim = vector_length(instance)

    def batch_obj(pop):
        out = np.empty(len(pop))
        for i, x in enumerate(pop):
            result = decode_vector(instance, x, mapping=mapping)
            mk = result.schedule.makespan
            recorder.record(mk, result.n_repairs, x)
            out[i] = mk
        return out

    start = recorder.start
    stop = lambda: (time.perf_counter() - start) >= time_limit  # noqa: E731

    crashed = False
    try:
        _PORTS[method](batch_obj, dim, 0.0, 1.0, pop_size, seed, max_cycles=10**9, stop=stop)
    except Exception:  # noqa: BLE001
        crashed = True
    wall = time.perf_counter() - start

    schedule = None
    if recorder.best_x is not None:
        schedule = decode_vector(instance, recorder.best_x, mapping=mapping).schedule
    feasible = schedule is not None

    return MethodResult(
        method=method,
        instance=instance.name,
        status="FEASIBLE" if feasible else "UNKNOWN",
        best_obj=recorder.best,
        best_bound=None,
        schedule=schedule,
        feasible_final=feasible,
        anytime=recorder.points(),
        time_to_first=recorder.time_to_first,
        time_to_best=recorder.time_to_best,
        wall_time=wall,
        n_decoder_calls=recorder.n_calls,
        n_repairs=recorder.n_repairs,
        crashed=crashed,
    )
