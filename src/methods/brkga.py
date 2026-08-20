"""Biased random-key genetic algorithm through the shared decoder.

The study's representation is a vector of random keys, and the biased random-key genetic
algorithm is the evolutionary method built specifically for that representation, so its
absence from a random-key comparison would leave the closest competitor untested.

The implementation follows Goncalves and Resende (2011). The population is partitioned each
generation into an elite set of size ``p_e * p`` and the remainder. The elite set is copied
unchanged into the next generation. A further ``p_m * p`` individuals are drawn uniformly at
random, which is the mutation mechanism. The remaining individuals come from parameterised
uniform crossover between one parent drawn from the elite set and one drawn from the
non-elite set, where each allele is inherited from the elite parent with probability
``rho_e``. Selection pressure therefore comes entirely from the biased parent choice and the
elite copy, and no allele-level mutation operator is applied.

Parameter values are the ones recommended in the source: an elite fraction of 0.20 inside
the recommended interval [0.10, 0.25], a mutant fraction of 0.15 inside [0.10, 0.30], and an
elite inheritance probability of 0.70 inside [0.5, 0.8]. The population size follows the
study-wide setting so that every population-based method carries the same value.
"""
from __future__ import annotations

import time

import numpy as np

from src.core.decoder import DEFAULT_MAPPING, decode_vector, vector_length
from src.core.instance import Instance
from src.methods.base import MethodResult, RunRecorder

ELITE_FRACTION = 0.20
MUTANT_FRACTION = 0.15
RHO_ELITE = 0.70


def solve_brkga(
    instance: Instance,
    time_limit: float,
    seed: int,
    pop_size: int = 50,
    mapping: str = DEFAULT_MAPPING,
) -> MethodResult:
    rng = np.random.default_rng(seed)
    dim = vector_length(instance)
    recorder = RunRecorder()

    n_elite = max(1, int(round(ELITE_FRACTION * pop_size)))
    n_mutant = max(1, int(round(MUTANT_FRACTION * pop_size)))
    n_cross = max(0, pop_size - n_elite - n_mutant)

    def evaluate(population: np.ndarray) -> np.ndarray:
        out = np.empty(len(population))
        for i, x in enumerate(population):
            result = decode_vector(instance, x, mapping=mapping)
            makespan = float(result.schedule.makespan)
            recorder.record(makespan, result.n_repairs, x)
            out[i] = makespan
        return out

    crashed = False
    start = recorder.start
    try:
        pop = rng.random((pop_size, dim))
        fit = evaluate(pop)
        while time.perf_counter() - start < time_limit:
            order = np.argsort(fit, kind="stable")
            elite, rest = pop[order[:n_elite]], pop[order[n_elite:]]
            elite_fit = fit[order[:n_elite]]

            offspring = rng.random((n_mutant, dim))
            if n_cross:
                ea = elite[rng.integers(0, n_elite, size=n_cross)]
                eb = rest[rng.integers(0, max(1, len(rest)), size=n_cross)]
                take_elite = rng.random((n_cross, dim)) < RHO_ELITE
                offspring = np.vstack([offspring, np.where(take_elite, ea, eb)])

            child_fit = evaluate(offspring)
            pop = np.vstack([elite, offspring])
            fit = np.concatenate([elite_fit, child_fit])
    except Exception:  # noqa: BLE001 - a crashing optimiser is logged, not fatal to the run
        crashed = True
    wall = time.perf_counter() - start

    schedule = None
    if recorder.best_x is not None:
        schedule = decode_vector(instance, recorder.best_x, mapping=mapping).schedule

    return MethodResult(
        method="brkga",
        instance=instance.name,
        status="FEASIBLE" if schedule is not None else "UNKNOWN",
        best_obj=recorder.best,
        best_bound=None,
        schedule=schedule,
        feasible_final=schedule is not None,
        anytime=recorder.points(),
        time_to_first=recorder.time_to_first,
        time_to_best=recorder.time_to_best,
        wall_time=wall,
        n_decoder_calls=recorder.n_calls,
        n_repairs=recorder.n_repairs,
        crashed=crashed,
    )
