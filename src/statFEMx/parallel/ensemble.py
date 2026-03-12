from __future__ import annotations

from collections.abc import Callable
import numpy as np
from mpi4py import MPI


def evaluate_samples_distributed(
    sample_values: np.ndarray,
    evaluator: Callable[[float], np.ndarray],
    *,
    comm: MPI.Intracomm = MPI.COMM_WORLD,
) -> np.ndarray | None:
    samples = np.asarray(sample_values, dtype=float)
    rank = comm.rank
    size = comm.size

    local_indices = np.arange(rank, samples.size, size, dtype=np.int32)
    local_results = []
    for idx in local_indices:
        local_results.append(np.asarray(evaluator(float(samples[idx])), dtype=float))

    gathered = comm.gather((local_indices, local_results), root=0)

    if rank != 0:
        return None

    if not gathered or not gathered[0][1]:
        raise RuntimeError("No sample evaluations were produced on rank 0.")

    qoi_size = int(np.asarray(gathered[0][1][0]).size)
    full = np.empty((samples.size, qoi_size), dtype=float)
    for indices, values in gathered:
        for idx, val in zip(indices, values, strict=False):
            full[int(idx), :] = np.asarray(val, dtype=float).reshape(-1)
    return full
