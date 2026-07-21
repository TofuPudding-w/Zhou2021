"""Deterministic small-system exact ground states for ADQC target preparation."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .models import validate_dense_hamiltonian


@dataclass(frozen=True, slots=True)
class GroundStateResult:
    """Phase-fixed representative and diagnostics of a dense ground space."""

    energy: float
    state: NDArray[np.complex128]
    residual: float
    gap: float
    degeneracy: int
    selection_basis_index: int


def _phase_fix(state: NDArray[np.complex128]) -> NDArray[np.complex128]:
    result = np.array(state, dtype=np.complex128, copy=True)
    pivot = int(np.argmax(np.abs(result)))
    if abs(result[pivot]) > 0.0:
        result *= np.exp(-1j * np.angle(result[pivot]))
    return result


def exact_ground_state(
    hamiltonian: ArrayLike,
    *,
    degeneracy_atol: float = 1e-10,
    degeneracy_rtol: float = 1e-12,
) -> GroundStateResult:
    """Return a deterministic representative of the lowest eigenspace."""

    if not math.isfinite(degeneracy_atol) or degeneracy_atol < 0.0:
        raise ValueError("degeneracy_atol must be finite and nonnegative.")
    if not math.isfinite(degeneracy_rtol) or degeneracy_rtol < 0.0:
        raise ValueError("degeneracy_rtol must be finite and nonnegative.")
    matrix = validate_dense_hamiltonian(hamiltonian)
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    energy = float(eigenvalues[0])
    threshold = degeneracy_atol + degeneracy_rtol * np.maximum(
        np.abs(eigenvalues),
        abs(energy),
    )
    degeneracy = int(np.count_nonzero(np.abs(eigenvalues - energy) <= threshold))
    if degeneracy < eigenvalues.size:
        gap = float(eigenvalues[degeneracy] - energy)
    else:
        gap = float("inf")
    basis = eigenvectors[:, :degeneracy]
    weights = np.sum(np.abs(basis) ** 2, axis=1)
    support = np.flatnonzero(weights >= 1e-12 * float(np.max(weights)))
    if support.size == 0:  # pragma: no cover - eigh orthonormality invariant
        raise ArithmeticError("Ground-space projector has no stable support.")
    selection_basis_index = int(support[0])
    state = basis @ basis[selection_basis_index, :].conj()
    state /= np.linalg.norm(state)
    state = _phase_fix(state)
    state /= np.linalg.norm(state)
    residual = float(np.linalg.norm(matrix @ state - energy * state))
    return GroundStateResult(
        energy=energy,
        state=state,
        residual=residual,
        gap=gap,
        degeneracy=degeneracy,
        selection_basis_index=selection_basis_index,
    )


__all__ = ["GroundStateResult", "exact_ground_state"]
