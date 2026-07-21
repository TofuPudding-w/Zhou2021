"""Numerical conventions shared by the Zhou 2021 ADQC implementation."""

from __future__ import annotations

import math
import numbers

import numpy as np
from numpy.typing import ArrayLike, NDArray

COMPLEX_DTYPE = np.dtype(np.complex128)
REAL_DTYPE = np.dtype(np.float64)
QUBIT_ORDER = "q0-most-significant"
TWO_QUBIT_BASIS = ("00", "01", "10", "11")


def validate_num_qubits(value: int, *, name: str = "num_qubits") -> int:
    """Return one strictly positive integer qubit count."""

    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise TypeError(f"{name} must be an integer.")
    result = int(value)
    if result < 1:
        raise ValueError(f"{name} must be positive, got {result}.")
    return result


def validate_real(value: float, *, name: str, positive: bool = False) -> float:
    """Validate one finite real scalar, optionally requiring strict positivity."""

    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f"{name} must be a real scalar.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    if positive and result <= 0.0:
        raise ValueError(f"{name} must be positive.")
    return result


def infer_num_qubits(statevector: ArrayLike) -> int:
    """Infer the q0-MSB register width from a nonempty power-of-two vector."""

    vector = np.asarray(statevector).reshape(-1)
    if vector.size < 2 or vector.size & (vector.size - 1):
        raise ValueError("Statevector length must be a qubit power of two >= 2.")
    return vector.size.bit_length() - 1


def normalized_statevector(
    statevector: ArrayLike,
    *,
    num_qubits: int | None = None,
    normalization_atol: float = 2e-11,
) -> NDArray[np.complex128]:
    """Validate a finite normalized q0-MSB statevector without phase fixing."""

    if num_qubits is None:
        num_qubits = infer_num_qubits(statevector)
    else:
        num_qubits = validate_num_qubits(num_qubits)
    normalization_atol = validate_real(
        normalization_atol,
        name="normalization_atol",
        positive=True,
    )
    vector = np.asarray(statevector, dtype=COMPLEX_DTYPE).reshape(-1)
    if vector.size != 1 << num_qubits:
        raise ValueError(
            f"Statevector length {vector.size} disagrees with N={num_qubits}."
        )
    if not np.all(np.isfinite(vector)):
        raise ValueError("Statevector contains non-finite entries.")
    norm = float(np.linalg.norm(vector))
    if not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=normalization_atol):
        raise ValueError(f"Statevector must be normalized; norm={norm:.17g}.")
    return np.array(vector / norm, dtype=COMPLEX_DTYPE, copy=True)


def phase_aligned_state(reference: ArrayLike, candidate: ArrayLike) -> NDArray[np.complex128]:
    """Return candidate after removing its overlap-defined global phase."""

    reference_vector = normalized_statevector(reference)
    candidate_vector = normalized_statevector(
        candidate,
        num_qubits=infer_num_qubits(reference_vector),
    )
    overlap = complex(np.vdot(reference_vector, candidate_vector))
    if abs(overlap) == 0.0:
        return candidate_vector
    return candidate_vector * np.exp(-1j * np.angle(overlap))


__all__ = [
    "COMPLEX_DTYPE",
    "QUBIT_ORDER",
    "REAL_DTYPE",
    "TWO_QUBIT_BASIS",
    "infer_num_qubits",
    "normalized_statevector",
    "phase_aligned_state",
    "validate_num_qubits",
    "validate_real",
]
