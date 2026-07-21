"""State-preparation metrics, including Zhou et al. Eq. (1)."""

from __future__ import annotations

import math
import numbers

import numpy as np
from numpy.typing import ArrayLike

from .conventions import infer_num_qubits, normalized_statevector, validate_num_qubits
from .models import validate_dense_hamiltonian


def overlap_amplitude(reference: ArrayLike, candidate: ArrayLike) -> float:
    """Return ``|<reference|candidate>|`` for normalized statevectors."""

    reference_vector = normalized_statevector(reference)
    candidate_vector = normalized_statevector(
        candidate,
        num_qubits=infer_num_qubits(reference_vector),
    )
    return float(abs(np.vdot(reference_vector, candidate_vector)))


def squared_state_fidelity(reference: ArrayLike, candidate: ArrayLike) -> float:
    """Return the global squared state fidelity without physical clipping."""

    amplitude = overlap_amplitude(reference, candidate)
    fidelity = amplitude * amplitude
    if fidelity > 1.0 + 2e-10:
        raise ArithmeticError(f"Normalized state fidelity exceeds one: {fidelity:.17g}.")
    return fidelity


def negative_log_fidelity_per_site_from_squared_fidelity(
    squared_fidelity: float,
    *,
    num_qubits: int,
) -> float:
    r"""Return Zhou Eq. (1) from a physical squared-state fidelity.

    This scalar form keeps dense and MPS reporting on exactly the same
    contract.  In particular, it does not apply the differentiable floor used
    by the optimizer: a zero physical overlap is represented by ``+inf``.
    """

    num_qubits = validate_num_qubits(num_qubits)
    if isinstance(squared_fidelity, bool) or not isinstance(
        squared_fidelity, numbers.Real
    ):
        raise TypeError("squared_fidelity must be a real scalar.")
    squared_fidelity = float(squared_fidelity)
    if not math.isfinite(squared_fidelity):
        raise ValueError("squared_fidelity must be finite.")
    if squared_fidelity < 0.0 or squared_fidelity > 1.0 + 2e-10:
        raise ValueError("squared_fidelity must lie in [0, 1] within tolerance.")
    if squared_fidelity == 0.0:
        return float("inf")
    return -0.5 * math.log(squared_fidelity) / num_qubits


def negative_log_fidelity_per_site(
    reference: ArrayLike,
    candidate: ArrayLike,
    *,
    num_qubits: int | None = None,
) -> float:
    r"""Return Zhou Eq. (1), ``-log(|<reference|candidate>|)/N``.

    A zero overlap is represented by positive infinity rather than a hidden
    numerical floor.  Training uses an explicitly recorded differentiable
    floor in :mod:`zhou2021.adqc`; that engineering detail is not silently
    folded into this physics metric.
    """

    inferred_num_qubits = infer_num_qubits(reference)
    if num_qubits is None:
        num_qubits = inferred_num_qubits
    else:
        num_qubits = validate_num_qubits(num_qubits)
        if num_qubits != inferred_num_qubits:
            raise ValueError(
                "num_qubits disagrees with the reference state dimension: "
                f"N={num_qubits}, inferred N={inferred_num_qubits}."
            )
    amplitude = overlap_amplitude(reference, candidate)
    return negative_log_fidelity_per_site_from_squared_fidelity(
        amplitude * amplitude,
        num_qubits=num_qubits,
    )


def energy_expectation(state: ArrayLike, hamiltonian: ArrayLike) -> float:
    """Return ``<state|H|state>`` and reject an unphysical imaginary part."""

    vector = normalized_statevector(state)
    matrix = validate_dense_hamiltonian(hamiltonian)
    if matrix.shape != (vector.size, vector.size):
        raise ValueError(
            f"Hamiltonian shape {matrix.shape} disagrees with state dimension {vector.size}."
        )
    value = complex(np.vdot(vector, matrix @ vector))
    if abs(value.imag) > 2e-10:
        raise ArithmeticError(f"Energy has an imaginary residual {value.imag:.3e}.")
    return float(value.real)


def energy_error_per_site(
    state: ArrayLike,
    hamiltonian: ArrayLike,
    ground_energy: float,
    *,
    num_qubits: int | None = None,
) -> float:
    """Return signed ``(<H>-E0)/N`` under the recorded Hamiltonian convention."""

    inferred_num_qubits = infer_num_qubits(state)
    if num_qubits is None:
        num_qubits = inferred_num_qubits
    else:
        num_qubits = validate_num_qubits(num_qubits)
        if num_qubits != inferred_num_qubits:
            raise ValueError(
                "num_qubits disagrees with the state dimension: "
                f"N={num_qubits}, inferred N={inferred_num_qubits}."
            )
    if isinstance(ground_energy, bool) or not np.isscalar(ground_energy):
        raise TypeError("ground_energy must be a real scalar.")
    ground_energy = float(ground_energy)
    if not math.isfinite(ground_energy):
        raise ValueError("ground_energy must be finite.")
    return (energy_expectation(state, hamiltonian) - ground_energy) / num_qubits


__all__ = [
    "energy_error_per_site",
    "energy_expectation",
    "negative_log_fidelity_per_site",
    "negative_log_fidelity_per_site_from_squared_fidelity",
    "overlap_amplitude",
    "squared_state_fidelity",
]
