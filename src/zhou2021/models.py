"""Open-boundary spin chains used by Zhou 2021 and explicit extensions."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .conventions import COMPLEX_DTYPE, validate_num_qubits, validate_real

IDENTITY = np.eye(2, dtype=COMPLEX_DTYPE)
PAULI_X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=COMPLEX_DTYPE)
PAULI_Y = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=COMPLEX_DTYPE)
PAULI_Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=COMPLEX_DTYPE)
SPIN_X = PAULI_X / 2.0
SPIN_Y = PAULI_Y / 2.0
SPIN_Z = PAULI_Z / 2.0

for _operator in (IDENTITY, PAULI_X, PAULI_Y, PAULI_Z, SPIN_X, SPIN_Y, SPIN_Z):
    _operator.flags.writeable = False


def _kron(factors: list[NDArray[np.complex128]]) -> NDArray[np.complex128]:
    result = np.array([[1.0 + 0.0j]], dtype=COMPLEX_DTYPE)
    for factor in factors:
        result = np.kron(result, factor)
    return np.asarray(result, dtype=COMPLEX_DTYPE)


def single_site_operator(
    operator: ArrayLike,
    site: int,
    num_qubits: int,
) -> NDArray[np.complex128]:
    """Embed a one-qubit matrix on q0-MSB site ``site``."""

    num_qubits = validate_num_qubits(num_qubits)
    if isinstance(site, bool) or not isinstance(site, (int, np.integer)):
        raise TypeError("site must be an integer.")
    site = int(site)
    if not 0 <= site < num_qubits:
        raise ValueError(f"site={site} is outside N={num_qubits}.")
    local = np.asarray(operator, dtype=COMPLEX_DTYPE)
    if local.shape != (2, 2) or not np.all(np.isfinite(local)):
        raise ValueError("operator must be a finite 2x2 matrix.")
    factors = [IDENTITY] * num_qubits
    factors[site] = local
    return _kron(factors)


def nearest_neighbor_operator(
    left_operator: ArrayLike,
    right_operator: ArrayLike,
    left_site: int,
    num_qubits: int,
) -> NDArray[np.complex128]:
    """Embed a product operator at adjacent q0-MSB sites ``i,i+1``."""

    num_qubits = validate_num_qubits(num_qubits)
    if isinstance(left_site, bool) or not isinstance(left_site, (int, np.integer)):
        raise TypeError("left_site must be an integer.")
    left_site = int(left_site)
    if not 0 <= left_site < num_qubits - 1:
        raise ValueError(f"left_site={left_site} is invalid for N={num_qubits}.")
    left = np.asarray(left_operator, dtype=COMPLEX_DTYPE)
    right = np.asarray(right_operator, dtype=COMPLEX_DTYPE)
    if left.shape != (2, 2) or right.shape != (2, 2):
        raise ValueError("Both local operators must have shape (2, 2).")
    if not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
        raise ValueError("Local operators must be finite.")
    factors = [IDENTITY] * num_qubits
    factors[left_site] = left
    factors[left_site + 1] = right
    return _kron(factors)


def xxz_hamiltonian(
    num_qubits: int,
    delta: float,
    *,
    coupling: float = 1.0,
) -> NDArray[np.complex128]:
    r"""Return OBC ``J Σ(SxSx+SySy+delta SzSz)`` with ``S=sigma/2``."""

    num_qubits = validate_num_qubits(num_qubits)
    delta = validate_real(delta, name="delta")
    coupling = validate_real(coupling, name="coupling")
    matrix = np.zeros((1 << num_qubits, 1 << num_qubits), dtype=COMPLEX_DTYPE)
    for site in range(num_qubits - 1):
        matrix += coupling * nearest_neighbor_operator(SPIN_X, SPIN_X, site, num_qubits)
        matrix += coupling * nearest_neighbor_operator(SPIN_Y, SPIN_Y, site, num_qubits)
        matrix += coupling * delta * nearest_neighbor_operator(SPIN_Z, SPIN_Z, site, num_qubits)
    return matrix


def heisenberg_hamiltonian(
    num_qubits: int,
    *,
    coupling: float = 1.0,
) -> NDArray[np.complex128]:
    """Return Zhou Eq. (4), the OBC isotropic ``S=sigma/2`` chain."""

    return xxz_hamiltonian(num_qubits, 1.0, coupling=coupling)


def xy_hamiltonian(
    num_qubits: int,
    *,
    coupling: float = 1.0,
) -> NDArray[np.complex128]:
    """Return Zhou Eq. (5), the OBC XY chain in the ``S=sigma/2`` convention."""

    return xxz_hamiltonian(num_qubits, 0.0, coupling=coupling)


def tfim_tndiff_hamiltonian(
    num_qubits: int,
    *,
    coupling_zz: float = 1.0,
    field_x: float = 1.0,
) -> NDArray[np.complex128]:
    r"""Return the explicit TN-Diff-compatible OBC Pauli TFIM extension.

    ``H=-J Σ Z_i Z_(i+1)-h_x Σ X_i`` is not one of the two Zhou-paper
    benchmark Hamiltonians.  It is separate so no factor-of-four spin versus
    Pauli convention can be hidden when comparing later baselines.
    """

    num_qubits = validate_num_qubits(num_qubits)
    coupling_zz = validate_real(coupling_zz, name="coupling_zz")
    field_x = validate_real(field_x, name="field_x")
    matrix = np.zeros((1 << num_qubits, 1 << num_qubits), dtype=COMPLEX_DTYPE)
    for site in range(num_qubits - 1):
        matrix -= coupling_zz * nearest_neighbor_operator(PAULI_Z, PAULI_Z, site, num_qubits)
    for site in range(num_qubits):
        matrix -= field_x * single_site_operator(PAULI_X, site, num_qubits)
    return matrix


def validate_dense_hamiltonian(
    hamiltonian: ArrayLike,
    *,
    hermiticity_atol: float = 1e-12,
) -> NDArray[np.complex128]:
    """Validate one finite Hermitian power-of-two dense Hamiltonian."""

    hermiticity_atol = validate_real(
        hermiticity_atol,
        name="hermiticity_atol",
    )
    if hermiticity_atol < 0.0:
        raise ValueError("hermiticity_atol must be nonnegative.")
    matrix = np.asarray(hamiltonian, dtype=COMPLEX_DTYPE)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"Hamiltonian must be square, got {matrix.shape}.")
    dimension = int(matrix.shape[0])
    if dimension < 2 or dimension & (dimension - 1):
        raise ValueError("Hamiltonian dimension must be a qubit power of two >= 2.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("Hamiltonian contains non-finite entries.")
    residual = float(np.linalg.norm(matrix - matrix.conj().T, ord=np.inf))
    if residual > hermiticity_atol:
        raise ValueError(f"Hamiltonian is not Hermitian; residual={residual:.3e}.")
    return np.array(matrix, dtype=COMPLEX_DTYPE, copy=True)


__all__ = [
    "IDENTITY",
    "PAULI_X",
    "PAULI_Y",
    "PAULI_Z",
    "SPIN_X",
    "SPIN_Y",
    "SPIN_Z",
    "heisenberg_hamiltonian",
    "nearest_neighbor_operator",
    "single_site_operator",
    "tfim_tndiff_hamiltonian",
    "validate_dense_hamiltonian",
    "xxz_hamiltonian",
    "xy_hamiltonian",
]
