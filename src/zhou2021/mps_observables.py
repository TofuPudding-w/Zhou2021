"""MPS-only local observables and spin-chain energies for Zhou 2021.

The ADQC paper uses open-boundary MPS targets for its large-system results.
This module evaluates local observables directly in that representation: it
never calls :meth:`zhou2021.mps.OpenBoundaryMPS.to_dense` and its largest
intermediate objects are virtual-bond transfer matrices.  The site ordering is
the project-wide q0-most-significant convention, i.e. the first MPS core is
q0 and an adjacent pair ``left_site, left_site + 1`` follows the chain order.

Only finite Hermitian one-qubit factors are accepted.  This is deliberately
stricter than a general complex transition-amplitude API: all public outputs
here are physical real expectation values, so accepting a non-Hermitian local
operator would silently invalidate the associated energy check.
"""

from __future__ import annotations

import math
import numbers
from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .conventions import COMPLEX_DTYPE, validate_num_qubits, validate_real
from .models import PAULI_X, PAULI_Z, SPIN_X, SPIN_Y, SPIN_Z
from .mps import OpenBoundaryMPS

Tensor = NDArray[np.complex128]

MPS_LOCAL_ENERGY_SCHEMA_VERSION = "zhou2021-mps-local-energy-v1"


def _validate_site(site: int, *, num_qubits: int, name: str = "site") -> int:
    if isinstance(site, bool) or not isinstance(site, numbers.Integral):
        raise TypeError(f"{name} must be an integer.")
    result = int(site)
    if not 0 <= result < num_qubits:
        raise ValueError(f"{name}={result} is outside N={num_qubits}.")
    return result


def _validate_left_site(left_site: int, *, num_qubits: int) -> int:
    left_site = _validate_site(left_site, num_qubits=num_qubits, name="left_site")
    if left_site >= num_qubits - 1:
        raise ValueError(f"left_site={left_site} is invalid for N={num_qubits}.")
    return left_site


def _validate_hermitian_one_qubit_operator(
    operator: ArrayLike,
    *,
    name: str,
    hermiticity_atol: float,
) -> Tensor:
    """Copy one finite Hermitian 2x2 operator after a strict residual check."""

    matrix = np.asarray(operator, dtype=COMPLEX_DTYPE)
    if matrix.shape != (2, 2):
        raise ValueError(f"{name} must have shape (2, 2), got {matrix.shape}.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} contains non-finite entries.")
    residual = float(np.linalg.norm(matrix - matrix.conj().T, ord=np.inf))
    if residual > hermiticity_atol:
        raise ValueError(
            f"{name} must be Hermitian; infinity-norm residual={residual:.3e}."
        )
    return np.array(matrix, dtype=COMPLEX_DTYPE, copy=True)


def _require_mps(mps: OpenBoundaryMPS) -> OpenBoundaryMPS:
    if not isinstance(mps, OpenBoundaryMPS):
        raise TypeError("mps must be an OpenBoundaryMPS.")
    return mps


def _finite_transfer(value: Tensor, *, context: str) -> Tensor:
    if not np.all(np.isfinite(value)):
        raise ArithmeticError(f"Non-finite MPS transfer matrix encountered at {context}.")
    return value


def _propagate_identity(environment: Tensor, tensor: Tensor, *, context: str) -> Tensor:
    return _finite_transfer(
        np.einsum("ab,air,bis->rs", environment, tensor.conj(), tensor, optimize=True),
        context=context,
    )


def _propagate_operator(
    environment: Tensor,
    tensor: Tensor,
    operator: Tensor,
    *,
    context: str,
) -> Tensor:
    return _finite_transfer(
        np.einsum(
            "ab,air,ij,bjs->rs",
            environment,
            tensor.conj(),
            operator,
            tensor,
            optimize=True,
        ),
        context=context,
    )


def _real_expectation(value: complex, *, label: str, imaginary_atol: float) -> float:
    value = complex(value)
    if not math.isfinite(value.real) or not math.isfinite(value.imag):
        raise ArithmeticError(f"{label} is non-finite.")
    scale = max(1.0, abs(value.real))
    if abs(value.imag) > imaginary_atol * scale:
        raise ArithmeticError(f"{label} has imaginary residual {value.imag:.3e}.")
    return float(value.real)


@dataclass(frozen=True, slots=True)
class _MPSLocalEnvironment:
    """Identity-transfer prefix/suffix cache for one immutable OBC MPS.

    ``left[site]`` is the contraction before ``site`` and ``right[site]`` is
    the contraction starting at ``site``.  Both are indexed in q0-to-q(N-1)
    order.  Building the cache is O(N chi^3), after which all single-site and
    nearest-neighbour expectations avoid a dense statevector.
    """

    mps: OpenBoundaryMPS
    left: tuple[Tensor, ...]
    right: tuple[Tensor, ...]
    norm_squared: float


def _build_environment(mps: OpenBoundaryMPS, *, imaginary_atol: float) -> _MPSLocalEnvironment:
    mps = _require_mps(mps)
    left: list[Tensor] = [np.ones((1, 1), dtype=COMPLEX_DTYPE)]
    for site, tensor in enumerate(mps.tensors):
        left.append(_propagate_identity(left[-1], tensor, context=f"left site {site}"))

    right: list[Tensor | None] = [None] * (mps.num_qubits + 1)
    right[mps.num_qubits] = np.ones((1, 1), dtype=COMPLEX_DTYPE)
    for site in range(mps.num_qubits - 1, -1, -1):
        suffix = right[site + 1]
        if suffix is None:  # pragma: no cover - backward sweep invariant
            raise RuntimeError("Missing MPS suffix transfer matrix.")
        right[site] = _finite_transfer(
            np.einsum(
                "air,bis,rs->ab",
                mps.tensors[site].conj(),
                mps.tensors[site],
                suffix,
                optimize=True,
            ),
            context=f"right site {site}",
        )

    terminal = complex(left[-1][0, 0])
    norm_squared = _real_expectation(
        terminal,
        label="MPS squared norm",
        imaginary_atol=imaginary_atol,
    )
    if norm_squared <= 0.0:
        raise ValueError("Local expectations are undefined for a zero-norm MPS.")
    initial = complex(right[0][0, 0]) if right[0] is not None else complex("nan")
    suffix_norm_squared = _real_expectation(
        initial,
        label="MPS reverse squared norm",
        imaginary_atol=imaginary_atol,
    )
    if not math.isclose(
        norm_squared,
        suffix_norm_squared,
        rel_tol=2e-11,
        abs_tol=2e-11,
    ):
        raise ArithmeticError(
            "Forward and reverse MPS norm contractions disagree: "
            f"{norm_squared:.17g} versus {suffix_norm_squared:.17g}."
        )
    return _MPSLocalEnvironment(
        mps=mps,
        left=tuple(left),
        right=tuple(
            suffix if suffix is not None else np.empty((0, 0), dtype=COMPLEX_DTYPE)
            for suffix in right
        ),
        norm_squared=norm_squared,
    )


def _expectation_from_cache(
    cache: _MPSLocalEnvironment,
    *,
    left_site: int,
    left_operator: Tensor,
    right_operator: Tensor | None,
    imaginary_atol: float,
) -> float:
    """Contract one local product term and divide by the MPS norm squared."""

    environment = _propagate_operator(
        cache.left[left_site],
        cache.mps.tensors[left_site],
        left_operator,
        context=f"operator site {left_site}",
    )
    suffix_site = left_site + 1
    if right_operator is not None:
        environment = _propagate_operator(
            environment,
            cache.mps.tensors[suffix_site],
            right_operator,
            context=f"operator site {suffix_site}",
        )
        suffix_site += 1
    raw_value = np.einsum(
        "ab,ab->",
        environment,
        cache.right[suffix_site],
        optimize=True,
    )
    return _real_expectation(
        complex(raw_value) / cache.norm_squared,
        label="Hermitian MPS expectation",
        imaginary_atol=imaginary_atol,
    )


def mps_single_site_expectation(
    mps: OpenBoundaryMPS,
    operator: ArrayLike,
    site: int,
    *,
    hermiticity_atol: float = 1e-12,
    imaginary_atol: float = 2e-10,
) -> float:
    """Return normalized ``<O_site>`` from an OBC MPS without densifying it.

    The local factor must be finite, 2x2 and Hermitian.  Its q0-MSB ``site``
    identifies the same site as the corresponding core of ``mps``.
    """

    mps = _require_mps(mps)
    hermiticity_atol = validate_real(hermiticity_atol, name="hermiticity_atol")
    imaginary_atol = validate_real(imaginary_atol, name="imaginary_atol")
    if hermiticity_atol < 0.0 or imaginary_atol < 0.0:
        raise ValueError("Hermiticity and imaginary tolerances must be nonnegative.")
    site = _validate_site(site, num_qubits=mps.num_qubits)
    checked_operator = _validate_hermitian_one_qubit_operator(
        operator,
        name="operator",
        hermiticity_atol=hermiticity_atol,
    )
    cache = _build_environment(mps, imaginary_atol=imaginary_atol)
    return _expectation_from_cache(
        cache,
        left_site=site,
        left_operator=checked_operator,
        right_operator=None,
        imaginary_atol=imaginary_atol,
    )


def mps_adjacent_product_expectation(
    mps: OpenBoundaryMPS,
    left_operator: ArrayLike,
    right_operator: ArrayLike,
    left_site: int,
    *,
    hermiticity_atol: float = 1e-12,
    imaginary_atol: float = 2e-10,
) -> float:
    """Return normalized ``<A_i B_(i+1)>`` without forming a dense operator.

    Each factor is separately required to be Hermitian.  Because they act on
    distinct sites, their product is then Hermitian and the result must be a
    real scalar up to ``imaginary_atol``.
    """

    mps = _require_mps(mps)
    hermiticity_atol = validate_real(hermiticity_atol, name="hermiticity_atol")
    imaginary_atol = validate_real(imaginary_atol, name="imaginary_atol")
    if hermiticity_atol < 0.0 or imaginary_atol < 0.0:
        raise ValueError("Hermiticity and imaginary tolerances must be nonnegative.")
    left_site = _validate_left_site(left_site, num_qubits=mps.num_qubits)
    checked_left = _validate_hermitian_one_qubit_operator(
        left_operator,
        name="left_operator",
        hermiticity_atol=hermiticity_atol,
    )
    checked_right = _validate_hermitian_one_qubit_operator(
        right_operator,
        name="right_operator",
        hermiticity_atol=hermiticity_atol,
    )
    cache = _build_environment(mps, imaginary_atol=imaginary_atol)
    return _expectation_from_cache(
        cache,
        left_site=left_site,
        left_operator=checked_left,
        right_operator=checked_right,
        imaginary_atol=imaginary_atol,
    )


@dataclass(frozen=True, slots=True)
class XXZParameters:
    """Explicit OBC ``J(SxSx + SySy + delta SzSz)`` parameter record.

    The spin convention is fixed to ``S=Pauli/2``.  It covers Zhou's
    Heisenberg (``delta=1``) and XY (``delta=0``) paper Hamiltonians as special
    cases, while preserving a distinct generic XXZ extension for comparison.
    """

    num_qubits: int
    delta: float
    coupling: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "num_qubits", validate_num_qubits(self.num_qubits))
        object.__setattr__(self, "delta", validate_real(self.delta, name="delta"))
        object.__setattr__(self, "coupling", validate_real(self.coupling, name="coupling"))

    def to_record(self) -> dict[str, str | int | float]:
        """Return JSON-ready model metadata for a result artifact."""

        return {
            "schema_version": MPS_LOCAL_ENERGY_SCHEMA_VERSION,
            "model": "xxz",
            "boundary": "open",
            "site_order": "q0-most-significant",
            "spin_convention": "S=Pauli/2",
            "num_qubits": self.num_qubits,
            "coupling": self.coupling,
            "delta": self.delta,
        }


@dataclass(frozen=True, slots=True)
class TNDiffTFIMParameters:
    """Explicit OBC Pauli-TFIM record, ``-J sum ZZ - hx sum X``.

    This is intentionally separate from the Zhou-paper spin-chain convention.
    It is the stated TN-Diff extension and therefore records its Pauli rather
    than spin-half normalization directly in its JSON-ready schema.
    """

    num_qubits: int
    coupling_zz: float = 1.0
    field_x: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "num_qubits", validate_num_qubits(self.num_qubits))
        object.__setattr__(
            self,
            "coupling_zz",
            validate_real(self.coupling_zz, name="coupling_zz"),
        )
        object.__setattr__(self, "field_x", validate_real(self.field_x, name="field_x"))

    def to_record(self) -> dict[str, str | int | float]:
        """Return JSON-ready model metadata for a result artifact."""

        return {
            "schema_version": MPS_LOCAL_ENERGY_SCHEMA_VERSION,
            "model": "tfim_tndiff",
            "boundary": "open",
            "site_order": "q0-most-significant",
            "operator_convention": "Pauli",
            "num_qubits": self.num_qubits,
            "coupling_zz": self.coupling_zz,
            "field_x": self.field_x,
        }


MPSModelParameters: TypeAlias = XXZParameters | TNDiffTFIMParameters


def _validate_model_for_mps(mps: OpenBoundaryMPS, model: MPSModelParameters) -> None:
    _require_mps(mps)
    if not isinstance(model, (XXZParameters, TNDiffTFIMParameters)):
        raise TypeError("model must be XXZParameters or TNDiffTFIMParameters.")
    if model.num_qubits != mps.num_qubits:
        raise ValueError(
            f"Model N={model.num_qubits} disagrees with MPS N={mps.num_qubits}."
        )


def _xxz_energy_from_cache(cache: _MPSLocalEnvironment, model: XXZParameters) -> float:
    energy = 0.0
    for left_site in range(model.num_qubits - 1):
        xx = _expectation_from_cache(
            cache,
            left_site=left_site,
            left_operator=SPIN_X,
            right_operator=SPIN_X,
            imaginary_atol=2e-10,
        )
        yy = _expectation_from_cache(
            cache,
            left_site=left_site,
            left_operator=SPIN_Y,
            right_operator=SPIN_Y,
            imaginary_atol=2e-10,
        )
        zz = _expectation_from_cache(
            cache,
            left_site=left_site,
            left_operator=SPIN_Z,
            right_operator=SPIN_Z,
            imaginary_atol=2e-10,
        )
        energy += model.coupling * (xx + yy + model.delta * zz)
    if not math.isfinite(energy):
        raise ArithmeticError("XXZ MPS energy is non-finite.")
    return float(energy)


def _tfim_energy_from_cache(cache: _MPSLocalEnvironment, model: TNDiffTFIMParameters) -> float:
    energy = 0.0
    for left_site in range(model.num_qubits - 1):
        energy -= model.coupling_zz * _expectation_from_cache(
            cache,
            left_site=left_site,
            left_operator=PAULI_Z,
            right_operator=PAULI_Z,
            imaginary_atol=2e-10,
        )
    for site in range(model.num_qubits):
        energy -= model.field_x * _expectation_from_cache(
            cache,
            left_site=site,
            left_operator=PAULI_X,
            right_operator=None,
            imaginary_atol=2e-10,
        )
    if not math.isfinite(energy):
        raise ArithmeticError("TN-Diff TFIM MPS energy is non-finite.")
    return float(energy)


def mps_model_energy(mps: OpenBoundaryMPS, model: MPSModelParameters) -> float:
    """Return the normalized OBC energy from local MPS contractions only."""

    _validate_model_for_mps(mps, model)
    cache = _build_environment(mps, imaginary_atol=2e-10)
    if isinstance(model, XXZParameters):
        return _xxz_energy_from_cache(cache, model)
    return _tfim_energy_from_cache(cache, model)


def xxz_mps_energy(
    mps: OpenBoundaryMPS,
    *,
    delta: float,
    coupling: float = 1.0,
) -> float:
    """Return the OBC spin-``1/2`` XXZ MPS energy."""

    mps = _require_mps(mps)
    return mps_model_energy(
        mps,
        XXZParameters(mps.num_qubits, delta=delta, coupling=coupling),
    )


def heisenberg_mps_energy(mps: OpenBoundaryMPS, *, coupling: float = 1.0) -> float:
    """Return Zhou Eq. (4), the ``delta=1`` spin-``1/2`` OBC energy."""

    return xxz_mps_energy(mps, delta=1.0, coupling=coupling)


def xy_mps_energy(mps: OpenBoundaryMPS, *, coupling: float = 1.0) -> float:
    """Return Zhou Eq. (5), the ``delta=0`` spin-``1/2`` OBC energy."""

    return xxz_mps_energy(mps, delta=0.0, coupling=coupling)


def tfim_tndiff_mps_energy(
    mps: OpenBoundaryMPS,
    *,
    coupling_zz: float = 1.0,
    field_x: float = 1.0,
) -> float:
    """Return the explicitly Pauli-normalized TN-Diff OBC TFIM MPS energy."""

    mps = _require_mps(mps)
    return mps_model_energy(
        mps,
        TNDiffTFIMParameters(
            mps.num_qubits,
            coupling_zz=coupling_zz,
            field_x=field_x,
        ),
    )


def mps_energy_error_per_site(
    mps: OpenBoundaryMPS,
    model: MPSModelParameters,
    ground_energy: float,
) -> float:
    """Return signed normalized-energy error ``(<H> - E0) / N``.

    ``ground_energy`` must be independently supplied (for example by dense ED
    in a small-system validation or an MPS/DMRG reference in a large-system
    production run).  The function intentionally does not materialise the
    Hamiltonian or a statevector to obtain it.
    """

    _validate_model_for_mps(mps, model)
    if isinstance(ground_energy, bool) or not isinstance(ground_energy, numbers.Real):
        raise TypeError("ground_energy must be a real scalar.")
    ground_energy = float(ground_energy)
    if not math.isfinite(ground_energy):
        raise ValueError("ground_energy must be finite.")
    return (mps_model_energy(mps, model) - ground_energy) / model.num_qubits


__all__ = [
    "MPS_LOCAL_ENERGY_SCHEMA_VERSION",
    "MPSModelParameters",
    "TNDiffTFIMParameters",
    "XXZParameters",
    "heisenberg_mps_energy",
    "mps_adjacent_product_expectation",
    "mps_energy_error_per_site",
    "mps_model_energy",
    "mps_single_site_expectation",
    "tfim_tndiff_mps_energy",
    "xxz_mps_energy",
    "xy_mps_energy",
]
