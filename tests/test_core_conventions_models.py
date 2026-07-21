"""Independent unit checks for conventions, spin Hamiltonians and ED targets."""

from __future__ import annotations

import math

import numpy as np
import pytest
from numpy.testing import assert_allclose

from zhou2021.conventions import (
    COMPLEX_DTYPE,
    infer_num_qubits,
    normalized_statevector,
    phase_aligned_state,
    validate_num_qubits,
    validate_real,
)
from zhou2021.ground_states import exact_ground_state
from zhou2021.metrics import (
    energy_error_per_site,
    energy_expectation,
    negative_log_fidelity_per_site,
    overlap_amplitude,
    squared_state_fidelity,
)
from zhou2021.models import (
    IDENTITY,
    PAULI_X,
    PAULI_Z,
    SPIN_X,
    SPIN_Y,
    SPIN_Z,
    heisenberg_hamiltonian,
    nearest_neighbor_operator,
    single_site_operator,
    tfim_tndiff_hamiltonian,
    validate_dense_hamiltonian,
    xxz_hamiltonian,
    xy_hamiltonian,
)


def _basis_state(index: int, num_qubits: int) -> np.ndarray:
    state = np.zeros(1 << num_qubits, dtype=np.complex128)
    state[index] = 1.0
    return state


@pytest.mark.parametrize("value", [1, np.int64(2)])
def test_validate_num_qubits_accepts_positive_integral_values(value: int) -> None:
    assert validate_num_qubits(value) == int(value)


@pytest.mark.parametrize(
    ("value", "exception"),
    [(True, TypeError), (1.0, TypeError), (0, ValueError), (-3, ValueError)],
)
def test_validate_num_qubits_rejects_invalid_values(
    value: object,
    exception: type[Exception],
) -> None:
    with pytest.raises(exception):
        validate_num_qubits(value)  # type: ignore[arg-type]


def test_validate_real_checks_type_finiteness_and_sign() -> None:
    assert validate_real(np.float64(0.25), name="x") == 0.25
    assert validate_real(0.25, name="x", positive=True) == 0.25
    for value, exception in [
        (True, TypeError),
        ("0.25", TypeError),
        (math.inf, ValueError),
        (math.nan, ValueError),
        (0.0, ValueError),
        (-0.25, ValueError),
    ]:
        with pytest.raises(exception):
            validate_real(value, name="x", positive=True)  # type: ignore[arg-type]


def test_statevector_normalization_inference_and_global_phase_alignment() -> None:
    reference = _basis_state(1, 2)
    candidate = np.exp(0.37j) * reference

    assert infer_num_qubits(reference.reshape(2, 2)) == 2
    normalized = normalized_statevector((1.0 + 1e-12) * reference)
    assert normalized.dtype == COMPLEX_DTYPE
    assert_allclose(normalized, reference, atol=2e-12)
    assert_allclose(phase_aligned_state(reference, candidate), reference, atol=2e-12)


@pytest.mark.parametrize(
    "state",
    [np.array([], dtype=np.complex128), np.ones(3), np.ones(8) / np.sqrt(2.0)],
)
def test_statevector_validation_rejects_invalid_dimensions_or_norms(state: np.ndarray) -> None:
    with pytest.raises(ValueError):
        normalized_statevector(state)


def test_metrics_are_global_phase_invariant_and_follow_zhou_equation_one() -> None:
    reference = _basis_state(0, 2)
    candidate = np.exp(-0.73j) * reference

    assert overlap_amplitude(reference, candidate) == pytest.approx(1.0, abs=2e-12)
    assert squared_state_fidelity(reference, candidate) == pytest.approx(1.0, abs=2e-12)
    assert negative_log_fidelity_per_site(reference, candidate) == pytest.approx(0.0, abs=2e-12)

    orthogonal = _basis_state(3, 2)
    assert squared_state_fidelity(reference, orthogonal) == pytest.approx(0.0, abs=2e-12)
    assert math.isinf(negative_log_fidelity_per_site(reference, orthogonal))

    half_overlap = (_basis_state(0, 2) + _basis_state(1, 2)) / np.sqrt(2.0)
    assert negative_log_fidelity_per_site(reference, half_overlap) == pytest.approx(
        math.log(2.0) / 4.0,
        abs=2e-12,
    )


def test_explicit_metric_num_qubits_must_match_the_state_dimension() -> None:
    reference = _basis_state(0, 2)
    candidate = (_basis_state(0, 2) + _basis_state(1, 2)) / np.sqrt(2.0)
    hamiltonian = np.diag([0.0, 1.0, 2.0, 3.0]).astype(np.complex128)

    assert negative_log_fidelity_per_site(
        reference,
        candidate,
        num_qubits=2,
    ) == pytest.approx(negative_log_fidelity_per_site(reference, candidate))
    assert energy_error_per_site(
        candidate,
        hamiltonian,
        0.0,
        num_qubits=2,
    ) == pytest.approx(energy_error_per_site(candidate, hamiltonian, 0.0))
    with pytest.raises(ValueError, match="num_qubits disagrees"):
        negative_log_fidelity_per_site(reference, candidate, num_qubits=3)
    with pytest.raises(ValueError, match="num_qubits disagrees"):
        energy_error_per_site(candidate, hamiltonian, 0.0, num_qubits=3)


def test_energy_metrics_use_normalized_states_and_reject_unphysical_imaginary_values() -> None:
    state = (_basis_state(0, 2) + _basis_state(3, 2)) / np.sqrt(2.0)
    hamiltonian = np.diag([1.0, 2.0, 3.0, 5.0]).astype(np.complex128)

    assert energy_expectation(state, hamiltonian) == pytest.approx(3.0, abs=2e-12)
    assert energy_error_per_site(state, hamiltonian, 1.0) == pytest.approx(1.0, abs=2e-12)
    with pytest.raises(ValueError, match="not Hermitian"):
        energy_expectation(_basis_state(0, 1), np.diag([1.0j, 0.0]))
    with pytest.raises(ValueError, match="not Hermitian"):
        energy_expectation(
            _basis_state(0, 1),
            np.array([[0.0, 1.0], [0.0, 0.0]], dtype=np.complex128),
        )
    with pytest.raises(ValueError, match="disagrees"):
        energy_expectation(state, np.eye(2, dtype=np.complex128))


def test_q0_msb_single_and_nearest_neighbor_embeddings_match_independent_kron() -> None:
    assert_allclose(
        single_site_operator(PAULI_X, 0, 3),
        np.kron(PAULI_X, np.kron(IDENTITY, IDENTITY)),
        atol=0.0,
    )
    assert_allclose(
        single_site_operator(PAULI_X, 2, 3),
        np.kron(IDENTITY, np.kron(IDENTITY, PAULI_X)),
        atol=0.0,
    )
    assert_allclose(
        nearest_neighbor_operator(PAULI_X, PAULI_Z, 1, 3),
        np.kron(IDENTITY, np.kron(PAULI_X, PAULI_Z)),
        atol=0.0,
    )


def test_spin_chain_hamiltonians_match_two_site_analytic_formulas() -> None:
    xx = np.kron(SPIN_X, SPIN_X)
    yy = np.kron(SPIN_Y, SPIN_Y)
    zz = np.kron(SPIN_Z, SPIN_Z)

    assert_allclose(xxz_hamiltonian(2, 0.5, coupling=1.7), 1.7 * (xx + yy + 0.5 * zz))
    assert_allclose(heisenberg_hamiltonian(2, coupling=1.7), 1.7 * (xx + yy + zz))
    assert_allclose(xy_hamiltonian(2, coupling=1.7), 1.7 * (xx + yy))
    assert_allclose(xxz_hamiltonian(1, 0.25), np.zeros((2, 2), dtype=np.complex128))


def test_tndiff_tfim_is_an_explicit_pauli_not_spin_convention() -> None:
    expected = -1.3 * np.kron(PAULI_Z, PAULI_Z)
    expected -= 0.4 * (np.kron(PAULI_X, IDENTITY) + np.kron(IDENTITY, PAULI_X))
    assert_allclose(
        tfim_tndiff_hamiltonian(2, coupling_zz=1.3, field_x=0.4),
        expected,
        atol=2e-15,
    )


def test_dense_hamiltonian_validation_is_hermitian_and_returns_an_independent_copy() -> None:
    hamiltonian = np.diag([1.0, -2.0]).astype(np.complex128)
    checked = validate_dense_hamiltonian(hamiltonian)
    hamiltonian[0, 0] = 99.0
    assert checked[0, 0] == 1.0

    with pytest.raises(ValueError, match="not Hermitian"):
        validate_dense_hamiltonian(np.array([[0.0, 1.0], [0.0, 0.0]]))
    with pytest.raises(ValueError, match="power of two"):
        validate_dense_hamiltonian(np.eye(3))
    with pytest.raises(ValueError, match="square"):
        validate_dense_hamiltonian(np.zeros((2, 3)))


def test_exact_ground_state_degenerate_selection_is_deterministic() -> None:
    hamiltonian = np.diag([-2.0, -2.0, 0.0, 1.0]).astype(np.complex128)
    first = exact_ground_state(hamiltonian)
    second = exact_ground_state(hamiltonian)

    assert first.energy == pytest.approx(-2.0, abs=2e-12)
    assert first.degeneracy == 2
    assert first.gap == pytest.approx(2.0, abs=2e-12)
    assert first.selection_basis_index == 0
    assert first.residual < 2e-12
    assert_allclose(first.state, _basis_state(0, 2), atol=2e-12)
    assert_allclose(first.state, second.state, atol=0.0)


def test_exact_ground_state_complex_eigenvector_has_positive_pivot() -> None:
    vector = np.array([1.0 + 2.0j, -0.5 + 1.0j], dtype=np.complex128)
    vector /= np.linalg.norm(vector)
    hamiltonian = -np.outer(vector, vector.conj())
    result = exact_ground_state(hamiltonian)
    pivot = int(np.argmax(np.abs(result.state)))

    assert result.energy == pytest.approx(-1.0, abs=2e-12)
    assert result.degeneracy == 1
    assert result.residual < 2e-12
    assert result.state[pivot].real > 0.0
    assert abs(result.state[pivot].imag) < 2e-12
    assert squared_state_fidelity(result.state, vector) == pytest.approx(1.0, abs=2e-12)


@pytest.mark.parametrize(
    ("atol", "rtol"), [(-1e-12, 0.0), (0.0, -1e-12), (math.inf, 0.0)],
)
def test_exact_ground_state_rejects_invalid_degeneracy_tolerances(atol: float, rtol: float) -> None:
    with pytest.raises(ValueError):
        exact_ground_state(np.diag([0.0, 1.0]), degeneracy_atol=atol, degeneracy_rtol=rtol)
