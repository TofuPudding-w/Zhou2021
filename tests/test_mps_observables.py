"""Independent dense small-N cross-checks for MPS-only local observables."""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from zhou2021.ground_states import exact_ground_state
from zhou2021.metrics import energy_error_per_site, energy_expectation
from zhou2021.models import (
    PAULI_X,
    PAULI_Z,
    SPIN_Y,
    SPIN_Z,
    heisenberg_hamiltonian,
    nearest_neighbor_operator,
    single_site_operator,
    tfim_tndiff_hamiltonian,
    xxz_hamiltonian,
    xy_hamiltonian,
)
from zhou2021.mps import OpenBoundaryMPS, random_open_boundary_mps, tt_svd_from_dense_statevector
from zhou2021.mps_observables import (
    MPS_LOCAL_ENERGY_SCHEMA_VERSION,
    TNDiffTFIMParameters,
    XXZParameters,
    heisenberg_mps_energy,
    mps_adjacent_product_expectation,
    mps_energy_error_per_site,
    mps_model_energy,
    mps_single_site_expectation,
    tfim_tndiff_mps_energy,
    xxz_mps_energy,
    xy_mps_energy,
)


def _random_state(num_qubits: int, seed: int) -> np.ndarray:
    generator = np.random.default_rng(seed)
    state = generator.standard_normal(1 << num_qubits) + 1j * generator.standard_normal(
        1 << num_qubits
    )
    return np.asarray(state / np.linalg.norm(state), dtype=np.complex128)


def _mps_from_random_state(num_qubits: int, seed: int):
    state = _random_state(num_qubits, seed)
    return state, tt_svd_from_dense_statevector(state).mps


@pytest.mark.parametrize("site", [0, 3])
def test_single_site_expectation_matches_independent_q0_msb_dense_embedding(site: int) -> None:
    state, mps = _mps_from_random_state(4, seed=71)

    observed = mps_single_site_expectation(mps, PAULI_X, site)
    expected = float(np.vdot(state, single_site_operator(PAULI_X, site, 4) @ state).real)

    assert observed == pytest.approx(expected, abs=4e-13)


@pytest.mark.parametrize("left_site", [0, 2])
def test_adjacent_product_expectation_matches_independent_q0_msb_dense_embedding(
    left_site: int,
) -> None:
    state, mps = _mps_from_random_state(4, seed=73)

    observed = mps_adjacent_product_expectation(mps, SPIN_Y, SPIN_Z, left_site)
    dense_operator = nearest_neighbor_operator(SPIN_Y, SPIN_Z, left_site, 4)
    expected = float(np.vdot(state, dense_operator @ state).real)

    assert observed == pytest.approx(expected, abs=4e-13)


def test_expectations_are_normalized_for_an_unnormalized_mps() -> None:
    state, mps = _mps_from_random_state(4, seed=79)
    scaled = mps.scaled(2.3 - 0.7j)

    expected_one = float(np.vdot(state, single_site_operator(PAULI_Z, 1, 4) @ state).real)
    expected_two = float(
        np.vdot(state, nearest_neighbor_operator(PAULI_X, PAULI_Z, 1, 4) @ state).real
    )

    assert mps_single_site_expectation(scaled, PAULI_Z, 1) == pytest.approx(
        expected_one,
        abs=5e-13,
    )
    assert mps_adjacent_product_expectation(scaled, PAULI_X, PAULI_Z, 1) == pytest.approx(
        expected_two,
        abs=5e-13,
    )


@pytest.mark.parametrize(
    ("operator", "message"),
    [
        (np.array([[0.0, 1.0], [0.0, 0.0]], dtype=np.complex128), "Hermitian"),
        (np.array([[np.nan, 0.0], [0.0, 1.0]], dtype=np.complex128), "non-finite"),
        (np.eye(3, dtype=np.complex128), "shape"),
    ],
)
def test_local_observable_validation_is_strict(operator: np.ndarray, message: str) -> None:
    _, mps = _mps_from_random_state(3, seed=83)
    with pytest.raises(ValueError, match=message):
        mps_single_site_expectation(mps, operator, 1)
    with pytest.raises(ValueError, match=message):
        mps_adjacent_product_expectation(mps, operator, PAULI_Z, 1)


def test_local_observable_rejects_invalid_sites_and_zero_norm_mps() -> None:
    _, mps = _mps_from_random_state(3, seed=89)
    with pytest.raises(ValueError, match="outside"):
        mps_single_site_expectation(mps, PAULI_X, 3)
    with pytest.raises(ValueError, match="invalid"):
        mps_adjacent_product_expectation(mps, PAULI_X, PAULI_Z, 2)

    zero = OpenBoundaryMPS(
        tuple(np.zeros_like(tensor) for tensor in mps.tensors)
    )
    with pytest.raises(ValueError, match="zero-norm"):
        mps_single_site_expectation(zero, PAULI_X, 0)


@pytest.mark.parametrize(
    ("model", "hamiltonian", "energy_function"),
    [
        (
            XXZParameters(4, delta=0.37, coupling=1.2),
            xxz_hamiltonian(4, 0.37, coupling=1.2),
            lambda mps: xxz_mps_energy(mps, delta=0.37, coupling=1.2),
        ),
        (
            XXZParameters(4, delta=1.0, coupling=-0.8),
            heisenberg_hamiltonian(4, coupling=-0.8),
            lambda mps: heisenberg_mps_energy(mps, coupling=-0.8),
        ),
        (
            XXZParameters(4, delta=0.0, coupling=0.65),
            xy_hamiltonian(4, coupling=0.65),
            lambda mps: xy_mps_energy(mps, coupling=0.65),
        ),
        (
            TNDiffTFIMParameters(4, coupling_zz=1.1, field_x=-0.4),
            tfim_tndiff_hamiltonian(4, coupling_zz=1.1, field_x=-0.4),
            lambda mps: tfim_tndiff_mps_energy(mps, coupling_zz=1.1, field_x=-0.4),
        ),
    ],
)
def test_mps_model_energies_cross_validate_against_dense_hamiltonians(
    model: XXZParameters | TNDiffTFIMParameters,
    hamiltonian: np.ndarray,
    energy_function,
) -> None:
    state, mps = _mps_from_random_state(4, seed=97)
    expected = energy_expectation(state, hamiltonian)
    ground_energy = exact_ground_state(hamiltonian).energy

    assert mps_model_energy(mps, model) == pytest.approx(expected, abs=8e-13)
    assert energy_function(mps) == pytest.approx(expected, abs=8e-13)
    assert mps_energy_error_per_site(mps, model, ground_energy) == pytest.approx(
        energy_error_per_site(state, hamiltonian, ground_energy),
        abs=8e-13,
    )


@pytest.mark.parametrize(
    ("model", "hamiltonian"),
    [
        (XXZParameters(4, delta=1.0), heisenberg_hamiltonian(4)),
        (XXZParameters(4, delta=0.0), xy_hamiltonian(4)),
        (
            TNDiffTFIMParameters(4, coupling_zz=0.9, field_x=0.6),
            tfim_tndiff_hamiltonian(4, coupling_zz=0.9, field_x=0.6),
        ),
    ],
)
def test_mps_ground_state_energy_and_error_per_site_match_dense_metrics(
    model: XXZParameters | TNDiffTFIMParameters,
    hamiltonian: np.ndarray,
) -> None:
    ground = exact_ground_state(hamiltonian)
    mps = tt_svd_from_dense_statevector(ground.state).mps

    assert mps_model_energy(mps, model) == pytest.approx(ground.energy, abs=1e-12)
    assert mps_energy_error_per_site(mps, model, ground.energy) == pytest.approx(0.0, abs=3e-13)
    assert mps_energy_error_per_site(mps, model, ground.energy) == pytest.approx(
        energy_error_per_site(ground.state, hamiltonian, ground.energy),
        abs=3e-13,
    )


def test_parameter_records_are_explicit_and_model_mismatch_is_rejected() -> None:
    xxz = XXZParameters(6, delta=0.5, coupling=1.25)
    tfim = TNDiffTFIMParameters(6, coupling_zz=1.1, field_x=0.75)

    assert xxz.to_record() == {
        "schema_version": MPS_LOCAL_ENERGY_SCHEMA_VERSION,
        "model": "xxz",
        "boundary": "open",
        "site_order": "q0-most-significant",
        "spin_convention": "S=Pauli/2",
        "num_qubits": 6,
        "coupling": 1.25,
        "delta": 0.5,
    }
    assert tfim.to_record()["operator_convention"] == "Pauli"
    assert tfim.to_record()["model"] == "tfim_tndiff"

    _, mps = _mps_from_random_state(4, seed=101)
    with pytest.raises(ValueError, match="disagrees"):
        mps_model_energy(mps, xxz)
    with pytest.raises(TypeError, match="real scalar"):
        mps_energy_error_per_site(
            mps,
            TNDiffTFIMParameters(4),
            1.0 + 0.0j,  # type: ignore[arg-type]
        )


def test_large_mps_energy_path_never_uses_dense_materialization(monkeypatch) -> None:
    mps = random_open_boundary_mps(24, 8, seed=103)

    def _forbidden_dense(*_args, **_kwargs):
        raise AssertionError("MPS observable code attempted dense materialization.")

    monkeypatch.setattr(OpenBoundaryMPS, "to_dense", _forbidden_dense)
    energy = heisenberg_mps_energy(mps)

    assert np.isfinite(energy)
    assert "to_dense" not in inspect.getsource(mps_model_energy)
