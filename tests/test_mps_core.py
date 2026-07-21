"""Independent small-system tests for strict NumPy MPS primitives."""

from __future__ import annotations

import numpy as np
import pytest

from zhou2021.mps import (
    OpenBoundaryMPS,
    apply_adjacent_two_qubit_gate,
    load_open_boundary_mps_npz,
    mps_squared_fidelity,
    open_boundary_bond_dimensions,
    product_zero_mps,
    random_mps_ensemble,
    random_open_boundary_mps,
    save_open_boundary_mps_npz,
    tt_svd_from_dense_statevector,
)


def _random_state(num_qubits: int, seed: int) -> np.ndarray:
    generator = np.random.default_rng(seed)
    vector = generator.standard_normal(1 << num_qubits) + 1j * generator.standard_normal(
        1 << num_qubits
    )
    return np.asarray(vector / np.linalg.norm(vector), dtype=np.complex128)


def _haar_u4(seed: int) -> np.ndarray:
    generator = np.random.default_rng(seed)
    matrix = generator.standard_normal((4, 4)) + 1j * generator.standard_normal((4, 4))
    q_factor, r_factor = np.linalg.qr(matrix)
    phases = np.diag(r_factor) / np.abs(np.diag(r_factor))
    return np.asarray(q_factor * phases, dtype=np.complex128)


def _dense_adjacent_operator(matrix: np.ndarray, left_site: int, num_qubits: int) -> np.ndarray:
    factors: list[np.ndarray] = []
    site = 0
    while site < num_qubits:
        if site == left_site:
            factors.append(matrix)
            site += 2
        else:
            factors.append(np.eye(2, dtype=np.complex128))
            site += 1
    result = np.array([[1.0 + 0.0j]], dtype=np.complex128)
    for factor in factors:
        result = np.kron(result, factor)
    return result


def test_tt_svd_exact_round_trip_and_norm_ledger() -> None:
    state = _random_state(6, seed=13)

    result = tt_svd_from_dense_statevector(state)

    assert result.retained_ranks == (2, 4, 8, 4, 2)
    assert result.discarded_weight == pytest.approx(0.0, abs=2e-14)
    assert result.input_norm_squared == pytest.approx(1.0, abs=2e-14)
    assert np.allclose(result.mps.to_dense(), state, rtol=0.0, atol=3e-13)
    assert result.mps.norm_squared() == pytest.approx(1.0, abs=3e-13)
    assert mps_squared_fidelity(result.mps, result.mps) == pytest.approx(1.0, abs=3e-13)


def test_tt_svd_rank_truncation_has_consistent_discarded_weight() -> None:
    state = _random_state(5, seed=17)

    result = tt_svd_from_dense_statevector(state, max_bond_dimension=1)

    assert result.mps.max_bond_dimension == 1
    assert result.discarded_weight > 0.0
    assert result.mps.norm_squared() == pytest.approx(
        result.retained_norm_squared,
        abs=3e-13,
    )
    assert 0.0 < mps_squared_fidelity(result.mps, result.mps) <= 1.0


def test_mps_gauge_operations_preserve_dense_state_and_global_fidelity() -> None:
    state = _random_state(5, seed=18)
    mps = tt_svd_from_dense_statevector(state).mps
    canonical = mps.left_canonicalized()
    phase = np.exp(0.731j)

    assert np.allclose(canonical.to_dense(), state, rtol=0.0, atol=4e-13)
    assert canonical.left_canonical_residual() < 2e-12
    assert np.allclose(canonical.normalized().to_dense(), state, rtol=0.0, atol=4e-13)
    assert mps_squared_fidelity(mps, canonical.scaled(phase)) == pytest.approx(1.0, abs=4e-13)
    assert np.allclose(mps.copy().to_dense(), state, rtol=0.0, atol=4e-13)

    mixed = mps.mixed_canonicalized_for_two_site(left_site=2)
    assert np.allclose(mixed.to_dense(), state, rtol=0.0, atol=5e-13)
    assert mixed.mixed_canonical_residual_for_two_site(2) < 3e-12


def test_two_site_mps_gate_matches_independent_dense_kron_reference() -> None:
    num_qubits = 4
    state = _random_state(num_qubits, seed=19)
    matrix = _haar_u4(seed=23)
    mps = tt_svd_from_dense_statevector(state).mps

    update = apply_adjacent_two_qubit_gate(mps, matrix, left_site=1)
    expected = _dense_adjacent_operator(matrix, left_site=1, num_qubits=num_qubits) @ state

    assert update.retained_rank <= 4
    assert update.discarded_weight == pytest.approx(0.0, abs=2e-13)
    assert np.allclose(update.mps.to_dense(), expected, rtol=0.0, atol=4e-13)
    assert update.mps.norm_squared() == pytest.approx(1.0, abs=4e-13)


def test_canonical_truncated_two_site_update_reports_actual_global_norm_loss() -> None:
    state = _random_state(4, seed=29)
    matrix = _haar_u4(seed=31)
    mps = tt_svd_from_dense_statevector(state).mps

    update = apply_adjacent_two_qubit_gate(
        mps,
        matrix,
        left_site=1,
        max_bond_dimension=1,
    )

    assert update.retained_rank == 1
    assert update.discarded_weight > 0.0
    assert update.discarded_weight_is_global_norm_loss
    exact = _dense_adjacent_operator(matrix, left_site=1, num_qubits=4) @ state
    assert np.linalg.norm(update.mps.to_dense() - exact) > 1e-6
    assert update.mps.norm_squared() == pytest.approx(
        1.0 - update.discarded_weight,
        abs=6e-13,
    )


def test_uncanonicalized_truncation_explicitly_marks_its_ledger_as_local() -> None:
    state = _random_state(4, seed=33)
    update = apply_adjacent_two_qubit_gate(
        tt_svd_from_dense_statevector(state).mps,
        _haar_u4(seed=37),
        left_site=1,
        max_bond_dimension=1,
        canonicalize_for_truncation=False,
    )

    assert not update.discarded_weight_is_global_norm_loss


def test_canonical_truncation_ledger_is_additive_along_a_serial_sweep() -> None:
    current = tt_svd_from_dense_statevector(_random_state(5, seed=41)).mps
    cumulative_discarded = 0.0
    initial_norm_squared = current.norm_squared()
    for left_site, seed in enumerate((43, 47, 53, 59)):
        norm_before = current.norm_squared()
        update = apply_adjacent_two_qubit_gate(
            current,
            _haar_u4(seed=seed),
            left_site=left_site,
            max_bond_dimension=2,
        )
        current = update.mps
        norm_after = current.norm_squared()
        assert update.discarded_weight_is_global_norm_loss
        assert norm_before - norm_after == pytest.approx(update.discarded_weight, abs=8e-13)
        cumulative_discarded += update.discarded_weight
    assert initial_norm_squared - current.norm_squared() == pytest.approx(
        cumulative_discarded,
        abs=1e-12,
    )


def test_random_mps_is_reproducible_canonical_and_never_requires_dense() -> None:
    first = random_open_boundary_mps(48, 64, seed=2021)
    second = random_open_boundary_mps(48, 64, seed=2021)

    assert first.num_qubits == 48
    assert first.bond_dimensions == open_boundary_bond_dimensions(48, 64)
    assert first.max_bond_dimension == 64
    assert first.left_canonical_residual() < 2e-12
    assert first.norm_squared() == pytest.approx(1.0, abs=2e-11)
    for left, right in zip(first.tensors, second.tensors, strict=True):
        assert np.array_equal(left, right)
    with pytest.raises(ValueError, match="Dense conversion is disabled"):
        first.to_dense()


def test_random_ensemble_records_child_seeds_and_is_reproducible() -> None:
    first = random_mps_ensemble(7, 8, 3, seed=222)
    second = random_mps_ensemble(7, 8, 3, seed=222)

    assert first.requested_seed == 222
    assert first.child_seeds == second.child_seeds
    assert len(first) == 3
    for left_mps, right_mps in zip(first, second, strict=True):
        assert mps_squared_fidelity(left_mps, right_mps) == pytest.approx(1.0, abs=3e-13)


def test_product_zero_and_open_boundary_validation() -> None:
    mps = product_zero_mps(3)
    assert np.array_equal(mps.to_dense(), np.array([1.0] + [0.0] * 7, dtype=np.complex128))
    assert mps.bond_dimensions == (1, 1, 1, 1)

    with pytest.raises(ValueError, match="bond mismatch"):
        OpenBoundaryMPS(
            (
                np.zeros((1, 2, 2), dtype=np.complex128),
                np.zeros((3, 2, 1), dtype=np.complex128),
            )
        )
    with pytest.raises(ValueError, match="physical dimension 2"):
        OpenBoundaryMPS((np.zeros((1, 3, 1), dtype=np.complex128),))


def test_strict_npz_round_trip_records_conventions_and_metadata(tmp_path) -> None:
    original = random_open_boundary_mps(6, 4, seed=77)
    path = tmp_path / "external_dmrg_target.npz"

    saved = save_open_boundary_mps_npz(
        path,
        original,
        metadata={"producer": "external-dmrg", "chi": 4, "seed": 77},
    )
    loaded = load_open_boundary_mps_npz(saved)

    assert saved == path
    assert loaded.path == path
    assert loaded.metadata == {"chi": 4, "producer": "external-dmrg", "seed": 77}
    assert mps_squared_fidelity(original, loaded.mps) == pytest.approx(1.0, abs=4e-13)
    with pytest.raises(ValueError, match="must end"):
        save_open_boundary_mps_npz(tmp_path / "missing_suffix", original)
