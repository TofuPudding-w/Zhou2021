"""Autograd and dense-reference tests for the paper-scale torch MPS backend."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from zhou2021.circuit import apply_adjacent_two_qubit_gate as dense_apply_gate
from zhou2021.circuit import zero_state
from zhou2021.mps import tt_svd_from_dense_statevector
from zhou2021.polar import polar_unitary
from zhou2021.torch_mps import (
    TorchMPSDifferentiabilityError,
    TorchOpenBoundaryMPS,
    estimated_mps_complex_elements,
    torch_apply_adjacent_two_qubit_gate,
    torch_apply_staircase_layer,
    torch_mixed_canonicalized_for_two_site,
    torch_mps_from_numpy,
    torch_mps_norm_squared,
    torch_mps_squared_fidelity,
    torch_mps_to_dense,
    torch_negative_log_fidelity_per_site,
    torch_product_zero_mps,
)


def _haar_u4_torch(seed: int) -> torch.Tensor:
    generator = np.random.default_rng(seed)
    matrix = generator.standard_normal((4, 4)) + 1j * generator.standard_normal((4, 4))
    q_factor, r_factor = np.linalg.qr(matrix)
    phases = np.diag(r_factor) / np.abs(np.diag(r_factor))
    return torch.tensor(q_factor * phases, dtype=torch.complex128)


def _random_complex_parameter(seed: int) -> torch.Tensor:
    generator = np.random.default_rng(seed)
    data = generator.standard_normal((4, 4)) + 1j * generator.standard_normal((4, 4))
    return torch.tensor(data, dtype=torch.complex128, requires_grad=True)


def test_torch_mps_gate_matches_dense_axis_reference_and_preserves_gradient() -> None:
    gate = _haar_u4_torch(31).requires_grad_(True)
    initial_mps = torch_product_zero_mps(3)

    result = torch_apply_adjacent_two_qubit_gate(initial_mps, gate, left_site=1)
    observed = torch_mps_to_dense(result.mps)
    expected = dense_apply_gate(
        zero_state(3),
        gate,
        left_site=1,
        num_qubits=3,
    )
    loss = torch.real(observed[3].conj() * observed[3])
    loss.backward()
    mps_gradient = gate.grad.detach().clone()

    dense_gate = gate.detach().clone().requires_grad_(True)
    dense_state = dense_apply_gate(
        zero_state(3),
        dense_gate,
        left_site=1,
        num_qubits=3,
    )
    dense_loss = torch.real(dense_state[3].conj() * dense_state[3])
    dense_loss.backward()

    assert result.split_method == "identity_exact"
    assert result.retained_rank == 2
    assert torch.allclose(observed, expected, rtol=0.0, atol=3e-13)
    assert gate.grad is not None
    assert bool(torch.all(torch.isfinite(gate.grad)))
    assert torch.allclose(mps_gradient, dense_gate.grad, rtol=2e-10, atol=2e-11)


def test_torch_mps_target_overlap_and_polar_gate_have_finite_gradient() -> None:
    target = torch_product_zero_mps(3)
    latent = _random_complex_parameter(37)
    projected = polar_unitary(latent)
    prepared = torch_apply_adjacent_two_qubit_gate(
        torch_product_zero_mps(3),
        projected,
        left_site=0,
    ).mps
    loss = torch_negative_log_fidelity_per_site(target, prepared)
    loss.backward()

    assert torch.isfinite(loss)
    assert latent.grad is not None
    assert bool(torch.all(torch.isfinite(latent.grad)))
    fidelity = torch_mps_squared_fidelity(target, prepared)
    assert 0.0 <= float(fidelity.detach()) <= 1.0 + 2e-12


def test_torch_mps_truncated_update_records_loss_and_rank_cap() -> None:
    first = _haar_u4_torch(41)
    second = _haar_u4_torch(43)
    state = torch_product_zero_mps(3)
    state = torch_apply_adjacent_two_qubit_gate(state, first, left_site=0).mps

    update = torch_apply_adjacent_two_qubit_gate(
        state,
        second,
        left_site=1,
        max_bond_dimension=1,
    )

    assert update.split_method == "svd_truncated"
    assert update.retained_rank == 1
    assert float(update.discarded_weight.detach()) >= 0.0
    assert update.discarded_weight_is_global_norm_loss
    # The rank cap applies to the updated (1, 2) bond only; the untouched
    # (0, 1) bond can legitimately retain rank two.
    assert update.mps.bond_dimensions[2] == 1
    assert torch_mps_norm_squared(update.mps) == pytest.approx(
        1.0 - float(update.discarded_weight.detach()),
        abs=7e-13,
    )


def test_torch_canonical_truncation_preserves_finite_gradients_to_prior_gates() -> None:
    first_latent = _random_complex_parameter(44)
    second_latent = _random_complex_parameter(46)
    state = torch_apply_adjacent_two_qubit_gate(
        torch_product_zero_mps(3),
        polar_unitary(first_latent),
        left_site=0,
    ).mps
    update = torch_apply_adjacent_two_qubit_gate(
        state,
        polar_unitary(second_latent),
        left_site=1,
        max_bond_dimension=1,
    )
    loss = torch_mps_norm_squared(update.mps)
    loss.backward()

    assert update.discarded_weight_is_global_norm_loss
    assert first_latent.grad is not None
    assert second_latent.grad is not None
    assert bool(torch.all(torch.isfinite(first_latent.grad)))
    assert bool(torch.all(torch.isfinite(second_latent.grad)))


def test_rank_deficient_capped_split_has_finite_tangent_gradient_and_matches_fd() -> None:
    """An identity gate on |00> has a degenerate SVD null space, not NaNs."""

    gate = torch.eye(4, dtype=torch.complex128, requires_grad=True)
    update = torch_apply_adjacent_two_qubit_gate(
        torch_product_zero_mps(2),
        gate,
        left_site=0,
        max_bond_dimension=1,
    )
    loss = torch.real(torch_mps_to_dense(update.mps)[2])
    loss.backward()

    def finite_difference_value(offset: float) -> float:
        perturbed = torch.eye(4, dtype=torch.complex128)
        # This direction remains on the exact rank-one manifold and therefore
        # has a well-defined derivative of the rank-one projection.
        perturbed[2, 0] += offset
        result = torch_apply_adjacent_two_qubit_gate(
            torch_product_zero_mps(2),
            perturbed,
            left_site=0,
            max_bond_dimension=1,
            require_unitary=False,
        )
        return float(torch.real(torch_mps_to_dense(result.mps)[2]))

    step = 1e-6
    finite_difference = (
        finite_difference_value(step) - finite_difference_value(-step)
    ) / (2.0 * step)

    assert update.split_method == "svd_rank_deficient_tangent"
    assert update.retained_rank == 1
    assert update.discarded_weight_is_global_norm_loss
    assert float(update.discarded_weight.detach()) == pytest.approx(0.0, abs=1e-25)
    assert gate.grad is not None
    assert bool(torch.all(torch.isfinite(gate.grad)))
    assert float(torch.real(gate.grad[2, 0])) == pytest.approx(
        finite_difference,
        abs=3e-9,
    )


def test_positive_singular_value_degeneracy_fails_before_backward() -> None:
    """A nonunique hard rank-one approximation must not silently yield NaNs."""

    bell = np.zeros(4, dtype=np.complex128)
    bell[0] = bell[3] = 1.0 / np.sqrt(2.0)
    bell_mps = torch_mps_from_numpy(tt_svd_from_dense_statevector(bell).mps)
    gate = torch.eye(4, dtype=torch.complex128, requires_grad=True)

    with pytest.raises(TorchMPSDifferentiabilityError, match="Unsafe differentiable"):
        torch_apply_adjacent_two_qubit_gate(
            bell_mps,
            gate,
            left_site=0,
            max_bond_dimension=1,
        )

    assert gate.grad is None
    assert bool(torch.all(torch.isfinite(gate)))


def test_rank_deficient_canonical_qr_fails_before_backward() -> None:
    """Gauge fixing detects a redundant bond before QR can contaminate gradients."""

    first = torch.zeros((1, 2, 2), dtype=torch.complex128, requires_grad=True)
    middle = torch.zeros((2, 2, 2), dtype=torch.complex128, requires_grad=True)
    last = torch.zeros((2, 2, 1), dtype=torch.complex128, requires_grad=True)
    with torch.no_grad():
        first[0, 0, 0] = first[0, 1, 1] = 1.0
        middle[0, 0, 0] = middle[1, 0, 0] = 1.0
        last[0, 0, 0] = last[1, 0, 0] = 1.0
    redundant = TorchOpenBoundaryMPS((first, middle, last))

    with pytest.raises(TorchMPSDifferentiabilityError, match="QR canonicalisation"):
        torch_mixed_canonicalized_for_two_site(redundant, left_site=0)

    assert bool(torch.all(torch.isfinite(first)))
    assert bool(torch.all(torch.isfinite(middle)))
    assert bool(torch.all(torch.isfinite(last)))


def test_torch_canonical_truncation_ledger_is_additive_along_serial_sweep() -> None:
    current = torch_product_zero_mps(4)
    initial_norm_squared = torch_mps_norm_squared(current)
    cumulative_discarded = torch.zeros((), dtype=torch.float64)
    for left_site, seed in enumerate((61, 67, 71)):
        norm_before = torch_mps_norm_squared(current)
        update = torch_apply_adjacent_two_qubit_gate(
            current,
            _haar_u4_torch(seed),
            left_site=left_site,
            max_bond_dimension=1,
        )
        current = update.mps
        norm_after = torch_mps_norm_squared(current)
        assert update.discarded_weight_is_global_norm_loss
        assert torch.allclose(
            norm_before - norm_after,
            update.discarded_weight,
            rtol=0.0,
            atol=8e-13,
        )
        cumulative_discarded = cumulative_discarded + update.discarded_weight
    assert torch.allclose(
        initial_norm_squared - torch_mps_norm_squared(current),
        cumulative_discarded,
        rtol=0.0,
        atol=1e-12,
    )


def test_torch_mps_overlap_matches_dense_fidelity_and_has_target_gradient() -> None:
    target_vector = np.zeros(8, dtype=np.complex128)
    target_vector[0] = 1.0
    target = torch_mps_from_numpy(tt_svd_from_dense_statevector(target_vector).mps)
    latent = _random_complex_parameter(45)
    prepared = torch_apply_adjacent_two_qubit_gate(
        torch_product_zero_mps(3),
        polar_unitary(latent),
        left_site=1,
    ).mps

    fidelity = torch_mps_squared_fidelity(target, prepared)
    dense_overlap = torch.vdot(torch_mps_to_dense(target), torch_mps_to_dense(prepared))
    dense_fidelity = torch.real(dense_overlap.conj() * dense_overlap)
    fidelity.backward()

    assert torch.allclose(fidelity, dense_fidelity, rtol=0.0, atol=4e-13)
    assert latent.grad is not None
    assert bool(torch.all(torch.isfinite(latent.grad)))


def test_torch_staircase_layer_matches_serial_dense_reference() -> None:
    gates = (_haar_u4_torch(47), _haar_u4_torch(53))
    result = torch_apply_staircase_layer(torch_product_zero_mps(3), gates)
    expected = dense_apply_gate(zero_state(3), gates[0], 0, num_qubits=3)
    expected = dense_apply_gate(expected, gates[1], 1, num_qubits=3)

    assert len(result.updates) == 2
    assert torch.allclose(torch_mps_to_dense(result.mps), expected, rtol=0.0, atol=4e-13)
    assert float(result.total_discarded_weight.detach()) == pytest.approx(0.0, abs=1e-15)


def test_numpy_target_bridge_and_paper_scale_memory_ledger() -> None:
    vector = np.zeros(16, dtype=np.complex128)
    vector[0] = 1.0
    target = torch_mps_from_numpy(tt_svd_from_dense_statevector(vector).mps)

    assert torch.allclose(torch_mps_to_dense(target), torch.tensor(vector))
    assert estimated_mps_complex_elements(48, 64) == 305832
    assert estimated_mps_complex_elements(48, 64) * 16 < 5_000_000
