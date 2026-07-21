"""Unitary-projection, autograd and q0-MSB staircase checks."""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import pytest
import torch
from numpy.testing import assert_allclose

from zhou2021.circuit import (
    LatentStaircase,
    apply_adjacent_two_qubit_gate,
    validate_torch_statevector,
    zero_state,
)
from zhou2021.polar import polar_unitary, project_latent_gate, validate_latent_gate


def _random_complex_tensor(seed: int, shape: tuple[int, ...]) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    real = torch.randn(shape, dtype=torch.float64, generator=generator)
    imag = torch.randn(shape, dtype=torch.float64, generator=generator)
    return torch.complex(real, imag)


def _normalized_random_state(seed: int, num_qubits: int) -> torch.Tensor:
    state = _random_complex_tensor(seed, (1 << num_qubits,))
    return state / torch.linalg.vector_norm(state)


def _q0_msb_dense_gate(matrix: np.ndarray, left_site: int, num_qubits: int) -> np.ndarray:
    """Independent q0-MSB Kronecker embedding for an adjacent U(4)."""

    before = np.eye(1 << left_site, dtype=np.complex128)
    after = np.eye(1 << (num_qubits - left_site - 2), dtype=np.complex128)
    return np.kron(before, np.kron(matrix, after))


def _squared_fidelity_loss(latent: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    prepared = apply_adjacent_two_qubit_gate(
        zero_state(2),
        polar_unitary(latent),
        0,
        num_qubits=2,
    )
    overlap = torch.vdot(target, prepared)
    return -torch.log(torch.real(overlap.conj() * overlap))


def test_polar_projection_is_unitary_and_attains_the_nuclear_norm_objective() -> None:
    latent = _random_complex_tensor(11, (4, 4))
    projection = project_latent_gate(latent)
    identity = torch.eye(4, dtype=torch.complex128)

    residual = torch.linalg.matrix_norm(
        projection.unitary.conj().T @ projection.unitary - identity
    )
    assert residual < 2e-12
    assert projection.unitarity_residual < 2e-12
    assert float(projection.objective) == pytest.approx(
        float(torch.sum(projection.singular_values)),
        rel=0.0,
        abs=2e-12,
    )


def test_polar_projection_is_equivariant_under_global_phase_and_idempotent_on_unitaries() -> None:
    latent = _random_complex_tensor(12, (4, 4))
    unitary = polar_unitary(latent)
    phase = torch.exp(torch.tensor(0.71j, dtype=torch.complex128))

    assert_allclose(
        polar_unitary(phase * latent).detach().numpy(),
        (phase * unitary).detach().numpy(),
        atol=3e-12,
    )
    assert_allclose(
        polar_unitary(unitary).detach().numpy(),
        unitary.detach().numpy(),
        atol=3e-12,
    )


def _central_coordinate_derivative(
    objective: Callable[[torch.Tensor], float],
    latent: torch.Tensor,
    coordinate: tuple[int, int],
    *,
    imaginary: bool,
    step: float = 1e-6,
) -> float:
    """Independently estimate one real-coordinate derivative."""

    plus = latent.detach().clone()
    minus = latent.detach().clone()
    perturbation = 1j * step if imaginary else step
    plus[coordinate] += perturbation
    minus[coordinate] -= perturbation
    return (objective(plus) - objective(minus)) / (2.0 * step)


def _assert_coordinate_gradient_matches_central_difference(
    latent: torch.Tensor,
    target: torch.Tensor,
    coordinate: tuple[int, int],
) -> None:
    """Check both real coordinates of a real scalar ADQC loss."""

    loss = _squared_fidelity_loss(latent, target)
    loss.backward()
    assert latent.grad is not None

    def objective(value: torch.Tensor) -> float:
        return float(_squared_fidelity_loss(value, target).detach().cpu())

    finite_real = _central_coordinate_derivative(objective, latent, coordinate, imaginary=False)
    finite_imag = _central_coordinate_derivative(objective, latent, coordinate, imaginary=True)
    # PyTorch's complex autograd exposes the real-coordinate derivatives in
    # ``grad.real`` and ``grad.imag`` for this real scalar loss.
    assert float(latent.grad[coordinate].real) == pytest.approx(
        finite_real,
        rel=2e-5,
        abs=2e-6,
    )
    assert float(latent.grad[coordinate].imag) == pytest.approx(
        finite_imag,
        rel=2e-5,
        abs=2e-6,
    )


@pytest.mark.parametrize(
    "latent, exception",
    [
        (np.eye(4), TypeError),
        (torch.eye(3, dtype=torch.complex128), ValueError),
        (torch.eye(4, dtype=torch.complex64), TypeError),
        (torch.zeros((4, 4), dtype=torch.complex128), ValueError),
    ],
)
def test_latent_gate_validation_rejects_invalid_inputs(
    latent: object,
    exception: type[Exception],
) -> None:
    with pytest.raises(exception):
        validate_latent_gate(latent)  # type: ignore[arg-type]


def test_polar_autograd_matches_central_finite_differences() -> None:
    latent = _random_complex_tensor(13, (4, 4)).requires_grad_(True)
    target = _normalized_random_state(14, 2)
    _assert_coordinate_gradient_matches_central_difference(latent, target, (1, 2))


def test_polar_identity_has_finite_difference_correct_gradient() -> None:
    """Identity has four repeated singular values but a smooth polar map."""

    latent = torch.eye(4, dtype=torch.complex128).requires_grad_(True)
    target = _normalized_random_state(141, 2)

    assert_allclose(polar_unitary(latent).detach().numpy(), np.eye(4), atol=2e-12)
    _assert_coordinate_gradient_matches_central_difference(latent, target, (1, 2))
    assert torch.isfinite(latent.grad).all()


def test_polar_identity_vjp_passes_full_coordinate_gradcheck() -> None:
    """All 32 real coordinates remain valid at the maximally degenerate point."""

    latent = torch.eye(4, dtype=torch.complex128).requires_grad_(True)
    assert torch.autograd.gradcheck(
        polar_unitary,
        (latent,),
        eps=1e-6,
        atol=2e-5,
        rtol=2e-3,
        check_grad_dtypes=True,
    )


def test_polar_repeated_singular_values_match_svd_forward_and_finite_differences() -> None:
    """The custom VJP must not inherit SVD's singular-vector 0/0 instability."""

    left_unitary = polar_unitary(_random_complex_tensor(142, (4, 4))).detach()
    right_unitary = polar_unitary(_random_complex_tensor(143, (4, 4))).detach()
    diagonal = torch.diag(torch.tensor([2.0, 2.0, 0.75, 0.75], dtype=torch.complex128))
    latent = (left_unitary @ diagonal @ right_unitary.conj().transpose(-2, -1)).requires_grad_(
        True
    )
    expected_u, _, expected_vh = torch.linalg.svd(latent.detach(), full_matrices=False)
    expected = expected_u @ expected_vh

    assert_allclose(polar_unitary(latent).detach().numpy(), expected.numpy(), atol=3e-12)
    _assert_coordinate_gradient_matches_central_difference(
        latent,
        _normalized_random_state(144, 2),
        (2, 1),
    )
    assert torch.isfinite(latent.grad).all()


def test_polar_rejects_rank_deficient_latent_without_basis_dependent_completion() -> None:
    rank_one = torch.diag(torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.complex128))
    with pytest.raises(ValueError, match="no unique U\\(4\\) polar factor"):
        polar_unitary(rank_one)


def test_polar_diagnostic_singular_values_are_deliberately_nondifferentiable() -> None:
    latent = torch.eye(4, dtype=torch.complex128).requires_grad_(True)
    projection = project_latent_gate(latent)
    assert not projection.singular_values.requires_grad


@pytest.mark.parametrize("left_site", [0, 1])
def test_adjacent_gate_matches_independent_q0_msb_dense_kron(left_site: int) -> None:
    state = _normalized_random_state(20, 3)
    unitary = polar_unitary(_random_complex_tensor(21, (4, 4)))
    observed = apply_adjacent_two_qubit_gate(state, unitary, left_site, num_qubits=3)
    expected = _q0_msb_dense_gate(unitary.detach().numpy(), left_site, 3) @ state.numpy()

    assert_allclose(observed.detach().numpy(), expected, atol=3e-12)
    assert float(torch.linalg.vector_norm(observed)) == pytest.approx(1.0, abs=3e-12)


def test_zero_state_and_torch_state_validation_cover_dtype_shape_norm_and_finiteness() -> None:
    state = zero_state(3)
    assert state.dtype == torch.complex128
    assert state.shape == (8,)
    assert torch.equal(state, torch.tensor([1.0] + [0.0] * 7, dtype=torch.complex128))
    assert validate_torch_statevector(state, num_qubits=3) is state

    invalid_states = [
        torch.zeros(8, dtype=torch.float64),
        torch.zeros((2, 4), dtype=torch.complex128),
        torch.zeros(8, dtype=torch.complex128),
        torch.full((8,), complex(math.inf, 0.0), dtype=torch.complex128),
    ]
    for invalid in invalid_states:
        with pytest.raises(ValueError):
            validate_torch_statevector(invalid, num_qubits=3)


def test_staircase_seeded_layers_order_counts_and_dense_forward() -> None:
    first = LatentStaircase(3)
    second = LatentStaircase(3)
    for model in (first, second):
        assert model.add_layer(initialization="random", seed=np.int64(31)) == 0
        assert model.add_layer(initialization="identity_perturbed", seed=32) == 1

    assert first.num_layers == 2
    assert first.gates_per_layer == 2
    assert first.paper_parameter_count == 64
    assert first.real_latent_parameter_count == 128
    assert first.maximum_mps_bond_bound == 16
    for expected, observed in zip(first.parameters(), second.parameters(), strict=True):
        assert torch.equal(expected, observed)

    gates = list(first.iter_projected_gates())
    assert [(gate.layer, gate.position, gate.left_site) for gate in gates] == [
        (0, 0, 0),
        (0, 1, 1),
        (1, 0, 0),
        (1, 1, 1),
    ]
    dense_expected = zero_state(3).numpy()
    for gate in gates:
        dense_expected = _q0_msb_dense_gate(
            gate.matrix.detach().numpy(),
            gate.left_site,
            3,
        ) @ dense_expected
    assert_allclose(first().detach().numpy(), dense_expected, atol=5e-12)

    first.set_trainable_layer(1)
    assert all(not parameter.requires_grad for parameter in first.layer_parameters(0))
    assert all(parameter.requires_grad for parameter in first.layer_parameters(1))
    first.set_trainable_layer(None)
    assert all(parameter.requires_grad for parameter in first.parameters())


def test_staircase_rejects_one_qubit_degenerate_topology() -> None:
    with pytest.raises(ValueError, match="at least two qubits"):
        LatentStaircase(1)


def test_identity_perturbed_layer_is_close_to_but_not_equal_to_identity_after_projection() -> None:
    model = LatentStaircase(2)
    model.add_layer(initialization="identity_perturbed", seed=41, perturbation=1e-3)
    gate = next(model.iter_projected_gates()).matrix.detach()
    identity = torch.eye(4, dtype=torch.complex128)

    assert 1e-8 < float(torch.linalg.matrix_norm(gate - identity)) < 2e-2


def test_staircase_and_adjacent_gate_reject_invalid_layer_and_gate_arguments() -> None:
    model = LatentStaircase(2)
    with pytest.raises(ValueError, match="initialization"):
        model.add_layer(initialization="wrong", seed=0)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="nonnegative"):
        model.add_layer(initialization="random", seed=-1)
    model.add_layer(initialization="random", seed=0)
    with pytest.raises(ValueError, match="outside"):
        model.layer_parameters(1)
    with pytest.raises(TypeError, match="integer"):
        model.set_trainable_layer(True)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="left_site"):
        apply_adjacent_two_qubit_gate(
            zero_state(2),
            torch.eye(4, dtype=torch.complex128),
            1,
            num_qubits=2,
        )
    with pytest.raises(ValueError, match="shape"):
        apply_adjacent_two_qubit_gate(
            zero_state(2),
            torch.eye(2, dtype=torch.complex128),
            0,
            num_qubits=2,
        )
    with pytest.raises(ValueError, match="unitary"):
        apply_adjacent_two_qubit_gate(
            zero_state(2),
            2.0 * torch.eye(4, dtype=torch.complex128),
            0,
            num_qubits=2,
        )
