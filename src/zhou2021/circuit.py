"""The 1D serial U(4) staircase specified by Zhou et al. (2021)."""

from __future__ import annotations

import math
import numbers
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn

from .conventions import validate_num_qubits, validate_real
from .polar import polar_unitary

Initialization = Literal["random", "identity_perturbed"]


def _validate_seed(value: int, *, name: str = "seed") -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise TypeError(f"{name} must be an integer.")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be nonnegative.")
    return result


def zero_state(
    num_qubits: int,
    *,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Return the normalized q0-MSB product state ``|0...0>`` in complex128."""

    num_qubits = validate_num_qubits(num_qubits)
    state = torch.zeros(1 << num_qubits, dtype=torch.complex128, device=device)
    state[0] = 1.0
    return state


def validate_torch_statevector(
    state: torch.Tensor,
    *,
    num_qubits: int,
    normalization_atol: float = 2e-11,
) -> torch.Tensor:
    """Validate a normalized complex128 flat statevector without detaching it."""

    num_qubits = validate_num_qubits(num_qubits)
    normalization_atol = validate_real(
        normalization_atol,
        name="normalization_atol",
        positive=True,
    )
    if not isinstance(state, torch.Tensor):
        raise TypeError("state must be a torch.Tensor.")
    if state.dtype != torch.complex128 or state.ndim != 1 or state.numel() != 1 << num_qubits:
        raise ValueError(
            "state must be a flat torch.complex128 vector of length "
            f"2**N; got dtype={state.dtype}, shape={tuple(state.shape)}."
        )
    if not bool(torch.all(torch.isfinite(state)).detach().cpu()):
        raise ValueError("state contains non-finite values.")
    norm = float(torch.linalg.vector_norm(state).detach().cpu())
    if not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=normalization_atol):
        raise ValueError(f"state must be normalized; norm={norm:.17g}.")
    return state


def apply_adjacent_two_qubit_gate(
    state: torch.Tensor,
    matrix: torch.Tensor,
    left_site: int,
    *,
    num_qubits: int,
    unitarity_atol: float = 2e-11,
) -> torch.Tensor:
    """Apply a U(4) in q0-MSB local order ``|left,right>`` without detaching.

    This is intentionally a direct tensor-axis implementation, distinct from
    Qiskit's little-endian representation.  Tests compare it against an
    independently constructed dense Kronecker operator.
    """

    num_qubits = validate_num_qubits(num_qubits)
    unitarity_atol = validate_real(
        unitarity_atol,
        name="unitarity_atol",
        positive=True,
    )
    validate_torch_statevector(state, num_qubits=num_qubits)
    if isinstance(left_site, bool) or not isinstance(left_site, int):
        raise TypeError("left_site must be an integer.")
    if not 0 <= left_site < num_qubits - 1:
        raise ValueError(f"left_site={left_site} is invalid for N={num_qubits}.")
    if not isinstance(matrix, torch.Tensor):
        raise TypeError("matrix must be a torch.Tensor.")
    if matrix.dtype != torch.complex128 or matrix.shape != (4, 4):
        raise ValueError("matrix must have dtype torch.complex128 and shape (4, 4).")
    if matrix.device != state.device:
        raise ValueError("state and matrix must be on the same device.")
    if not bool(torch.all(torch.isfinite(matrix)).detach().cpu()):
        raise ValueError("matrix contains non-finite entries.")
    with torch.no_grad():
        identity = torch.eye(4, dtype=torch.complex128, device=matrix.device)
        residual = float(
            torch.linalg.matrix_norm(matrix.conj().transpose(-2, -1) @ matrix - identity)
            .cpu()
        )
    if residual > unitarity_atol:
        raise ValueError(
            "matrix must be unitary in the declared |00>,|01>,|10>,|11> basis; "
            f"residual={residual:.3e}."
        )

    remaining_axes = [axis for axis in range(num_qubits) if axis not in (left_site, left_site + 1)]
    axes = [left_site, left_site + 1, *remaining_axes]
    inverse_axes = [axes.index(axis) for axis in range(num_qubits)]
    local_matrix = state.reshape((2,) * num_qubits).permute(axes).reshape(4, -1)
    updated = (matrix @ local_matrix).reshape((2, 2, *((2,) * (num_qubits - 2))))
    return updated.permute(inverse_axes).reshape(-1)


@dataclass(frozen=True, slots=True)
class ProjectedGate:
    """One ordered physical U(4) gate derived from a latent parameter."""

    layer: int
    position: int
    left_site: int
    matrix: torch.Tensor


class LatentStaircase(nn.Module):
    """Zhou's serial nearest-neighbor U(4) ADQC ansatz.

    A layer has exactly ``N-1`` latent U(4) gates.  The module deliberately
    has no terminal U(2), no brickwall recoloring and no hidden random jitter.
    """

    def __init__(
        self,
        num_qubits: int,
        *,
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__()
        self.num_qubits = validate_num_qubits(num_qubits)
        if self.num_qubits < 2:
            raise ValueError(
                "Zhou's serial U(4) staircase requires at least two qubits; "
                "N=1 has no two-qubit gate and is outside this ansatz."
            )
        self.device = torch.device(device)
        self.latent_layers = nn.ModuleList()

    @property
    def num_layers(self) -> int:
        return len(self.latent_layers)

    @property
    def gates_per_layer(self) -> int:
        return self.num_qubits - 1

    @property
    def paper_parameter_count(self) -> int:
        """Return the paper's matrix-element count ``16(N-1)L`` (Eq. 6)."""

        return 16 * self.gates_per_layer * self.num_layers

    @property
    def real_latent_parameter_count(self) -> int:
        """Return real scalar coordinates of the unconstrained complex latents."""

        return 2 * self.paper_parameter_count

    @property
    def maximum_mps_bond_bound(self) -> int:
        """Return the paper's staircase upper bound ``chi<=4**L``."""

        return 4**self.num_layers

    def add_layer(
        self,
        *,
        initialization: Initialization,
        seed: int,
        perturbation: float = 1e-3,
    ) -> int:
        """Append a seeded random or identity-perturbed latent U(4) layer."""

        if initialization not in {"random", "identity_perturbed"}:
            raise ValueError("initialization must be 'random' or 'identity_perturbed'.")
        seed = _validate_seed(seed)
        perturbation = validate_real(
            perturbation,
            name="perturbation",
            positive=True,
        )
        generator = torch.Generator(device=self.device)
        generator.manual_seed(seed)
        gates = nn.ParameterList()
        identity = torch.eye(4, dtype=torch.complex128, device=self.device)
        for _ in range(self.gates_per_layer):
            real = torch.randn(
                (4, 4),
                generator=generator,
                dtype=torch.float64,
                device=self.device,
            )
            imaginary = torch.randn(
                (4, 4),
                generator=generator,
                dtype=torch.float64,
                device=self.device,
            )
            noise = torch.complex(real, imaginary)
            if initialization == "random":
                latent = noise / math.sqrt(8.0)
            else:
                latent = identity + perturbation * noise
            gates.append(nn.Parameter(latent))
        self.latent_layers.append(gates)
        return self.num_layers - 1

    def layer_parameters(self, layer: int) -> tuple[nn.Parameter, ...]:
        """Return the ordered latent parameters in one layer."""

        if isinstance(layer, bool) or not isinstance(layer, int):
            raise TypeError("layer must be an integer.")
        if not 0 <= layer < self.num_layers:
            raise ValueError(f"layer={layer} is outside [0, {self.num_layers}).")
        return tuple(self.latent_layers[layer])

    def set_trainable_layer(self, active_layer: int | None) -> None:
        """Freeze all but one layer for the paper's layer-by-layer BP schedule."""

        if active_layer is not None:
            self.layer_parameters(active_layer)
        for layer_index, layer in enumerate(self.latent_layers):
            enabled = active_layer is None or layer_index == active_layer
            for parameter in layer:
                parameter.requires_grad_(enabled)

    def iter_projected_gates(self) -> Iterator[ProjectedGate]:
        """Yield U(4) gates in the exact physical execution order."""

        for layer_index, layer in enumerate(self.latent_layers):
            for position, latent in enumerate(layer):
                yield ProjectedGate(
                    layer=layer_index,
                    position=position,
                    left_site=position,
                    matrix=polar_unitary(latent),
                )

    def forward(self, initial_state: torch.Tensor | None = None) -> torch.Tensor:
        """Prepare a state with serial staircase gates in layer-major order."""

        if initial_state is None:
            state = zero_state(self.num_qubits, device=self.device)
        else:
            state = validate_torch_statevector(initial_state, num_qubits=self.num_qubits)
            if state.device != self.device:
                raise ValueError("initial_state must be on the staircase device.")
        for gate in self.iter_projected_gates():
            state = apply_adjacent_two_qubit_gate(
                state,
                gate.matrix,
                gate.left_site,
                num_qubits=self.num_qubits,
            )
        return state


__all__ = [
    "Initialization",
    "LatentStaircase",
    "ProjectedGate",
    "apply_adjacent_two_qubit_gate",
    "validate_torch_statevector",
    "zero_state",
]
