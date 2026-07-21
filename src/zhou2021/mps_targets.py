"""MPS target helpers for Zhou 2021 paper-scale and validation workflows.

The paper's ``N=48`` targets are MPSs (obtained by a tensor-network ground
state solver), so a production-scale reproduction should hand an externally
prepared :class:`~zhou2021.mps.OpenBoundaryMPS` directly to the ADQC/MPS
backend.  This module intentionally does *not* pretend that dense exact
diagonalisation is viable at that scale.

``dense_model_ground_state_mps`` is instead a small-N cross-validation helper:
it constructs one of the explicitly conventioned model Hamiltonians, runs the
existing dense exact diagonaliser, then TT-SVDs the result.  It is an
extension/test oracle, not a substitute for the paper's DMRG target pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .conventions import validate_num_qubits, validate_real
from .ground_states import GroundStateResult, exact_ground_state
from .models import (
    heisenberg_hamiltonian,
    tfim_tndiff_hamiltonian,
    xxz_hamiltonian,
    xy_hamiltonian,
)
from .mps import OpenBoundaryMPS, TTSVDResult, tt_svd_from_dense_statevector

ModelName = Literal["heisenberg", "xy", "xxz", "tfim_tndiff"]


@dataclass(frozen=True, slots=True)
class DenseModelGroundStateMPSTarget:
    """Small-system dense oracle and its TT-SVD MPS representation."""

    model: ModelName
    num_qubits: int
    ground_state: GroundStateResult
    decomposition: TTSVDResult

    @property
    def mps(self) -> OpenBoundaryMPS:
        """Ground state as an MPS; no new dense materialisation is performed."""

        return self.decomposition.mps


def _validate_dense_width(num_qubits: int, max_dense_qubits: int) -> None:
    max_dense_qubits = validate_num_qubits(max_dense_qubits, name="max_dense_qubits")
    if num_qubits > max_dense_qubits:
        raise ValueError(
            "Dense exact diagonalisation is a small-system validation path only: "
            f"N={num_qubits} exceeds max_dense_qubits={max_dense_qubits}. "
            "Supply an externally prepared OpenBoundaryMPS for paper-scale targets."
        )


def dense_model_ground_state_mps(
    model: ModelName,
    num_qubits: int,
    *,
    coupling: float = 1.0,
    delta: float = 1.0,
    coupling_zz: float = 1.0,
    field_x: float = 1.0,
    max_dense_qubits: int = 12,
    max_bond_dimension: int | None = None,
    singular_value_cutoff: float = 0.0,
) -> DenseModelGroundStateMPSTarget:
    """Build a convention-explicit small-N model ground state and TT-SVD it.

    ``heisenberg`` and ``xy`` are the two paper Hamiltonians.  ``xxz`` and
    ``tfim_tndiff`` are explicit comparison extensions, retained here solely
    so the later multi-paper ground-state protocol can use one target API.
    """

    num_qubits = validate_num_qubits(num_qubits)
    _validate_dense_width(num_qubits, max_dense_qubits)
    coupling = validate_real(coupling, name="coupling")
    delta = validate_real(delta, name="delta")
    coupling_zz = validate_real(coupling_zz, name="coupling_zz")
    field_x = validate_real(field_x, name="field_x")
    if model == "heisenberg":
        hamiltonian = heisenberg_hamiltonian(num_qubits, coupling=coupling)
    elif model == "xy":
        hamiltonian = xy_hamiltonian(num_qubits, coupling=coupling)
    elif model == "xxz":
        hamiltonian = xxz_hamiltonian(num_qubits, delta, coupling=coupling)
    elif model == "tfim_tndiff":
        hamiltonian = tfim_tndiff_hamiltonian(
            num_qubits,
            coupling_zz=coupling_zz,
            field_x=field_x,
        )
    else:
        raise ValueError(
            "model must be one of 'heisenberg', 'xy', 'xxz', or 'tfim_tndiff'; "
            f"got {model!r}."
        )
    ground_state = exact_ground_state(hamiltonian)
    decomposition = tt_svd_from_dense_statevector(
        ground_state.state,
        max_bond_dimension=max_bond_dimension,
        singular_value_cutoff=singular_value_cutoff,
    )
    return DenseModelGroundStateMPSTarget(
        model=model,
        num_qubits=num_qubits,
        ground_state=ground_state,
        decomposition=decomposition,
    )


__all__ = [
    "DenseModelGroundStateMPSTarget",
    "ModelName",
    "dense_model_ground_state_mps",
]
