"""End-to-end native-gate/FakeWashington evaluation for Zhou ADQC states.

The Zhou 2021 training path remains strictly noiseless and differentiable.
This module is an explicitly downstream protocol:

``trained serial U(4) staircase -> exact block-native circuit -> compact
FakeWashingtonV2 target compilation -> Aer density-matrix channel``.

It is intentionally separate from :mod:`zhou2021.adqc` so no noise gradient,
native transpilation, routing choice, or calibration artifact can silently
change the paper-method result.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from .adqc import ADQCTrainingResult
from .circuit import LatentStaircase
from .gate_archive import (
    OrderedU4Archive,
    replay_ordered_u4_archive,
    validate_ordered_u4_archive,
)
from .qiskit_adapter import (
    CompilationDiagnostics,
    StaircaseGateRecord,
    compile_gate_records_native,
    compile_staircase_native,
    diagnose_gate_records_compilation,
    diagnose_staircase_compilation,
    qiskit_statevector_q0_msb,
    to_qiskit_logical,
    to_qiskit_logical_from_records,
)
from .washington_noise import (
    CompactWashingtonDevice,
    NativeCompilation,
    NoisyEncodingResult,
    build_compact_washington_device,
    compile_to_native_target,
    evaluate_native_density_matrix,
)


@dataclass(frozen=True, slots=True)
class StaircaseNativeNoiseEvaluation:
    """All three ideal/noisy stages for one frozen Zhou ADQC state."""

    block_synthesis: CompilationDiagnostics
    target_compilation: NativeCompilation
    noisy: NoisyEncodingResult
    compact_device: CompactWashingtonDevice = field(repr=False, compare=False)
    logical_circuit: Any = field(repr=False, compare=False)
    block_native_circuit: Any = field(repr=False, compare=False)

    def as_dict(self) -> dict[str, Any]:
        """Return JSON-ready scalar/provenance records (not opaque circuits)."""

        return {
            "schema_version": "zhou2021-staircase-native-noise-evaluation-v2",
            "scope": (
                "downstream evaluation of unchanged Zhou 2021 serial-U4 ADQC; "
                "no noise-aware training or recompilation feedback"
            ),
            "block_synthesis": {
                "num_qubits": self.block_synthesis.num_qubits,
                "num_layers": self.block_synthesis.num_layers,
                "num_logical_u4": self.block_synthesis.num_logical_u4,
                "logical_replay_fidelity": self.block_synthesis.logical_replay_fidelity,
                "native_replay_fidelity": self.block_synthesis.native_replay_fidelity,
                "compile_fidelity": self.block_synthesis.compile_fidelity,
                "logical_phase_over_dense": self.block_synthesis.logical_phase_over_dense,
                "native_phase_over_logical": self.block_synthesis.native_phase_over_logical,
                "logical_phase_aligned_max_abs_error": (
                    self.block_synthesis.logical_phase_aligned_max_abs_error
                ),
                "native_phase_aligned_max_abs_error": (
                    self.block_synthesis.native_phase_aligned_max_abs_error
                ),
                "native_operation_counts": dict(self.block_synthesis.native_operation_counts),
                "native_basis_only": self.block_synthesis.native_basis_only,
            },
            "target_compilation": self.target_compilation.provenance_dict(),
            "compact_washington_device": self.compact_device.provenance_dict(),
            "noise_evaluation": self.noisy.as_dict(),
        }


def _validate_compilation_diagnostics(
    diagnostics: CompilationDiagnostics,
    *,
    atol: float,
) -> None:
    if not diagnostics.native_basis_only:
        raise ArithmeticError("Exact block synthesis emitted an instruction outside rz/sx/x/cx.")
    for name, value in {
        "logical replay": diagnostics.logical_replay_fidelity,
        "native replay": diagnostics.native_replay_fidelity,
        "block compilation": diagnostics.compile_fidelity,
    }.items():
        if abs(value - 1.0) > atol:
            raise ArithmeticError(f"{name} fidelity={value:.17g}, outside atol={atol:.3e}.")


def evaluate_staircase_under_washington(
    staircase: LatentStaircase,
    target_state: ArrayLike,
    *,
    hamiltonian: ArrayLike | None = None,
    ground_energy: float | None = None,
    device: CompactWashingtonDevice | None = None,
    physical_path: Iterable[int] | None = None,
    optimization_level: int = 0,
    seed_transpiler: int = 2021,
    seed_simulator: int = 2021,
    max_density_matrix_qubits: int = 12,
    compilation_fidelity_atol: float = 2e-9,
) -> StaircaseNativeNoiseEvaluation:
    """Evaluate one frozen serial staircase with compact FakeWashington noise.

    If ``device`` is omitted, a deterministic compact physical path is selected
    from a freshly instantiated ``FakeWashingtonV2`` snapshot.  Pass an
    explicit ``physical_path`` in a formal study after inspecting calibration
    provenance.  A supplied device and a supplied path are mutually exclusive
    to avoid ambiguous hardware provenance.
    """

    if not isinstance(staircase, LatentStaircase):
        raise TypeError("staircase must be a zhou2021.circuit.LatentStaircase.")
    if device is not None and not isinstance(device, CompactWashingtonDevice):
        raise TypeError("device must be CompactWashingtonDevice or None.")
    if device is not None and physical_path is not None:
        raise ValueError("Provide device or physical_path, not both.")
    if isinstance(compilation_fidelity_atol, bool) or not isinstance(
        compilation_fidelity_atol, (float, int, np.floating, np.integer)
    ):
        raise TypeError("compilation_fidelity_atol must be a real scalar.")
    compilation_fidelity_atol = float(compilation_fidelity_atol)
    if not np.isfinite(compilation_fidelity_atol) or compilation_fidelity_atol <= 0.0:
        raise ValueError("compilation_fidelity_atol must be finite and positive.")
    if device is None:
        device = build_compact_washington_device(
            staircase.num_qubits,
            physical_path=physical_path,
            gate_error=True,
            thermal_relaxation=True,
            readout_error=False,
        )
    if device.num_qubits != staircase.num_qubits:
        raise ValueError("staircase and compact Washington device disagree in qubit count.")

    # Freeze the logical circuit and independently replay the exact local-block
    # synthesis before the hardware-target compiler has a chance to modify it.
    logical_circuit = to_qiskit_logical(staircase)
    block_native_circuit = compile_staircase_native(staircase)
    block_synthesis = diagnose_staircase_compilation(staircase, block_native_circuit)
    _validate_compilation_diagnostics(block_synthesis, atol=compilation_fidelity_atol)

    target_compilation = compile_to_native_target(
        block_native_circuit,
        device,
        optimization_level=optimization_level,
        seed_transpiler=seed_transpiler,
        enforce_no_swaps=True,
    )
    logical_state = qiskit_statevector_q0_msb(logical_circuit)
    noisy = evaluate_native_density_matrix(
        target_state,
        logical_state,
        target_compilation,
        device=device,
        hamiltonian=hamiltonian,
        ground_energy=ground_energy,
        num_sites=staircase.num_qubits,
        seed_simulator=seed_simulator,
        max_density_matrix_qubits=max_density_matrix_qubits,
        compute_minimum_eigenvalue=True,
    )
    if abs(noisy.compilation_fidelity - 1.0) > compilation_fidelity_atol:
        raise ArithmeticError(
            "Compact-target compilation failed ideal fidelity validation: "
            f"{noisy.compilation_fidelity:.17g}."
        )
    return StaircaseNativeNoiseEvaluation(
        block_synthesis=block_synthesis,
        target_compilation=target_compilation,
        noisy=noisy,
        compact_device=device,
        logical_circuit=logical_circuit,
        block_native_circuit=block_native_circuit,
    )


def evaluate_ordered_u4_archive_under_washington(
    archive: OrderedU4Archive,
    target_state: ArrayLike,
    *,
    hamiltonian: ArrayLike | None = None,
    ground_energy: float | None = None,
    device: CompactWashingtonDevice | None = None,
    physical_path: Iterable[int] | None = None,
    optimization_level: int = 0,
    seed_transpiler: int = 2021,
    seed_simulator: int = 2021,
    max_density_matrix_qubits: int = 12,
    compilation_fidelity_atol: float = 2e-9,
) -> StaircaseNativeNoiseEvaluation:
    """Evaluate a saved physical-U(4) archive without recreating latent gates.

    This is the post-training hand-off path: the archive is first validated
    against its strict serial order and semantic SHA, then its stored U(4)
    matrices are translated directly to Qiskit, exactly decomposed to the
    native basis, and evaluated under FakeWashingtonV2.  No PyTorch module,
    optimizer state, polar projection, or noise-aware update is reconstructed.
    """

    validated = validate_ordered_u4_archive(archive)
    if device is not None and not isinstance(device, CompactWashingtonDevice):
        raise TypeError("device must be CompactWashingtonDevice or None.")
    if device is not None and physical_path is not None:
        raise ValueError("Provide device or physical_path, not both.")
    if isinstance(compilation_fidelity_atol, bool) or not isinstance(
        compilation_fidelity_atol, (float, int, np.floating, np.integer)
    ):
        raise TypeError("compilation_fidelity_atol must be a real scalar.")
    compilation_fidelity_atol = float(compilation_fidelity_atol)
    if not np.isfinite(compilation_fidelity_atol) or compilation_fidelity_atol <= 0.0:
        raise ValueError("compilation_fidelity_atol must be finite and positive.")
    if device is None:
        device = build_compact_washington_device(
            validated.num_qubits,
            physical_path=physical_path,
            gate_error=True,
            thermal_relaxation=True,
            readout_error=False,
        )
    if device.num_qubits != validated.num_qubits:
        raise ValueError(
            "ordered-U4 archive and compact Washington device disagree in qubit count."
        )

    records = tuple(
        StaircaseGateRecord(
            layer=gate.layer,
            position=gate.position,
            left_site=gate.left_site,
            matrix=gate.matrix,
        )
        for gate in validated.gates
    )
    logical_circuit = to_qiskit_logical_from_records(records, num_qubits=validated.num_qubits)
    block_native_circuit = compile_gate_records_native(records, num_qubits=validated.num_qubits)
    block_synthesis = diagnose_gate_records_compilation(
        records,
        num_qubits=validated.num_qubits,
        dense_state=replay_ordered_u4_archive(validated),
        native_circuit=block_native_circuit,
    )
    _validate_compilation_diagnostics(block_synthesis, atol=compilation_fidelity_atol)

    target_compilation = compile_to_native_target(
        block_native_circuit,
        device,
        optimization_level=optimization_level,
        seed_transpiler=seed_transpiler,
        enforce_no_swaps=True,
    )
    logical_state = qiskit_statevector_q0_msb(logical_circuit)
    noisy = evaluate_native_density_matrix(
        target_state,
        logical_state,
        target_compilation,
        device=device,
        hamiltonian=hamiltonian,
        ground_energy=ground_energy,
        num_sites=validated.num_qubits,
        seed_simulator=seed_simulator,
        max_density_matrix_qubits=max_density_matrix_qubits,
        compute_minimum_eigenvalue=True,
    )
    if abs(noisy.compilation_fidelity - 1.0) > compilation_fidelity_atol:
        raise ArithmeticError(
            "Compact-target compilation failed ideal fidelity validation: "
            f"{noisy.compilation_fidelity:.17g}."
        )
    return StaircaseNativeNoiseEvaluation(
        block_synthesis=block_synthesis,
        target_compilation=target_compilation,
        noisy=noisy,
        compact_device=device,
        logical_circuit=logical_circuit,
        block_native_circuit=block_native_circuit,
    )


def evaluate_training_result_under_washington(
    training: ADQCTrainingResult,
    *,
    hamiltonian: ArrayLike | None = None,
    ground_energy: float | None = None,
    device: CompactWashingtonDevice | None = None,
    physical_path: Iterable[int] | None = None,
    optimization_level: int = 0,
    seed_transpiler: int = 2021,
    seed_simulator: int = 2021,
    max_density_matrix_qubits: int = 12,
    compilation_fidelity_atol: float = 2e-9,
) -> StaircaseNativeNoiseEvaluation:
    """Convenience wrapper evaluating a completed noiseless ADQC optimization.

    The wrapper deliberately takes the target from the saved training result;
    it does not inspect gradients, optimizer states, or modify any latent
    parameters.  For a ground-state study pass the exact Hamiltonian and its
    matching ground energy to obtain noisy energy error per site.
    """

    if not isinstance(training, ADQCTrainingResult):
        raise TypeError("training must be an ADQCTrainingResult.")
    target = training.target_state.detach().cpu().numpy()
    return evaluate_staircase_under_washington(
        training.model,
        target,
        hamiltonian=hamiltonian,
        ground_energy=ground_energy,
        device=device,
        physical_path=physical_path,
        optimization_level=optimization_level,
        seed_transpiler=seed_transpiler,
        seed_simulator=seed_simulator,
        max_density_matrix_qubits=max_density_matrix_qubits,
        compilation_fidelity_atol=compilation_fidelity_atol,
    )


__all__ = [
    "StaircaseNativeNoiseEvaluation",
    "evaluate_ordered_u4_archive_under_washington",
    "evaluate_staircase_under_washington",
    "evaluate_training_result_under_washington",
]
