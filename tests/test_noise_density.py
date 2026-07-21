"""Pure density-matrix and optional FakeWashington protocol checks."""

from __future__ import annotations

import importlib.util
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from zhou2021.circuit import LatentStaircase
from zhou2021.gate_archive import ordered_u4_archive_from_staircase
from zhou2021.metrics import energy_expectation
from zhou2021.models import heisenberg_hamiltonian
from zhou2021.native_noise_protocol import (
    StaircaseNativeNoiseEvaluation,
    evaluate_ordered_u4_archive_under_washington,
    evaluate_staircase_under_washington,
)
from zhou2021.qiskit_adapter import CompilationDiagnostics, compile_staircase_native
from zhou2021.washington_noise import (
    CompactWashingtonDevice,
    InstructionCalibrationAudit,
    NativeCompilation,
    QiskitNoiseDependencyError,
    QubitCalibrationAudit,
    _layout_indices,
    _native_compilation_from_circuit,
    _validate_native_compilation_ledger,
    _validate_protocol_native_operation_counts,
    build_compact_washington_device,
    density_matrix_energy_expectation,
    diagnose_density_matrix,
    evaluate_native_density_matrix,
)


def test_density_matrix_diagnostics_cover_trace_hermiticity_purity_and_psd() -> None:
    pure = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.complex128)
    mixed = np.eye(2, dtype=np.complex128) / 2.0
    pure_diagnostics = diagnose_density_matrix(pure)
    mixed_diagnostics = diagnose_density_matrix(mixed)

    assert pure_diagnostics.purity == pytest.approx(1.0, abs=2e-12)
    assert pure_diagnostics.minimum_eigenvalue == pytest.approx(0.0, abs=2e-12)
    assert mixed_diagnostics.purity == pytest.approx(0.5, abs=2e-12)
    assert mixed_diagnostics.minimum_eigenvalue == pytest.approx(0.5, abs=2e-12)
    assert mixed_diagnostics.trace_error == pytest.approx(0.0, abs=2e-12)
    assert mixed_diagnostics.hermiticity_residual == pytest.approx(0.0, abs=2e-12)


@pytest.mark.parametrize(
    "rho, expected",
    [
        (np.array([[1.0, 1.0], [0.0, 0.0]], dtype=np.complex128), "trace/Hermiticity"),
        # The diagnostics intentionally reject unphysical purity before
        # performing the more expensive PSD eigensolve, hence either wording
        # is an acceptable physicality failure for this deliberately bad rho.
        (np.array([[1.1, 0.0], [0.0, -0.1]], dtype=np.complex128), "purity|PSD"),
    ],
)
def test_density_matrix_diagnostics_reject_unphysical_inputs(
    rho: np.ndarray,
    expected: str,
) -> None:
    with pytest.raises(ArithmeticError, match=expected):
        diagnose_density_matrix(rho)


def test_density_matrix_energy_is_tr_h_rho_with_no_hidden_normalization() -> None:
    rho = np.array([[0.75, 0.0], [0.0, 0.25]], dtype=np.complex128)
    hamiltonian = np.diag([2.0, -2.0]).astype(np.complex128)
    assert density_matrix_energy_expectation(rho, hamiltonian) == pytest.approx(1.0, abs=2e-12)
    with pytest.raises(ValueError, match="disagrees"):
        density_matrix_energy_expectation(rho, np.eye(4, dtype=np.complex128))
    with pytest.raises(ValueError, match="not Hermitian"):
        density_matrix_energy_expectation(
            rho,
            np.array([[0.0, 1.0], [0.0, 0.0]], dtype=np.complex128),
        )


def test_noise_dependency_failure_is_explicit_when_aer_is_not_installed() -> None:
    if importlib.util.find_spec("qiskit_aer") is None:
        with pytest.raises(QiskitNoiseDependencyError, match="qiskit-aer"):
            build_compact_washington_device(2)


def test_protocol_native_operation_validation_fails_closed() -> None:
    _validate_protocol_native_operation_counts(
        {"rz": 2, "sx": 1, "x": 1, "cx": 3, "barrier": 1}
    )
    for operation in ("unitary", "swap", "h", "cz", "measure", "reset", "delay"):
        with pytest.raises(ValueError, match="opaque, non-native"):
            _validate_protocol_native_operation_counts({operation: 1})
    with pytest.raises(ValueError, match="compact target"):
        _validate_protocol_native_operation_counts(
            {"cx": 1},
            target_operation_names=("rz", "sx", "x"),
        )


def test_layout_audit_only_uses_identity_fallback_when_layout_is_absent() -> None:
    assert _layout_indices(SimpleNamespace(num_qubits=2, layout=None)) == ((0, 1), (0, 1))
    with pytest.raises(ArithmeticError, match="Unable to audit"):
        _layout_indices(SimpleNamespace(num_qubits=2, layout=object()))

    malformed = SimpleNamespace(
        initial_index_layout=lambda **_: [0, 0],
        final_index_layout=lambda **_: [0, 1],
    )
    with pytest.raises(ArithmeticError, match="not a permutation"):
        _layout_indices(SimpleNamespace(num_qubits=2, layout=malformed))

    audited = SimpleNamespace(
        initial_index_layout=lambda **_: [1, 0],
        final_index_layout=lambda **_: [0, 1],
    )
    assert _layout_indices(SimpleNamespace(num_qubits=2, layout=audited)) == (
        (1, 0),
        (0, 1),
    )

    circuit = SimpleNamespace(
        num_qubits=2,
        layout=audited,
        count_ops=lambda: {"x": 1},
        depth=lambda: 1,
        size=lambda: 1,
    )
    compilation = NativeCompilation(
        circuit=circuit,
        gate_counts=(("x", 1),),
        depth=1,
        size=1,
        swap_count=0,
        unitary_count=0,
        initial_index_layout=(1, 0),
        final_index_layout=(0, 1),
        optimization_level=0,
        seed_transpiler=1,
    )
    with pytest.raises(ArithmeticError, match="identity compact-wire layouts"):
        _validate_native_compilation_ledger(compilation, device=None)


def test_noise_manifest_embeds_complete_compact_device_provenance() -> None:
    snapshot_sha = "a" * 64
    device = CompactWashingtonDevice(
        source_backend_name="fake_washington",
        source_backend_class="test.FakeWashingtonV2",
        source_num_qubits=127,
        physical_path=(7, 8),
        path_selection_policy="test_path",
        two_qubit_gate="cx",
        target=SimpleNamespace(operation_names=("rz", "sx", "x", "cx")),
        noise_model=object(),
        qubit_calibrations=(
            QubitCalibrationAudit(0, 7, 1e-4, 8e-5, 5e9),
            QubitCalibrationAudit(1, 8, 1.1e-4, 9e-5, 5.1e9),
        ),
        instruction_calibrations=(
            InstructionCalibrationAudit("cx", (0, 1), (7, 8), 3e-7, 0.01),
        ),
        include_gate_error=True,
        include_thermal_relaxation=True,
        include_readout_error=False,
        temperature_millikelvin=0.0,
        snapshot_sha256=snapshot_sha,
    )
    diagnostics = CompilationDiagnostics(
        num_qubits=2,
        num_layers=1,
        num_logical_u4=1,
        logical_replay_fidelity=1.0,
        native_replay_fidelity=1.0,
        compile_fidelity=1.0,
        logical_phase_over_dense=0.0,
        native_phase_over_logical=0.0,
        logical_phase_aligned_max_abs_error=0.0,
        native_phase_aligned_max_abs_error=0.0,
        native_operation_counts=(("cx", 1),),
        native_basis_only=True,
    )
    evaluation = StaircaseNativeNoiseEvaluation(
        block_synthesis=diagnostics,
        target_compilation=SimpleNamespace(provenance_dict=lambda: {"depth": 1}),
        noisy=SimpleNamespace(
            as_dict=lambda: {"hardware": {"backend_snapshot_sha256": snapshot_sha}}
        ),
        compact_device=device,
        logical_circuit=object(),
        block_native_circuit=object(),
    )

    payload = evaluation.as_dict()
    provenance = payload["compact_washington_device"]
    assert payload["schema_version"] == "zhou2021-staircase-native-noise-evaluation-v2"
    assert provenance["source_backend_class"] == "test.FakeWashingtonV2"
    assert provenance["snapshot_sha256"] == snapshot_sha
    assert provenance["noise_options"] == {
        "gate_error": True,
        "thermal_relaxation": True,
        "readout_error": False,
        "temperature_millikelvin": 0.0,
    }
    assert len(provenance["qubit_calibrations"]) == 2
    assert len(provenance["instruction_calibrations"]) == 1
    assert (
        provenance["snapshot_sha256"]
        == payload["noise_evaluation"]["hardware"]["backend_snapshot_sha256"]
    )


@pytest.mark.qiskit
def test_fake_washington_compaction_and_noisy_staircase_protocol() -> None:
    pytest.importorskip("qiskit")
    pytest.importorskip("qiskit_aer")
    if importlib.util.find_spec("qiskit_ibm_runtime") is None:
        pytest.skip("FakeWashingtonV2 requires qiskit-ibm-runtime")

    model = LatentStaircase(2)
    model.add_layer(initialization="identity_perturbed", seed=777)
    target = model().detach().cpu().numpy()
    evaluation = evaluate_staircase_under_washington(
        model,
        target,
        hamiltonian=np.zeros((4, 4), dtype=np.complex128),
        ground_energy=0.0,
        seed_transpiler=777,
        seed_simulator=777,
        max_density_matrix_qubits=2,
    )

    assert evaluation.block_synthesis.compile_fidelity == pytest.approx(1.0, abs=2e-9)
    assert evaluation.noisy.compilation_fidelity == pytest.approx(1.0, abs=2e-9)
    assert evaluation.noisy.density.trace_error < 2e-8
    assert evaluation.noisy.density.hermiticity_residual < 2e-8
    assert evaluation.noisy.density.minimum_eigenvalue is not None
    assert evaluation.noisy.density.minimum_eigenvalue >= -2e-8
    assert 0.0 <= evaluation.noisy.noisy_target_fidelity <= 1.0 + 2e-8
    assert evaluation.noisy.noisy_energy == pytest.approx(0.0, abs=2e-10)
    assert evaluation.noisy.energy_error_per_site == pytest.approx(0.0, abs=2e-10)
    assert evaluation.noisy.physical_path is not None
    assert len(evaluation.noisy.physical_path) == 2
    assert (
        evaluation.noisy.path_selection_policy
        == "deterministic_lexicographically_first_simple_path"
    )
    assert evaluation.noisy.qiskit_ibm_runtime_version is not None


@pytest.mark.qiskit
def test_fake_washington_evaluates_frozen_ordered_u4_archive_without_latents() -> None:
    pytest.importorskip("qiskit")
    pytest.importorskip("qiskit_aer")
    if importlib.util.find_spec("qiskit_ibm_runtime") is None:
        pytest.skip("FakeWashingtonV2 requires qiskit-ibm-runtime")

    model = LatentStaircase(2)
    model.add_layer(initialization="identity_perturbed", seed=778)
    target = model().detach().cpu().numpy()
    archive = ordered_u4_archive_from_staircase(model)
    evaluation = evaluate_ordered_u4_archive_under_washington(
        archive,
        target,
        hamiltonian=np.zeros((4, 4), dtype=np.complex128),
        ground_energy=0.0,
        seed_transpiler=778,
        seed_simulator=778,
        max_density_matrix_qubits=2,
    )

    assert evaluation.block_synthesis.logical_replay_fidelity == pytest.approx(1.0, abs=2e-9)
    assert evaluation.block_synthesis.native_replay_fidelity == pytest.approx(1.0, abs=2e-9)
    assert evaluation.noisy.compilation_fidelity == pytest.approx(1.0, abs=2e-9)
    assert evaluation.noisy.density.trace_error < 2e-8
    assert evaluation.noisy.density.hermiticity_residual < 2e-8
    assert evaluation.noisy.physical_path is not None
    assert len(evaluation.noisy.physical_path) == 2


@pytest.mark.qiskit
def test_raw_qiskit_circuit_rejects_opaque_and_non_native_operations() -> None:
    qiskit = pytest.importorskip("qiskit")
    pytest.importorskip("qiskit_aer")
    target = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.complex128)

    non_native = qiskit.QuantumCircuit(2)
    non_native.h(0)
    with pytest.raises(ValueError, match="opaque, non-native"):
        evaluate_native_density_matrix(target, target, non_native, max_density_matrix_qubits=2)

    opaque = qiskit.QuantumCircuit(2)
    opaque.unitary(np.eye(4, dtype=np.complex128), [0, 1])
    with pytest.raises(ValueError, match="opaque, non-native"):
        evaluate_native_density_matrix(target, target, opaque, max_density_matrix_qubits=2)

    native = qiskit.QuantumCircuit(2)
    native.x(0)
    with pytest.raises(ValueError, match="num_sites disagrees"):
        evaluate_native_density_matrix(
            target,
            target,
            native,
            hamiltonian=np.eye(4, dtype=np.complex128),
            ground_energy=0.0,
            num_sites=3,
            max_density_matrix_qubits=2,
        )
    compilation = _native_compilation_from_circuit(native)
    tampered = replace(compilation, depth=compilation.depth + 1)
    with pytest.raises(ArithmeticError, match="depth ledger disagrees"):
        evaluate_native_density_matrix(target, target, tampered, max_density_matrix_qubits=2)


@pytest.mark.qiskit
def test_noiseless_aer_limit_preserves_nonzero_hamiltonian_energy() -> None:
    pytest.importorskip("qiskit")
    pytest.importorskip("qiskit_aer")
    model = LatentStaircase(2)
    model.add_layer(initialization="random", seed=779)
    target = model().detach().cpu().numpy()
    native = compile_staircase_native(model)
    hamiltonian = heisenberg_hamiltonian(2)
    reference_energy = energy_expectation(target, hamiltonian)

    evaluation = evaluate_native_density_matrix(
        target,
        target,
        native,
        hamiltonian=hamiltonian,
        ground_energy=reference_energy,
        num_sites=2,
        max_density_matrix_qubits=2,
        seed_simulator=779,
    )

    assert evaluation.logical_target_fidelity == pytest.approx(1.0, abs=2e-10)
    assert evaluation.compilation_fidelity == pytest.approx(1.0, abs=2e-9)
    assert evaluation.noisy_target_fidelity == pytest.approx(1.0, abs=2e-9)
    assert evaluation.noisy_energy == pytest.approx(reference_energy, abs=2e-9)
    assert evaluation.energy_error_per_site == pytest.approx(0.0, abs=2e-9)
