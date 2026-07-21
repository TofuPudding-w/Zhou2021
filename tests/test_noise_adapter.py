"""Qiskit-bridge tests, with optional executable Qiskit checks."""

from __future__ import annotations

import inspect

import numpy as np
import pytest
from numpy.testing import assert_allclose

from zhou2021.circuit import LatentStaircase
from zhou2021.qiskit_adapter import (
    QiskitUnavailableError,
    StaircaseGateRecord,
    compile_gate_records_native,
    compile_staircase_native,
    diagnose_gate_records_compilation,
    diagnose_staircase_compilation,
    project_adjacent_sites_to_qiskit_qargs,
    project_site_to_qiskit,
    qiskit_available,
    qiskit_statevector_q0_msb,
    require_qiskit,
    staircase_gate_records,
    statevector_fidelity,
    to_qiskit_logical,
    to_qiskit_logical_from_records,
)


def _staircase() -> LatentStaircase:
    model = LatentStaircase(3)
    model.add_layer(initialization="random", seed=101)
    model.add_layer(initialization="identity_perturbed", seed=102)
    return model


def test_q0_msb_to_qiskit_mapping_and_u4_qarg_order_are_explicit() -> None:
    assert [project_site_to_qiskit(site, num_qubits=4) for site in range(4)] == [3, 2, 1, 0]
    # Project U4 basis is |site=1,site=2>; Qiskit qargs must intentionally be
    # (mapped site 2, mapped site 1), not merely the mapped site order.
    assert project_adjacent_sites_to_qiskit_qargs(1, num_qubits=4) == (1, 2)
    with pytest.raises(ValueError, match="right neighbour"):
        project_adjacent_sites_to_qiskit_qargs(3, num_qubits=4)


def test_frozen_staircase_gate_records_have_the_paper_serial_order_and_unitaries() -> None:
    records = staircase_gate_records(_staircase())
    assert [(item.layer, item.position, item.left_site) for item in records] == [
        (0, 0, 0),
        (0, 1, 1),
        (1, 0, 0),
        (1, 1, 1),
    ]
    for record in records:
        assert_allclose(record.matrix.conj().T @ record.matrix, np.eye(4), atol=3e-12)


def test_frozen_records_reject_nonserial_order_before_optional_qiskit_import() -> None:
    records = list(staircase_gate_records(_staircase()))
    records[0] = StaircaseGateRecord(
        layer=0,
        position=1,
        left_site=1,
        matrix=records[0].matrix,
    )
    with pytest.raises(ValueError, match="exact layer-major order"):
        to_qiskit_logical_from_records(records, num_qubits=3)


def test_statevector_fidelity_is_phase_invariant_and_validates_input() -> None:
    reference = np.array([1.0, 1j], dtype=np.complex128) / np.sqrt(2.0)
    candidate = np.exp(0.31j) * reference
    assert statevector_fidelity(reference, candidate) == pytest.approx(1.0, abs=2e-12)
    with pytest.raises(ValueError, match="shape"):
        statevector_fidelity(reference, np.ones(4, dtype=np.complex128) / 2.0)


def test_missing_or_invalid_qiskit_installation_has_an_actionable_error() -> None:
    if not qiskit_available():
        with pytest.raises(QiskitUnavailableError, match="optional dependency"):
            require_qiskit()


def test_native_compile_api_exposes_no_unused_simplify_coordinate() -> None:
    assert "simplify" not in inspect.signature(compile_gate_records_native).parameters
    assert "simplify" not in inspect.signature(compile_staircase_native).parameters


@pytest.mark.qiskit
@pytest.mark.skipif(
    not qiskit_available(),
    reason="requires a valid optional Qiskit installation",
)
def test_logical_qiskit_replay_matches_pytorch_q0_msb_statevector() -> None:
    model = _staircase()
    circuit = to_qiskit_logical(model)
    observed = qiskit_statevector_q0_msb(circuit)
    expected = model().detach().cpu().numpy()

    assert statevector_fidelity(expected, observed) == pytest.approx(1.0, abs=2e-11)
    # This direct component comparison is stronger than fidelity and detects a
    # hidden bit reversal despite global phase being fixed by the logical path.
    overlap = np.vdot(expected, observed)
    observed *= np.exp(-1j * np.angle(overlap))
    assert_allclose(observed, expected, atol=3e-11)


@pytest.mark.qiskit
@pytest.mark.skipif(
    not qiskit_available(),
    reason="requires a valid optional Qiskit installation",
)
def test_exact_u4_native_synthesis_and_frozen_record_replay_are_faithful() -> None:
    model = _staircase()
    logical = to_qiskit_logical(model)
    frozen = to_qiskit_logical_from_records(
        staircase_gate_records(model),
        num_qubits=model.num_qubits,
    )
    native = compile_staircase_native(model)
    diagnostics = diagnose_staircase_compilation(model, native)

    assert statevector_fidelity(
        qiskit_statevector_q0_msb(logical), qiskit_statevector_q0_msb(frozen)
    ) == pytest.approx(1.0, abs=2e-11)
    assert diagnostics.logical_replay_fidelity == pytest.approx(1.0, abs=2e-10)
    assert diagnostics.native_replay_fidelity == pytest.approx(1.0, abs=2e-10)
    assert diagnostics.compile_fidelity == pytest.approx(1.0, abs=2e-10)
    assert diagnostics.native_basis_only
    assert {name for name, _ in diagnostics.native_operation_counts} <= {"rz", "sx", "x", "cx"}


@pytest.mark.qiskit
@pytest.mark.skipif(
    not qiskit_available(),
    reason="requires a valid optional Qiskit installation",
)
def test_frozen_record_native_compilation_matches_independent_dense_replay() -> None:
    model = _staircase()
    records = staircase_gate_records(model)
    native = compile_gate_records_native(records, num_qubits=model.num_qubits)
    diagnostics = diagnose_gate_records_compilation(
        records,
        num_qubits=model.num_qubits,
        dense_state=model().detach().cpu().numpy(),
        native_circuit=native,
    )
    assert diagnostics.logical_replay_fidelity == pytest.approx(1.0, abs=2e-10)
    assert diagnostics.native_replay_fidelity == pytest.approx(1.0, abs=2e-10)
    assert diagnostics.compile_fidelity == pytest.approx(1.0, abs=2e-10)
    assert diagnostics.native_basis_only
