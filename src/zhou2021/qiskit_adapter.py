"""Convention-safe Qiskit translation for Zhou's serial U(4) ADQC.

The differentiable implementation uses a ``q0``-most-significant tensor
convention.  Qiskit stores qubit zero as the least-significant subsystem of a
statevector.  This adapter keeps the numerical statevector order unchanged by
mapping project site ``i`` to Qiskit qubit ``N - 1 - i``.  A Zhou U(4) matrix
is defined in the ordered local basis ``|left,right>``; its Qiskit qargs must
therefore be supplied in the reverse mapped order.

Qiskit is intentionally optional.  Importing this module does not import it,
so the paper's noiseless PyTorch reproduction remains usable without the
hardware-evaluation extras.
"""

from __future__ import annotations

import importlib
import numbers
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .circuit import LatentStaircase
from .conventions import validate_num_qubits

ComplexArray = NDArray[np.complex128]
NATIVE_BASIS = frozenset({"rz", "sx", "x", "cx"})


class QiskitUnavailableError(ImportError):
    """Raised when a caller explicitly requests an optional Qiskit feature."""


@dataclass(frozen=True, slots=True)
class StaircaseGateRecord:
    """One ordered projected U(4) gate frozen from a Zhou staircase."""

    layer: int
    position: int
    left_site: int
    matrix: ComplexArray


@dataclass(frozen=True, slots=True)
class CompilationDiagnostics:
    """Independent ideal-replay checks for a Zhou serial staircase."""

    num_qubits: int
    num_layers: int
    num_logical_u4: int
    logical_replay_fidelity: float
    native_replay_fidelity: float
    compile_fidelity: float
    logical_phase_over_dense: float
    native_phase_over_logical: float
    logical_phase_aligned_max_abs_error: float
    native_phase_aligned_max_abs_error: float
    native_operation_counts: tuple[tuple[str, int], ...]
    native_basis_only: bool


def qiskit_available() -> bool:
    """Return whether a *working* Qiskit import is available.

    ``find_spec`` alone is deliberately not used: mixed pre-/post-Qiskit-1.0
    installations can be discoverable but fail during import, and should be
    reported as unavailable rather than producing a misleading later error.
    """

    try:
        importlib.import_module("qiskit")
    except (ImportError, ModuleNotFoundError):
        return False
    return True


def require_qiskit() -> Any:
    """Import Qiskit or raise one actionable optional-dependency error."""

    try:
        return importlib.import_module("qiskit")
    except (ImportError, ModuleNotFoundError) as exc:
        raise QiskitUnavailableError(
            "Qiskit is required for Zhou circuit translation/native synthesis. "
            "Install the optional dependency group with `pip install -e '.[noise]'` "
            "in a clean Python 3.11 or 3.12 environment."
        ) from exc


def _validated_site(site: int, *, num_qubits: int) -> int:
    if isinstance(site, bool) or not isinstance(site, numbers.Integral):
        raise TypeError("site must be an integer.")
    site = int(site)
    if not 0 <= site < num_qubits:
        raise ValueError(f"site={site} is outside an N={num_qubits} register.")
    return site


def project_site_to_qiskit(site: int, *, num_qubits: int) -> int:
    """Map one q0-MSB Zhou site to a Qiskit little-endian qubit index."""

    num_qubits = validate_num_qubits(num_qubits)
    return num_qubits - 1 - _validated_site(site, num_qubits=num_qubits)


def project_adjacent_sites_to_qiskit_qargs(
    left_site: int,
    *,
    num_qubits: int,
) -> tuple[int, int]:
    """Map ordered ``(left,left+1)`` sites while preserving the U(4) basis.

    Qiskit's first qarg is its locally least-significant subsystem.  Returning
    the project sites in reverse mapped order makes an input U(4) represented
    in ``|left,right>`` act on exactly the same statevector coordinates.
    """

    num_qubits = validate_num_qubits(num_qubits)
    left_site = _validated_site(left_site, num_qubits=num_qubits)
    if left_site >= num_qubits - 1:
        raise ValueError(f"left_site={left_site} has no right neighbour for N={num_qubits}.")
    return (
        project_site_to_qiskit(left_site + 1, num_qubits=num_qubits),
        project_site_to_qiskit(left_site, num_qubits=num_qubits),
    )


def _validated_u4(matrix: ArrayLike, *, name: str = "matrix") -> ComplexArray:
    value = np.asarray(matrix, dtype=np.complex128)
    if value.shape != (4, 4):
        raise ValueError(f"{name} must have shape (4, 4), got {value.shape}.")
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{name} contains non-finite entries.")
    residual = float(np.linalg.norm(value.conj().T @ value - np.eye(4), ord=np.inf))
    if residual > 2e-10:
        raise ValueError(f"{name} is not unitary; residual={residual:.3e}.")
    return np.array(value, dtype=np.complex128, copy=True)


def staircase_gate_records(staircase: LatentStaircase) -> tuple[StaircaseGateRecord, ...]:
    """Freeze projected gates in Zhou's physical layer-major serial order."""

    if not isinstance(staircase, LatentStaircase):
        raise TypeError("staircase must be a zhou2021.circuit.LatentStaircase.")
    records: list[StaircaseGateRecord] = []
    for gate in staircase.iter_projected_gates():
        matrix = _validated_u4(
            gate.matrix.detach().cpu().numpy(),
            name=f"projected U4 at layer={gate.layer}, position={gate.position}",
        )
        records.append(
            StaircaseGateRecord(
                layer=gate.layer,
                position=gate.position,
                left_site=gate.left_site,
                matrix=matrix,
            )
        )
    expected = staircase.num_layers * staircase.gates_per_layer
    if len(records) != expected:
        raise ArithmeticError(
            f"Serial staircase produced {len(records)} gates, expected {expected}."
        )
    return tuple(records)


def _validated_gate_records(
    gate_records: Iterable[StaircaseGateRecord],
    *,
    num_qubits: int,
) -> tuple[StaircaseGateRecord, ...]:
    num_qubits = validate_num_qubits(num_qubits)
    if num_qubits < 2:
        raise ValueError("Zhou serial U(4) staircase records require num_qubits >= 2.")
    records = tuple(gate_records)
    gates_per_layer = num_qubits - 1
    if len(records) % gates_per_layer != 0:
        raise ValueError(
            "Zhou serial staircase records must contain an integer multiple of N-1 gates."
        )
    for index, record in enumerate(records):
        if not isinstance(record, StaircaseGateRecord):
            raise TypeError(
                "gate_records must contain StaircaseGateRecord values; "
                f"element {index} is {type(record).__name__}."
            )
        _validated_u4(record.matrix, name=f"gate_records[{index}].matrix")
        expected_layer, expected_position = divmod(index, gates_per_layer)
        if (
            record.layer != expected_layer
            or record.position != expected_position
            or record.left_site != expected_position
        ):
            raise ValueError(
                "Zhou serial staircase records must be exact layer-major order; "
                f"index {index} requires (layer={expected_layer}, "
                f"position={expected_position}, left_site={expected_position})."
            )
    return records


def to_qiskit_logical(
    staircase: LatentStaircase,
    *,
    insert_barriers: bool = False,
) -> Any:
    """Translate projected Zhou U(4) gates into exact Qiskit ``unitary`` blocks."""

    if not isinstance(insert_barriers, bool):
        raise TypeError("insert_barriers must be bool.")
    if not isinstance(staircase, LatentStaircase):
        raise TypeError("staircase must be a zhou2021.circuit.LatentStaircase.")
    records = staircase_gate_records(staircase)
    require_qiskit()
    from qiskit import QuantumCircuit

    circuit = QuantumCircuit(staircase.num_qubits, name="zhou2021_logical")
    for index, record in enumerate(records):
        qargs = project_adjacent_sites_to_qiskit_qargs(
            record.left_site,
            num_qubits=staircase.num_qubits,
        )
        circuit.unitary(
            record.matrix,
            list(qargs),
            label=f"U4_L{record.layer}_P{record.position}",
        )
        if insert_barriers and index + 1 < len(records):
            circuit.barrier()
    return circuit


def to_qiskit_logical_from_records(
    gate_records: Iterable[StaircaseGateRecord],
    *,
    num_qubits: int,
    insert_barriers: bool = False,
) -> Any:
    """Translate an explicitly frozen sequence of Zhou U(4) records.

    This is useful for provenance-stable replay after a PyTorch training
    process has ended.  It retains the same q0-MSB mapping as
    :func:`to_qiskit_logical` and intentionally accepts no arbitrary topology.
    """

    if not isinstance(insert_barriers, bool):
        raise TypeError("insert_barriers must be bool.")
    num_qubits = validate_num_qubits(num_qubits)
    records = _validated_gate_records(gate_records, num_qubits=num_qubits)
    require_qiskit()
    from qiskit import QuantumCircuit

    circuit = QuantumCircuit(num_qubits, name="zhou2021_logical_frozen")
    for index, record in enumerate(records):
        circuit.unitary(
            record.matrix,
            list(
                project_adjacent_sites_to_qiskit_qargs(
                    record.left_site,
                    num_qubits=num_qubits,
                )
            ),
            label=f"U4_L{record.layer}_P{record.position}",
        )
        if insert_barriers and index + 1 < len(records):
            circuit.barrier()
    return circuit


def compile_gate_records_native(
    gate_records: Iterable[StaircaseGateRecord],
    *,
    num_qubits: int,
    insert_barriers: bool = False,
    remove_inserted_barriers: bool = True,
) -> Any:
    """Exactly synthesize frozen serial U(4) records into ``rz/sx/x/cx``.

    Unlike :func:`compile_staircase_native`, this function requires no live
    PyTorch latent parameters.  It is the provenance-safe bridge used when a
    separately saved ordered-U(4) archive is later handed to Qiskit.  The
    strict record validator preserves the same layer-major Zhou execution
    order, q0-MSB convention and local U(4) basis as the live staircase path.
    """

    if not isinstance(insert_barriers, bool):
        raise TypeError("insert_barriers must be bool.")
    if not isinstance(remove_inserted_barriers, bool):
        raise TypeError("remove_inserted_barriers must be bool.")
    num_qubits = validate_num_qubits(num_qubits)
    records = _validated_gate_records(gate_records, num_qubits=num_qubits)
    require_qiskit()
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import CXGate
    from qiskit.synthesis import TwoQubitBasisDecomposer

    decomposer = TwoQubitBasisDecomposer(
        CXGate(),
        basis_fidelity=1.0,
        euler_basis="ZSX",
        pulse_optimize=False,
    )
    circuit = QuantumCircuit(num_qubits, name="zhou2021_native_frozen_blocks")
    for index, record in enumerate(records):
        block = decomposer(record.matrix, basis_fidelity=1.0, approximate=False)
        qargs = project_adjacent_sites_to_qiskit_qargs(
            record.left_site,
            num_qubits=num_qubits,
        )
        circuit.compose(block, list(qargs), inplace=True)
        if insert_barriers and index + 1 < len(records):
            circuit.barrier()
    if insert_barriers and remove_inserted_barriers:
        circuit = remove_barriers(circuit)

    unexpected = set(circuit.count_ops()) - NATIVE_BASIS - {"barrier"}
    if unexpected:
        joined = ", ".join(sorted(unexpected))
        raise RuntimeError(f"Exact U(4) synthesis emitted non-native operations: {joined}.")
    return circuit


def remove_barriers(qiskit_circuit: Any) -> Any:
    """Return a barrier-free copy while preserving the global phase."""

    require_qiskit()
    from qiskit import QuantumCircuit
    from qiskit.transpiler.passes import RemoveBarriers

    if not isinstance(qiskit_circuit, QuantumCircuit):
        raise TypeError("qiskit_circuit must be a qiskit.QuantumCircuit.")
    return RemoveBarriers()(qiskit_circuit.copy())


def compile_staircase_native(
    staircase: LatentStaircase,
    *,
    insert_barriers: bool = False,
    remove_inserted_barriers: bool = True,
) -> Any:
    """Exactly synthesize every serial Zhou U(4) into ``rz/sx/x/cx``.

    Each U(4) is decomposed independently in physical execution order.  This
    avoids a black-box whole-circuit approximation and leaves a direct
    block-synthesis fidelity oracle for the protocol module.
    """

    if not isinstance(staircase, LatentStaircase):
        raise TypeError("staircase must be a zhou2021.circuit.LatentStaircase.")
    return compile_gate_records_native(
        staircase_gate_records(staircase),
        num_qubits=staircase.num_qubits,
        insert_barriers=insert_barriers,
        remove_inserted_barriers=remove_inserted_barriers,
    )


def qiskit_statevector_q0_msb(qiskit_circuit: Any) -> ComplexArray:
    """Return ideal Qiskit statevector in the project's direct q0-MSB order."""

    require_qiskit()
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector

    if not isinstance(qiskit_circuit, QuantumCircuit):
        raise TypeError("qiskit_circuit must be a qiskit.QuantumCircuit.")
    # Site reversal in ``to_qiskit_logical`` makes Qiskit's little-endian
    # numeric array coincide with the q0-MSB project's array indexing.
    return np.array(
        Statevector.from_instruction(qiskit_circuit).data,
        dtype=np.complex128,
        copy=True,
    )


def statevector_fidelity(reference: ArrayLike, candidate: ArrayLike) -> float:
    """Return normalized squared pure-state fidelity without hidden clipping."""

    left = np.asarray(reference, dtype=np.complex128).reshape(-1)
    right = np.asarray(candidate, dtype=np.complex128).reshape(-1)
    if left.shape != right.shape:
        raise ValueError(f"Statevectors disagree in shape: {left.shape} != {right.shape}.")
    if left.size == 0:
        raise ValueError("Statevectors must not be empty.")
    if not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
        raise ValueError("Statevectors must contain finite entries.")
    left_norm_sq = float(np.vdot(left, left).real)
    right_norm_sq = float(np.vdot(right, right).real)
    if left_norm_sq <= 0.0 or right_norm_sq <= 0.0:
        raise ValueError("Statevectors must have nonzero norm.")
    overlap = complex(np.vdot(left, right))
    value = (overlap.real * overlap.real + overlap.imag * overlap.imag) / (
        left_norm_sq * right_norm_sq
    )
    if value < -2e-12 or value > 1.0 + 2e-10:
        raise ArithmeticError(f"Invalid normalized state fidelity {value:.17g}.")
    return float(value)


def _phase_aligned_comparison(reference: ArrayLike, candidate: ArrayLike) -> tuple[float, float]:
    left = np.asarray(reference, dtype=np.complex128).reshape(-1)
    right = np.asarray(candidate, dtype=np.complex128).reshape(-1)
    if left.shape != right.shape or left.size == 0:
        raise ValueError("Phase comparison requires equally sized nonempty statevectors.")
    left = left / np.linalg.norm(left)
    right = right / np.linalg.norm(right)
    overlap = complex(np.vdot(left, right))
    phase = float(np.angle(overlap)) if abs(overlap) > 0.0 else 0.0
    aligned = right * np.exp(-1j * phase)
    return phase, float(np.max(np.abs(left - aligned)))


def diagnose_gate_records_compilation(
    gate_records: Iterable[StaircaseGateRecord],
    *,
    num_qubits: int,
    dense_state: ArrayLike,
    native_circuit: Any | None = None,
) -> CompilationDiagnostics:
    """Cross-check a frozen ordered-U(4) circuit against a dense replay.

    This is the archived-circuit counterpart of
    :func:`diagnose_staircase_compilation`.  It deliberately requires the
    caller to provide an independently obtained q0-MSB dense state (for
    example the NumPy replay of a validated archive), so the audit does not
    reconstruct or trust latent PyTorch parameters after training has ended.
    """

    num_qubits = validate_num_qubits(num_qubits)
    records = _validated_gate_records(gate_records, num_qubits=num_qubits)
    logical_circuit = to_qiskit_logical_from_records(records, num_qubits=num_qubits)
    if native_circuit is None:
        native_circuit = compile_gate_records_native(records, num_qubits=num_qubits)
    require_qiskit()
    from qiskit import QuantumCircuit

    if not isinstance(native_circuit, QuantumCircuit):
        raise TypeError("native_circuit must be a qiskit.QuantumCircuit.")
    if native_circuit.num_qubits != num_qubits:
        raise ValueError("native_circuit qubit count disagrees with frozen gate records.")
    dense = np.asarray(dense_state, dtype=np.complex128).reshape(-1)
    logical_state = qiskit_statevector_q0_msb(logical_circuit)
    native_state = qiskit_statevector_q0_msb(native_circuit)
    # Statevector-fidelity validates dimension, finiteness and nonzero norm
    # before the stricter phase-aligned comparison below.
    logical_fidelity = statevector_fidelity(dense, logical_state)
    native_fidelity = statevector_fidelity(dense, native_state)
    compile_fidelity = statevector_fidelity(logical_state, native_state)
    logical_phase, logical_error = _phase_aligned_comparison(dense, logical_state)
    native_phase, native_error = _phase_aligned_comparison(logical_state, native_state)
    counts = tuple(
        sorted((str(name), int(count)) for name, count in native_circuit.count_ops().items())
    )
    names = {name for name, _ in counts}
    return CompilationDiagnostics(
        num_qubits=num_qubits,
        num_layers=len(records) // (num_qubits - 1),
        num_logical_u4=len(records),
        logical_replay_fidelity=logical_fidelity,
        native_replay_fidelity=native_fidelity,
        compile_fidelity=compile_fidelity,
        logical_phase_over_dense=logical_phase,
        native_phase_over_logical=native_phase,
        logical_phase_aligned_max_abs_error=logical_error,
        native_phase_aligned_max_abs_error=native_error,
        native_operation_counts=counts,
        native_basis_only=(names - {"barrier"}) <= NATIVE_BASIS,
    )


def diagnose_staircase_compilation(
    staircase: LatentStaircase,
    native_circuit: Any | None = None,
) -> CompilationDiagnostics:
    """Compare PyTorch, logical-Qiskit, and native-Qiskit ideal replays."""

    if not isinstance(staircase, LatentStaircase):
        raise TypeError("staircase must be a zhou2021.circuit.LatentStaircase.")
    return diagnose_gate_records_compilation(
        staircase_gate_records(staircase),
        num_qubits=staircase.num_qubits,
        dense_state=staircase().detach().cpu().numpy(),
        native_circuit=native_circuit,
    )


__all__ = [
    "CompilationDiagnostics",
    "NATIVE_BASIS",
    "QiskitUnavailableError",
    "StaircaseGateRecord",
    "compile_gate_records_native",
    "compile_staircase_native",
    "diagnose_gate_records_compilation",
    "diagnose_staircase_compilation",
    "project_adjacent_sites_to_qiskit_qargs",
    "project_site_to_qiskit",
    "qiskit_available",
    "qiskit_statevector_q0_msb",
    "remove_barriers",
    "require_qiskit",
    "staircase_gate_records",
    "statevector_fidelity",
    "to_qiskit_logical",
    "to_qiskit_logical_from_records",
]
