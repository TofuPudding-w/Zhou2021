"""Compact FakeWashingtonV2 density-matrix noise evaluation.

``FakeWashingtonV2`` exposes a 127-qubit backend.  Simulating a full 127-wire
density matrix would be both meaningless for a small ADQC preparation circuit
and computationally impossible.  This module instead retains calibration data
on one selected contiguous physical path, re-indexs that path to a compact
backend target, and derives an Aer noise model from the compact target.

This is a downstream hardware-noise extension, not part of Zhou et al. (2021).
It never changes the trained PyTorch ansatz or optimizes it against noise.
All Qiskit imports are lazy so optional hardware packages do not affect the
paper-method implementation.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import json
import math
import numbers
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .conventions import normalized_statevector, validate_num_qubits
from .models import validate_dense_hamiltonian
from .qiskit_adapter import statevector_fidelity

ComplexArray = NDArray[np.complex128]
COMPACT_DEVICE_SCHEMA = "zhou2021-compact-washington-v1"
NOISY_RESULT_SCHEMA = "zhou2021-native-density-matrix-v1"
DEFAULT_NATIVE_OPERATIONS = ("rz", "sx", "x", "cx")


class QiskitNoiseDependencyError(ImportError):
    """Raised when a caller explicitly requests unavailable noise dependencies."""


def _qiskit_dependencies() -> dict[str, Any]:
    """Load Qiskit/Aer objects required by this module or raise one clear error."""

    try:
        from qiskit.quantum_info import Statevector
        from qiskit.transpiler import InstructionProperties, Target
        from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
        from qiskit_aer import AerSimulator
        from qiskit_aer.noise import NoiseModel
    except (ImportError, ModuleNotFoundError) as exc:
        raise QiskitNoiseDependencyError(
            "FakeWashingtonV2 noise evaluation requires compatible `qiskit`, "
            "`qiskit-aer`, and `qiskit-ibm-runtime` packages. Install the Zhou "
            "optional noise extras in a clean environment: `pip install -e '.[noise]'`."
        ) from exc
    return {
        "AerSimulator": AerSimulator,
        "InstructionProperties": InstructionProperties,
        "NoiseModel": NoiseModel,
        "Statevector": Statevector,
        "Target": Target,
        "generate_preset_pass_manager": generate_preset_pass_manager,
    }


def load_fake_washington_v2() -> Any:
    """Load the real calibration-bearing ``FakeWashingtonV2`` provider object."""

    errors: list[BaseException] = []
    try:
        from qiskit_ibm_runtime.fake_provider import FakeWashingtonV2

        return FakeWashingtonV2()
    except (ImportError, ModuleNotFoundError, AttributeError) as exc:
        errors.append(exc)
    try:
        from qiskit.providers.fake_provider import FakeWashingtonV2

        return FakeWashingtonV2()
    except (ImportError, ModuleNotFoundError, AttributeError) as exc:
        errors.append(exc)
    raise QiskitNoiseDependencyError(
        "FakeWashingtonV2 is unavailable. Install a supported `qiskit-ibm-runtime` "
        "release providing qiskit_ibm_runtime.fake_provider.FakeWashingtonV2."
    ) from errors[-1]


def _backend_name(backend: Any) -> str:
    value = getattr(backend, "name", type(backend).__name__)
    return str(value() if callable(value) else value)


def _backend_num_qubits(backend: Any) -> int:
    value = getattr(backend, "num_qubits", None)
    if value is None:
        value = getattr(getattr(backend, "target", None), "num_qubits", None)
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise TypeError("backend must expose an integer num_qubits through BackendV2.")
    value = int(value)
    if value < 1:
        raise ValueError("backend.num_qubits must be positive.")
    return value


def _target_two_qubit_edges(target: Any, operation: str) -> set[tuple[int, int]]:
    if operation not in tuple(target.operation_names):
        raise ValueError(f"Backend Target does not support {operation!r}.")
    edges: set[tuple[int, int]] = set()
    for qargs in target[operation]:
        if qargs is not None and len(qargs) == 2:
            edges.add((int(qargs[0]), int(qargs[1])))
    if not edges:
        raise ValueError(f"Backend Target contains no two-qubit {operation!r} edges.")
    return edges


def validate_physical_path(
    backend: Any,
    physical_path: Iterable[int],
    *,
    two_qubit_gate: str = "cx",
) -> tuple[int, ...]:
    """Validate an undirected contiguous simple path in a BackendV2 target.

    Directional calibration entries are preserved in the compact target.  The
    path test only needs one CX direction per adjacent physical pair because
    Qiskit's direction-correction pass can implement the opposite direction
    without inserting a routing SWAP.
    """

    if not isinstance(two_qubit_gate, str) or not two_qubit_gate:
        raise TypeError("two_qubit_gate must be a nonempty instruction name.")
    try:
        raw_path = tuple(physical_path)
    except TypeError as exc:
        raise TypeError("physical_path must be an iterable of integer qubit indices.") from exc
    if not raw_path:
        raise ValueError("physical_path must contain at least one physical qubit.")
    if any(isinstance(item, bool) or not isinstance(item, numbers.Integral) for item in raw_path):
        raise TypeError("physical_path must contain integer qubit indices.")
    path = tuple(int(item) for item in raw_path)
    if len(set(path)) != len(path):
        raise ValueError("physical_path must not repeat a physical qubit.")
    total = _backend_num_qubits(backend)
    if any(item < 0 or item >= total for item in path):
        raise ValueError(f"physical_path={path} leaves backend range [0, {total - 1}].")
    if len(path) == 1:
        return path
    directed = _target_two_qubit_edges(backend.target, two_qubit_gate)
    undirected = {frozenset(edge) for edge in directed}
    for left, right in zip(path[:-1], path[1:], strict=True):
        if frozenset((left, right)) not in undirected:
            raise ValueError(
                f"physical_path is not contiguous: {two_qubit_gate}{(left, right)} "
                "is unsupported in both directions."
            )
    return path


def select_physical_path(
    backend: Any,
    num_qubits: int,
    *,
    two_qubit_gate: str = "cx",
) -> tuple[int, ...]:
    """Choose a deterministic lexicographically first simple backend path."""

    num_qubits = validate_num_qubits(num_qubits)
    available = _backend_num_qubits(backend)
    if num_qubits > available:
        raise ValueError(f"N={num_qubits} exceeds backend width {available}.")
    if num_qubits == 1:
        return (0,)
    directed = _target_two_qubit_edges(backend.target, two_qubit_gate)
    adjacency: dict[int, set[int]] = {index: set() for index in range(available)}
    for left, right in directed:
        adjacency[left].add(right)
        adjacency[right].add(left)

    def extend(path: tuple[int, ...]) -> tuple[int, ...] | None:
        if len(path) == num_qubits:
            return path
        for neighbour in sorted(adjacency[path[-1]]):
            if neighbour not in path:
                result = extend((*path, neighbour))
                if result is not None:
                    return result
        return None

    for start in range(available):
        result = extend((start,))
        if result is not None:
            return result
    raise ValueError(
        f"No simple {two_qubit_gate!r} path of length {num_qubits} exists on "
        f"backend {_backend_name(backend)!r}."
    )


def _finite_float_or_none(value: Any, *, label: str) -> float | None:
    if value is None:
        return None
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"Backend Target has non-finite {label}={value!r}.")
    return value


@dataclass(frozen=True, slots=True)
class QubitCalibrationAudit:
    """Calibration fields retained for one compact physical-wire mapping."""

    compact_qubit: int
    physical_qubit: int
    t1_seconds: float | None
    t2_seconds: float | None
    frequency_hz: float | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "compact_qubit": self.compact_qubit,
            "physical_qubit": self.physical_qubit,
            "t1_seconds": self.t1_seconds,
            "t2_seconds": self.t2_seconds,
            "frequency_hz": self.frequency_hz,
        }


@dataclass(frozen=True, slots=True)
class InstructionCalibrationAudit:
    """One operation calibration retained on the compact selected path."""

    name: str
    compact_qubits: tuple[int, ...]
    physical_qubits: tuple[int, ...]
    duration_seconds: float | None
    error: float | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "compact_qubits": list(self.compact_qubits),
            "physical_qubits": list(self.physical_qubits),
            "duration_seconds": self.duration_seconds,
            "error": self.error,
        }


class _CompactBackendV2View:
    """Minimal public BackendV2 surface required by ``NoiseModel.from_backend``."""

    version = 2

    def __init__(self, target: Any, name: str) -> None:
        self.target = target
        self.name = name

    @property
    def operation_names(self) -> list[str]:
        return list(self.target.operation_names)

    @property
    def dt(self) -> float | None:
        return self.target.dt

    @property
    def num_qubits(self) -> int:
        return int(self.target.num_qubits)


@dataclass(frozen=True, slots=True)
class CompactWashingtonDevice:
    """Compact calibration snapshot plus Aer noise model for one physical path."""

    source_backend_name: str
    source_backend_class: str
    source_num_qubits: int
    physical_path: tuple[int, ...]
    path_selection_policy: str
    two_qubit_gate: str
    target: Any = field(repr=False, compare=False)
    noise_model: Any = field(repr=False, compare=False)
    qubit_calibrations: tuple[QubitCalibrationAudit, ...]
    instruction_calibrations: tuple[InstructionCalibrationAudit, ...]
    include_gate_error: bool
    include_thermal_relaxation: bool
    include_readout_error: bool
    temperature_millikelvin: float
    snapshot_sha256: str

    @property
    def num_qubits(self) -> int:
        return len(self.physical_path)

    @property
    def compact_to_physical(self) -> tuple[int, ...]:
        return self.physical_path

    @property
    def project_site_to_physical_q0_msb(self) -> tuple[int, ...]:
        """Physical wire for each q0-MSB project site, in project-site order."""

        return tuple(reversed(self.physical_path))

    def provenance_dict(self) -> dict[str, Any]:
        return {
            "schema_version": COMPACT_DEVICE_SCHEMA,
            "source_backend_name": self.source_backend_name,
            "source_backend_class": self.source_backend_class,
            "source_num_qubits": self.source_num_qubits,
            "physical_path": list(self.physical_path),
            "path_selection_policy": self.path_selection_policy,
            "compact_to_physical": list(self.compact_to_physical),
            "project_site_to_physical_q0_msb": list(self.project_site_to_physical_q0_msb),
            "project_to_qiskit_convention": "site_i_to_compact_qubit_N_minus_1_minus_i",
            "two_qubit_gate": self.two_qubit_gate,
            "target_operation_names": sorted(self.target.operation_names),
            "noise_options": {
                "gate_error": self.include_gate_error,
                "thermal_relaxation": self.include_thermal_relaxation,
                "readout_error": self.include_readout_error,
                "temperature_millikelvin": self.temperature_millikelvin,
            },
            "qubit_calibrations": [item.as_dict() for item in self.qubit_calibrations],
            "instruction_calibrations": [item.as_dict() for item in self.instruction_calibrations],
            "snapshot_sha256": self.snapshot_sha256,
        }


def _compact_target_and_audit(
    backend: Any,
    path: tuple[int, ...],
) -> tuple[Any, tuple[QubitCalibrationAudit, ...], tuple[InstructionCalibrationAudit, ...]]:
    dependencies = _qiskit_dependencies()
    InstructionProperties = dependencies["InstructionProperties"]
    Target = dependencies["Target"]
    source = backend.target
    source_properties = source.qubit_properties
    if source_properties is None:
        compact_properties = None
        qubit_audit = tuple(
            QubitCalibrationAudit(index, physical, None, None, None)
            for index, physical in enumerate(path)
        )
    else:
        compact_properties = [copy.deepcopy(source_properties[physical]) for physical in path]
        qubit_audit = tuple(
            QubitCalibrationAudit(
                compact_qubit=index,
                physical_qubit=physical,
                t1_seconds=_finite_float_or_none(
                    getattr(source_properties[physical], "t1", None), label="T1"
                ),
                t2_seconds=_finite_float_or_none(
                    getattr(source_properties[physical], "t2", None), label="T2"
                ),
                frequency_hz=_finite_float_or_none(
                    getattr(source_properties[physical], "frequency", None), label="frequency"
                ),
            )
            for index, physical in enumerate(path)
        )
    target = Target(
        description=f"Compact path extracted from {_backend_name(backend)}",
        num_qubits=len(path),
        dt=source.dt,
        granularity=getattr(source, "granularity", 1),
        min_length=getattr(source, "min_length", 1),
        pulse_alignment=getattr(source, "pulse_alignment", 1),
        acquire_alignment=getattr(source, "acquire_alignment", 1),
        qubit_properties=compact_properties,
    )
    physical_to_compact = {physical: compact for compact, physical in enumerate(path)}
    path_edges = {frozenset((path[index], path[index + 1])) for index in range(len(path) - 1)}
    instruction_audit: list[InstructionCalibrationAudit] = []
    for name in source.operation_names:
        copied: dict[tuple[int, ...], Any] = {}
        for physical_qargs, properties in source[name].items():
            if physical_qargs is None:
                continue
            physical_qargs = tuple(int(qubit) for qubit in physical_qargs)
            if len(physical_qargs) == 1:
                if physical_qargs[0] not in physical_to_compact:
                    continue
            elif len(physical_qargs) == 2:
                if any(qubit not in physical_to_compact for qubit in physical_qargs):
                    continue
                if frozenset(physical_qargs) not in path_edges:
                    continue
            else:
                # The protocol needs only one- and two-qubit preparation operations.
                continue
            compact_qargs = tuple(physical_to_compact[qubit] for qubit in physical_qargs)
            duration = _finite_float_or_none(
                getattr(properties, "duration", None), label=f"{name} duration"
            )
            error = _finite_float_or_none(getattr(properties, "error", None), label=f"{name} error")
            copied[compact_qargs] = InstructionProperties(duration=duration, error=error)
            instruction_audit.append(
                InstructionCalibrationAudit(
                    name=str(name),
                    compact_qubits=compact_qargs,
                    physical_qubits=physical_qargs,
                    duration_seconds=duration,
                    error=error,
                )
            )
        if copied:
            target.add_instruction(
                copy.deepcopy(source.operation_from_name(name)),
                copied,
                name=name,
            )
    instruction_audit.sort(key=lambda item: (item.name, item.compact_qubits))
    return target, qubit_audit, tuple(instruction_audit)


def _snapshot_sha256(
    *,
    backend: Any,
    path: tuple[int, ...],
    two_qubit_gate: str,
    target: Any,
    qubits: tuple[QubitCalibrationAudit, ...],
    instructions: tuple[InstructionCalibrationAudit, ...],
    gate_error: bool,
    thermal_relaxation: bool,
    readout_error: bool,
    temperature_millikelvin: float,
) -> str:
    payload = {
        "schema_version": COMPACT_DEVICE_SCHEMA,
        "source_backend_name": _backend_name(backend),
        "source_backend_class": f"{type(backend).__module__}.{type(backend).__qualname__}",
        "source_num_qubits": _backend_num_qubits(backend),
        "physical_path": list(path),
        "two_qubit_gate": two_qubit_gate,
        "target_operation_names": sorted(target.operation_names),
        "noise_options": {
            "gate_error": gate_error,
            "thermal_relaxation": thermal_relaxation,
            "readout_error": readout_error,
            "temperature_millikelvin": temperature_millikelvin,
        },
        "qubit_calibrations": [item.as_dict() for item in qubits],
        "instruction_calibrations": [item.as_dict() for item in instructions],
    }
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def build_compact_washington_device(
    num_qubits: int | None = None,
    *,
    backend: Any | None = None,
    physical_path: Iterable[int] | None = None,
    two_qubit_gate: str = "cx",
    gate_error: bool = True,
    thermal_relaxation: bool = True,
    readout_error: bool = False,
    temperature_millikelvin: float = 0.0,
) -> CompactWashingtonDevice:
    """Build a calibration-preserving compact FakeWashingtonV2 noise device."""

    dependencies = _qiskit_dependencies()
    if backend is None:
        backend = load_fake_washington_v2()
    if not hasattr(backend, "target"):
        raise TypeError("backend must implement the BackendV2 Target interface.")
    if not isinstance(two_qubit_gate, str) or not two_qubit_gate:
        raise TypeError("two_qubit_gate must be a nonempty instruction name.")
    for name, value in {
        "gate_error": gate_error,
        "thermal_relaxation": thermal_relaxation,
        "readout_error": readout_error,
    }.items():
        if not isinstance(value, bool):
            raise TypeError(f"{name} must be bool.")
    if isinstance(temperature_millikelvin, bool) or not isinstance(
        temperature_millikelvin, numbers.Real
    ):
        raise TypeError("temperature_millikelvin must be a real scalar.")
    temperature_millikelvin = float(temperature_millikelvin)
    if not math.isfinite(temperature_millikelvin) or temperature_millikelvin < 0.0:
        raise ValueError("temperature_millikelvin must be finite and nonnegative.")
    if physical_path is None:
        if num_qubits is None:
            raise ValueError("num_qubits is required when physical_path is omitted.")
        path = select_physical_path(
            backend,
            validate_num_qubits(num_qubits),
            two_qubit_gate=two_qubit_gate,
        )
        path_selection_policy = "deterministic_lexicographically_first_simple_path"
    else:
        path = validate_physical_path(backend, physical_path, two_qubit_gate=two_qubit_gate)
        if num_qubits is not None and validate_num_qubits(num_qubits) != len(path):
            raise ValueError("num_qubits disagrees with physical_path length.")
        path_selection_policy = "explicit_user_supplied_contiguous_path"
    compact_target, qubits, instructions = _compact_target_and_audit(backend, path)
    if len(path) > 1 and two_qubit_gate not in compact_target.operation_names:
        raise ValueError(f"Compact target unexpectedly contains no {two_qubit_gate!r} operation.")
    view = _CompactBackendV2View(
        compact_target,
        f"compact-{_backend_name(backend)}-{'-'.join(str(item) for item in path)}",
    )
    noise_model = dependencies["NoiseModel"].from_backend(
        view,
        gate_error=gate_error,
        thermal_relaxation=thermal_relaxation,
        readout_error=readout_error,
        temperature=temperature_millikelvin,
    )
    digest = _snapshot_sha256(
        backend=backend,
        path=path,
        two_qubit_gate=two_qubit_gate,
        target=compact_target,
        qubits=qubits,
        instructions=instructions,
        gate_error=gate_error,
        thermal_relaxation=thermal_relaxation,
        readout_error=readout_error,
        temperature_millikelvin=temperature_millikelvin,
    )
    return CompactWashingtonDevice(
        source_backend_name=_backend_name(backend),
        source_backend_class=f"{type(backend).__module__}.{type(backend).__qualname__}",
        source_num_qubits=_backend_num_qubits(backend),
        physical_path=path,
        path_selection_policy=path_selection_policy,
        two_qubit_gate=two_qubit_gate,
        target=compact_target,
        noise_model=noise_model,
        qubit_calibrations=qubits,
        instruction_calibrations=instructions,
        include_gate_error=gate_error,
        include_thermal_relaxation=thermal_relaxation,
        include_readout_error=readout_error,
        temperature_millikelvin=temperature_millikelvin,
        snapshot_sha256=digest,
    )


@dataclass(frozen=True, slots=True)
class NativeCompilation:
    """Target-native compact circuit and routing/operation audit."""

    circuit: Any = field(repr=False, compare=False)
    gate_counts: tuple[tuple[str, int], ...]
    depth: int
    size: int
    swap_count: int
    unitary_count: int
    initial_index_layout: tuple[int, ...]
    final_index_layout: tuple[int, ...]
    optimization_level: int
    seed_transpiler: int

    def provenance_dict(self) -> dict[str, Any]:
        return {
            "gate_counts": dict(self.gate_counts),
            "depth": self.depth,
            "size": self.size,
            "swap_count": self.swap_count,
            "unitary_count": self.unitary_count,
            "initial_index_layout": list(self.initial_index_layout),
            "final_index_layout": list(self.final_index_layout),
            "optimization_level": self.optimization_level,
            "seed_transpiler": self.seed_transpiler,
        }


def _layout_indices(circuit: Any) -> tuple[tuple[int, ...], tuple[int, ...]]:
    expected = tuple(range(int(circuit.num_qubits)))
    layout = getattr(circuit, "layout", None)
    if layout is None:
        return expected, expected
    try:
        initial = tuple(int(item) for item in layout.initial_index_layout(filter_ancillas=True))
        final = tuple(int(item) for item in layout.final_index_layout(filter_ancillas=True))
    except (AttributeError, TypeError, ValueError) as error:
        raise ArithmeticError(
            "Unable to audit the transpiler layout through the supported index-layout API."
        ) from error
    for name, indices in (("initial", initial), ("final", final)):
        if len(indices) != len(expected) or set(indices) != set(expected):
            raise ArithmeticError(
                f"The {name} index layout is not a permutation of compact wires: "
                f"{indices!r}."
            )
    return initial, final


def compile_to_native_target(
    logical_circuit: Any,
    device: CompactWashingtonDevice,
    *,
    optimization_level: int = 0,
    seed_transpiler: int = 2021,
    enforce_no_swaps: bool = True,
) -> NativeCompilation:
    """Compile a compact circuit to its selected Washington-path target exactly."""

    dependencies = _qiskit_dependencies()
    from qiskit import QuantumCircuit

    if not isinstance(logical_circuit, QuantumCircuit):
        raise TypeError("logical_circuit must be a qiskit.QuantumCircuit.")
    if not isinstance(device, CompactWashingtonDevice):
        raise TypeError("device must be a CompactWashingtonDevice.")
    if logical_circuit.num_qubits != device.num_qubits:
        raise ValueError("logical_circuit and compact device disagree in qubit count.")
    if isinstance(optimization_level, bool) or not isinstance(optimization_level, numbers.Integral):
        raise TypeError("optimization_level must be an integer.")
    optimization_level = int(optimization_level)
    if optimization_level not in (0, 1, 2, 3):
        raise ValueError("optimization_level must be 0, 1, 2, or 3.")
    if isinstance(seed_transpiler, bool) or not isinstance(seed_transpiler, numbers.Integral):
        raise TypeError("seed_transpiler must be an integer.")
    seed_transpiler = int(seed_transpiler)
    if seed_transpiler < 0:
        raise ValueError("seed_transpiler must be nonnegative.")
    if not isinstance(enforce_no_swaps, bool):
        raise TypeError("enforce_no_swaps must be bool.")
    pass_manager = dependencies["generate_preset_pass_manager"](
        optimization_level=optimization_level,
        target=device.target,
        initial_layout=list(range(device.num_qubits)),
        routing_method="none",
        approximation_degree=1.0,
        seed_transpiler=seed_transpiler,
    )
    circuit = pass_manager.run(logical_circuit)
    counts = {str(name): int(count) for name, count in circuit.count_ops().items()}
    swaps = counts.get("swap", 0)
    unitaries = counts.get("unitary", 0)
    if enforce_no_swaps and swaps:
        raise ArithmeticError(f"Native compilation inserted {swaps} SWAP gates.")
    if unitaries:
        raise ArithmeticError("Native compilation left opaque U(4) unitary instructions.")
    unsupported = sorted(set(counts) - set(device.target.operation_names) - {"barrier"})
    if unsupported:
        raise ArithmeticError(
            f"Native target circuit contains unsupported operations: {unsupported}."
        )
    if circuit.num_qubits != device.num_qubits:
        raise ArithmeticError("Target compilation changed compact register width.")
    initial, final = _layout_indices(circuit)
    expected = tuple(range(device.num_qubits))
    if final != expected:
        raise ArithmeticError(
            "routing_method='none' changed logical compact-wire order: "
            f"{final!r} != {expected!r}."
        )
    return NativeCompilation(
        circuit=circuit,
        gate_counts=tuple(sorted(counts.items())),
        depth=int(circuit.depth()),
        size=int(circuit.size()),
        swap_count=swaps,
        unitary_count=unitaries,
        initial_index_layout=initial,
        final_index_layout=final,
        optimization_level=optimization_level,
        seed_transpiler=seed_transpiler,
    )


def _normalized_vector(state: ArrayLike, *, num_qubits: int, name: str) -> ComplexArray:
    try:
        return normalized_statevector(state, num_qubits=num_qubits)
    except (TypeError, ValueError) as exc:
        raise type(exc)(f"{name}: {exc}") from exc


def _mixed_fidelity(state: ComplexArray, density_matrix: ComplexArray) -> float:
    value = complex(np.vdot(state, density_matrix @ state))
    if abs(value.imag) > 2e-8:
        raise ArithmeticError(f"Mixed-state fidelity has imaginary residual {value.imag:.3e}.")
    result = float(value.real)
    if result < -2e-8 or result > 1.0 + 2e-8:
        raise ArithmeticError(f"Invalid mixed-state fidelity {result:.17g}.")
    return result


@dataclass(frozen=True, slots=True)
class DensityMatrixDiagnostics:
    """Physicality checks for Aer output in exact density-matrix simulation."""

    purity: float
    trace_real: float
    trace_imag: float
    trace_error: float
    hermiticity_residual: float
    minimum_eigenvalue: float | None
    eigenvalue_check_performed: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "purity": self.purity,
            "trace_real": self.trace_real,
            "trace_imag": self.trace_imag,
            "trace_error": self.trace_error,
            "hermiticity_residual": self.hermiticity_residual,
            "minimum_eigenvalue": self.minimum_eigenvalue,
            "eigenvalue_check_performed": self.eigenvalue_check_performed,
        }


def diagnose_density_matrix(
    density_matrix: ArrayLike,
    *,
    compute_minimum_eigenvalue: bool = True,
    atol: float = 2e-8,
) -> DensityMatrixDiagnostics:
    """Return trace, Hermiticity, purity, and optional PSD diagnostics.

    This pure NumPy helper is deliberately public so density-matrix physicality
    can be unit-tested independently of a Qiskit/Aer installation.
    """

    if not isinstance(compute_minimum_eigenvalue, bool):
        raise TypeError("compute_minimum_eigenvalue must be bool.")
    if isinstance(atol, bool) or not isinstance(atol, numbers.Real):
        raise TypeError("atol must be a real scalar.")
    atol = float(atol)
    if not math.isfinite(atol) or atol <= 0.0:
        raise ValueError("atol must be finite and positive.")
    matrix = np.asarray(density_matrix, dtype=np.complex128)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] < 1:
        raise ValueError("density_matrix must be a nonempty square matrix.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("density_matrix contains non-finite entries.")
    trace = complex(np.trace(matrix))
    trace_error = float(abs(trace - 1.0))
    hermiticity = float(np.max(np.abs(matrix - matrix.conj().T)))
    purity_value = complex(np.einsum("ij,ji->", matrix, matrix, optimize=True))
    if abs(purity_value.imag) > atol:
        raise ArithmeticError(
            f"Density-matrix purity has imaginary residual {purity_value.imag:.3e}."
        )
    purity = float(purity_value.real)
    if trace_error > atol or hermiticity > atol:
        raise ArithmeticError(
            "Density matrix failed trace/Hermiticity checks: "
            f"trace_error={trace_error:.3e}, hermiticity={hermiticity:.3e}."
        )
    if purity < -atol or purity > 1.0 + atol:
        raise ArithmeticError(f"Density-matrix purity is unphysical: {purity:.17g}.")
    minimum = None
    if compute_minimum_eigenvalue:
        minimum = float(np.linalg.eigvalsh(matrix)[0])
        if minimum < -atol:
            raise ArithmeticError(f"Density matrix is not PSD; min eigenvalue={minimum:.3e}.")
    return DensityMatrixDiagnostics(
        purity=purity,
        trace_real=float(trace.real),
        trace_imag=float(trace.imag),
        trace_error=trace_error,
        hermiticity_residual=hermiticity,
        minimum_eigenvalue=minimum,
        eigenvalue_check_performed=compute_minimum_eigenvalue,
    )


def density_matrix_energy_expectation(density_matrix: ArrayLike, hamiltonian: ArrayLike) -> float:
    """Return ``Tr(H rho)`` while rejecting dimension/Hermiticity artifacts."""

    rho = np.asarray(density_matrix, dtype=np.complex128)
    hamiltonian = validate_dense_hamiltonian(hamiltonian)
    if rho.ndim != 2 or rho.shape[0] != rho.shape[1]:
        raise ValueError("density_matrix must be square.")
    if hamiltonian.shape != rho.shape:
        raise ValueError(
            f"hamiltonian shape {hamiltonian.shape} disagrees with density matrix {rho.shape}."
        )
    if not np.all(np.isfinite(rho)):
        raise ValueError("density_matrix contains non-finite entries.")
    value = complex(np.einsum("ij,ji->", hamiltonian, rho, optimize=True))
    if abs(value.imag) > 2e-8:
        raise ArithmeticError(f"Noisy energy has imaginary residual {value.imag:.3e}.")
    return float(value.real)


def _package_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


@dataclass(frozen=True, slots=True)
class NoisyEncodingResult:
    """Target, compiler, and noisy-state metrics for one encoded ground state."""

    num_qubits: int
    logical_target_fidelity: float
    compilation_fidelity: float
    native_target_fidelity: float
    noisy_target_fidelity: float
    noisy_native_retention_fidelity: float
    density: DensityMatrixDiagnostics
    noisy_energy: float | None
    energy_error_per_site: float | None
    gate_counts: tuple[tuple[str, int], ...]
    native_depth: int
    native_size: int
    initial_index_layout: tuple[int, ...]
    final_index_layout: tuple[int, ...]
    physical_path: tuple[int, ...] | None
    path_selection_policy: str | None
    backend_snapshot_sha256: str | None
    simulator_method: str
    seed_simulator: int
    runtime_seconds: float
    qiskit_version: str | None
    qiskit_aer_version: str | None
    qiskit_ibm_runtime_version: str | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": NOISY_RESULT_SCHEMA,
            "num_qubits": self.num_qubits,
            "fidelity": {
                "logical_target": self.logical_target_fidelity,
                "compilation": self.compilation_fidelity,
                "native_target": self.native_target_fidelity,
                "noisy_target": self.noisy_target_fidelity,
                "noisy_native_retention": self.noisy_native_retention_fidelity,
            },
            "density_matrix_diagnostics": self.density.as_dict(),
            "energy": {
                "noisy": self.noisy_energy,
                "error_per_site": self.energy_error_per_site,
            },
            "native_circuit": {
                "gate_counts": dict(self.gate_counts),
                "depth": self.native_depth,
                "size": self.native_size,
                "initial_index_layout": list(self.initial_index_layout),
                "final_index_layout": list(self.final_index_layout),
            },
            "hardware": {
                "physical_path": None if self.physical_path is None else list(self.physical_path),
                "path_selection_policy": self.path_selection_policy,
                "backend_snapshot_sha256": self.backend_snapshot_sha256,
            },
            "simulation": {
                "method": self.simulator_method,
                "seed_simulator": self.seed_simulator,
                "runtime_seconds": self.runtime_seconds,
                "qiskit": self.qiskit_version,
                "qiskit_aer": self.qiskit_aer_version,
                "qiskit_ibm_runtime": self.qiskit_ibm_runtime_version,
            },
        }


def _native_compilation_from_circuit(circuit: Any) -> NativeCompilation:
    counts = tuple(sorted((str(name), int(count)) for name, count in circuit.count_ops().items()))
    initial, final = _layout_indices(circuit)
    return NativeCompilation(
        circuit=circuit,
        gate_counts=counts,
        depth=int(circuit.depth()),
        size=int(circuit.size()),
        swap_count=dict(counts).get("swap", 0),
        unitary_count=dict(counts).get("unitary", 0),
        initial_index_layout=initial,
        final_index_layout=final,
        optimization_level=-1,
        seed_transpiler=-1,
    )


def _validate_protocol_native_operation_counts(
    counts: Mapping[str, int],
    *,
    target_operation_names: Iterable[str] | None = None,
) -> None:
    """Fail closed unless a preparation circuit uses the declared native basis."""

    if not isinstance(counts, Mapping):
        raise TypeError("counts must be a mapping of operation names to counts.")
    normalized: dict[str, int] = {}
    for raw_name, raw_count in counts.items():
        name = str(raw_name)
        if isinstance(raw_count, bool) or not isinstance(raw_count, numbers.Integral):
            raise TypeError("every operation count must be an integer.")
        count = int(raw_count)
        if count < 0:
            raise ValueError("operation counts must be nonnegative.")
        if count:
            normalized[name] = count
    permitted = set(DEFAULT_NATIVE_OPERATIONS) | {"barrier"}
    unsupported = sorted(set(normalized) - permitted)
    if unsupported:
        raise ValueError(
            "native circuit contains opaque, non-native, or non-preparation operations: "
            f"{unsupported}."
        )
    if target_operation_names is not None:
        target_names = {str(name) for name in target_operation_names}
        target_unsupported = sorted((set(normalized) - {"barrier"}) - target_names)
        if target_unsupported:
            raise ValueError(
                "native circuit contains operations unsupported by the compact target: "
                f"{target_unsupported}."
            )


def _validate_native_compilation_ledger(
    compilation: NativeCompilation,
    *,
    device: CompactWashingtonDevice | None,
) -> None:
    """Recompute every circuit-derived ledger coordinate before simulation."""

    circuit = compilation.circuit
    actual_counts = tuple(
        sorted((str(name), int(count)) for name, count in circuit.count_ops().items())
    )
    target_names = None if device is None else tuple(device.target.operation_names)
    _validate_protocol_native_operation_counts(
        dict(actual_counts),
        target_operation_names=target_names,
    )
    actual_initial, actual_final = _layout_indices(circuit)
    identity_layout = tuple(range(int(circuit.num_qubits)))
    if actual_initial != identity_layout or actual_final != identity_layout:
        raise ArithmeticError(
            "Native density-matrix evaluation requires identity compact-wire layouts; "
            f"initial={actual_initial!r}, final={actual_final!r}."
        )
    expected_values = {
        "gate_counts": actual_counts,
        "depth": int(circuit.depth()),
        "size": int(circuit.size()),
        "swap_count": dict(actual_counts).get("swap", 0),
        "unitary_count": dict(actual_counts).get("unitary", 0),
        "initial_index_layout": actual_initial,
        "final_index_layout": actual_final,
    }
    for name, actual in expected_values.items():
        recorded = getattr(compilation, name)
        if recorded != actual:
            raise ArithmeticError(
                f"NativeCompilation {name} ledger disagrees with its circuit: "
                f"recorded={recorded!r}, actual={actual!r}."
            )
    if expected_values["swap_count"] or expected_values["unitary_count"]:
        raise ValueError(
            "native preparation circuits must contain neither SWAP nor unitary blocks."
        )


def evaluate_native_density_matrix(
    target_state: ArrayLike,
    logical_state: ArrayLike,
    native: NativeCompilation | Any,
    *,
    noise_model: Any | None = None,
    device: CompactWashingtonDevice | None = None,
    hamiltonian: ArrayLike | None = None,
    ground_energy: float | None = None,
    num_sites: int | None = None,
    seed_simulator: int = 2021,
    max_density_matrix_qubits: int = 12,
    compute_minimum_eigenvalue: bool | None = None,
) -> NoisyEncodingResult:
    """Evaluate a native preparation circuit using Aer density-matrix evolution.

    No measurements are added: reported noisy fidelity is exactly
    ``<psi_target|rho_noisy|psi_target>``, rather than a finite-shot
    tomography estimator.  Passing a compact device uses its calibrated Aer
    noise model and records its physical path; an injected ``noise_model`` is
    allowed for unit tests but cannot be combined with a device.
    """

    dependencies = _qiskit_dependencies()
    from qiskit import QuantumCircuit

    if device is not None and not isinstance(device, CompactWashingtonDevice):
        raise TypeError("device must be CompactWashingtonDevice or None.")
    if device is not None and noise_model is not None:
        raise ValueError("Pass either device or noise_model, not both.")
    if isinstance(max_density_matrix_qubits, bool) or not isinstance(
        max_density_matrix_qubits, numbers.Integral
    ):
        raise TypeError("max_density_matrix_qubits must be an integer.")
    max_density_matrix_qubits = int(max_density_matrix_qubits)
    if max_density_matrix_qubits < 1:
        raise ValueError("max_density_matrix_qubits must be positive.")
    if isinstance(seed_simulator, bool) or not isinstance(seed_simulator, numbers.Integral):
        raise TypeError("seed_simulator must be an integer.")
    seed_simulator = int(seed_simulator)
    if seed_simulator < 0:
        raise ValueError("seed_simulator must be nonnegative.")
    if compute_minimum_eigenvalue is not None and not isinstance(compute_minimum_eigenvalue, bool):
        raise TypeError("compute_minimum_eigenvalue must be bool or None.")
    if isinstance(native, NativeCompilation):
        compilation = native
        circuit = native.circuit
    else:
        if not isinstance(native, QuantumCircuit):
            raise TypeError("native must be NativeCompilation or qiskit.QuantumCircuit.")
        compilation = _native_compilation_from_circuit(native)
        circuit = native
    if not isinstance(circuit, QuantumCircuit):
        raise TypeError("native circuit must be a qiskit.QuantumCircuit.")
    _validate_native_compilation_ledger(compilation, device=device)
    num_qubits = int(circuit.num_qubits)
    if num_qubits < 1:
        raise ValueError("native circuit must contain at least one qubit.")
    if num_qubits > max_density_matrix_qubits:
        raise ValueError(
            f"Refusing a {num_qubits}-qubit full density matrix; "
            f"max_density_matrix_qubits={max_density_matrix_qubits}."
        )
    if device is not None and device.num_qubits != num_qubits:
        raise ValueError("native circuit and compact device disagree in qubit count.")
    target = _normalized_vector(target_state, num_qubits=num_qubits, name="target_state")
    logical = _normalized_vector(logical_state, num_qubits=num_qubits, name="logical_state")
    if (hamiltonian is None) != (ground_energy is None):
        raise ValueError("hamiltonian and ground_energy must be provided together or both omitted.")
    sites: int | None = None
    if hamiltonian is not None:
        if isinstance(ground_energy, bool) or not isinstance(ground_energy, numbers.Real):
            raise TypeError("ground_energy must be a finite real scalar.")
        ground_energy = float(ground_energy)
        if not math.isfinite(ground_energy):
            raise ValueError("ground_energy must be finite.")
        sites = (
            num_qubits
            if num_sites is None
            else validate_num_qubits(num_sites, name="num_sites")
        )
        if sites != num_qubits:
            raise ValueError(
                "num_sites disagrees with the native circuit width: "
                f"N={sites}, circuit N={num_qubits}."
            )
    nonunitary = {
        instruction.operation.name
        for instruction in circuit.data
        if instruction.operation.name in {"measure", "reset", "initialize"}
        or instruction.operation.name.startswith("save_")
    }
    if nonunitary:
        raise ValueError(
            "native circuit must contain preparation unitaries only; "
            f"found {sorted(nonunitary)}."
        )
    native_state = _normalized_vector(
        np.asarray(dependencies["Statevector"].from_instruction(circuit).data, dtype=np.complex128),
        num_qubits=num_qubits,
        name="native ideal state",
    )
    if noise_model is None and device is not None:
        noise_model = device.noise_model
    simulation = circuit.copy()
    simulation.save_density_matrix(label="rho_noisy")
    simulator = dependencies["AerSimulator"](method="density_matrix", noise_model=noise_model)
    started = time.perf_counter()
    result = simulator.run(simulation, shots=1, seed_simulator=seed_simulator).result()
    elapsed = time.perf_counter() - started
    rho = np.asarray(result.data(0)["rho_noisy"], dtype=np.complex128)
    expected_shape = (1 << num_qubits, 1 << num_qubits)
    if rho.shape != expected_shape:
        raise ArithmeticError(f"Aer returned rho shape {rho.shape}, expected {expected_shape}.")
    diagnostics = diagnose_density_matrix(
        rho,
        compute_minimum_eigenvalue=(rho.shape[0] <= 256)
        if compute_minimum_eigenvalue is None
        else compute_minimum_eigenvalue,
    )
    noisy_energy = None
    error_per_site = None
    if hamiltonian is not None:
        assert ground_energy is not None
        assert sites is not None
        noisy_energy = density_matrix_energy_expectation(rho, hamiltonian)
        error_per_site = (noisy_energy - ground_energy) / sites
    return NoisyEncodingResult(
        num_qubits=num_qubits,
        logical_target_fidelity=statevector_fidelity(target, logical),
        compilation_fidelity=statevector_fidelity(logical, native_state),
        native_target_fidelity=statevector_fidelity(target, native_state),
        noisy_target_fidelity=_mixed_fidelity(target, rho),
        noisy_native_retention_fidelity=_mixed_fidelity(native_state, rho),
        density=diagnostics,
        noisy_energy=noisy_energy,
        energy_error_per_site=error_per_site,
        gate_counts=compilation.gate_counts,
        native_depth=compilation.depth,
        native_size=compilation.size,
        initial_index_layout=compilation.initial_index_layout,
        final_index_layout=compilation.final_index_layout,
        physical_path=None if device is None else device.physical_path,
        path_selection_policy=None if device is None else device.path_selection_policy,
        backend_snapshot_sha256=None if device is None else device.snapshot_sha256,
        simulator_method="density_matrix",
        seed_simulator=seed_simulator,
        runtime_seconds=elapsed,
        qiskit_version=_package_version("qiskit"),
        qiskit_aer_version=_package_version("qiskit-aer"),
        qiskit_ibm_runtime_version=_package_version("qiskit-ibm-runtime"),
    )


__all__ = [
    "COMPACT_DEVICE_SCHEMA",
    "DEFAULT_NATIVE_OPERATIONS",
    "NOISY_RESULT_SCHEMA",
    "CompactWashingtonDevice",
    "DensityMatrixDiagnostics",
    "InstructionCalibrationAudit",
    "NativeCompilation",
    "NoisyEncodingResult",
    "QiskitNoiseDependencyError",
    "QubitCalibrationAudit",
    "build_compact_washington_device",
    "compile_to_native_target",
    "density_matrix_energy_expectation",
    "diagnose_density_matrix",
    "evaluate_native_density_matrix",
    "load_fake_washington_v2",
    "select_physical_path",
    "validate_physical_path",
]
