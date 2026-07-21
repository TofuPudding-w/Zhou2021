"""Qiskit-independent archival and replay of Zhou serial ordered U(4) gates.

The native/noise extension used to retain only a hash for each projected U(4).
That is useful for detection of an accidental change, but it is insufficient to
independently replay a trained circuit.  This module defines a deliberately
small, strict NPZ protocol which stores the physical gates themselves without
depending on Qiskit or on the latent PyTorch parameters.

The protocol stores a layer-major serial staircase.  Gate ``(layer, position)``
always acts on project sites ``(position, position + 1)`` in the q0-most-
significant convention and uses the local basis ``|00>, |01>, |10>, |11>``.
It is therefore suitable as an immutable hand-off artifact for a separate
compiler, simulator, or audit implementation.
"""

from __future__ import annotations

import hashlib
import json
import math
import numbers
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .circuit import LatentStaircase
from .conventions import COMPLEX_DTYPE, QUBIT_ORDER, TWO_QUBIT_BASIS, normalized_statevector

ORDERED_U4_NPZ_SCHEMA_VERSION = "zhou2021-ordered-u4-npz-v1"
ORDERED_U4_EXECUTION_ORDER = (
    "layer-major serial staircase: for each layer, position=left_site runs from 0 through N-2"
)
ORDERED_U4_MATRIX_DTYPE = "complex128"
ORDERED_U4_INDEX_DTYPE = "int64"
ORDERED_U4_UNITARITY_ATOL = 2e-10
_PAYLOAD_SEMANTIC_SHA256_KEY = "payload_semantic_sha256"
_REQUIRED_NPZ_KEYS = frozenset(
    {"metadata_json", "matrices", "layer", "position", "left_site"}
)

ComplexMatrix = NDArray[np.complex128]


@dataclass(frozen=True, slots=True)
class OrderedU4Gate:
    """One frozen physical U(4) gate in declared serial execution order."""

    layer: int
    position: int
    left_site: int
    matrix: ComplexMatrix


@dataclass(frozen=True, slots=True)
class OrderedU4Archive:
    """Validated ordered-U(4) payload plus canonical JSON metadata."""

    num_qubits: int
    gates: tuple[OrderedU4Gate, ...]
    metadata: dict[str, Any]

    @property
    def num_layers(self) -> int:
        """Return the declared serial staircase depth."""

        return int(self.metadata["num_layers"])

    @property
    def semantic_sha256(self) -> str:
        """Return the validated semantic SHA-256 declared by this archive."""

        return str(self.metadata[_PAYLOAD_SEMANTIC_SHA256_KEY])


@dataclass(frozen=True, slots=True)
class LoadedOrderedU4Archive:
    """An ordered-U(4) archive parsed from one validated NPZ file."""

    archive: OrderedU4Archive
    path: Path


def _validate_nonnegative_integer(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise TypeError(f"{name} must be an integer.")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be nonnegative.")
    return result


def _validate_archive_num_qubits(value: Any) -> int:
    result = _validate_nonnegative_integer(value, name="num_qubits")
    if result < 2:
        raise ValueError("ordered-U4 staircase archives require num_qubits >= 2.")
    return result


def _validate_json_value(value: Any, *, path: str) -> None:
    """Require portable finite JSON values without silent NumPy coercions."""

    if value is None or isinstance(value, (str, bool)):
        return
    if isinstance(value, int):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"metadata contains a non-finite float at {path}.")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json_value(item, path=f"{path}[{index}]")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"metadata key at {path} must be a string.")
            _validate_json_value(item, path=f"{path}.{key}")
        return
    raise TypeError(
        f"metadata value at {path} has unsupported type {type(value).__name__}; "
        "use only JSON-native values."
    )


def canonical_json_metadata(metadata: Mapping[str, Any] | None) -> str:
    """Return canonical finite JSON for protocol metadata.

    Canonicalization is part of the semantic SHA definition.  It also rejects
    NumPy scalar/object values that JSON would otherwise convert or encode in a
    platform-dependent manner.
    """

    if metadata is None:
        metadata = {}
    if not isinstance(metadata, Mapping):
        raise TypeError("metadata must be a mapping or None.")
    materialized = dict(metadata)
    _validate_json_value(materialized, path="metadata")
    return json.dumps(
        materialized,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _frozen_matrix(matrix: ArrayLike, *, name: str) -> ComplexMatrix:
    value = np.asarray(matrix)
    if value.dtype != COMPLEX_DTYPE:
        raise ValueError(f"{name} must have dtype complex128, got {value.dtype}.")
    if value.shape != (4, 4):
        raise ValueError(f"{name} must have shape (4, 4), got {value.shape}.")
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{name} contains non-finite entries.")
    residual = float(np.linalg.norm(value.conj().T @ value - np.eye(4), ord=np.inf))
    if residual > ORDERED_U4_UNITARITY_ATOL:
        raise ValueError(
            f"{name} is not unitary in the declared |00>,|01>,|10>,|11> basis; "
            f"residual={residual:.3e}."
        )
    frozen = np.ascontiguousarray(value, dtype=COMPLEX_DTYPE)
    frozen.setflags(write=False)
    return frozen


def _validated_gate_sequence(
    num_qubits: int,
    gates: Iterable[OrderedU4Gate],
    *,
    expected_num_layers: int | None = None,
) -> tuple[tuple[OrderedU4Gate, ...], int]:
    """Validate exact Zhou layer-major serial order and return frozen copies."""

    num_qubits = _validate_archive_num_qubits(num_qubits)
    if isinstance(gates, (str, bytes)):
        raise TypeError("gates must be an iterable of OrderedU4Gate values.")
    records = tuple(gates)
    gates_per_layer = num_qubits - 1
    if expected_num_layers is None:
        if len(records) % gates_per_layer != 0:
            raise ValueError(
                "ordered-U4 gate count must be an integer multiple of N-1 for a serial "
                "staircase."
            )
        num_layers = len(records) // gates_per_layer
    else:
        num_layers = _validate_nonnegative_integer(
            expected_num_layers,
            name="metadata.num_layers",
        )
        if len(records) != num_layers * gates_per_layer:
            raise ValueError(
                "ordered-U4 gate count disagrees with metadata.num_layers*(N-1); "
                f"got {len(records)} versus {num_layers * gates_per_layer}."
            )

    frozen: list[OrderedU4Gate] = []
    for index, record in enumerate(records):
        if not isinstance(record, OrderedU4Gate):
            raise TypeError(
                "gates must contain OrderedU4Gate values; "
                f"element {index} is {type(record).__name__}."
            )
        layer = _validate_nonnegative_integer(record.layer, name=f"gates[{index}].layer")
        position = _validate_nonnegative_integer(
            record.position,
            name=f"gates[{index}].position",
        )
        left_site = _validate_nonnegative_integer(
            record.left_site,
            name=f"gates[{index}].left_site",
        )
        expected_layer, expected_position = divmod(index, gates_per_layer)
        if (layer, position, left_site) != (
            expected_layer,
            expected_position,
            expected_position,
        ):
            raise ValueError(
                "ordered-U4 records must be exact layer-major serial staircase order; "
                f"index {index} must be (layer={expected_layer}, "
                f"position={expected_position}, left_site={expected_position}), got "
                f"(layer={layer}, position={position}, left_site={left_site})."
            )
        frozen.append(
            OrderedU4Gate(
                layer=layer,
                position=position,
                left_site=left_site,
                matrix=_frozen_matrix(record.matrix, name=f"gates[{index}].matrix"),
            )
        )
    return tuple(frozen), num_layers


def _base_metadata(
    *,
    num_qubits: int,
    num_layers: int,
    gate_count: int,
    user_metadata: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Build all immutable protocol metadata except the self-checking digest."""

    user_json = canonical_json_metadata(user_metadata)
    return {
        "schema_version": ORDERED_U4_NPZ_SCHEMA_VERSION,
        "num_qubits": num_qubits,
        "num_layers": num_layers,
        "gates_per_layer": num_qubits - 1,
        "gate_count": gate_count,
        "qubit_order": QUBIT_ORDER,
        "two_qubit_basis": list(TWO_QUBIT_BASIS),
        "matrix_dtype": ORDERED_U4_MATRIX_DTYPE,
        "matrix_shape": [gate_count, 4, 4],
        "index_dtype": ORDERED_U4_INDEX_DTYPE,
        "execution_order": ORDERED_U4_EXECUTION_ORDER,
        "global_phase_convention": "each stored U(4) includes its own global phase",
        "digest_algorithm": "sha256",
        "user_metadata": json.loads(user_json),
    }


def _semantic_sha256_from_parts(
    *,
    num_qubits: int,
    gates: tuple[OrderedU4Gate, ...],
    metadata_without_digest: Mapping[str, Any],
) -> str:
    """Hash every replay-relevant coordinate under one canonical byte protocol."""

    digest = hashlib.sha256()
    digest.update(b"zhou2021-ordered-u4-semantic-v1\n")
    digest.update(canonical_json_metadata(metadata_without_digest).encode("utf-8"))
    digest.update(b"\n")
    digest.update(str(num_qubits).encode("ascii"))
    digest.update(b"\n")
    for index, gate in enumerate(gates):
        header = json.dumps(
            {
                "index": index,
                "layer": gate.layer,
                "position": gate.position,
                "left_site": gate.left_site,
                "dtype": gate.matrix.dtype.str,
                "shape": list(gate.matrix.shape),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        digest.update(header)
        digest.update(b"\n")
        digest.update(memoryview(np.ascontiguousarray(gate.matrix)).cast("B"))
    return digest.hexdigest()


def _validate_protocol_metadata(
    metadata: Mapping[str, Any],
    *,
    num_qubits: int,
    num_layers: int,
    gate_count: int,
) -> dict[str, Any]:
    """Validate fixed conventions and canonicalize the full metadata object."""

    if not isinstance(metadata, Mapping):
        raise TypeError("archive metadata must be a mapping.")
    canonical = canonical_json_metadata(metadata)
    normal = json.loads(canonical)
    if not isinstance(normal, dict):  # pragma: no cover - mapping invariant
        raise ValueError("archive metadata must encode one JSON object.")
    if _PAYLOAD_SEMANTIC_SHA256_KEY not in normal:
        raise ValueError("ordered-U4 metadata is missing payload_semantic_sha256.")
    expected = _base_metadata(
        num_qubits=num_qubits,
        num_layers=num_layers,
        gate_count=gate_count,
        user_metadata=normal.get("user_metadata"),
    )
    expected_keys = {*expected, _PAYLOAD_SEMANTIC_SHA256_KEY}
    if set(normal) != expected_keys:
        missing = sorted(expected_keys - set(normal))
        unexpected = sorted(set(normal) - expected_keys)
        raise ValueError(
            "ordered-U4 metadata key set is invalid; "
            f"missing={missing}, unexpected={unexpected}."
        )
    for key, value in expected.items():
        if normal[key] != value:
            raise ValueError(
                f"ordered-U4 metadata field {key!r} disagrees with the strict protocol."
            )
    declared = normal[_PAYLOAD_SEMANTIC_SHA256_KEY]
    if not isinstance(declared, str) or len(declared) != 64:
        raise ValueError("payload_semantic_sha256 must be one 64-character SHA-256 string.")
    try:
        int(declared, 16)
    except ValueError as error:
        raise ValueError("payload_semantic_sha256 must be hexadecimal.") from error
    if declared != declared.lower():
        raise ValueError("payload_semantic_sha256 must use lowercase hexadecimal.")
    return normal


def build_ordered_u4_archive(
    num_qubits: int,
    gates: Iterable[OrderedU4Gate],
    *,
    user_metadata: Mapping[str, Any] | None = None,
) -> OrderedU4Archive:
    """Create one self-validating, replayable strict ordered-U(4) archive."""

    num_qubits = _validate_archive_num_qubits(num_qubits)
    frozen_gates, num_layers = _validated_gate_sequence(num_qubits, gates)
    metadata = _base_metadata(
        num_qubits=num_qubits,
        num_layers=num_layers,
        gate_count=len(frozen_gates),
        user_metadata=user_metadata,
    )
    metadata[_PAYLOAD_SEMANTIC_SHA256_KEY] = _semantic_sha256_from_parts(
        num_qubits=num_qubits,
        gates=frozen_gates,
        metadata_without_digest=metadata,
    )
    return OrderedU4Archive(
        num_qubits=num_qubits,
        gates=frozen_gates,
        metadata=json.loads(canonical_json_metadata(metadata)),
    )


def validate_ordered_u4_archive(archive: OrderedU4Archive) -> OrderedU4Archive:
    """Return a frozen validated copy and detect any semantic payload tampering."""

    if not isinstance(archive, OrderedU4Archive):
        raise TypeError("archive must be an OrderedU4Archive.")
    num_qubits = _validate_archive_num_qubits(archive.num_qubits)
    declared_layers = None
    if isinstance(archive.metadata, Mapping) and "num_layers" in archive.metadata:
        declared_layers = archive.metadata["num_layers"]
    gates, num_layers = _validated_gate_sequence(
        num_qubits,
        archive.gates,
        expected_num_layers=declared_layers,
    )
    metadata = _validate_protocol_metadata(
        archive.metadata,
        num_qubits=num_qubits,
        num_layers=num_layers,
        gate_count=len(gates),
    )
    metadata_without_digest = dict(metadata)
    declared = str(metadata_without_digest.pop(_PAYLOAD_SEMANTIC_SHA256_KEY))
    observed = _semantic_sha256_from_parts(
        num_qubits=num_qubits,
        gates=gates,
        metadata_without_digest=metadata_without_digest,
    )
    if observed != declared:
        raise ValueError(
            "ordered-U4 payload semantic SHA-256 mismatch; the gate payload or metadata "
            "was modified after export."
        )
    return OrderedU4Archive(num_qubits=num_qubits, gates=gates, metadata=metadata)


def ordered_u4_archive_from_staircase(
    staircase: LatentStaircase,
    *,
    user_metadata: Mapping[str, Any] | None = None,
) -> OrderedU4Archive:
    """Freeze one trained latent staircase to physical U(4) matrices only."""

    if not isinstance(staircase, LatentStaircase):
        raise TypeError("staircase must be a zhou2021.circuit.LatentStaircase.")
    gates = tuple(
        OrderedU4Gate(
            layer=gate.layer,
            position=gate.position,
            left_site=gate.left_site,
            matrix=np.asarray(gate.matrix.detach().cpu().numpy(), dtype=COMPLEX_DTYPE),
        )
        for gate in staircase.iter_projected_gates()
    )
    return build_ordered_u4_archive(
        staircase.num_qubits,
        gates,
        user_metadata=user_metadata,
    )


def ordered_u4_semantic_sha256(archive: OrderedU4Archive) -> str:
    """Validate and return the semantic SHA-256 of one ordered-U(4) archive."""

    return validate_ordered_u4_archive(archive).semantic_sha256


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 of the exact bytes in one already-written artifact."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Artifact does not exist: {source}")
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_ordered_u4_npz(path: str | Path, archive: OrderedU4Archive) -> Path:
    """Save a strict replayable ordered-U(4) NPZ archive without pickle data."""

    validated = validate_ordered_u4_archive(archive)
    destination = Path(path)
    if destination.suffix.lower() != ".npz":
        raise ValueError("ordered-U4 NPZ destination path must end in '.npz'.")
    if not destination.parent.is_dir():
        raise FileNotFoundError(f"Parent directory does not exist: {destination.parent}")
    count = len(validated.gates)
    matrices = np.empty((count, 4, 4), dtype=COMPLEX_DTYPE)
    layer = np.empty(count, dtype=np.int64)
    position = np.empty(count, dtype=np.int64)
    left_site = np.empty(count, dtype=np.int64)
    for index, gate in enumerate(validated.gates):
        matrices[index] = gate.matrix
        layer[index] = gate.layer
        position[index] = gate.position
        left_site[index] = gate.left_site
    np.savez_compressed(
        destination,
        metadata_json=np.array(canonical_json_metadata(validated.metadata)),
        matrices=matrices,
        layer=layer,
        position=position,
        left_site=left_site,
    )
    return destination


def _required_unicode_scalar(archive: Any, key: str) -> str:
    if key not in archive.files:
        raise ValueError(f"ordered-U4 NPZ artifact is missing required key {key!r}.")
    value = np.asarray(archive[key])
    if value.ndim != 0 or value.dtype.kind != "U":
        raise ValueError(
            f"ordered-U4 NPZ key {key!r} must be one Unicode scalar, "
            f"got dtype={value.dtype}, shape={value.shape}."
        )
    return str(value.item())


def load_ordered_u4_npz(path: str | Path) -> LoadedOrderedU4Archive:
    """Safely load, validate, and integrity-check one ordered-U(4) NPZ artifact."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"ordered-U4 NPZ artifact does not exist: {source}")
    try:
        archive_context = np.load(source, allow_pickle=False)
    except (OSError, ValueError) as error:
        raise ValueError(
            f"Unable to open safe ordered-U4 NPZ artifact {source}: {error}"
        ) from error
    if not hasattr(archive_context, "files"):
        raise ValueError(f"ordered-U4 artifact {source} is not an NPZ archive.")
    with archive_context as npz:
        actual_keys = set(npz.files)
        if actual_keys != _REQUIRED_NPZ_KEYS:
            missing = sorted(_REQUIRED_NPZ_KEYS - actual_keys)
            unexpected = sorted(actual_keys - _REQUIRED_NPZ_KEYS)
            raise ValueError(
                "ordered-U4 NPZ key set is invalid; "
                f"missing={missing}, unexpected={unexpected}."
            )
        metadata_raw = _required_unicode_scalar(npz, "metadata_json")
        try:
            metadata = json.loads(metadata_raw)
        except json.JSONDecodeError as error:
            raise ValueError("ordered-U4 metadata_json is not valid JSON.") from error
        if not isinstance(metadata, dict):
            raise ValueError("ordered-U4 metadata_json must encode one JSON object.")
        if metadata_raw != canonical_json_metadata(metadata):
            raise ValueError("ordered-U4 metadata_json must use canonical JSON encoding.")
        if "num_qubits" not in metadata or "num_layers" not in metadata:
            raise ValueError("ordered-U4 metadata must declare num_qubits and num_layers.")
        num_qubits = _validate_archive_num_qubits(metadata["num_qubits"])
        num_layers = _validate_nonnegative_integer(
            metadata["num_layers"],
            name="metadata.num_layers",
        )
        expected_count = num_layers * (num_qubits - 1)
        matrices = np.asarray(npz["matrices"])
        if matrices.dtype != COMPLEX_DTYPE or matrices.shape != (expected_count, 4, 4):
            raise ValueError(
                "ordered-U4 matrices must have dtype complex128 and shape "
                f"({expected_count}, 4, 4); got dtype={matrices.dtype}, "
                f"shape={matrices.shape}."
            )
        index_arrays: dict[str, NDArray[np.int64]] = {}
        for name in ("layer", "position", "left_site"):
            value = np.asarray(npz[name])
            if value.dtype != np.dtype(np.int64) or value.shape != (expected_count,):
                raise ValueError(
                    f"ordered-U4 {name} must have dtype int64 and shape ({expected_count},); "
                    f"got dtype={value.dtype}, shape={value.shape}."
                )
            index_arrays[name] = value
        gates = tuple(
            OrderedU4Gate(
                layer=int(index_arrays["layer"][index]),
                position=int(index_arrays["position"][index]),
                left_site=int(index_arrays["left_site"][index]),
                matrix=np.array(matrices[index], dtype=COMPLEX_DTYPE, copy=True),
            )
            for index in range(expected_count)
        )
    validated = validate_ordered_u4_archive(
        OrderedU4Archive(num_qubits=num_qubits, gates=gates, metadata=metadata)
    )
    return LoadedOrderedU4Archive(archive=validated, path=source)


def replay_ordered_u4_archive(
    archive: OrderedU4Archive,
    *,
    initial_state: ArrayLike | None = None,
) -> NDArray[np.complex128]:
    """Replay an archive with NumPy in the declared q0-MSB tensor convention."""

    validated = validate_ordered_u4_archive(archive)
    num_qubits = validated.num_qubits
    if initial_state is None:
        state = np.zeros(1 << num_qubits, dtype=COMPLEX_DTYPE)
        state[0] = 1.0
    else:
        state = normalized_statevector(initial_state, num_qubits=num_qubits)
    for gate in validated.gates:
        remaining_axes = [
            axis for axis in range(num_qubits) if axis not in (gate.left_site, gate.left_site + 1)
        ]
        axes = [gate.left_site, gate.left_site + 1, *remaining_axes]
        inverse_axes = [axes.index(axis) for axis in range(num_qubits)]
        local = state.reshape((2,) * num_qubits).transpose(axes).reshape(4, -1)
        updated = (gate.matrix @ local).reshape((2, 2, *((2,) * (num_qubits - 2))))
        state = updated.transpose(inverse_axes).reshape(-1)
    return np.asarray(state, dtype=COMPLEX_DTYPE)


__all__ = [
    "LoadedOrderedU4Archive",
    "ORDERED_U4_EXECUTION_ORDER",
    "ORDERED_U4_INDEX_DTYPE",
    "ORDERED_U4_MATRIX_DTYPE",
    "ORDERED_U4_NPZ_SCHEMA_VERSION",
    "ORDERED_U4_UNITARITY_ATOL",
    "OrderedU4Archive",
    "OrderedU4Gate",
    "build_ordered_u4_archive",
    "canonical_json_metadata",
    "load_ordered_u4_npz",
    "ordered_u4_archive_from_staircase",
    "ordered_u4_semantic_sha256",
    "replay_ordered_u4_archive",
    "save_ordered_u4_npz",
    "sha256_file",
    "validate_ordered_u4_archive",
]
