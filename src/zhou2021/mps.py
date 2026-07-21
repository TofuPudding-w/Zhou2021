"""Strict open-boundary matrix-product-state primitives for Zhou 2021.

The paper obtains its large-system targets as MPSs, whereas the first ADQC
smoke tests use dense statevectors.  This module is the deliberately separate
bridge between those two regimes.  Its tensor convention is
``(chi_left, physical, chi_right)`` with a qubit physical dimension of two and
the first tensor corresponding to q0 (the most-significant qubit).

No method below silently materialises a dense vector.  Dense conversion is an
explicit small-system diagnostic with a conservative width guard.  Therefore a
valid ``N=48, chi=64`` target can be stored, normalised, contracted, and
updated by a nearest-neighbour gate without allocating ``2**48`` amplitudes.

``random_open_boundary_mps`` is a reproducible *random-MPS ensemble choice*
for the paper's random-MPS experiment.  The article does not specify a unique
sampling measure, so this implementation records the choice explicitly:
i.i.d. complex-normal cores followed by exact QR canonicalisation and global
normalisation.  It is not presented as an author-supplied ensemble.
"""

from __future__ import annotations

import json
import math
import numbers
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .conventions import COMPLEX_DTYPE, infer_num_qubits, validate_num_qubits, validate_real

Tensor = NDArray[np.complex128]

MPS_NPZ_SCHEMA_VERSION = "zhou2021-obc-mps-npz-v1"
MPS_NPZ_AXIS_ORDER = "chi_left,physical,chi_right"
MPS_NPZ_QUBIT_ORDER = "q0-most-significant"


def _validate_positive_integer(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise TypeError(f"{name} must be an integer.")
    result = int(value)
    if result < 1:
        raise ValueError(f"{name} must be positive, got {result}.")
    return result


def _validate_optional_bond_dimension(value: int | None) -> int | None:
    if value is None:
        return None
    return _validate_positive_integer(value, name="max_bond_dimension")


def _validate_seed(seed: int | None) -> int | None:
    if seed is None:
        return None
    if isinstance(seed, bool) or not isinstance(seed, numbers.Integral):
        raise TypeError("seed must be an integer or None.")
    return int(seed)


def _frozen_complex_copy(value: ArrayLike) -> Tensor:
    result = np.array(value, dtype=COMPLEX_DTYPE, copy=True)
    result.flags.writeable = False
    return result


def _validate_two_qubit_matrix(
    matrix: ArrayLike,
    *,
    require_unitary: bool,
    unitary_atol: float,
) -> Tensor:
    unitary_atol = validate_real(unitary_atol, name="unitary_atol")
    if unitary_atol < 0.0:
        raise ValueError("unitary_atol must be nonnegative.")
    result = np.asarray(matrix, dtype=COMPLEX_DTYPE)
    if result.shape != (4, 4):
        raise ValueError(f"two-qubit matrix must have shape (4, 4), got {result.shape}.")
    if not np.all(np.isfinite(result)):
        raise ValueError("two-qubit matrix contains non-finite entries.")
    if require_unitary:
        residual = float(np.linalg.norm(result.conj().T @ result - np.eye(4), ord=np.inf))
        if residual > unitary_atol:
            raise ValueError(
                "two-qubit matrix must be unitary; "
                f"infinity-norm residual={residual:.3e}."
            )
    return np.array(result, dtype=COMPLEX_DTYPE, copy=True)


def open_boundary_bond_dimensions(
    num_qubits: int,
    max_bond_dimension: int,
) -> tuple[int, ...]:
    """Return the rank-feasible OBC bond dimensions for qubit MPS cores.

    The result has length ``N+1`` and includes the two boundary dimensions.
    Each internal bond is bounded both by ``chi`` and by the Hilbert-space
    dimensions available to its left and right.  This prevents a random MPS
    generator from allocating mathematically unreachable bond indices.
    """

    num_qubits = validate_num_qubits(num_qubits)
    max_bond_dimension = _validate_positive_integer(
        max_bond_dimension,
        name="max_bond_dimension",
    )
    return tuple(
        1
        if bond in (0, num_qubits)
        else min(max_bond_dimension, 1 << bond, 1 << (num_qubits - bond))
        for bond in range(num_qubits + 1)
    )


@dataclass(frozen=True, slots=True)
class OpenBoundaryMPS:
    """Validated qubit MPS with immutable core storage and strict OBC edges."""

    tensors: tuple[Tensor, ...]

    def __post_init__(self) -> None:
        raw_tensors = tuple(self.tensors)
        if not raw_tensors:
            raise ValueError("An open-boundary MPS must contain at least one tensor.")

        validated: list[Tensor] = []
        previous_right = 1
        for site, raw_tensor in enumerate(raw_tensors):
            tensor = np.asarray(raw_tensor, dtype=COMPLEX_DTYPE)
            if tensor.ndim != 3:
                raise ValueError(
                    f"MPS tensor at site {site} must have rank 3, got shape {tensor.shape}."
                )
            left, physical, right = (int(dim) for dim in tensor.shape)
            if left < 1 or right < 1:
                raise ValueError(f"MPS tensor at site {site} has a nonpositive bond dimension.")
            if physical != 2:
                raise ValueError(
                    "This qubit MPS implementation requires physical dimension 2; "
                    f"site {site} has {physical}."
                )
            if left != previous_right:
                raise ValueError(
                    f"MPS bond mismatch at site {site}: left={left}, "
                    f"previous right={previous_right}."
                )
            if not np.all(np.isfinite(tensor)):
                raise ValueError(f"MPS tensor at site {site} contains non-finite entries.")
            validated.append(_frozen_complex_copy(tensor))
            previous_right = right

        if previous_right != 1:
            raise ValueError(
                "An open-boundary MPS must have right boundary bond dimension 1; "
                f"got {previous_right}."
            )
        object.__setattr__(self, "tensors", tuple(validated))

    @property
    def num_qubits(self) -> int:
        """Number of physical qubit sites."""

        return len(self.tensors)

    @property
    def bond_dimensions(self) -> tuple[int, ...]:
        """Boundary-inclusive OBC bond dimensions ``(1, ..., 1)``."""

        return (1, *(int(tensor.shape[2]) for tensor in self.tensors))

    @property
    def max_bond_dimension(self) -> int:
        """Largest virtual bond dimension, including the boundary value one."""

        return max(self.bond_dimensions)

    def copy(self) -> OpenBoundaryMPS:
        """Return an independent immutable-core copy."""

        return OpenBoundaryMPS(tuple(np.array(tensor, copy=True) for tensor in self.tensors))

    def inner_product(self, other: OpenBoundaryMPS) -> complex:
        """Return ``<self|other>`` by transfer-matrix contraction only."""

        if not isinstance(other, OpenBoundaryMPS):
            raise TypeError("other must be an OpenBoundaryMPS.")
        if self.num_qubits != other.num_qubits:
            raise ValueError(
                "Cannot contract MPSs with different site counts: "
                f"{self.num_qubits} and {other.num_qubits}."
            )
        environment = np.ones((1, 1), dtype=COMPLEX_DTYPE)
        for bra, ket in zip(self.tensors, other.tensors, strict=True):
            environment = np.einsum(
                "ab,asr,bsq->rq",
                environment,
                bra.conj(),
                ket,
                optimize=True,
            )
        return complex(environment[0, 0])

    def norm_squared(self) -> float:
        """Return the nonnegative squared norm from an MPS-only contraction."""

        value = self.inner_product(self)
        imaginary_residual = abs(value.imag)
        scale = max(1.0, abs(value.real))
        if imaginary_residual > 1e-10 * scale:
            raise ArithmeticError(
                "Self-overlap acquired an unexpectedly large imaginary component: "
                f"{value.imag:.3e}."
            )
        if value.real < -1e-12 * scale:
            raise ArithmeticError(f"Self-overlap is negative: {value.real:.17g}.")
        return max(0.0, float(value.real))

    def norm(self) -> float:
        """Return the Euclidean norm without creating a statevector."""

        return math.sqrt(self.norm_squared())

    def normalized(self) -> OpenBoundaryMPS:
        """Return an equivalent unit-norm MPS by scaling only its last core.

        Choosing the last core preserves a preceding left-canonical QR gauge,
        which is important for stable random-MPS targets.
        """

        norm = self.norm()
        if not math.isfinite(norm) or norm == 0.0:
            raise ValueError("Cannot normalise a zero-norm MPS.")
        updated = [np.array(tensor, copy=True) for tensor in self.tensors]
        updated[-1] /= norm
        return OpenBoundaryMPS(tuple(updated))

    def scaled(self, scalar: complex) -> OpenBoundaryMPS:
        """Return the MPS multiplied by one finite complex scalar."""

        scalar = complex(scalar)
        if not math.isfinite(scalar.real) or not math.isfinite(scalar.imag):
            raise ValueError("scalar must have finite real and imaginary parts.")
        updated = [np.array(tensor, copy=True) for tensor in self.tensors]
        updated[0] *= scalar
        return OpenBoundaryMPS(tuple(updated))

    def left_canonicalized(self) -> OpenBoundaryMPS:
        """QR-canonicalise all but the last core while preserving the state exactly.

        The operation can reduce an overcomplete bond dimension.  It never
        truncates nonzero singular directions, so it is suitable for stable
        random-MPS generation and independent small-system checks.
        """

        updated = [np.array(tensor, dtype=COMPLEX_DTYPE, copy=True) for tensor in self.tensors]
        for site in range(self.num_qubits - 1):
            left, physical, right = updated[site].shape
            matrix = updated[site].reshape(left * physical, right)
            q_factor, r_factor = np.linalg.qr(matrix, mode="reduced")
            retained_right = q_factor.shape[1]
            updated[site] = q_factor.reshape(left, physical, retained_right)
            updated[site + 1] = np.tensordot(r_factor, updated[site + 1], axes=(1, 0))
        return OpenBoundaryMPS(tuple(updated))

    def left_canonical_residual(self) -> float:
        """Return the largest ``||A^dagger A-I||_inf`` over left-canonical cores."""

        residual = 0.0
        for tensor in self.tensors[:-1]:
            matrix = tensor.reshape(tensor.shape[0] * 2, tensor.shape[2])
            local = float(
                np.linalg.norm(
                    matrix.conj().T @ matrix - np.eye(matrix.shape[1]),
                    ord=np.inf,
                )
            )
            residual = max(residual, local)
        return residual

    def mixed_canonicalized_for_two_site(self, left_site: int) -> OpenBoundaryMPS:
        """Gauge-fix around ``left_site,left_site+1`` without truncating state data.

        Cores strictly left of ``left_site`` become left isometric and cores
        strictly right of ``left_site+1`` become right isometric.  Consequently
        the singular values of the centre's two-site matrix are physical
        Schmidt values and a subsequent local SVD truncation has an exact
        global squared-norm ledger.  This costs one QR sweep from each edge.
        """

        if isinstance(left_site, bool) or not isinstance(left_site, numbers.Integral):
            raise TypeError("left_site must be an integer.")
        left_site = int(left_site)
        if not 0 <= left_site < self.num_qubits - 1:
            raise ValueError(
                f"left_site={left_site} is invalid for N={self.num_qubits}."
            )
        tensors = [np.array(tensor, dtype=COMPLEX_DTYPE, copy=True) for tensor in self.tensors]

        # Put the prefix in left-canonical gauge, absorbing each R rightward.
        for site in range(left_site):
            left, physical, _ = tensors[site].shape
            q_factor, r_factor = np.linalg.qr(
                tensors[site].reshape(left * physical, -1),
                mode="reduced",
            )
            retained_right = q_factor.shape[1]
            tensors[site] = q_factor.reshape(left, physical, retained_right)
            tensors[site + 1] = np.tensordot(r_factor, tensors[site + 1], axes=(1, 0))

        # Put the suffix in right-canonical gauge, absorbing each L leftward.
        for site in range(self.num_qubits - 1, left_site + 1, -1):
            left, physical, right = tensors[site].shape
            q_factor, r_factor = np.linalg.qr(
                tensors[site].reshape(left, physical * right).T,
                mode="reduced",
            )
            retained_left = q_factor.shape[1]
            tensors[site] = q_factor.T.reshape(retained_left, physical, right)
            tensors[site - 1] = np.tensordot(tensors[site - 1], r_factor.T, axes=(2, 0))
        return OpenBoundaryMPS(tuple(tensors))

    def mixed_canonical_residual_for_two_site(self, left_site: int) -> float:
        """Return the isometry residual outside a declared two-site centre."""

        if isinstance(left_site, bool) or not isinstance(left_site, numbers.Integral):
            raise TypeError("left_site must be an integer.")
        left_site = int(left_site)
        if not 0 <= left_site < self.num_qubits - 1:
            raise ValueError(
                f"left_site={left_site} is invalid for N={self.num_qubits}."
            )
        residual = 0.0
        for tensor in self.tensors[:left_site]:
            matrix = tensor.reshape(tensor.shape[0] * 2, tensor.shape[2])
            residual = max(
                residual,
                float(
                    np.linalg.norm(
                        matrix.conj().T @ matrix - np.eye(matrix.shape[1]),
                        ord=np.inf,
                    )
                ),
            )
        for tensor in self.tensors[left_site + 2 :]:
            matrix = tensor.reshape(tensor.shape[0], 2 * tensor.shape[2])
            residual = max(
                residual,
                float(
                    np.linalg.norm(
                        matrix @ matrix.conj().T - np.eye(matrix.shape[0]),
                        ord=np.inf,
                    )
                ),
            )
        return residual

    def to_dense(self, *, max_num_qubits: int | None = 20) -> Tensor:
        """Materialise q0-MSB amplitudes for an explicitly permitted small system.

        ``max_num_qubits=20`` is intentionally conservative.  Passing
        ``None`` is an explicit caller acknowledgement that dense allocation
        may be infeasible; no large-system workflow should do that.
        """

        if max_num_qubits is not None:
            max_num_qubits = _validate_positive_integer(
                max_num_qubits,
                name="max_num_qubits",
            )
            if self.num_qubits > max_num_qubits:
                raise ValueError(
                    "Dense conversion is disabled for this MPS: "
                    f"N={self.num_qubits} exceeds max_num_qubits={max_num_qubits}."
                )

        state = np.array(self.tensors[0][0, :, :], dtype=COMPLEX_DTYPE, copy=True)
        for tensor in self.tensors[1:]:
            state = np.tensordot(state, tensor, axes=(-1, 0))
        return np.array(state[..., 0].reshape(-1), dtype=COMPLEX_DTYPE, copy=True)


@dataclass(frozen=True, slots=True)
class TTSVDResult:
    """Auditable result of a left-to-right TT-SVD statevector decomposition."""

    mps: OpenBoundaryMPS
    input_norm_squared: float
    discarded_weight: float
    retained_ranks: tuple[int, ...]
    max_bond_dimension: int | None
    singular_value_cutoff: float

    @property
    def retained_norm_squared(self) -> float:
        """Norm predicted by the SVD truncation ledger before roundoff."""

        return max(0.0, self.input_norm_squared - self.discarded_weight)


@dataclass(frozen=True, slots=True)
class TwoSiteUpdateResult:
    """Result and audited SVD accounting for one neighbouring two-qubit update."""

    mps: OpenBoundaryMPS
    left_site: int
    retained_rank: int
    discarded_weight: float
    discarded_weight_is_global_norm_loss: bool


@dataclass(frozen=True, slots=True)
class RandomMPSEnsemble:
    """Seed-recorded collection from the explicit complex-normal MPS ensemble."""

    members: tuple[OpenBoundaryMPS, ...]
    requested_seed: int | None
    child_seeds: tuple[int, ...]
    max_bond_dimension: int

    def __len__(self) -> int:
        return len(self.members)

    def __iter__(self):
        return iter(self.members)

    def __getitem__(self, index: int) -> OpenBoundaryMPS:
        return self.members[index]


@dataclass(frozen=True, slots=True)
class LoadedOpenBoundaryMPS:
    """Strictly parsed external MPS artifact and its JSON-only provenance metadata."""

    mps: OpenBoundaryMPS
    metadata: dict[str, Any]
    path: Path


def _canonical_metadata_json(metadata: Mapping[str, Any] | None) -> str:
    if metadata is None:
        metadata = {}
    if not isinstance(metadata, Mapping):
        raise TypeError("metadata must be a mapping or None.")
    try:
        encoded = json.dumps(
            dict(metadata),
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as error:
        raise ValueError("metadata must be JSON serialisable with finite values.") from error
    decoded = json.loads(encoded)
    if not isinstance(decoded, dict):  # pragma: no cover - mapping-to-dict invariant
        raise ValueError("metadata must encode to one JSON object.")
    return encoded


def save_open_boundary_mps_npz(
    path: str | Path,
    mps: OpenBoundaryMPS,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Save an externally portable strict-OBC MPS target as a safe NPZ artifact.

    Schema ``zhou2021-obc-mps-npz-v1`` contains scalar convention fields,
    ``tensor_000`` through ``tensor_{N-1:03d}``, and canonical JSON metadata.
    No pickle/object arrays are written.  Archive bytes are not claimed to be
    deterministic across NumPy/ZIP versions; record a SHA-256 of the written
    file in the experiment manifest when ingesting an external DMRG target.
    """

    if not isinstance(mps, OpenBoundaryMPS):
        raise TypeError("mps must be an OpenBoundaryMPS.")
    destination = Path(path)
    if destination.suffix.lower() != ".npz":
        raise ValueError("MPS NPZ destination path must end in '.npz'.")
    if not destination.parent.is_dir():
        raise FileNotFoundError(f"Parent directory does not exist: {destination.parent}")
    payload: dict[str, Any] = {
        "schema_version": np.array(MPS_NPZ_SCHEMA_VERSION),
        "axis_order": np.array(MPS_NPZ_AXIS_ORDER),
        "qubit_order": np.array(MPS_NPZ_QUBIT_ORDER),
        "dtype": np.array("complex128"),
        "num_qubits": np.array(mps.num_qubits, dtype=np.int64),
        "tensor_count": np.array(mps.num_qubits, dtype=np.int64),
        "metadata_json": np.array(_canonical_metadata_json(metadata)),
    }
    payload.update(
        {
            f"tensor_{site:03d}": np.array(tensor, dtype=COMPLEX_DTYPE, copy=True)
            for site, tensor in enumerate(mps.tensors)
        }
    )
    np.savez_compressed(destination, **payload)
    return destination


def _required_npz_scalar(archive: Any, key: str) -> str:
    if key not in archive.files:
        raise ValueError(f"MPS NPZ artifact is missing required key {key!r}.")
    value = np.asarray(archive[key])
    if value.ndim != 0:
        raise ValueError(f"MPS NPZ key {key!r} must be a scalar, got shape {value.shape}.")
    return str(value.item())


def load_open_boundary_mps_npz(path: str | Path) -> LoadedOpenBoundaryMPS:
    """Safely load and fully validate an external MPS target NPZ artifact."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"MPS NPZ artifact does not exist: {source}")
    try:
        archive_context = np.load(source, allow_pickle=False)
    except (OSError, ValueError) as error:
        raise ValueError(f"Unable to open safe MPS NPZ artifact {source}: {error}") from error
    if not hasattr(archive_context, "files"):
        raise ValueError(f"MPS artifact {source} is not an NPZ archive.")
    with archive_context as archive:
        schema = _required_npz_scalar(archive, "schema_version")
        if schema != MPS_NPZ_SCHEMA_VERSION:
            raise ValueError(
                "Unsupported MPS NPZ schema: "
                f"expected {MPS_NPZ_SCHEMA_VERSION!r}, got {schema!r}."
            )
        if _required_npz_scalar(archive, "axis_order") != MPS_NPZ_AXIS_ORDER:
            raise ValueError("MPS NPZ axis_order disagrees with chi_left,physical,chi_right.")
        if _required_npz_scalar(archive, "qubit_order") != MPS_NPZ_QUBIT_ORDER:
            raise ValueError("MPS NPZ qubit_order disagrees with q0-most-significant.")
        if _required_npz_scalar(archive, "dtype") != "complex128":
            raise ValueError("MPS NPZ dtype must be complex128.")
        try:
            num_qubits = int(_required_npz_scalar(archive, "num_qubits"))
            tensor_count = int(_required_npz_scalar(archive, "tensor_count"))
        except ValueError as error:
            raise ValueError("MPS NPZ num_qubits/tensor_count must be integer scalars.") from error
        num_qubits = validate_num_qubits(num_qubits)
        if tensor_count != num_qubits:
            raise ValueError(
                "MPS NPZ tensor_count must equal num_qubits; "
                f"got {tensor_count} and {num_qubits}."
            )
        tensor_keys = {f"tensor_{site:03d}" for site in range(num_qubits)}
        required_keys = {
            "schema_version",
            "axis_order",
            "qubit_order",
            "dtype",
            "num_qubits",
            "tensor_count",
            "metadata_json",
            *tensor_keys,
        }
        actual_keys = set(archive.files)
        if actual_keys != required_keys:
            missing = sorted(required_keys - actual_keys)
            unexpected = sorted(actual_keys - required_keys)
            raise ValueError(
                "MPS NPZ key set is invalid; "
                f"missing={missing}, unexpected={unexpected}."
            )
        metadata_raw = _required_npz_scalar(archive, "metadata_json")
        try:
            metadata = json.loads(metadata_raw)
        except json.JSONDecodeError as error:
            raise ValueError("MPS NPZ metadata_json is not valid JSON.") from error
        if not isinstance(metadata, dict):
            raise ValueError("MPS NPZ metadata_json must encode one JSON object.")
        # ``json.loads`` accepts NaN/Infinity spellings by default; pass the
        # decoded object through the same strict serializer used by writers so
        # externally supplied provenance cannot contain non-finite values.
        _canonical_metadata_json(metadata)
        mps = OpenBoundaryMPS(
            tuple(
                np.array(archive[f"tensor_{site:03d}"], dtype=COMPLEX_DTYPE, copy=True)
                for site in range(num_qubits)
            )
        )
    return LoadedOpenBoundaryMPS(mps=mps, metadata=metadata, path=source)


def _validate_statevector_for_tt_svd(statevector: ArrayLike) -> tuple[Tensor, int, float]:
    vector = np.asarray(statevector, dtype=COMPLEX_DTYPE).reshape(-1)
    num_qubits = infer_num_qubits(vector)
    if not np.all(np.isfinite(vector)):
        raise ValueError("Statevector contains non-finite entries.")
    norm_squared = float(np.vdot(vector, vector).real)
    if not math.isfinite(norm_squared) or norm_squared <= 0.0:
        raise ValueError("Statevector must have a finite nonzero norm.")
    return np.array(vector, dtype=COMPLEX_DTYPE, copy=True), num_qubits, norm_squared


def tt_svd_from_dense_statevector(
    statevector: ArrayLike,
    *,
    max_bond_dimension: int | None = None,
    singular_value_cutoff: float = 0.0,
) -> TTSVDResult:
    """Decompose a q0-MSB dense statevector by deterministic left-to-right TT-SVD.

    The cutoff is absolute in the statevector norm convention.  If it would
    discard every singular value at one split, the leading value is retained so
    the approximation remains a nonzero state.  With the defaults this is an
    exact (up to SVD roundoff) MPS representation for small systems.
    """

    max_bond_dimension = _validate_optional_bond_dimension(max_bond_dimension)
    singular_value_cutoff = validate_real(
        singular_value_cutoff,
        name="singular_value_cutoff",
    )
    if singular_value_cutoff < 0.0:
        raise ValueError("singular_value_cutoff must be nonnegative.")

    vector, num_qubits, input_norm_squared = _validate_statevector_for_tt_svd(statevector)
    if num_qubits == 1:
        return TTSVDResult(
            mps=OpenBoundaryMPS((vector.reshape(1, 2, 1),)),
            input_norm_squared=input_norm_squared,
            discarded_weight=0.0,
            retained_ranks=(),
            max_bond_dimension=max_bond_dimension,
            singular_value_cutoff=singular_value_cutoff,
        )

    work = vector.reshape(1, -1)
    left_bond = 1
    tensors: list[Tensor] = []
    retained_ranks: list[int] = []
    discarded_weight = 0.0
    for _site in range(num_qubits - 1):
        work = work.reshape(left_bond * 2, -1)
        left_vectors, singular_values, right_vectors = np.linalg.svd(
            work,
            full_matrices=False,
        )
        cutoff_rank = int(np.count_nonzero(singular_values > singular_value_cutoff))
        retained_rank = max(1, cutoff_rank)
        if max_bond_dimension is not None:
            retained_rank = min(retained_rank, max_bond_dimension)
        discarded_weight += float(np.sum(np.abs(singular_values[retained_rank:]) ** 2))
        tensors.append(
            np.asarray(
                left_vectors[:, :retained_rank].reshape(left_bond, 2, retained_rank),
                dtype=COMPLEX_DTYPE,
            )
        )
        work = singular_values[:retained_rank, np.newaxis] * right_vectors[:retained_rank, :]
        left_bond = retained_rank
        retained_ranks.append(retained_rank)

    tensors.append(np.asarray(work.reshape(left_bond, 2, 1), dtype=COMPLEX_DTYPE))
    return TTSVDResult(
        mps=OpenBoundaryMPS(tuple(tensors)),
        input_norm_squared=input_norm_squared,
        discarded_weight=discarded_weight,
        retained_ranks=tuple(retained_ranks),
        max_bond_dimension=max_bond_dimension,
        singular_value_cutoff=singular_value_cutoff,
    )


def mps_from_dense_statevector(
    statevector: ArrayLike,
    *,
    max_bond_dimension: int | None = None,
    singular_value_cutoff: float = 0.0,
) -> OpenBoundaryMPS:
    """Convenience wrapper returning only the MPS part of TT-SVD."""

    return tt_svd_from_dense_statevector(
        statevector,
        max_bond_dimension=max_bond_dimension,
        singular_value_cutoff=singular_value_cutoff,
    ).mps


def product_zero_mps(num_qubits: int) -> OpenBoundaryMPS:
    """Return the exact MPS representation of ``|0...0>`` without dense storage."""

    num_qubits = validate_num_qubits(num_qubits)
    core = np.zeros((1, 2, 1), dtype=COMPLEX_DTYPE)
    core[0, 0, 0] = 1.0
    return OpenBoundaryMPS(tuple(np.array(core, copy=True) for _ in range(num_qubits)))


def random_open_boundary_mps(
    num_qubits: int,
    max_bond_dimension: int,
    *,
    seed: int | None = None,
) -> OpenBoundaryMPS:
    """Sample one normalized complex-normal OBC MPS at feasible bond ranks.

    This is the documented random-MPS ensemble used for reproduction sweeps,
    not a claim about a uniquely specified distribution in the paper.
    """

    num_qubits = validate_num_qubits(num_qubits)
    max_bond_dimension = _validate_positive_integer(
        max_bond_dimension,
        name="max_bond_dimension",
    )
    seed = _validate_seed(seed)
    bonds = open_boundary_bond_dimensions(num_qubits, max_bond_dimension)
    generator = np.random.default_rng(seed)
    tensors = []
    for site in range(num_qubits):
        shape = (bonds[site], 2, bonds[site + 1])
        tensor = (
            generator.standard_normal(shape) + 1j * generator.standard_normal(shape)
        ) / math.sqrt(2.0)
        tensors.append(np.asarray(tensor, dtype=COMPLEX_DTYPE))
    return OpenBoundaryMPS(tuple(tensors)).left_canonicalized().normalized()


def random_mps_ensemble(
    num_qubits: int,
    max_bond_dimension: int,
    num_samples: int,
    *,
    seed: int | None = None,
) -> RandomMPSEnsemble:
    """Return a provenance-recorded independent random-MPS collection."""

    num_qubits = validate_num_qubits(num_qubits)
    max_bond_dimension = _validate_positive_integer(
        max_bond_dimension,
        name="max_bond_dimension",
    )
    num_samples = _validate_positive_integer(num_samples, name="num_samples")
    seed = _validate_seed(seed)
    root = np.random.SeedSequence(seed)
    children = root.spawn(num_samples)
    child_seeds = tuple(
        int(child.generate_state(1, dtype=np.uint64)[0]) for child in children
    )
    members = tuple(
        random_open_boundary_mps(
            num_qubits,
            max_bond_dimension,
            seed=child_seed,
        )
        for child_seed in child_seeds
    )
    return RandomMPSEnsemble(
        members=members,
        requested_seed=seed,
        child_seeds=child_seeds,
        max_bond_dimension=max_bond_dimension,
    )


def apply_adjacent_two_qubit_gate(
    mps: OpenBoundaryMPS,
    matrix: ArrayLike,
    left_site: int,
    *,
    max_bond_dimension: int | None = None,
    singular_value_cutoff: float = 0.0,
    require_unitary: bool = True,
    unitary_atol: float = 1e-11,
    canonicalize_for_truncation: bool = True,
) -> TwoSiteUpdateResult:
    """Apply an ordered U(4) on q0-MSB sites ``left_site,left_site+1``.

    The matrix uses rows and columns ordered as ``00, 01, 10, 11``.  A local
    SVD restores OBC MPS form and records every discarded squared singular
    value.  By default a rank-limited/cutoff update first performs a QR
    mixed-canonical gauge sweep, so ``discarded_weight`` equals the global
    squared-norm loss exactly (up to roundoff).  Disabling that sweep is an
    advanced performance option; its result marks the ledger as local only.
    With no rank limit/cutoff, a unitary update is exact up to SVD roundoff.
    """

    if not isinstance(mps, OpenBoundaryMPS):
        raise TypeError("mps must be an OpenBoundaryMPS.")
    if isinstance(left_site, bool) or not isinstance(left_site, numbers.Integral):
        raise TypeError("left_site must be an integer.")
    left_site = int(left_site)
    if not 0 <= left_site < mps.num_qubits - 1:
        raise ValueError(
            f"left_site={left_site} is invalid for N={mps.num_qubits}."
        )
    max_bond_dimension = _validate_optional_bond_dimension(max_bond_dimension)
    singular_value_cutoff = validate_real(
        singular_value_cutoff,
        name="singular_value_cutoff",
    )
    if singular_value_cutoff < 0.0:
        raise ValueError("singular_value_cutoff must be nonnegative.")
    if not isinstance(canonicalize_for_truncation, bool):
        raise TypeError("canonicalize_for_truncation must be a bool.")
    gate = _validate_two_qubit_matrix(
        matrix,
        require_unitary=require_unitary,
        unitary_atol=unitary_atol,
    )

    truncation_requested = max_bond_dimension is not None or singular_value_cutoff > 0.0
    if truncation_requested and canonicalize_for_truncation:
        working_mps = mps.mixed_canonicalized_for_two_site(left_site)
    else:
        working_mps = mps
    first = working_mps.tensors[left_site]
    second = working_mps.tensors[left_site + 1]
    left_bond, _, _ = first.shape
    _, _, right_bond = second.shape
    merged = np.einsum("asm,mtn->astn", first, second, optimize=True)
    reshaped_gate = gate.reshape(2, 2, 2, 2)
    updated = np.einsum("uvst,lstr->luvr", reshaped_gate, merged, optimize=True)
    matrix_form = updated.reshape(left_bond * 2, 2 * right_bond)
    left_vectors, singular_values, right_vectors = np.linalg.svd(
        matrix_form,
        full_matrices=False,
    )
    cutoff_rank = int(np.count_nonzero(singular_values > singular_value_cutoff))
    retained_rank = max(1, cutoff_rank)
    if max_bond_dimension is not None:
        retained_rank = min(retained_rank, max_bond_dimension)
    discarded_weight = float(np.sum(np.abs(singular_values[retained_rank:]) ** 2))
    first_updated = left_vectors[:, :retained_rank].reshape(left_bond, 2, retained_rank)
    second_updated = (
        singular_values[:retained_rank, np.newaxis] * right_vectors[:retained_rank, :]
    ).reshape(retained_rank, 2, right_bond)

    tensors = [
        np.array(tensor, dtype=COMPLEX_DTYPE, copy=True) for tensor in working_mps.tensors
    ]
    tensors[left_site] = np.asarray(first_updated, dtype=COMPLEX_DTYPE)
    tensors[left_site + 1] = np.asarray(second_updated, dtype=COMPLEX_DTYPE)
    return TwoSiteUpdateResult(
        mps=OpenBoundaryMPS(tuple(tensors)),
        left_site=left_site,
        retained_rank=retained_rank,
        discarded_weight=discarded_weight,
        discarded_weight_is_global_norm_loss=truncation_requested
        and canonicalize_for_truncation,
    )


def mps_squared_fidelity(left: OpenBoundaryMPS, right: OpenBoundaryMPS) -> float:
    """Return global squared-state fidelity from MPS contractions only."""

    if not isinstance(left, OpenBoundaryMPS) or not isinstance(right, OpenBoundaryMPS):
        raise TypeError("left and right must both be OpenBoundaryMPS instances.")
    denominator = left.norm_squared() * right.norm_squared()
    if denominator <= 0.0:
        raise ValueError("Squared fidelity is undefined for a zero-norm MPS.")
    value = abs(left.inner_product(right)) ** 2 / denominator
    if value > 1.0 and value <= 1.0 + 2e-12:
        return 1.0
    return float(value)


__all__ = [
    "LoadedOpenBoundaryMPS",
    "MPS_NPZ_AXIS_ORDER",
    "MPS_NPZ_QUBIT_ORDER",
    "MPS_NPZ_SCHEMA_VERSION",
    "OpenBoundaryMPS",
    "RandomMPSEnsemble",
    "TTSVDResult",
    "TwoSiteUpdateResult",
    "apply_adjacent_two_qubit_gate",
    "load_open_boundary_mps_npz",
    "mps_from_dense_statevector",
    "mps_squared_fidelity",
    "open_boundary_bond_dimensions",
    "product_zero_mps",
    "random_mps_ensemble",
    "random_open_boundary_mps",
    "save_open_boundary_mps_npz",
    "tt_svd_from_dense_statevector",
]
