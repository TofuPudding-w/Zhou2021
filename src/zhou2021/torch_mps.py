"""Autograd-preserving torch MPS operations for Zhou 2021 paper-scale ADQC.

This module is intentionally independent of :mod:`zhou2021.adqc`, whose first
implementation is a dense-statevector smoke path.  It provides the primitive
needed to move the same latent U(4) staircase to MPS targets such as the
paper's ``N=48, chi=64`` ground states:

* adjacent ordered U(4) application with a rank-robust exact split or a
  guarded differentiable truncated-SVD split;
* MPS-only overlap, global squared fidelity, and Zhou Eq. (1) loss;
* a serial staircase executor accepting autograd-connected projected gates.

Tensor order and gate order are the same as the NumPy MPS module: cores are
``(chi_left, physical, chi_right)``, q0 is the first/M.S.B. site, and a U(4)
uses ``00,01,10,11`` row/column order.  ``max_bond_dimension`` is an explicit
approximation coordinate.  It must be recorded in any paper-scale run: an
uncompressed L-layer staircase can require a bond up to ``4**L`` and is not
feasible at L=8 for N=48.
"""

from __future__ import annotations

import math
import numbers
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
import torch

from .conventions import validate_num_qubits, validate_real
from .mps import OpenBoundaryMPS, open_boundary_bond_dimensions

# ``torch.linalg.svd`` differentiates singular *vectors*.  Its documented
# backward therefore becomes undefined at repeated singular values and, in
# particular, at a rank-deficient matrix with a degenerate null space.  An MPS
# update sees those matrices naturally (for example an identity U(4) acting on
# |00>), so silently delegating every split to ``torch.linalg.svd`` or QR can
# turn an otherwise finite optimisation into all-NaN gradients.  These
# thresholds are deliberately conservative: the alternative at a hard rank
# boundary is an explicit diagnostic, never an unrecorded arbitrary gradient.
_SPECTRAL_ZERO_RELATIVE_TOLERANCE = 1024.0 * np.finfo(np.float64).eps
_SPECTRAL_GAP_RELATIVE_TOLERANCE = 1e-10


class TorchMPSDifferentiabilityError(RuntimeError):
    """Raised before an ill-defined MPS split can poison an ADQC backward pass.

    A hard truncated SVD is not differentiable when the retained/discarded
    singular subspaces meet at a degenerate positive singular value.  PyTorch
    also returns NaNs for some otherwise well-defined maps when its full SVD
    backward encounters a degenerate null space.  Exact rank-deficient states
    with a separated nonzero subspace use the stable tangent construction
    below; all remaining unsafe cases fail before ``loss.backward()`` and
    therefore before any optimiser can update a latent U(4) parameter.
    """


@dataclass(frozen=True, slots=True)
class _DetachedSVDDiagnostics:
    """Detached spectral facts used only to select a safe local split chart."""

    singular_values: torch.Tensor
    numerical_rank: int
    zero_tolerance: float
    gap_tolerance: float
    minimum_adjacent_gap: float


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


def _validate_cutoff(value: float) -> float:
    value = validate_real(value, name="singular_value_cutoff")
    if value < 0.0:
        raise ValueError("singular_value_cutoff must be nonnegative.")
    return value


def _detached_svd_diagnostics(matrix: torch.Tensor) -> _DetachedSVDDiagnostics:
    """Return reproducible rank/gap diagnostics without creating an autograd graph.

    The singular values are intentionally calculated from a detached matrix:
    they decide a discrete rank chart and are not part of the loss derivative.
    This avoids constructing the known-unsafe full-SVD backward graph merely
    to discover that it cannot be used.
    """

    singular_values = torch.linalg.svdvals(matrix.detach())
    if singular_values.ndim != 1 or singular_values.numel() == 0:
        raise AssertionError("A two-site matrix must have at least one singular value.")
    if not bool(torch.all(torch.isfinite(singular_values)).cpu()):
        raise ValueError("Two-site matrix has non-finite singular values.")
    scale = max(float(singular_values[0].cpu()), np.finfo(np.float64).tiny)
    zero_tolerance = _SPECTRAL_ZERO_RELATIVE_TOLERANCE * scale
    gap_tolerance = _SPECTRAL_GAP_RELATIVE_TOLERANCE * scale
    numerical_rank = int(torch.count_nonzero(singular_values > zero_tolerance).cpu())
    if singular_values.numel() == 1:
        minimum_adjacent_gap = float("inf")
    else:
        minimum_adjacent_gap = float(
            torch.min(torch.abs(singular_values[:-1] - singular_values[1:])).cpu()
        )
    return _DetachedSVDDiagnostics(
        singular_values=singular_values,
        numerical_rank=numerical_rank,
        zero_tolerance=zero_tolerance,
        gap_tolerance=gap_tolerance,
        minimum_adjacent_gap=minimum_adjacent_gap,
    )


def _spectral_summary(diagnostics: _DetachedSVDDiagnostics) -> str:
    """Format compact, actionable detached SVD diagnostics for an exception."""

    values = diagnostics.singular_values.detach().cpu().numpy()
    shown = ", ".join(f"{value:.3e}" for value in values[:8])
    if values.size > 8:
        shown += ", ..."
    return (
        f"singular_values=[{shown}], numerical_rank={diagnostics.numerical_rank}, "
        f"zero_tolerance={diagnostics.zero_tolerance:.3e}, "
        f"minimum_adjacent_gap={diagnostics.minimum_adjacent_gap:.3e}, "
        f"gap_tolerance={diagnostics.gap_tolerance:.3e}"
    )


def _raise_unsafe_truncated_svd(
    *,
    left_site: int,
    retained_rank: int,
    max_bond_dimension: int | None,
    singular_value_cutoff: float,
    diagnostics: _DetachedSVDDiagnostics,
    reason: str,
) -> None:
    """Fail before backward when a hard-SVD chart has no reliable gradient."""

    raise TorchMPSDifferentiabilityError(
        "Unsafe differentiable truncated SVD at serial staircase bond "
        f"({left_site}, {left_site + 1}): {reason}. "
        f"retained_rank={retained_rank}, max_bond_dimension={max_bond_dimension}, "
        f"singular_value_cutoff={singular_value_cutoff:.3e}; "
        f"{_spectral_summary(diagnostics)}. "
        "No optimiser step has been taken. Use a nondegenerate initialisation, "
        "relax/disable the cap or cutoff for this update, or perturb the state "
        "and restart from a recorded finite checkpoint."
    )


def _assert_qr_backward_safe(
    matrix: torch.Tensor,
    *,
    context: str,
) -> None:
    """Reject rank-deficient QR gauge transformations before they yield NaNs.

    QR has a well-defined autograd backward only for full numerical rank in
    the reduced dimension.  Canonicalisation is a gauge operation, so a
    conservative early error is preferable to contaminating all older layer
    parameters with NaNs.  The check is skipped for detached/evaluation-only
    paths, where no backward is possible.
    """

    if not matrix.requires_grad:
        return
    diagnostics = _detached_svd_diagnostics(matrix)
    required_rank = min(int(matrix.shape[0]), int(matrix.shape[1]))
    if diagnostics.numerical_rank < required_rank:
        raise TorchMPSDifferentiabilityError(
            f"Unsafe differentiable QR canonicalisation in {context}: "
            f"matrix shape={tuple(matrix.shape)} requires numerical rank "
            f"{required_rank}, but found {diagnostics.numerical_rank}; "
            f"{_spectral_summary(diagnostics)}. No optimiser step has been taken. "
            "Remove redundant zero bond sectors or restart from a finite, "
            "full-rank MPS gauge before requesting canonical truncation."
        )


def _identity_gauge_exact_split(matrix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return an exact rank-feasible split whose backward is the identity map.

    A QR split is unnecessary for an untruncated update and its backward is
    singular at a perfectly valid rank-deficient two-site tensor.  Splitting
    with an identity gauge has the same reduced bond dimension
    ``min(rows, columns)``, represents *exactly* the same matrix, and has no
    decomposition backward at all.
    """

    rows, columns = (int(dimension) for dimension in matrix.shape)
    if rows >= columns:
        right = torch.eye(columns, dtype=matrix.dtype, device=matrix.device)
        return matrix, right
    left = torch.eye(rows, dtype=matrix.dtype, device=matrix.device)
    return left, matrix


def _rank_deficient_tangent_split(
    matrix: torch.Tensor,
    *,
    retained_rank: int,
    diagnostics: _DetachedSVDDiagnostics,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split an exactly rank-``r`` matrix with a finite tangent-space backward.

    For a matrix of exact numerical rank ``r`` with a separated nonzero
    singular subspace, the nearest-rank-``r`` map has the local derivative

    ``dA = P_left dM + dM P_right - P_left dM P_right``.

    ``torch.linalg.svd`` nevertheless emits NaNs because the unused null-space
    singular vectors are degenerate.  We obtain detached SVD factors for the
    exact forward value and inject the analytic tangent derivative through a
    zero-valued straight-through correction.  The returned factors retain bond
    ``r`` and their product is exactly the truncated forward matrix.
    """

    if retained_rank < 1:
        raise AssertionError("retained_rank must be positive in the tangent split.")
    if retained_rank != diagnostics.numerical_rank:
        raise AssertionError("The tangent split is valid only at exact retained rank.")
    left_vectors, singular_values, right_vectors = torch.linalg.svd(
        matrix.detach(),
        full_matrices=False,
    )
    left_vectors = left_vectors[:, :retained_rank]
    singular_values = singular_values[:retained_rank]
    right_vectors = right_vectors[:retained_rank, :]
    if bool(torch.any(singular_values <= diagnostics.zero_tolerance).cpu()):
        raise AssertionError("Tangent split retained a numerically zero singular value.")

    square_root = torch.sqrt(singular_values)
    inverse_square_root = torch.reciprocal(square_root)
    right_columns = right_vectors.conj().transpose(-2, -1)
    left_projector = left_vectors @ left_vectors.conj().transpose(-2, -1)
    tangent_only = matrix - matrix.detach()
    left_factor = left_vectors * square_root.unsqueeze(0)
    left_factor = left_factor + (
        (tangent_only - left_projector @ tangent_only)
        @ right_columns
        * inverse_square_root.unsqueeze(0)
    )
    right_factor = square_root.unsqueeze(1) * right_vectors
    right_factor = right_factor + inverse_square_root.unsqueeze(1) * (
        left_vectors.conj().transpose(-2, -1) @ tangent_only
    )
    return left_factor, right_factor


@dataclass(frozen=True, slots=True)
class TorchOpenBoundaryMPS:
    """Strict OBC torch MPS that preserves tensor autograd connections.

    The constructor deliberately retains the supplied tensors rather than
    cloning/detaching them.  It can consequently hold leaf parameters,
    projected-gate outputs, or a fully differentiable circuit state.
    """

    tensors: tuple[torch.Tensor, ...]

    def __post_init__(self) -> None:
        tensors = tuple(self.tensors)
        if not tensors:
            raise ValueError("An open-boundary MPS must contain at least one tensor.")
        first_device: torch.device | None = None
        previous_right = 1
        for site, tensor in enumerate(tensors):
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"MPS tensor at site {site} must be a torch.Tensor.")
            if tensor.dtype != torch.complex128:
                raise TypeError(
                    "Torch MPS tensors must use torch.complex128 under the frozen "
                    f"reproduction protocol; site {site} has {tensor.dtype}."
                )
            if tensor.ndim != 3:
                raise ValueError(
                    f"MPS tensor at site {site} must have rank 3, got {tuple(tensor.shape)}."
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
            if first_device is None:
                first_device = tensor.device
            elif tensor.device != first_device:
                raise ValueError("All Torch MPS tensors must share one device.")
            if not bool(torch.all(torch.isfinite(tensor)).detach().cpu()):
                raise ValueError(f"MPS tensor at site {site} contains non-finite entries.")
            previous_right = right
        if previous_right != 1:
            raise ValueError(
                "An open-boundary MPS must have right boundary bond dimension 1; "
                f"got {previous_right}."
            )
        object.__setattr__(self, "tensors", tensors)

    @property
    def num_qubits(self) -> int:
        return len(self.tensors)

    @property
    def device(self) -> torch.device:
        return self.tensors[0].device

    @property
    def bond_dimensions(self) -> tuple[int, ...]:
        return (1, *(int(tensor.shape[2]) for tensor in self.tensors))

    @property
    def max_bond_dimension(self) -> int:
        return max(self.bond_dimensions)


@dataclass(frozen=True, slots=True)
class TorchTwoSiteUpdateResult:
    """Autograd-connected result and audited rank ledger for one U(4) update."""

    mps: TorchOpenBoundaryMPS
    left_site: int
    retained_rank: int
    discarded_weight: torch.Tensor
    split_method: str
    discarded_weight_is_global_norm_loss: bool


@dataclass(frozen=True, slots=True)
class TorchStaircaseUpdateResult:
    """Result and per-gate rank ledgers of a serial nearest-neighbour layer."""

    mps: TorchOpenBoundaryMPS
    updates: tuple[TorchTwoSiteUpdateResult, ...]

    @property
    def total_discarded_weight(self) -> torch.Tensor:
        """Differentiable sum of per-update discarded squared weights."""

        total = torch.zeros((), dtype=torch.float64, device=self.mps.device)
        for update in self.updates:
            total = total + update.discarded_weight
        return total

    @property
    def discarded_weight_is_global_norm_loss(self) -> bool:
        """Whether every ledger term was produced after a canonical gauge sweep."""

        return all(update.discarded_weight_is_global_norm_loss for update in self.updates)


def torch_product_zero_mps(
    num_qubits: int,
    *,
    device: torch.device | str = "cpu",
) -> TorchOpenBoundaryMPS:
    """Return ``|0...0>`` as a complex128 MPS without dense allocation."""

    num_qubits = validate_num_qubits(num_qubits)
    device = torch.device(device)
    core = torch.zeros((1, 2, 1), dtype=torch.complex128, device=device)
    core[0, 0, 0] = 1.0
    return TorchOpenBoundaryMPS(tuple(core.clone() for _ in range(num_qubits)))


def torch_mps_from_numpy(
    mps: OpenBoundaryMPS,
    *,
    device: torch.device | str = "cpu",
) -> TorchOpenBoundaryMPS:
    """Copy an immutable NumPy target MPS into a non-parameter torch MPS."""

    if not isinstance(mps, OpenBoundaryMPS):
        raise TypeError("mps must be an OpenBoundaryMPS.")
    device = torch.device(device)
    tensors = tuple(
        torch.as_tensor(np.array(tensor, copy=True), dtype=torch.complex128, device=device)
        for tensor in mps.tensors
    )
    return TorchOpenBoundaryMPS(tensors)


def torch_mps_to_numpy(mps: TorchOpenBoundaryMPS) -> OpenBoundaryMPS:
    """Detach/copy a torch MPS for non-differentiable serialisation or testing."""

    if not isinstance(mps, TorchOpenBoundaryMPS):
        raise TypeError("mps must be a TorchOpenBoundaryMPS.")
    return OpenBoundaryMPS(
        tuple(tensor.detach().cpu().numpy().copy() for tensor in mps.tensors)
    )


def torch_mps_inner_product(
    bra: TorchOpenBoundaryMPS,
    ket: TorchOpenBoundaryMPS,
) -> torch.Tensor:
    """Return autograd-connected ``<bra|ket>`` via transfer contractions."""

    if not isinstance(bra, TorchOpenBoundaryMPS) or not isinstance(ket, TorchOpenBoundaryMPS):
        raise TypeError("bra and ket must both be TorchOpenBoundaryMPS instances.")
    if bra.num_qubits != ket.num_qubits:
        raise ValueError(
            "Cannot contract MPSs with different site counts: "
            f"{bra.num_qubits} and {ket.num_qubits}."
        )
    if bra.device != ket.device:
        raise ValueError("bra and ket must be on the same device.")
    environment = torch.ones((1, 1), dtype=torch.complex128, device=bra.device)
    for bra_tensor, ket_tensor in zip(bra.tensors, ket.tensors, strict=True):
        environment = torch.einsum("ab,asr,bsq->rq", environment, bra_tensor.conj(), ket_tensor)
    return environment[0, 0]


def torch_mps_norm_squared(mps: TorchOpenBoundaryMPS) -> torch.Tensor:
    """Return a differentiable real squared norm without a dense statevector."""

    if not isinstance(mps, TorchOpenBoundaryMPS):
        raise TypeError("mps must be a TorchOpenBoundaryMPS.")
    return torch.real(torch_mps_inner_product(mps, mps))


def torch_mps_normalized(mps: TorchOpenBoundaryMPS) -> TorchOpenBoundaryMPS:
    """Return an autograd-connected unit-norm MPS by scaling its first core."""

    if not isinstance(mps, TorchOpenBoundaryMPS):
        raise TypeError("mps must be a TorchOpenBoundaryMPS.")
    norm_squared = torch_mps_norm_squared(mps)
    with torch.no_grad():
        numeric = float(norm_squared.detach().cpu())
    if not math.isfinite(numeric) or numeric <= 0.0:
        raise ValueError("Cannot normalise a zero-norm or non-finite MPS.")
    tensors = list(mps.tensors)
    tensors[0] = tensors[0] / torch.sqrt(norm_squared)
    return TorchOpenBoundaryMPS(tuple(tensors))


def torch_mixed_canonicalized_for_two_site(
    mps: TorchOpenBoundaryMPS,
    left_site: int,
) -> TorchOpenBoundaryMPS:
    """Differentiably QR-gauge-fix an MPS around one neighbouring two-site centre.

    The prefix left of ``left_site`` becomes left isometric and the suffix
    right of ``left_site + 1`` becomes right isometric.  Therefore an SVD of
    the centre after a unitary update has physical Schmidt values: discarded
    squared singular values are exactly the global squared-norm loss.  The
    operation preserves the represented state and its existing autograd graph.
    """

    if not isinstance(mps, TorchOpenBoundaryMPS):
        raise TypeError("mps must be a TorchOpenBoundaryMPS.")
    if isinstance(left_site, bool) or not isinstance(left_site, numbers.Integral):
        raise TypeError("left_site must be an integer.")
    left_site = int(left_site)
    if not 0 <= left_site < mps.num_qubits - 1:
        raise ValueError(
            f"left_site={left_site} is invalid for N={mps.num_qubits}."
        )
    tensors = list(mps.tensors)
    for site in range(left_site):
        left, physical, _ = tensors[site].shape
        qr_input = tensors[site].reshape(left * physical, -1)
        _assert_qr_backward_safe(
            qr_input,
            context=f"left-canonical site {site}",
        )
        q_factor, r_factor = torch.linalg.qr(
            qr_input,
            mode="reduced",
        )
        retained_right = int(q_factor.shape[1])
        tensors[site] = q_factor.reshape(left, physical, retained_right)
        tensors[site + 1] = torch.tensordot(r_factor, tensors[site + 1], dims=([1], [0]))
    for site in range(mps.num_qubits - 1, left_site + 1, -1):
        left, physical, right = tensors[site].shape
        qr_input = tensors[site].reshape(left, physical * right).transpose(0, 1)
        _assert_qr_backward_safe(
            qr_input,
            context=f"right-canonical site {site}",
        )
        q_factor, r_factor = torch.linalg.qr(
            qr_input,
            mode="reduced",
        )
        retained_left = int(q_factor.shape[1])
        tensors[site] = q_factor.transpose(0, 1).reshape(retained_left, physical, right)
        tensors[site - 1] = torch.tensordot(
            tensors[site - 1],
            r_factor.transpose(0, 1),
            dims=([2], [0]),
        )
    return TorchOpenBoundaryMPS(tuple(tensors))


def torch_mps_squared_fidelity(
    left: TorchOpenBoundaryMPS,
    right: TorchOpenBoundaryMPS,
) -> torch.Tensor:
    """Return differentiable global squared-state fidelity from MPS overlaps."""

    if not isinstance(left, TorchOpenBoundaryMPS) or not isinstance(right, TorchOpenBoundaryMPS):
        raise TypeError("left and right must both be TorchOpenBoundaryMPS instances.")
    denominator = torch_mps_norm_squared(left) * torch_mps_norm_squared(right)
    with torch.no_grad():
        numeric = float(denominator.detach().cpu())
    if not math.isfinite(numeric) or numeric <= 0.0:
        raise ValueError("Squared fidelity is undefined for a zero-norm MPS.")
    overlap = torch_mps_inner_product(left, right)
    return torch.real(overlap.conj() * overlap) / denominator


def torch_negative_log_fidelity_per_site(
    target: TorchOpenBoundaryMPS,
    prepared: TorchOpenBoundaryMPS,
    *,
    squared_fidelity_floor: float = 1e-30,
) -> torch.Tensor:
    r"""Return Zhou Eq. (1), ``-log(|<t|p>|)/N``, with an explicit floor."""

    if target.num_qubits != prepared.num_qubits:
        raise ValueError("target and prepared must have the same number of qubits.")
    squared_fidelity_floor = validate_real(
        squared_fidelity_floor,
        name="squared_fidelity_floor",
        positive=True,
    )
    if squared_fidelity_floor > 1.0:
        raise ValueError("squared_fidelity_floor must not exceed one.")
    fidelity = torch_mps_squared_fidelity(target, prepared)
    return -0.5 * torch.log(torch.clamp(fidelity, min=squared_fidelity_floor)) / target.num_qubits


def _validate_gate(
    matrix: torch.Tensor,
    *,
    device: torch.device,
    require_unitary: bool,
    unitary_atol: float,
) -> torch.Tensor:
    unitary_atol = validate_real(unitary_atol, name="unitary_atol")
    if unitary_atol < 0.0:
        raise ValueError("unitary_atol must be nonnegative.")
    if not isinstance(matrix, torch.Tensor):
        raise TypeError("matrix must be a torch.Tensor.")
    if matrix.dtype != torch.complex128 or matrix.shape != (4, 4):
        raise ValueError("matrix must have dtype torch.complex128 and shape (4, 4).")
    if matrix.device != device:
        raise ValueError("matrix and MPS must be on the same device.")
    if not bool(torch.all(torch.isfinite(matrix)).detach().cpu()):
        raise ValueError("matrix contains non-finite entries.")
    if require_unitary:
        with torch.no_grad():
            identity = torch.eye(4, dtype=torch.complex128, device=device)
            residual = float(
                torch.linalg.matrix_norm(
                    matrix.detach().conj().transpose(-2, -1) @ matrix.detach() - identity,
                    ord=float("inf"),
                ).cpu()
            )
        if residual > unitary_atol:
            raise ValueError(
                "matrix must be unitary in the declared |00>,|01>,|10>,|11> basis; "
                f"residual={residual:.3e}."
            )
    return matrix


def torch_apply_adjacent_two_qubit_gate(
    mps: TorchOpenBoundaryMPS,
    matrix: torch.Tensor,
    left_site: int,
    *,
    max_bond_dimension: int | None = None,
    singular_value_cutoff: float = 0.0,
    require_unitary: bool = True,
    unitary_atol: float = 1e-11,
    canonicalize_for_truncation: bool = True,
) -> TorchTwoSiteUpdateResult:
    """Apply a q0-MSB ordered U(4), preserving gradients through the MPS split.

    With the default uncompressed settings, an identity-gauge exact split
    avoids decomposition backward altogether.  Supplying a rank cap or a
    positive cutoff activates a truncated SVD.  By default a differentiable
    mixed-canonical QR sweep happens first, making the discarded squared
    singular value sum an exact global norm-loss ledger.  Rank-deficient
    no-loss truncations use a stable analytic tangent split; a genuinely
    unsafe hard-SVD degeneracy fails before backward rather than returning
    non-finite gradients.  Disabling the sweep is an advanced performance
    option and marks the returned ledger as local only.
    """

    if not isinstance(mps, TorchOpenBoundaryMPS):
        raise TypeError("mps must be a TorchOpenBoundaryMPS.")
    if isinstance(left_site, bool) or not isinstance(left_site, numbers.Integral):
        raise TypeError("left_site must be an integer.")
    left_site = int(left_site)
    if not 0 <= left_site < mps.num_qubits - 1:
        raise ValueError(
            f"left_site={left_site} is invalid for N={mps.num_qubits}."
        )
    max_bond_dimension = _validate_optional_bond_dimension(max_bond_dimension)
    singular_value_cutoff = _validate_cutoff(singular_value_cutoff)
    if not isinstance(canonicalize_for_truncation, bool):
        raise TypeError("canonicalize_for_truncation must be a bool.")
    matrix = _validate_gate(
        matrix,
        device=mps.device,
        require_unitary=require_unitary,
        unitary_atol=unitary_atol,
    )

    truncation_requested = max_bond_dimension is not None or singular_value_cutoff > 0.0
    if truncation_requested and canonicalize_for_truncation:
        working_mps = torch_mixed_canonicalized_for_two_site(mps, left_site)
    else:
        working_mps = mps
    first = working_mps.tensors[left_site]
    second = working_mps.tensors[left_site + 1]
    left_bond, _, _ = first.shape
    _, _, right_bond = second.shape
    merged = torch.einsum("asm,mtn->astn", first, second)
    updated = torch.einsum("uvst,lstr->luvr", matrix.reshape(2, 2, 2, 2), merged)
    matrix_form = updated.reshape(left_bond * 2, 2 * right_bond)

    exact_untruncated = max_bond_dimension is None and singular_value_cutoff == 0.0
    full_split_rank = min(int(matrix_form.shape[0]), int(matrix_form.shape[1]))
    if exact_untruncated:
        left_factor, right_factor = _identity_gauge_exact_split(matrix_form)
        retained_rank = full_split_rank
        first_updated = left_factor.reshape(left_bond, 2, retained_rank)
        second_updated = right_factor.reshape(retained_rank, 2, right_bond)
        discarded_weight = torch.zeros((), dtype=torch.float64, device=mps.device)
        split_method = "identity_exact"
    else:
        diagnostics = _detached_svd_diagnostics(matrix_form)
        cutoff_rank = int(
            torch.count_nonzero(diagnostics.singular_values > singular_value_cutoff).cpu()
        )
        retained_rank = max(1, cutoff_rank)
        if max_bond_dimension is not None:
            retained_rank = min(retained_rank, max_bond_dimension)
        if retained_rank >= full_split_rank:
            # A declared cap/cutoff that retains every representable singular
            # direction is an exact update.  Avoid SVD/QR backward exactly as
            # in the uncompressed path while still recording the requested
            # truncation protocol and its zero loss ledger.
            left_factor, right_factor = _identity_gauge_exact_split(matrix_form)
            retained_rank = full_split_rank
            first_updated = left_factor.reshape(left_bond, 2, retained_rank)
            second_updated = right_factor.reshape(retained_rank, 2, right_bond)
            discarded_weight = torch.zeros((), dtype=torch.float64, device=mps.device)
            split_method = "identity_exact"
        elif not matrix_form.requires_grad:
            # Evaluation-only MPS preparation has no backward pass to poison;
            # use the literal SVD even at an exact degeneracy.
            left_vectors, singular_values, right_vectors = torch.linalg.svd(
                matrix_form,
                full_matrices=False,
            )
            discarded_weight = torch.sum(singular_values[retained_rank:] ** 2)
            first_updated = left_vectors[:, :retained_rank].reshape(
                left_bond,
                2,
                retained_rank,
            )
            second_updated = (
                singular_values[:retained_rank, None] * right_vectors[:retained_rank, :]
            ).reshape(retained_rank, 2, right_bond)
            split_method = "svd_truncated"
        elif retained_rank == diagnostics.numerical_rank:
            # The retained nonzero Schmidt sector is separated from a
            # degenerate null space.  The rank-r projection is differentiable
            # here, but PyTorch's full SVD backward is not; use its analytic
            # tangent derivative without differentiating singular vectors.
            left_factor, right_factor = _rank_deficient_tangent_split(
                matrix_form,
                retained_rank=retained_rank,
                diagnostics=diagnostics,
            )
            discarded_weight = torch.sum(diagnostics.singular_values[retained_rank:] ** 2)
            first_updated = left_factor.reshape(left_bond, 2, retained_rank)
            second_updated = right_factor.reshape(retained_rank, 2, right_bond)
            split_method = "svd_rank_deficient_tangent"
        else:
            if diagnostics.numerical_rank < full_split_rank:
                _raise_unsafe_truncated_svd(
                    left_site=left_site,
                    retained_rank=retained_rank,
                    max_bond_dimension=max_bond_dimension,
                    singular_value_cutoff=singular_value_cutoff,
                    diagnostics=diagnostics,
                    reason=(
                        "the requested retained rank cuts a matrix with a "
                        "degenerate numerical null space"
                    ),
                )
            if diagnostics.minimum_adjacent_gap <= diagnostics.gap_tolerance:
                _raise_unsafe_truncated_svd(
                    left_site=left_site,
                    retained_rank=retained_rank,
                    max_bond_dimension=max_bond_dimension,
                    singular_value_cutoff=singular_value_cutoff,
                    diagnostics=diagnostics,
                    reason=(
                        "the singular spectrum is degenerate or too close for "
                        "a gauge-invariant hard-SVD backward"
                    ),
                )
            left_vectors, singular_values, right_vectors = torch.linalg.svd(
                matrix_form,
                full_matrices=False,
            )
            discarded_weight = torch.sum(singular_values[retained_rank:] ** 2)
            first_updated = left_vectors[:, :retained_rank].reshape(
                left_bond,
                2,
                retained_rank,
            )
            second_updated = (
                singular_values[:retained_rank, None] * right_vectors[:retained_rank, :]
            ).reshape(retained_rank, 2, right_bond)
            split_method = "svd_truncated"

    tensors = list(working_mps.tensors)
    tensors[left_site] = first_updated
    tensors[left_site + 1] = second_updated
    return TorchTwoSiteUpdateResult(
        mps=TorchOpenBoundaryMPS(tuple(tensors)),
        left_site=left_site,
        retained_rank=retained_rank,
        discarded_weight=discarded_weight,
        split_method=split_method,
        discarded_weight_is_global_norm_loss=truncation_requested
        and canonicalize_for_truncation,
    )


def torch_apply_staircase_layer(
    mps: TorchOpenBoundaryMPS,
    matrices: Sequence[torch.Tensor] | Iterable[torch.Tensor],
    *,
    max_bond_dimension: int | None = None,
    singular_value_cutoff: float = 0.0,
    require_unitary: bool = True,
    unitary_atol: float = 1e-11,
    canonicalize_for_truncation: bool = True,
) -> TorchStaircaseUpdateResult:
    """Apply one Zhou serial layer in the physical order ``(0,1),...,(N-2,N-1)``."""

    if not isinstance(mps, TorchOpenBoundaryMPS):
        raise TypeError("mps must be a TorchOpenBoundaryMPS.")
    matrices = tuple(matrices)
    expected = mps.num_qubits - 1
    if len(matrices) != expected:
        raise ValueError(
            f"One serial staircase layer for N={mps.num_qubits} needs {expected} U(4) gates; "
            f"got {len(matrices)}."
        )
    current = mps
    updates: list[TorchTwoSiteUpdateResult] = []
    for left_site, matrix in enumerate(matrices):
        update = torch_apply_adjacent_two_qubit_gate(
            current,
            matrix,
            left_site,
            max_bond_dimension=max_bond_dimension,
            singular_value_cutoff=singular_value_cutoff,
            require_unitary=require_unitary,
            unitary_atol=unitary_atol,
            canonicalize_for_truncation=canonicalize_for_truncation,
        )
        current = update.mps
        updates.append(update)
    return TorchStaircaseUpdateResult(mps=current, updates=tuple(updates))


def torch_mps_to_dense(
    mps: TorchOpenBoundaryMPS,
    *,
    max_num_qubits: int | None = 20,
) -> torch.Tensor:
    """Explicit small-N dense diagnostic retaining the autograd graph."""

    if not isinstance(mps, TorchOpenBoundaryMPS):
        raise TypeError("mps must be a TorchOpenBoundaryMPS.")
    if max_num_qubits is not None:
        max_num_qubits = _validate_positive_integer(
            max_num_qubits,
            name="max_num_qubits",
        )
        if mps.num_qubits > max_num_qubits:
            raise ValueError(
                "Dense conversion is disabled for this MPS: "
                f"N={mps.num_qubits} exceeds max_num_qubits={max_num_qubits}."
            )
    state = mps.tensors[0][0, :, :]
    for tensor in mps.tensors[1:]:
        state = torch.tensordot(state, tensor, dims=([-1], [0]))
    return state[..., 0].reshape(-1)


def estimated_mps_complex_elements(num_qubits: int, max_bond_dimension: int) -> int:
    """Return qubit-MPS core storage at rank-feasible OBC bond dimensions.

    Multiply this number by 16 for the raw complex128 bytes.  Optimiser state,
    SVD/QR workspaces, and autograd saved tensors are additional runtime costs;
    this is a lower-level representation ledger, not a peak-memory promise.
    """

    num_qubits = validate_num_qubits(num_qubits)
    max_bond_dimension = _validate_positive_integer(
        max_bond_dimension,
        name="max_bond_dimension",
    )
    bonds = open_boundary_bond_dimensions(num_qubits, max_bond_dimension)
    return sum(2 * bonds[site] * bonds[site + 1] for site in range(num_qubits))


__all__ = [
    "TorchMPSDifferentiabilityError",
    "TorchOpenBoundaryMPS",
    "TorchStaircaseUpdateResult",
    "TorchTwoSiteUpdateResult",
    "estimated_mps_complex_elements",
    "torch_apply_adjacent_two_qubit_gate",
    "torch_apply_staircase_layer",
    "torch_mps_from_numpy",
    "torch_mps_inner_product",
    "torch_mixed_canonicalized_for_two_site",
    "torch_mps_normalized",
    "torch_mps_norm_squared",
    "torch_mps_squared_fidelity",
    "torch_mps_to_dense",
    "torch_mps_to_numpy",
    "torch_negative_log_fidelity_per_site",
    "torch_product_zero_mps",
]
