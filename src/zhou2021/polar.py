"""Numerically stable differentiable polar projection for Zhou latent gates.

Zhou, Hong and Ran parameterise a two-qubit gate by an unconstrained complex
matrix and use its unitary polar factor.  The forward expression is the usual
``U @ Vh`` from an SVD.  Differentiating *that expression* with PyTorch's SVD
backward is, however, not valid at repeated singular values: its derivative of
the individual singular vectors contains ``1 / (s_i**2 - s_j**2)`` terms even
though the polar factor itself is smooth at every nonsingular matrix.

``_StablePolarProjection`` therefore keeps the exact SVD forward value, but
implements the first-order VJP of the polar map through a Lyapunov equation.
This makes identity and repeated-singular-value initialisations usable without
adding non-paper random jitter.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

# Numerical rank necessarily needs a finite-precision convention.  Freeze a
# conservative complex128 relative threshold rather than allowing an arbitrary
# SVD completion on a null space to silently define a gate.
POLAR_NUMERICAL_RANK_RCOND = 64.0 * torch.finfo(torch.float64).eps


@dataclass(frozen=True, slots=True)
class PolarProjection:
    """Forward polar diagnostics while retaining the unitary autograd graph.

    ``singular_values`` is intentionally a detached diagnostic.  Individual
    ordered singular values are not differentiable at degeneracy; the ADQC
    optimisation differentiates the physical polar unitary instead.
    """

    unitary: torch.Tensor
    singular_values: torch.Tensor
    unitarity_residual: torch.Tensor
    objective: torch.Tensor


def validate_latent_gate(latent: torch.Tensor) -> torch.Tensor:
    """Validate structural constraints of one finite, nonzero complex U(4) latent.

    Numerical full rank is checked together with the SVD in
    :func:`project_latent_gate`, so ordinary ADQC forward passes do not perform
    the same 4x4 decomposition twice.
    """

    if not isinstance(latent, torch.Tensor):
        raise TypeError("latent must be a torch.Tensor.")
    if latent.shape != (4, 4):
        raise ValueError(f"A Zhou latent gate must have shape (4, 4), got {tuple(latent.shape)}.")
    if not torch.is_complex(latent):
        raise TypeError("latent must have a complex dtype.")
    if latent.dtype != torch.complex128:
        raise TypeError("latent must use torch.complex128 for the frozen reproduction protocol.")
    if not bool(torch.all(torch.isfinite(latent)).detach().cpu()):
        raise ValueError("latent contains non-finite values.")
    if float(torch.linalg.vector_norm(latent).detach().cpu()) == 0.0:
        raise ValueError(
            "The zero latent gate has a non-unique polar unitary; initialize with a "
            "nonzero seeded perturbation instead."
        )
    return latent


def _validate_numerical_full_rank(singular_values: torch.Tensor) -> None:
    """Reject a numerically rank-deficient latent under the frozen float64 rule.

    A rank-deficient square matrix has a partial isometry as polar factor, but
    no unique U(4) polar factor: completing its null space depends on the
    arbitrary singular-vector basis returned by an SVD.  Such a completion is
    neither a physically specified Zhou gate nor a differentiable map.  The
    correct reproducibility rule is consequently to reject it rather than
    conceal a basis-dependent choice.
    """

    largest = float(singular_values[0].detach().cpu())
    smallest = float(singular_values[-1].detach().cpu())
    threshold = POLAR_NUMERICAL_RANK_RCOND * largest
    if smallest <= threshold:
        raise ValueError(
            "latent is numerically rank deficient under the frozen complex128 "
            f"rule: sigma_min={smallest:.3e}, threshold={threshold:.3e}, "
            f"rcond={POLAR_NUMERICAL_RANK_RCOND:.3e}. A rank-deficient matrix has "
            "no unique U(4) polar factor; use a full-rank seeded perturbation."
        )


class _StablePolarProjection(torch.autograd.Function):
    """Exact SVD polar forward with a repeated-singular-value-safe VJP.

    For a nonsingular matrix ``A = Q H`` with ``Q`` unitary and ``H`` positive
    definite, the differential of the polar unitary is ``dQ = Q Omega`` where

    ``H Omega + Omega H = Q^dagger dA - dA^dagger Q``.

    If ``G`` is the incoming real-Frobenius cotangent, the adjoint equation is
    solved in the right-singular-vector basis without dividing by singular
    value *differences*:

    ``Z_ij = skew((Q^dagger G)_ij) / (s_i + s_j)``,
    ``grad_A = 2 Q Z``.

    The denominators are strictly positive for accepted full-rank inputs and
    remain finite when any singular values are equal.
    """

    @staticmethod
    def forward(ctx: torch.autograd.function.FunctionCtx, latent: torch.Tensor) -> tuple[
        torch.Tensor, torch.Tensor
    ]:
        u_matrix, singular_values, vh_matrix = torch.linalg.svd(latent, full_matrices=False)
        _validate_numerical_full_rank(singular_values)
        unitary = u_matrix @ vh_matrix
        ctx.save_for_backward(unitary, singular_values, vh_matrix)
        # SVD values are exported only as diagnostics.  Marking them
        # nondifferentiable prevents an accidental use of the non-smooth ordered
        # singular-value derivative at a degeneracy.
        ctx.mark_non_differentiable(singular_values)
        return unitary, singular_values

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_unitary: torch.Tensor | None,
        _grad_singular_values: torch.Tensor | None,
    ) -> torch.Tensor:
        unitary, singular_values, vh_matrix = ctx.saved_tensors
        if grad_unitary is None:
            return torch.zeros_like(unitary)

        co_tangent = unitary.conj().transpose(-2, -1) @ grad_unitary
        skew_co_tangent = 0.5 * (co_tangent - co_tangent.conj().transpose(-2, -1))

        v_matrix = vh_matrix.conj().transpose(-2, -1)
        skew_in_singular_basis = vh_matrix @ skew_co_tangent @ v_matrix
        denominators = singular_values[:, None] + singular_values[None, :]
        lyapunov_solution = skew_in_singular_basis / denominators
        lyapunov_solution = 0.5 * (
            lyapunov_solution - lyapunov_solution.conj().transpose(-2, -1)
        )
        solution = v_matrix @ lyapunov_solution @ vh_matrix
        return 2.0 * unitary @ solution


def _stable_polar_unitary_and_singular_values(latent: torch.Tensor) -> tuple[
    torch.Tensor, torch.Tensor
]:
    """Return the physical polar unitary and detached SVD diagnostics."""

    return _StablePolarProjection.apply(latent)


def project_latent_gate(latent: torch.Tensor) -> PolarProjection:
    r"""Return the Eq. (2)-preceding projection ``G=U@Vh``.

    For ``latent=U diag(s) Vh``, the returned gate maximizes
    ``Re Tr(latent^dagger G)`` over U(4).  No forward-pass random jitter is
    used: it would make the paper method and exported circuits non-deterministic.
    Repeated *nonzero* singular values are handled by the stable polar VJP.
    Numerically rank-deficient matrices are rejected because they do not define
    a unique U(4) polar factor.
    """

    latent = validate_latent_gate(latent)
    unitary, singular_values = _stable_polar_unitary_and_singular_values(latent)
    identity = torch.eye(4, dtype=torch.complex128, device=latent.device)
    residual = torch.linalg.matrix_norm(unitary.conj().transpose(-2, -1) @ unitary - identity)
    objective = torch.real(torch.sum(latent.conj() * unitary))
    return PolarProjection(
        unitary=unitary,
        singular_values=singular_values,
        unitarity_residual=residual,
        objective=objective,
    )


def polar_unitary(latent: torch.Tensor) -> torch.Tensor:
    """Return only the autograd-connected unitary projection."""

    return project_latent_gate(latent).unitary


__all__ = [
    "POLAR_NUMERICAL_RANK_RCOND",
    "PolarProjection",
    "polar_unitary",
    "project_latent_gate",
    "validate_latent_gate",
]
