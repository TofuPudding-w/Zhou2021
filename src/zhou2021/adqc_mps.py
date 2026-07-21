"""MPS-native layerwise ADQC training for the paper-scale Zhou 2021 path.

The dense :mod:`zhou2021.adqc` implementation is an exact small-system oracle.
This module executes the *same* latent-U(4), polar-projected serial staircase
directly on open-boundary MPS cores, so an N=48 target never requires a
``2**48`` statevector.  A bond cap is an explicit approximation coordinate;
it is never silently selected and is recorded in the result object.
"""

from __future__ import annotations

import numbers
from dataclasses import dataclass, field

import torch

from .adqc import ADQCTrainingConfig
from .circuit import LatentStaircase
from .conventions import validate_real
from .metrics import negative_log_fidelity_per_site_from_squared_fidelity
from .mps import OpenBoundaryMPS, mps_squared_fidelity
from .torch_mps import (
    TorchOpenBoundaryMPS,
    torch_apply_adjacent_two_qubit_gate,
    torch_mps_from_numpy,
    torch_mps_squared_fidelity,
    torch_mps_to_numpy,
    torch_negative_log_fidelity_per_site,
    torch_product_zero_mps,
)


def _validate_optional_bond_dimension(value: int | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise TypeError("max_bond_dimension must be an integer or None.")
    result = int(value)
    if result < 1:
        raise ValueError("max_bond_dimension must be positive.")
    return result


@dataclass(frozen=True, slots=True)
class ADQCMPSConfig:
    """All numerical coordinates of one MPS-native ADQC optimization.

    ``training`` supplies the optimizer/warm-start coordinates.  Leaving
    ``max_bond_dimension`` as ``None`` is exact in the local MPS splits but
    can grow exponentially with ADQC depth.  A paper-scale run should declare
    a cap after a separate bond-convergence study rather than relying on an
    undocumented default.
    """

    training: ADQCTrainingConfig = field(default_factory=ADQCTrainingConfig)
    max_bond_dimension: int | None = None
    singular_value_cutoff: float = 0.0
    unitary_atol: float = 1e-11

    def __post_init__(self) -> None:
        if not isinstance(self.training, ADQCTrainingConfig):
            raise TypeError("training must be an ADQCTrainingConfig.")
        _validate_optional_bond_dimension(self.max_bond_dimension)
        cutoff = validate_real(
            self.singular_value_cutoff,
            name="singular_value_cutoff",
        )
        if cutoff < 0.0:
            raise ValueError("singular_value_cutoff must be nonnegative.")
        validate_real(self.unitary_atol, name="unitary_atol", positive=True)


@dataclass(frozen=True, slots=True)
class MPSLayerUpdateRecord:
    """One layer-local MPS backpropagation update and truncation ledger."""

    stage: int
    epoch: int
    active_layer: int
    negative_log_fidelity_per_site_before: float
    negative_log_fidelity_per_site_after: float
    squared_fidelity_before: float
    squared_fidelity_after: float
    learning_rate: float
    prepared_max_bond_dimension_after: int
    discarded_weight_after: float
    training_floored_negative_log_fidelity_per_site_before: float
    training_floored_negative_log_fidelity_per_site_after: float


@dataclass(frozen=True, slots=True)
class MPSStageRecord:
    """One warm-start depth stage of an MPS-native ADQC run."""

    num_layers: int
    initial_negative_log_fidelity_per_site: float
    final_negative_log_fidelity_per_site: float
    initial_squared_fidelity: float
    final_squared_fidelity: float
    final_prepared_max_bond_dimension: int
    final_discarded_weight: float
    updates: tuple[MPSLayerUpdateRecord, ...]
    initial_training_floored_negative_log_fidelity_per_site: float
    final_training_floored_negative_log_fidelity_per_site: float


@dataclass(frozen=True, slots=True)
class ADQCMPSResult:
    """Frozen MPS output and a complete layerwise optimizer trajectory."""

    model: LatentStaircase
    target_mps: OpenBoundaryMPS
    prepared_mps: OpenBoundaryMPS
    stages: tuple[MPSStageRecord, ...]
    config: ADQCMPSConfig
    converged: bool
    stopping_reason: str

    @property
    def final_squared_fidelity(self) -> float:
        """Return the normalized global fidelity by MPS-only contraction."""

        return mps_squared_fidelity(self.target_mps, self.prepared_mps)

    @property
    def final_negative_log_fidelity_per_site(self) -> float:
        """Return Eq. (1) without materialising a statevector."""

        return negative_log_fidelity_per_site_from_squared_fidelity(
            self.final_squared_fidelity,
            num_qubits=self.target_mps.num_qubits,
        )

    @property
    def final_physical_negative_log_fidelity_per_site(self) -> float:
        """Explicit alias for the floor-free physical reporting metric."""

        return self.final_negative_log_fidelity_per_site

    @property
    def final_training_floored_negative_log_fidelity_per_site(self) -> float:
        """Return the finite differentiable scalar used by MPS training."""

        target = torch_mps_from_numpy(self.target_mps)
        prepared = torch_mps_from_numpy(self.prepared_mps)
        with torch.no_grad():
            loss = torch_negative_log_fidelity_per_site(
                target,
                prepared,
                squared_fidelity_floor=(
                    self.config.training.loss_floor_squared_fidelity
                ),
            )
        return float(loss.detach().cpu())


def _as_normalized_target(
    target_mps: OpenBoundaryMPS | TorchOpenBoundaryMPS,
) -> OpenBoundaryMPS:
    if isinstance(target_mps, TorchOpenBoundaryMPS):
        target_mps = torch_mps_to_numpy(target_mps)
    if not isinstance(target_mps, OpenBoundaryMPS):
        raise TypeError("target_mps must be an OpenBoundaryMPS or TorchOpenBoundaryMPS.")
    return target_mps.normalized()


def prepare_staircase_mps(
    model: LatentStaircase,
    *,
    max_bond_dimension: int | None = None,
    singular_value_cutoff: float = 0.0,
    unitary_atol: float = 1e-11,
) -> tuple[TorchOpenBoundaryMPS, torch.Tensor]:
    """Prepare ``|0...0>`` through a Zhou staircase without dense amplitudes.

    Returns the autograd-connected MPS and a scalar truncation ledger.  The
    latter is exactly zero in the untruncated QR path; with a declared cap or
    cutoff, the underlying default mixed-canonical sweeps make its terms the
    physical global norm-squared losses rather than gauge-local SVD scores.
    """

    if not isinstance(model, LatentStaircase):
        raise TypeError("model must be a LatentStaircase.")
    max_bond_dimension = _validate_optional_bond_dimension(max_bond_dimension)
    singular_value_cutoff = validate_real(
        singular_value_cutoff,
        name="singular_value_cutoff",
    )
    if singular_value_cutoff < 0.0:
        raise ValueError("singular_value_cutoff must be nonnegative.")
    unitary_atol = validate_real(unitary_atol, name="unitary_atol", positive=True)

    prepared = torch_product_zero_mps(model.num_qubits, device=model.device)
    discarded = torch.zeros((), dtype=torch.float64, device=model.device)
    for gate in model.iter_projected_gates():
        update = torch_apply_adjacent_two_qubit_gate(
            prepared,
            gate.matrix,
            gate.left_site,
            max_bond_dimension=max_bond_dimension,
            singular_value_cutoff=singular_value_cutoff,
            require_unitary=True,
            unitary_atol=unitary_atol,
        )
        prepared = update.mps
        discarded = discarded + update.discarded_weight
    return prepared, discarded


def _mps_metrics(
    target: TorchOpenBoundaryMPS,
    prepared: TorchOpenBoundaryMPS,
    *,
    squared_fidelity_floor: float,
) -> tuple[float, float, float]:
    with torch.no_grad():
        fidelity = torch_mps_squared_fidelity(target, prepared)
        loss = torch_negative_log_fidelity_per_site(
            target,
            prepared,
            squared_fidelity_floor=squared_fidelity_floor,
        )
    training_floored_nlf = float(loss.detach().cpu())
    squared = float(fidelity.detach().cpu())
    if squared < -2e-12 or squared > 1.0 + 2e-10:
        raise ArithmeticError(f"Invalid normalized MPS squared fidelity: {squared:.17g}.")
    physical_nlf = negative_log_fidelity_per_site_from_squared_fidelity(
        squared,
        num_qubits=target.num_qubits,
    )
    return physical_nlf, training_floored_nlf, squared


def train_adqc_mps(
    target_mps: OpenBoundaryMPS | TorchOpenBoundaryMPS,
    *,
    config: ADQCMPSConfig | None = None,
    device: torch.device | str = "cpu",
) -> ADQCMPSResult:
    """Train the Zhou warm-start schedule directly against an MPS target."""

    if config is None:
        config = ADQCMPSConfig()
    if not isinstance(config, ADQCMPSConfig):
        raise TypeError("config must be an ADQCMPSConfig.")
    target_numpy = _as_normalized_target(target_mps)
    device = torch.device(device)
    target = torch_mps_from_numpy(target_numpy, device=device)
    model = LatentStaircase(target.num_qubits, device=device)
    training = config.training
    optimizers: list[torch.optim.Adam] = []
    stages: list[MPSStageRecord] = []
    converged = False
    stopping_reason = "max_layers_reached"

    def current() -> tuple[float, float, float, int, float]:
        with torch.no_grad():
            prepared, discarded = prepare_staircase_mps(
                model,
                max_bond_dimension=config.max_bond_dimension,
                singular_value_cutoff=config.singular_value_cutoff,
                unitary_atol=config.unitary_atol,
            )
            nlf, training_nlf, fidelity = _mps_metrics(
                target,
                prepared,
                squared_fidelity_floor=training.loss_floor_squared_fidelity,
            )
        return (
            nlf,
            training_nlf,
            fidelity,
            prepared.max_bond_dimension,
            float(discarded.detach().cpu()),
        )

    try:
        for stage_index in range(training.max_layers):
            initialization = (
                training.first_layer_initialization
                if stage_index == 0
                else "identity_perturbed"
            )
            new_layer = model.add_layer(
                initialization=initialization,
                seed=int(training.seed) + stage_index,
                perturbation=training.new_layer_perturbation,
            )
            learning_rate = training.learning_rate * training.learning_rate_decay**stage_index
            for optimizer in optimizers:
                for parameter_group in optimizer.param_groups:
                    parameter_group["lr"] = learning_rate
            optimizers.append(
                torch.optim.Adam(
                    model.layer_parameters(new_layer),
                    lr=learning_rate,
                    betas=training.adam_betas,
                    eps=training.adam_eps,
                    weight_decay=training.adam_weight_decay,
                    amsgrad=training.adam_amsgrad,
                )
            )

            initial_nlf, initial_training_nlf, initial_fidelity, _, _ = current()
            updates: list[MPSLayerUpdateRecord] = []
            for epoch in range(training.epochs_per_stage):
                for active_layer in range(model.num_layers):
                    model.set_trainable_layer(active_layer)
                    before_nlf, before_training_nlf, before_fidelity, _, _ = current()
                    model.zero_grad(set_to_none=True)
                    prepared, _ = prepare_staircase_mps(
                        model,
                        max_bond_dimension=config.max_bond_dimension,
                        singular_value_cutoff=config.singular_value_cutoff,
                        unitary_atol=config.unitary_atol,
                    )
                    loss = torch_negative_log_fidelity_per_site(
                        target,
                        prepared,
                        squared_fidelity_floor=training.loss_floor_squared_fidelity,
                    )
                    loss.backward()
                    optimizer = optimizers[active_layer]
                    optimizer.step()
                    (
                        after_nlf,
                        after_training_nlf,
                        after_fidelity,
                        after_bond,
                        after_discarded,
                    ) = current()
                    updates.append(
                        MPSLayerUpdateRecord(
                            stage=stage_index,
                            epoch=epoch,
                            active_layer=active_layer,
                            negative_log_fidelity_per_site_before=before_nlf,
                            negative_log_fidelity_per_site_after=after_nlf,
                            squared_fidelity_before=before_fidelity,
                            squared_fidelity_after=after_fidelity,
                            learning_rate=float(optimizer.param_groups[0]["lr"]),
                            prepared_max_bond_dimension_after=after_bond,
                            discarded_weight_after=after_discarded,
                            training_floored_negative_log_fidelity_per_site_before=(
                                before_training_nlf
                            ),
                            training_floored_negative_log_fidelity_per_site_after=(
                                after_training_nlf
                            ),
                        )
                    )

                _, final_training_nlf, _, _, _ = current()
                if (
                    training.convergence_nlf is not None
                    and final_training_nlf <= training.convergence_nlf
                ):
                    converged = True
                    stopping_reason = "convergence_nlf_reached"
                    break

            (
                final_nlf,
                final_training_nlf,
                final_fidelity,
                final_bond,
                final_discarded,
            ) = current()
            stages.append(
                MPSStageRecord(
                    num_layers=model.num_layers,
                    initial_negative_log_fidelity_per_site=initial_nlf,
                    final_negative_log_fidelity_per_site=final_nlf,
                    initial_squared_fidelity=initial_fidelity,
                    final_squared_fidelity=final_fidelity,
                    final_prepared_max_bond_dimension=final_bond,
                    final_discarded_weight=final_discarded,
                    updates=tuple(updates),
                    initial_training_floored_negative_log_fidelity_per_site=(
                        initial_training_nlf
                    ),
                    final_training_floored_negative_log_fidelity_per_site=(
                        final_training_nlf
                    ),
                )
            )
            if converged:
                break
    finally:
        model.set_trainable_layer(None)

    with torch.no_grad():
        final_prepared, _ = prepare_staircase_mps(
            model,
            max_bond_dimension=config.max_bond_dimension,
            singular_value_cutoff=config.singular_value_cutoff,
            unitary_atol=config.unitary_atol,
        )
    return ADQCMPSResult(
        model=model,
        target_mps=target_numpy,
        prepared_mps=torch_mps_to_numpy(final_prepared),
        stages=tuple(stages),
        config=config,
        converged=converged,
        stopping_reason=stopping_reason,
    )


__all__ = [
    "ADQCMPSConfig",
    "ADQCMPSResult",
    "MPSLayerUpdateRecord",
    "MPSStageRecord",
    "prepare_staircase_mps",
    "train_adqc_mps",
]
