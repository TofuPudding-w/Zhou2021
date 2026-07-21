"""Layerwise automatic differentiation for Zhou et al.'s ADQC staircase."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import ArrayLike

from .circuit import Initialization, LatentStaircase, validate_torch_statevector
from .conventions import validate_num_qubits, validate_real
from .metrics import negative_log_fidelity_per_site_from_squared_fidelity


def _validate_nonnegative_integer(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer.")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be nonnegative.")
    return result


def _validate_adam_betas(value: tuple[float, float]) -> tuple[float, float]:
    """Return Adam beta coordinates after rejecting implicit/malformed values."""

    if not isinstance(value, tuple) or len(value) != 2:
        raise TypeError("adam_betas must be a tuple of exactly two real scalars.")
    beta1 = validate_real(value[0], name="adam_betas[0]")
    beta2 = validate_real(value[1], name="adam_betas[1]")
    if not 0.0 <= beta1 < 1.0 or not 0.0 <= beta2 < 1.0:
        raise ValueError("each Adam beta must satisfy 0 <= beta < 1.")
    return beta1, beta2


@dataclass(frozen=True, slots=True)
class ADQCTrainingConfig:
    """Explicit non-paper numerical coordinates for a reproducible ADQC run.

    The paper specifies Adam and a learning-rate schedule but does not publish
    its values.  These fields therefore belong in every exported manifest and
    are never presented as author-supplied hyperparameters.

    ``convergence_nlf`` is evaluated against the same explicitly floored
    differentiable NLF used for training.  Floor-free physical Eq. (1) values
    are retained separately in every result and trajectory record.
    """

    max_layers: int = 3
    epochs_per_stage: int = 100
    learning_rate: float = 0.02
    learning_rate_decay: float = 1.0
    adam_betas: tuple[float, float] = (0.9, 0.999)
    adam_eps: float = 1e-8
    adam_weight_decay: float = 0.0
    adam_amsgrad: bool = False
    first_layer_initialization: Initialization = "random"
    new_layer_perturbation: float = 1e-3
    seed: int = 2021
    loss_floor_squared_fidelity: float = 1e-30
    convergence_nlf: float | None = None

    def __post_init__(self) -> None:
        _validate_nonnegative_integer(self.max_layers, name="max_layers")
        if self.max_layers < 1:
            raise ValueError("max_layers must be positive.")
        _validate_nonnegative_integer(self.epochs_per_stage, name="epochs_per_stage")
        if self.epochs_per_stage < 1:
            raise ValueError("epochs_per_stage must be positive.")
        validate_real(self.learning_rate, name="learning_rate", positive=True)
        decay = validate_real(self.learning_rate_decay, name="learning_rate_decay", positive=True)
        if decay > 1.0:
            raise ValueError("learning_rate_decay must not exceed one.")
        object.__setattr__(self, "adam_betas", _validate_adam_betas(self.adam_betas))
        object.__setattr__(
            self,
            "adam_eps",
            validate_real(self.adam_eps, name="adam_eps", positive=True),
        )
        weight_decay = validate_real(self.adam_weight_decay, name="adam_weight_decay")
        if weight_decay < 0.0:
            raise ValueError("adam_weight_decay must be nonnegative.")
        object.__setattr__(self, "adam_weight_decay", weight_decay)
        if not isinstance(self.adam_amsgrad, bool):
            raise TypeError("adam_amsgrad must be a bool.")
        if self.first_layer_initialization not in {"random", "identity_perturbed"}:
            raise ValueError("first_layer_initialization is invalid.")
        validate_real(
            self.new_layer_perturbation,
            name="new_layer_perturbation",
            positive=True,
        )
        _validate_nonnegative_integer(self.seed, name="seed")
        floor = validate_real(
            self.loss_floor_squared_fidelity,
            name="loss_floor_squared_fidelity",
            positive=True,
        )
        if floor > 1.0:
            raise ValueError("loss_floor_squared_fidelity must not exceed one.")
        if self.convergence_nlf is not None:
            if validate_real(self.convergence_nlf, name="convergence_nlf") < 0.0:
                raise ValueError("convergence_nlf must be nonnegative.")


@dataclass(frozen=True, slots=True)
class LayerUpdateRecord:
    """One layer-local backpropagation update in physical trajectory order."""

    stage: int
    epoch: int
    active_layer: int
    negative_log_fidelity_per_site_before: float
    negative_log_fidelity_per_site_after: float
    squared_fidelity_before: float
    squared_fidelity_after: float
    learning_rate: float
    training_floored_negative_log_fidelity_per_site_before: float
    training_floored_negative_log_fidelity_per_site_after: float


@dataclass(frozen=True, slots=True)
class StageRecord:
    """One warm-start depth stage of the Zhou schedule."""

    num_layers: int
    initial_negative_log_fidelity_per_site: float
    final_negative_log_fidelity_per_site: float
    initial_squared_fidelity: float
    final_squared_fidelity: float
    updates: tuple[LayerUpdateRecord, ...]
    initial_training_floored_negative_log_fidelity_per_site: float
    final_training_floored_negative_log_fidelity_per_site: float


@dataclass(frozen=True, slots=True)
class ADQCTrainingResult:
    """Frozen outcome and all scalar trajectory data of an ADQC optimization."""

    model: LatentStaircase
    target_state: torch.Tensor
    prepared_state: torch.Tensor
    stages: tuple[StageRecord, ...]
    config: ADQCTrainingConfig
    converged: bool
    stopping_reason: str

    @property
    def final_squared_fidelity(self) -> float:
        return _squared_fidelity_tensor(self.target_state, self.prepared_state)

    @property
    def final_negative_log_fidelity_per_site(self) -> float:
        """Return the physical Zhou Eq. (1), without the optimizer floor."""

        return negative_log_fidelity_per_site_from_squared_fidelity(
            self.final_squared_fidelity,
            num_qubits=int(math.log2(self.target_state.numel())),
        )

    @property
    def final_physical_negative_log_fidelity_per_site(self) -> float:
        """Explicit alias for the floor-free physical reporting metric."""

        return self.final_negative_log_fidelity_per_site

    @property
    def final_training_floored_negative_log_fidelity_per_site(self) -> float:
        """Return the finite differentiable scalar used by the optimizer."""

        return _negative_log_fidelity_tensor(
            self.target_state,
            self.prepared_state,
            loss_floor_squared_fidelity=self.config.loss_floor_squared_fidelity,
        )


def _target_tensor(
    target_state: ArrayLike | torch.Tensor,
    *,
    num_qubits: int,
    device: torch.device,
) -> torch.Tensor:
    if isinstance(target_state, torch.Tensor):
        target = target_state.detach().to(device=device, dtype=torch.complex128).clone()
    else:
        target = torch.as_tensor(
            np.asarray(target_state, dtype=np.complex128),
            dtype=torch.complex128,
            device=device,
        ).clone()
    return validate_torch_statevector(target, num_qubits=num_qubits)


def _overlap_squared(target: torch.Tensor, prepared: torch.Tensor) -> torch.Tensor:
    overlap = torch.vdot(target, prepared)
    return torch.real(overlap.conj() * overlap)


def _squared_fidelity_tensor(target: torch.Tensor, prepared: torch.Tensor) -> float:
    value = float(_overlap_squared(target, prepared).detach().cpu())
    if value < -2e-12 or value > 1.0 + 2e-10:
        raise ArithmeticError(f"Invalid normalized squared fidelity: {value:.17g}.")
    return value


def _negative_log_fidelity_loss(
    target: torch.Tensor,
    prepared: torch.Tensor,
    *,
    num_qubits: int,
    loss_floor_squared_fidelity: float,
) -> torch.Tensor:
    r"""Differentiable Eq. (1) loss with an explicitly recorded zero-overlap floor."""

    floor = validate_real(
        loss_floor_squared_fidelity,
        name="loss_floor_squared_fidelity",
        positive=True,
    )
    if floor > 1.0:
        raise ValueError("loss_floor_squared_fidelity must not exceed one.")
    squared = _overlap_squared(target, prepared)
    return -0.5 * torch.log(torch.clamp(squared, min=floor)) / num_qubits


def _negative_log_fidelity_tensor(
    target: torch.Tensor,
    prepared: torch.Tensor,
    *,
    loss_floor_squared_fidelity: float,
) -> float:
    num_qubits = int(math.log2(target.numel()))
    with torch.no_grad():
        value = _negative_log_fidelity_loss(
            target,
            prepared,
            num_qubits=num_qubits,
            loss_floor_squared_fidelity=loss_floor_squared_fidelity,
        )
    return float(value.detach().cpu())


def _current_metrics(
    model: LatentStaircase,
    target: torch.Tensor,
    *,
    loss_floor_squared_fidelity: float,
) -> tuple[float, float, float]:
    with torch.no_grad():
        prepared = model()
        squared_fidelity = _squared_fidelity_tensor(target, prepared)
        physical_nlf = negative_log_fidelity_per_site_from_squared_fidelity(
            squared_fidelity,
            num_qubits=model.num_qubits,
        )
        training_floored_nlf = _negative_log_fidelity_tensor(
            target,
            prepared,
            loss_floor_squared_fidelity=loss_floor_squared_fidelity,
        )
        return physical_nlf, training_floored_nlf, squared_fidelity


def train_adqc(
    target_state: ArrayLike | torch.Tensor,
    *,
    num_qubits: int,
    config: ADQCTrainingConfig | None = None,
    device: torch.device | str = "cpu",
) -> ADQCTrainingResult:
    """Train the paper's warm-start, layer-by-layer ADQC schedule.

    At a depth stage, one backward graph is constructed for one active layer;
    all other latent layers are frozen.  This is the literal engineering
    interpretation of the paper's statement that gates are optimized "layer by
    layer" to avoid retaining all latent-gate computation graphs at once.
    """

    num_qubits = validate_num_qubits(num_qubits)
    if config is None:
        config = ADQCTrainingConfig()
    if not isinstance(config, ADQCTrainingConfig):
        raise TypeError("config must be an ADQCTrainingConfig.")
    device = torch.device(device)
    target = _target_tensor(target_state, num_qubits=num_qubits, device=device)
    model = LatentStaircase(num_qubits, device=device)
    optimizers: list[torch.optim.Adam] = []
    stages: list[StageRecord] = []
    converged = False
    stopping_reason = "max_layers_reached"

    try:
        for stage_index in range(config.max_layers):
            if stage_index == 0:
                initialization = config.first_layer_initialization
                perturbation = config.new_layer_perturbation
            else:
                initialization = "identity_perturbed"
                perturbation = config.new_layer_perturbation
            new_layer = model.add_layer(
                initialization=initialization,
                seed=config.seed + stage_index,
                perturbation=perturbation,
            )
            learning_rate = config.learning_rate * config.learning_rate_decay**stage_index
            # Keep Adam's accumulated moments for warm-started layers, while
            # applying the explicitly requested stage learning-rate schedule
            # consistently to every layer that is revisited at this depth.
            for optimizer in optimizers:
                for parameter_group in optimizer.param_groups:
                    parameter_group["lr"] = learning_rate
            optimizers.append(
                torch.optim.Adam(
                    model.layer_parameters(new_layer),
                    lr=learning_rate,
                    betas=config.adam_betas,
                    eps=config.adam_eps,
                    weight_decay=config.adam_weight_decay,
                    amsgrad=config.adam_amsgrad,
                )
            )
            initial_nlf, initial_training_nlf, initial_fidelity = _current_metrics(
                model,
                target,
                loss_floor_squared_fidelity=config.loss_floor_squared_fidelity,
            )
            updates: list[LayerUpdateRecord] = []

            for epoch in range(config.epochs_per_stage):
                for active_layer in range(model.num_layers):
                    model.set_trainable_layer(active_layer)
                    before_nlf, before_training_nlf, before_fidelity = _current_metrics(
                        model,
                        target,
                        loss_floor_squared_fidelity=config.loss_floor_squared_fidelity,
                    )
                    model.zero_grad(set_to_none=True)
                    prepared = model()
                    loss = _negative_log_fidelity_loss(
                        target,
                        prepared,
                        num_qubits=num_qubits,
                        loss_floor_squared_fidelity=config.loss_floor_squared_fidelity,
                    )
                    loss.backward()
                    optimizer = optimizers[active_layer]
                    optimizer.step()
                    after_nlf, after_training_nlf, after_fidelity = _current_metrics(
                        model,
                        target,
                        loss_floor_squared_fidelity=config.loss_floor_squared_fidelity,
                    )
                    updates.append(
                        LayerUpdateRecord(
                            stage=stage_index,
                            epoch=epoch,
                            active_layer=active_layer,
                            negative_log_fidelity_per_site_before=before_nlf,
                            negative_log_fidelity_per_site_after=after_nlf,
                            squared_fidelity_before=before_fidelity,
                            squared_fidelity_after=after_fidelity,
                            learning_rate=float(optimizer.param_groups[0]["lr"]),
                            training_floored_negative_log_fidelity_per_site_before=(
                                before_training_nlf
                            ),
                            training_floored_negative_log_fidelity_per_site_after=(
                                after_training_nlf
                            ),
                        )
                    )

                _, final_training_nlf, final_fidelity = _current_metrics(
                    model,
                    target,
                    loss_floor_squared_fidelity=config.loss_floor_squared_fidelity,
                )
                if (
                    config.convergence_nlf is not None
                    and final_training_nlf <= config.convergence_nlf
                ):
                    converged = True
                    stopping_reason = "convergence_nlf_reached"
                    break

            final_nlf, final_training_nlf, final_fidelity = _current_metrics(
                model,
                target,
                loss_floor_squared_fidelity=config.loss_floor_squared_fidelity,
            )
            stages.append(
                StageRecord(
                    num_layers=model.num_layers,
                    initial_negative_log_fidelity_per_site=initial_nlf,
                    final_negative_log_fidelity_per_site=final_nlf,
                    initial_squared_fidelity=initial_fidelity,
                    final_squared_fidelity=final_fidelity,
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
        prepared = model().detach().clone()
    return ADQCTrainingResult(
        model=model,
        target_state=target.detach().clone(),
        prepared_state=prepared,
        stages=tuple(stages),
        config=config,
        converged=converged,
        stopping_reason=stopping_reason,
    )


__all__ = [
    "ADQCTrainingConfig",
    "ADQCTrainingResult",
    "LayerUpdateRecord",
    "StageRecord",
    "train_adqc",
]
