"""Layer-by-layer ADQC training checks for the Zhou 2021 core implementation."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from zhou2021.adqc import (
    ADQCTrainingConfig,
    ADQCTrainingResult,
    _negative_log_fidelity_loss,
    train_adqc,
)
from zhou2021.circuit import LatentStaircase, zero_state
from zhou2021.metrics import negative_log_fidelity_per_site, squared_state_fidelity


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_layers": 0},
        {"max_layers": True},
        {"epochs_per_stage": 0},
        {"learning_rate": 0.0},
        {"learning_rate_decay": 1.01},
        {"adam_betas": (1.0, 0.999)},
        {"adam_betas": [0.9, 0.999]},
        {"adam_eps": 0.0},
        {"adam_weight_decay": -1e-12},
        {"adam_amsgrad": 1},
        {"first_layer_initialization": "not-an-initialization"},
        {"new_layer_perturbation": 0.0},
        {"seed": -1},
        {"loss_floor_squared_fidelity": 0.0},
        {"loss_floor_squared_fidelity": 1.01},
        {"convergence_nlf": -1e-12},
    ],
)
def test_training_config_rejects_invalid_numerical_coordinates(kwargs: dict[str, object]) -> None:
    with pytest.raises((TypeError, ValueError)):
        ADQCTrainingConfig(**kwargs)  # type: ignore[arg-type]


def test_differentiable_training_loss_agrees_with_zhou_metric_away_from_the_floor() -> None:
    target = zero_state(2)
    prepared = torch.tensor([1.0, 1.0, 0.0, 0.0], dtype=torch.complex128) / math.sqrt(2.0)
    observed = _negative_log_fidelity_loss(
        target,
        prepared,
        num_qubits=2,
        loss_floor_squared_fidelity=1e-30,
    )
    expected = negative_log_fidelity_per_site(target.numpy(), prepared.numpy())

    assert float(observed.detach().cpu()) == pytest.approx(expected, abs=2e-12)


def test_differentiable_training_loss_uses_the_explicit_recorded_zero_overlap_floor() -> None:
    target = zero_state(2)
    orthogonal = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.complex128)
    floor = 1e-12
    observed = _negative_log_fidelity_loss(
        target,
        orthogonal,
        num_qubits=2,
        loss_floor_squared_fidelity=floor,
    )

    assert float(observed.detach().cpu()) == pytest.approx(-0.5 * math.log(floor) / 2.0)
    assert math.isinf(negative_log_fidelity_per_site(target.numpy(), orthogonal.numpy()))
    with pytest.raises(ValueError, match="must not exceed one"):
        _negative_log_fidelity_loss(
            target,
            orthogonal,
            num_qubits=2,
            loss_floor_squared_fidelity=1.01,
        )


def test_result_separates_physical_nlf_from_the_floored_training_scalar() -> None:
    floor = 1e-12
    config = ADQCTrainingConfig(
        max_layers=1,
        epochs_per_stage=1,
        loss_floor_squared_fidelity=floor,
    )
    result = ADQCTrainingResult(
        model=LatentStaircase(2),
        target_state=zero_state(2),
        prepared_state=torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.complex128),
        stages=(),
        config=config,
        converged=False,
        stopping_reason="test_fixture",
    )

    assert result.final_squared_fidelity == 0.0
    assert math.isinf(result.final_negative_log_fidelity_per_site)
    assert math.isinf(result.final_physical_negative_log_fidelity_per_site)
    assert result.final_training_floored_negative_log_fidelity_per_site == pytest.approx(
        -0.5 * math.log(floor) / 2.0
    )


def test_one_stage_training_improves_a_simple_target_and_records_each_layer_local_update() -> None:
    config = ADQCTrainingConfig(
        max_layers=1,
        epochs_per_stage=2,
        learning_rate=0.05,
        seed=np.int64(7),
    )
    result = train_adqc(zero_state(2), num_qubits=2, config=config)
    stage = result.stages[0]

    assert result.stopping_reason == "max_layers_reached"
    assert not result.converged
    assert result.model.num_layers == 1
    assert len(stage.updates) == 2
    assert [update.active_layer for update in stage.updates] == [0, 0]
    assert stage.final_squared_fidelity > stage.initial_squared_fidelity
    assert stage.final_negative_log_fidelity_per_site < stage.initial_negative_log_fidelity_per_site
    assert result.final_squared_fidelity == pytest.approx(stage.final_squared_fidelity, abs=2e-12)
    assert result.final_negative_log_fidelity_per_site == pytest.approx(
        stage.final_negative_log_fidelity_per_site,
        abs=2e-12,
    )
    assert result.final_training_floored_negative_log_fidelity_per_site == pytest.approx(
        stage.final_training_floored_negative_log_fidelity_per_site,
        abs=2e-12,
    )
    assert all(parameter.requires_grad for parameter in result.model.parameters())


def test_adam_coordinates_are_explicit_and_retained_in_the_training_config() -> None:
    config = ADQCTrainingConfig(
        max_layers=1,
        epochs_per_stage=1,
        learning_rate=0.03,
        adam_betas=(0.8, 0.95),
        adam_eps=3e-9,
        adam_weight_decay=0.04,
        adam_amsgrad=True,
        seed=8,
    )
    result = train_adqc(zero_state(2), num_qubits=2, config=config)

    assert result.config.adam_betas == (0.8, 0.95)
    assert result.config.adam_eps == pytest.approx(3e-9)
    assert result.config.adam_weight_decay == pytest.approx(0.04)
    assert result.config.adam_amsgrad is True


def test_two_stage_training_is_bitwise_deterministic_and_preserves_warm_start_structure() -> None:
    config = ADQCTrainingConfig(
        max_layers=2,
        epochs_per_stage=1,
        learning_rate=0.05,
        learning_rate_decay=0.5,
        seed=7,
    )
    target = zero_state(2).numpy()
    first = train_adqc(target, num_qubits=2, config=config)
    second = train_adqc(target, num_qubits=2, config=config)

    assert len(first.stages) == 2
    assert [len(stage.updates) for stage in first.stages] == [1, 2]
    assert [update.active_layer for update in first.stages[0].updates] == [0]
    assert [update.active_layer for update in first.stages[1].updates] == [0, 1]
    assert first.stages[0].updates[0].learning_rate == pytest.approx(0.05)
    assert all(
        update.learning_rate == pytest.approx(0.025)
        for update in first.stages[1].updates
    )
    assert first.stages[-1].final_squared_fidelity > first.stages[0].initial_squared_fidelity

    assert torch.equal(first.prepared_state, second.prepared_state)
    assert first.final_squared_fidelity == second.final_squared_fidelity
    for left, right in zip(first.model.parameters(), second.model.parameters(), strict=True):
        assert torch.equal(left, right)
    assert all(parameter.requires_grad for parameter in first.model.parameters())

    assert squared_state_fidelity(target, first.prepared_state.numpy()) == pytest.approx(
        first.final_squared_fidelity,
        abs=2e-12,
    )


def test_convergence_threshold_stops_after_the_first_completed_epoch() -> None:
    config = ADQCTrainingConfig(
        max_layers=3,
        epochs_per_stage=5,
        learning_rate=0.05,
        seed=9,
        convergence_nlf=10.0,
    )
    result = train_adqc(zero_state(2), num_qubits=2, config=config)

    assert result.converged
    assert result.stopping_reason == "convergence_nlf_reached"
    assert len(result.stages) == 1
    assert len(result.stages[0].updates) == 1


def test_training_rejects_unnormalized_targets_and_wrong_config_type() -> None:
    with pytest.raises(ValueError, match="normalized"):
        train_adqc(np.ones(4, dtype=np.complex128), num_qubits=2)
    with pytest.raises(TypeError, match="ADQCTrainingConfig"):
        train_adqc(zero_state(2), num_qubits=2, config="wrong")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="at least two qubits"):
        train_adqc(np.array([1.0, 0.0], dtype=np.complex128), num_qubits=1)
