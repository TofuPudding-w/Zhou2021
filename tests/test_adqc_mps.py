"""Cross-validation of the dense and MPS-native Zhou ADQC executors."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from zhou2021.adqc import ADQCTrainingConfig, train_adqc
from zhou2021.adqc_mps import (
    ADQCMPSConfig,
    _mps_metrics,
    prepare_staircase_mps,
    train_adqc_mps,
)
from zhou2021.circuit import LatentStaircase
from zhou2021.mps import mps_from_dense_statevector, random_open_boundary_mps
from zhou2021.mps_targets import dense_model_ground_state_mps
from zhou2021.torch_mps import torch_mps_from_numpy, torch_mps_to_dense


def test_mps_executor_matches_dense_staircase_before_optimization() -> None:
    model = LatentStaircase(4)
    model.add_layer(initialization="random", seed=401)
    model.add_layer(initialization="identity_perturbed", seed=402)

    prepared_mps, discarded = prepare_staircase_mps(model)

    assert torch.allclose(
        torch_mps_to_dense(prepared_mps),
        model(),
        rtol=0.0,
        atol=7e-13,
    )
    assert float(discarded.detach()) == pytest.approx(0.0, abs=1e-15)


def test_mps_metrics_report_true_fidelity_below_the_training_floor() -> None:
    target_state = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.complex128)
    prepared_state = np.array(
        [0.01, math.sqrt(1.0 - 0.01**2), 0.0, 0.0],
        dtype=np.complex128,
    )
    target = torch_mps_from_numpy(mps_from_dense_statevector(target_state))
    prepared = torch_mps_from_numpy(mps_from_dense_statevector(prepared_state))

    physical_nlf, training_nlf, fidelity = _mps_metrics(
        target,
        prepared,
        squared_fidelity_floor=1e-2,
    )

    assert fidelity == pytest.approx(1e-4, abs=2e-14)
    assert physical_nlf == pytest.approx(-0.5 * math.log(1e-4) / 2.0, abs=2e-12)
    assert training_nlf == pytest.approx(-0.5 * math.log(1e-2) / 2.0, abs=2e-12)


def test_mps_training_agrees_with_the_independent_dense_oracle_at_small_n() -> None:
    target = dense_model_ground_state_mps("heisenberg", 3)
    training = ADQCTrainingConfig(
        max_layers=2,
        epochs_per_stage=2,
        learning_rate=0.02,
        seed=5,
    )

    dense = train_adqc(
        target.ground_state.state,
        num_qubits=3,
        config=training,
    )
    mps = train_adqc_mps(target.mps, config=ADQCMPSConfig(training=training))

    assert mps.final_squared_fidelity == pytest.approx(
        dense.final_squared_fidelity,
        abs=3e-12,
    )
    assert mps.final_negative_log_fidelity_per_site == pytest.approx(
        dense.final_negative_log_fidelity_per_site,
        abs=3e-12,
    )
    assert mps.final_training_floored_negative_log_fidelity_per_site == pytest.approx(
        dense.final_training_floored_negative_log_fidelity_per_site,
        abs=3e-12,
    )
    assert [len(stage.updates) for stage in mps.stages] == [2, 4]
    assert all(
        stage.final_discarded_weight == pytest.approx(0.0, abs=1e-15)
        for stage in mps.stages
    )
    assert all(parameter.requires_grad for parameter in mps.model.parameters())


def test_mps_training_rejects_invalid_config_and_records_explicit_bond_cap() -> None:
    target = dense_model_ground_state_mps("xy", 3).mps
    with pytest.raises(TypeError, match="ADQCMPSConfig"):
        train_adqc_mps(target, config="wrong")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="positive"):
        ADQCMPSConfig(max_bond_dimension=0)

    result = train_adqc_mps(
        target,
        config=ADQCMPSConfig(
            training=ADQCTrainingConfig(
                max_layers=1,
                epochs_per_stage=1,
                learning_rate=0.02,
                seed=17,
            ),
            max_bond_dimension=1,
        ),
    )
    assert result.config.max_bond_dimension == 1
    assert result.stages[0].final_prepared_max_bond_dimension == 1


def test_mps_training_rejects_one_qubit_degenerate_staircase() -> None:
    with pytest.raises(ValueError, match="at least two qubits"):
        train_adqc_mps(random_open_boundary_mps(1, 1, seed=18))
