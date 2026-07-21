"""Cross-check dense model-oracle MPS target helpers."""

from __future__ import annotations

import numpy as np
import pytest

from zhou2021.mps import mps_squared_fidelity
from zhou2021.mps_targets import dense_model_ground_state_mps


@pytest.mark.parametrize("model", ["heisenberg", "xy", "xxz", "tfim_tndiff"])
def test_dense_model_ground_state_mps_is_exact_without_truncation(model: str) -> None:
    target = dense_model_ground_state_mps(
        model,  # type: ignore[arg-type]
        4,
        delta=0.7,
        field_x=0.9,
    )

    reconstructed = target.mps.to_dense()
    assert target.ground_state.residual < 2e-12
    assert target.decomposition.discarded_weight == pytest.approx(0.0, abs=2e-13)
    assert np.allclose(reconstructed, target.ground_state.state, rtol=0.0, atol=3e-13)
    assert mps_squared_fidelity(target.mps, target.mps) == pytest.approx(1.0, abs=3e-13)


def test_dense_model_helper_refuses_paper_scale_dense_diagonalisation() -> None:
    with pytest.raises(ValueError, match="small-system validation path"):
        dense_model_ground_state_mps("heisenberg", 13)


def test_dense_model_helper_reports_truncation_explicitly() -> None:
    target = dense_model_ground_state_mps("heisenberg", 6, max_bond_dimension=1)

    assert target.mps.max_bond_dimension == 1
    assert target.decomposition.discarded_weight >= 0.0
    assert target.mps.norm_squared() == pytest.approx(
        target.decomposition.retained_norm_squared,
        abs=3e-12,
    )
