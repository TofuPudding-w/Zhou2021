"""Tests for the optional TeNPy-DMRG target path.

The fake finite-MPS tests exercise the order and norm contract without TeNPy.
The actual DMRG test is intentionally skipped unless ``physics-tenpy`` is
installed, so the base dense/torch workflow remains dependency-light.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from zhou2021.ground_states import exact_ground_state
from zhou2021.models import (
    heisenberg_hamiltonian,
    tfim_tndiff_hamiltonian,
    xxz_hamiltonian,
    xy_hamiltonian,
)
from zhou2021.mps import mps_from_dense_statevector
from zhou2021.tenpy_targets import (
    DMRGConfig,
    dmrg_ground_state_mps,
    tenpy_mps_to_open_boundary,
)


class _FakeTensor:
    def __init__(self, array: np.ndarray, labels: tuple[str, str, str]) -> None:
        self._array = np.array(array, dtype=np.complex128, copy=True)
        self._labels = labels

    def get_leg_labels(self) -> tuple[str, str, str]:
        return self._labels

    def to_ndarray(self) -> np.ndarray:
        return np.array(self._array, copy=True)


class _FakeFiniteMPS:
    finite = True

    def __init__(self, tensors: tuple[np.ndarray, ...], *, norm: complex = 1.0) -> None:
        self.L = len(tensors)
        self.norm = norm
        self._tensors = tensors

    def get_B(self, site: int, *, form: str) -> _FakeTensor:
        assert form == "B"
        return _FakeTensor(self._tensors[site], ("vL", "p", "vR"))


def test_tenpy_converter_preserves_q0_msb_tensor_order_and_global_norm() -> None:
    q0_msb_vector = np.array([1.0, 2.0 + 1.0j, -3.0j, 4.0 - 2.0j])
    source = mps_from_dense_statevector(q0_msb_vector)
    converted = tenpy_mps_to_open_boundary(
        _FakeFiniteMPS(source.tensors, norm=2.0 - 1.0j)
    )

    assert np.allclose(
        converted.to_dense(),
        (2.0 - 1.0j) * q0_msb_vector,
        rtol=0.0,
        atol=2e-13,
    )
    assert converted.bond_dimensions == source.bond_dimensions


def test_tenpy_converter_rejects_unknown_leg_order_and_nonfinite_boundary() -> None:
    source = mps_from_dense_statevector(np.array([1.0, 0.0, 0.0, 0.0]))

    class _BadLabels(_FakeFiniteMPS):
        def get_B(self, site: int, *, form: str) -> _FakeTensor:
            assert form == "B"
            return _FakeTensor(self._tensors[site], ("p", "vL", "vR"))

    with pytest.raises(ValueError, match="leg order"):
        tenpy_mps_to_open_boundary(_BadLabels(source.tensors))

    nonfinite = _FakeFiniteMPS(source.tensors)
    nonfinite.finite = False
    with pytest.raises(ValueError, match="only finite"):
        tenpy_mps_to_open_boundary(nonfinite)


def test_dmrg_config_validates_and_returns_a_fresh_engine_mapping() -> None:
    config = DMRGConfig(chi_max=7, max_sweeps=9, min_sweeps=2, mixer=False)
    first = config.engine_options()
    second = config.engine_options()
    first["trunc_params"]["chi_max"] = 99

    assert second["trunc_params"]["chi_max"] == 7
    assert second["max_E_err"] == pytest.approx(1e-10)

    with pytest.raises(ValueError, match="must not exceed"):
        DMRGConfig(max_sweeps=2, min_sweeps=3)
    with pytest.raises(ValueError, match="positive"):
        DMRGConfig(chi_max=0)
    with pytest.raises(ValueError, match="nonnegative"):
        DMRGConfig(svd_min=-1.0)


def test_missing_tenpy_raises_a_clear_optional_dependency_error() -> None:
    if importlib.util.find_spec("tenpy") is not None:
        pytest.skip("This check is specific to a base environment without TeNPy.")
    with pytest.raises(RuntimeError, match="physics-tenpy"):
        dmrg_ground_state_mps("heisenberg", 4)


def test_dmrg_input_validation_happens_before_optional_backend_loading() -> None:
    with pytest.raises(ValueError, match="model must be one of"):
        dmrg_ground_state_mps("not-a-model", 4)
    with pytest.raises(ValueError, match="must be positive"):
        dmrg_ground_state_mps("xy", 0)
    with pytest.raises(TypeError, match="DMRGConfig"):
        dmrg_ground_state_mps("xy", 4, config=object())  # type: ignore[arg-type]


@pytest.mark.skipif(
    importlib.util.find_spec("tenpy") is None,
    reason="requires optional physics-tenpy",
)
@pytest.mark.parametrize(
    ("model", "kwargs", "hamiltonian"),
    [
        ("heisenberg", {}, heisenberg_hamiltonian(4)),
        ("xy", {}, xy_hamiltonian(4)),
        ("xxz", {"delta": 0.6}, xxz_hamiltonian(4, 0.6)),
        (
            "tfim",
            {"coupling_zz": 0.8, "field_x": 1.1},
            tfim_tndiff_hamiltonian(4, coupling_zz=0.8, field_x=1.1),
        ),
    ],
)
def test_tenpy_dmrg_models_match_small_dense_ground_energy(
    model: str,
    kwargs: dict[str, float],
    hamiltonian: np.ndarray,
) -> None:
    config = DMRGConfig(
        chi_max=16,
        max_sweeps=20,
        min_sweeps=4,
        energy_tolerance=1e-12,
    )
    result = dmrg_ground_state_mps(model, 4, config=config, **kwargs)
    exact = exact_ground_state(hamiltonian)
    dense_target = result.mps.to_dense()
    dense_energy = float(np.vdot(dense_target, hamiltonian @ dense_target).real)

    assert result.mps.norm() == pytest.approx(1.0, abs=2e-12)
    assert result.energy == pytest.approx(exact.energy, abs=2e-8)
    assert dense_energy == pytest.approx(exact.energy, abs=2e-8)
    assert result.max_bond_dimension <= config.chi_max
    assert result.provenance["boundary_condition"] == "open"
    assert result.provenance["site_order"] == "tenpy_site_0_equals_q0_most_significant"
    assert result.provenance["hamiltonian"]["spin_convention"] == "S=sigma/2 for heisenberg/xy/xxz"
    assert result.provenance["convergence_certification"]["status"] == (
        "single_run_not_independently_budget_certified"
    )
