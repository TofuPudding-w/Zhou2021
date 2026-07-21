r"""Optional TeNPy-DMRG ground-state targets for the Zhou 2021 ADQC study.

Zhou, Hong and Ran use finite open-boundary MPS ground states as the target
states for their large-system experiments.  This module provides that target
path without making TeNPy a mandatory dependency of the small dense-state
reproduction.  It deliberately keeps the paper Hamiltonians and the later
comparison extensions separate:

* ``heisenberg`` and ``xy`` implement Zhou Eqs. (4) and (5), with
  :math:`S^\alpha=\sigma^\alpha/2` and open boundaries;
* ``xxz`` and ``tfim``/``tfim_tndiff`` are explicitly labelled extensions.

The TeNPy MPS site order is mapped directly to this repository's q0-MSB
convention: TeNPy site 0 becomes q0 and its ``B`` tensor is retained in the
``(vL, p, vR) == (chi_left, physical, chi_right)`` axis order.  The converter
checks this rather than silently transposing unknown tensor layouts.

``physics-tenpy`` is optional.  Importing this module is always safe; calling
the DMRG runner without TeNPy raises a precise installation error.
"""

from __future__ import annotations

import importlib.metadata
import math
import numbers
from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np

from .conventions import validate_num_qubits, validate_real
from .mps import OpenBoundaryMPS

TenpyModelName = Literal["heisenberg", "xy", "xxz", "tfim", "tfim_tndiff"]
InitialProductState = Literal["neel", "all_up", "all_down"]


def _validate_positive_integer(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise TypeError(f"{name} must be an integer.")
    result = int(value)
    if result < 1:
        raise ValueError(f"{name} must be positive, got {result}.")
    return result


@dataclass(frozen=True, slots=True)
class DMRGConfig:
    """Explicit finite two-site DMRG controls for a TeNPy MPS target.

    The Zhou paper specifies the target MPS bond dimensions but not a complete
    public DMRG convergence schedule.  These values are therefore controlled
    reproduction settings, not author-supplied hyperparameters.  Every value
    is returned in :class:`TenpyDMRGGroundStateTarget` provenance.
    """

    chi_max: int = 64
    max_sweeps: int = 30
    min_sweeps: int = 4
    energy_tolerance: float = 1e-10
    svd_min: float = 1e-14
    mixer: bool = True
    initial_product_state: InitialProductState = "neel"

    def __post_init__(self) -> None:
        chi_max = _validate_positive_integer(self.chi_max, name="chi_max")
        max_sweeps = _validate_positive_integer(self.max_sweeps, name="max_sweeps")
        if isinstance(self.min_sweeps, bool) or not isinstance(
            self.min_sweeps, numbers.Integral
        ):
            raise TypeError("min_sweeps must be an integer.")
        min_sweeps = int(self.min_sweeps)
        if min_sweeps < 0:
            raise ValueError("min_sweeps must be nonnegative.")
        if min_sweeps > max_sweeps:
            raise ValueError("min_sweeps must not exceed max_sweeps.")
        energy_tolerance = validate_real(
            self.energy_tolerance,
            name="energy_tolerance",
            positive=True,
        )
        svd_min = validate_real(self.svd_min, name="svd_min")
        if svd_min < 0.0:
            raise ValueError("svd_min must be nonnegative.")
        if not isinstance(self.mixer, bool):
            raise TypeError("mixer must be a bool.")
        if self.initial_product_state not in {"neel", "all_up", "all_down"}:
            raise ValueError(
                "initial_product_state must be 'neel', 'all_up', or 'all_down'."
            )

        object.__setattr__(self, "chi_max", chi_max)
        object.__setattr__(self, "max_sweeps", max_sweeps)
        object.__setattr__(self, "min_sweeps", min_sweeps)
        object.__setattr__(self, "energy_tolerance", energy_tolerance)
        object.__setattr__(self, "svd_min", svd_min)

    def engine_options(self) -> dict[str, Any]:
        """Return a fresh TeNPy ``TwoSiteDMRGEngine`` option mapping."""

        return {
            "mixer": self.mixer,
            "max_E_err": self.energy_tolerance,
            "max_sweeps": self.max_sweeps,
            "min_sweeps": self.min_sweeps,
            "trunc_params": {
                "chi_max": self.chi_max,
                "svd_min": self.svd_min,
            },
        }


@dataclass(frozen=True, slots=True)
class TenpyDMRGGroundStateTarget:
    """A normalized TeNPy-DMRG ground-state target plus audit metadata."""

    model: str
    requested_model: str
    num_qubits: int
    mps: OpenBoundaryMPS
    energy: float
    engine_energy: float
    engine_expectation_difference: float
    chi: tuple[int, ...]
    max_bond_dimension: int
    tenpy_mps_norm_before_normalization: float
    config: DMRGConfig
    provenance: dict[str, Any]


def _physics_tenpy_version() -> str | None:
    try:
        return importlib.metadata.version("physics-tenpy")
    except importlib.metadata.PackageNotFoundError:
        return None


def _require_tenpy() -> tuple[Any, Any, Any]:
    """Load the minimal TeNPy API only when an optional DMRG run is requested."""

    try:
        from tenpy.algorithms.dmrg import TwoSiteDMRGEngine
        from tenpy.models.model import CouplingMPOModel
        from tenpy.networks.mps import MPS
        from tenpy.networks.site import SpinHalfSite
    except ImportError as error:  # pragma: no cover - exercised without optional dependency
        raise RuntimeError(
            "TeNPy DMRG target generation requires the optional 'physics-tenpy' "
            "dependency. Install it in the reproduction environment, for example "
            "with `python -m pip install physics-tenpy`."
        ) from error
    return TwoSiteDMRGEngine, CouplingMPOModel, (MPS, SpinHalfSite)


def _canonical_model_name(model: str) -> str:
    if model == "tfim":
        return "tfim_tndiff"
    if model in {"heisenberg", "xy", "xxz", "tfim_tndiff"}:
        return model
    raise ValueError(
        "model must be one of 'heisenberg', 'xy', 'xxz', 'tfim', or "
        f"'tfim_tndiff'; got {model!r}."
    )


def _build_tenpy_model(
    model: str,
    num_qubits: int,
    *,
    coupling: float,
    delta: float,
    coupling_zz: float,
    field_x: float,
) -> Any:
    """Construct a finite OBC TeNPy model in the repository's conventions."""

    _two_site_dmrg, coupling_mpo_model, components = _require_tenpy()
    _mps, spin_half_site = components
    canonical_model = _canonical_model_name(model)

    class _OpenBoundarySpinChain(coupling_mpo_model):
        def init_sites(self, model_params: Any) -> Any:
            del model_params
            # ``conserve=None`` avoids silently constraining the target to the
            # initial product state's magnetisation sector.
            return spin_half_site(conserve=None)

        def init_terms(self, model_params: Any) -> None:
            resolved_model = model_params["resolved_model"]
            if resolved_model in {"heisenberg", "xy", "xxz"}:
                resolved_coupling = model_params["coupling"]
                self.add_coupling(resolved_coupling, 0, "Sx", 0, "Sx", 1)
                self.add_coupling(resolved_coupling, 0, "Sy", 0, "Sy", 1)
                if resolved_model == "heisenberg":
                    resolved_delta = 1.0
                elif resolved_model == "xy":
                    resolved_delta = 0.0
                else:
                    resolved_delta = model_params["delta"]
                if resolved_delta != 0.0:
                    self.add_coupling(
                        resolved_coupling * resolved_delta,
                        0,
                        "Sz",
                        0,
                        "Sz",
                        1,
                    )
                return

            # Explicit extension: H = -J sum ZZ - h_x sum X with Pauli,
            # not spin, operators.  This matches ``tfim_tndiff_hamiltonian``.
            self.add_coupling(
                -model_params["coupling_zz"],
                0,
                "Sigmaz",
                0,
                "Sigmaz",
                1,
            )
            self.add_onsite(-model_params["field_x"], 0, "Sigmax")

    return _OpenBoundarySpinChain(
        {
            "L": num_qubits,
            # In TeNPy's CouplingMPOModel construction, finite MPS boundary
            # conditions generate a non-wrapping one-dimensional chain.
            "bc_MPS": "finite",
            "resolved_model": canonical_model,
            "coupling": coupling,
            "delta": delta,
            "coupling_zz": coupling_zz,
            "field_x": field_x,
        }
    )


def _initial_product_state_labels(
    num_qubits: int,
    initial_product_state: InitialProductState,
) -> list[str]:
    if initial_product_state == "neel":
        return ["up" if site % 2 == 0 else "down" for site in range(num_qubits)]
    if initial_product_state == "all_up":
        return ["up"] * num_qubits
    if initial_product_state == "all_down":
        return ["down"] * num_qubits
    raise AssertionError("DMRGConfig validation should have rejected this state label.")


def tenpy_mps_to_open_boundary(psi: Any) -> OpenBoundaryMPS:
    """Convert a finite TeNPy MPS to q0-MSB :class:`OpenBoundaryMPS`.

    The conversion is intentionally conservative.  TeNPy's ``B`` tensors are
    requested in their documented ``(vL, p, vR)`` layout; a changed or unknown
    leg layout raises rather than creating an incorrectly ordered target.
    The separate ``psi.norm`` factor is placed on the first core, so the
    returned MPS represents the same (possibly unnormalized) state.
    """

    if not bool(getattr(psi, "finite", False)):
        raise ValueError("only finite TeNPy MPS objects can be converted")
    length = getattr(psi, "L", None)
    if isinstance(length, bool) or not isinstance(length, numbers.Integral):
        raise TypeError("finite TeNPy MPS must expose an integer L site count")
    length = int(length)
    if length < 1:
        raise ValueError("finite TeNPy MPS must contain at least one site")

    tensors: list[np.ndarray] = []
    for site in range(length):
        tensor = psi.get_B(site, form="B")
        labels = tuple(tensor.get_leg_labels())
        if labels != ("vL", "p", "vR"):
            raise ValueError(
                "unexpected TeNPy B-tensor leg order at site "
                f"{site}: {labels!r}; expected ('vL', 'p', 'vR')."
            )
        array = np.asarray(tensor.to_ndarray(), dtype=np.complex128)
        tensors.append(np.array(array, dtype=np.complex128, copy=True))

    try:
        global_norm = complex(getattr(psi, "norm", 1.0))
    except (TypeError, ValueError) as error:
        raise ValueError("TeNPy MPS norm must be convertible to a finite scalar") from error
    if not math.isfinite(global_norm.real) or not math.isfinite(global_norm.imag):
        raise ValueError("TeNPy MPS norm must be finite")
    if global_norm == 0.0:
        raise ValueError("cannot convert a zero-norm TeNPy MPS")
    tensors[0] *= global_norm
    return OpenBoundaryMPS(tuple(tensors))


def _real_finite(value: Any, *, name: str) -> float:
    complex_value = complex(value)
    if not math.isfinite(complex_value.real) or not math.isfinite(complex_value.imag):
        raise ArithmeticError(f"{name} is non-finite: {complex_value!r}.")
    allowed_imaginary = 1e-9 * (1.0 + abs(complex_value.real))
    if abs(complex_value.imag) > allowed_imaginary:
        raise ArithmeticError(
            f"{name} acquired an unexpectedly large imaginary component: "
            f"{complex_value.imag:.3e}."
        )
    return float(complex_value.real)


def dmrg_ground_state_mps(
    model: TenpyModelName | str,
    num_qubits: int,
    *,
    config: DMRGConfig | None = None,
    coupling: float = 1.0,
    delta: float = 1.0,
    coupling_zz: float = 1.0,
    field_x: float = 1.0,
) -> TenpyDMRGGroundStateTarget:
    """Produce a normalized finite-DMRG MPS target with complete provenance.

    For ``heisenberg`` and ``xy`` this is the paper-scale target-generation
    path.  ``xxz`` and ``tfim`` (the latter aliased to ``tfim_tndiff`` in the
    result) are explicit comparison extensions.  No dense statevector is ever
    materialized here, so the function remains usable at the paper's ``N=48``.
    """

    num_qubits = validate_num_qubits(num_qubits)
    coupling = validate_real(coupling, name="coupling")
    delta = validate_real(delta, name="delta")
    coupling_zz = validate_real(coupling_zz, name="coupling_zz")
    field_x = validate_real(field_x, name="field_x")
    requested_model = str(model)
    canonical_model = _canonical_model_name(requested_model)
    resolved_config = config if config is not None else DMRGConfig()
    if not isinstance(resolved_config, DMRGConfig):
        raise TypeError("config must be a DMRGConfig or None.")

    two_site_dmrg, _coupling_mpo_model, components = _require_tenpy()
    mps_class, _spin_half_site = components
    tenpy_model = _build_tenpy_model(
        canonical_model,
        num_qubits,
        coupling=coupling,
        delta=delta,
        coupling_zz=coupling_zz,
        field_x=field_x,
    )
    initial_labels = _initial_product_state_labels(
        num_qubits,
        resolved_config.initial_product_state,
    )
    psi = mps_class.from_product_state(
        tenpy_model.lat.mps_sites(),
        initial_labels,
        bc="finite",
    )
    engine = two_site_dmrg(psi, tenpy_model, resolved_config.engine_options())
    engine_energy_raw, optimized_psi = engine.run()
    if optimized_psi is not psi:
        psi = optimized_psi
    if not bool(getattr(psi, "finite", False)):
        raise ArithmeticError("TeNPy DMRG returned a non-finite-boundary MPS.")
    if int(getattr(psi, "L", -1)) != num_qubits:
        raise ArithmeticError("TeNPy DMRG returned an MPS with an unexpected site count.")

    # Canonical form puts TeNPy's global norm in ``psi.norm``.  Preserve that
    # factor during conversion, then normalize the repository target exactly.
    psi.canonical_form()
    converted = tenpy_mps_to_open_boundary(psi)
    norm_before_normalization = converted.norm()
    target_mps = converted.normalized()

    engine_energy = _real_finite(engine_energy_raw, name="DMRG engine energy")
    expectation_energy = _real_finite(
        tenpy_model.H_MPO.expectation_value(psi),
        name="TeNPy MPO expectation energy",
    )
    chi = tuple(int(value) for value in getattr(psi, "chi", ()))
    if len(chi) != num_qubits - 1 or any(value < 1 for value in chi):
        raise ArithmeticError(
            "TeNPy MPS chi ledger is inconsistent with a finite "
            f"N={num_qubits} chain: {chi!r}."
        )

    provenance: dict[str, Any] = {
        "schema_version": "zhou2021-tenpy-dmrg-target-v1",
        "backend": "tenpy.algorithms.dmrg.TwoSiteDMRGEngine",
        "physics_tenpy_version": _physics_tenpy_version(),
        "boundary_condition": "open",
        "mps_boundary_condition": "finite",
        "site_order": "tenpy_site_0_equals_q0_most_significant",
        "tensor_axis_order": "vL,p,vR == chi_left,physical,chi_right",
        "physical_basis": {"up": 0, "down": 1},
        "paper_model": canonical_model in {"heisenberg", "xy"},
        "hamiltonian": {
            "model": canonical_model,
            "spin_convention": "S=sigma/2 for heisenberg/xy/xxz",
            "tfim_convention": "-J*ZZ-hx*X (Pauli) for tfim_tndiff",
            "coupling": coupling,
            "delta": delta,
            "coupling_zz": coupling_zz,
            "field_x": field_x,
        },
        "dmrg_config": asdict(resolved_config),
        "initial_product_state_labels": initial_labels,
        "chi": list(chi),
        "max_bond_dimension": max((1, *chi)),
        "tenpy_mps_norm_before_normalization": norm_before_normalization,
        "target_mps_norm": target_mps.norm(),
        "engine_energy": engine_energy,
        "expectation_energy": expectation_energy,
        "engine_expectation_difference": abs(engine_energy - expectation_energy),
        "energy_reference": {
            "value": expectation_energy,
            "role": (
                "TeNPy MPO expectation of the normalized DMRG target MPS; "
                "a converged ground-state approximation, not dense exact diagonalization"
            ),
        },
        "convergence_certification": {
            "status": "single_run_not_independently_budget_certified",
            "reason": (
                "Engine/MPO-energy agreement validates this exported state evaluation, "
                "not convergence with respect to chi_max, sweep budget, or svd_min. "
                "A paper-scale claim requires a separately recorded multi-budget study."
            ),
            "configured_max_sweeps": resolved_config.max_sweeps,
            "configured_min_sweeps": resolved_config.min_sweeps,
            "configured_chi_max": resolved_config.chi_max,
            "configured_svd_min": resolved_config.svd_min,
        },
    }
    return TenpyDMRGGroundStateTarget(
        model=canonical_model,
        requested_model=requested_model,
        num_qubits=num_qubits,
        mps=target_mps,
        energy=expectation_energy,
        engine_energy=engine_energy,
        engine_expectation_difference=abs(engine_energy - expectation_energy),
        chi=chi,
        max_bond_dimension=max((1, *chi)),
        tenpy_mps_norm_before_normalization=norm_before_normalization,
        config=resolved_config,
        provenance=provenance,
    )


__all__ = [
    "DMRGConfig",
    "InitialProductState",
    "TenpyDMRGGroundStateTarget",
    "TenpyModelName",
    "dmrg_ground_state_mps",
    "tenpy_mps_to_open_boundary",
]
