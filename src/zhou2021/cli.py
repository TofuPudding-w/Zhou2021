"""Command line entry points for the convention-explicit Zhou 2021 reproduction.

The command line intentionally separates the paper's supervised ADQC compiler
from the later, explicitly labelled many-body and noise extensions.  It is a
small deterministic front end around the independently testable library API;
it does not hide optimizer settings or silently generate a random target.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import numbers
import platform
import sys
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .adqc import ADQCTrainingConfig, ADQCTrainingResult, train_adqc
from .adqc_mps import ADQCMPSConfig, ADQCMPSResult, train_adqc_mps
from .gate_archive import (
    load_ordered_u4_npz,
    ordered_u4_archive_from_staircase,
    replay_ordered_u4_archive,
    save_ordered_u4_npz,
)
from .ground_states import GroundStateResult, exact_ground_state
from .metrics import energy_error_per_site, energy_expectation, squared_state_fidelity
from .models import (
    heisenberg_hamiltonian,
    tfim_tndiff_hamiltonian,
    xxz_hamiltonian,
    xy_hamiltonian,
)
from .mps import (
    LoadedOpenBoundaryMPS,
    OpenBoundaryMPS,
    load_open_boundary_mps_npz,
    save_open_boundary_mps_npz,
)
from .mps_observables import (
    MPSModelParameters,
    TNDiffTFIMParameters,
    XXZParameters,
    mps_energy_error_per_site,
    mps_model_energy,
)
from .mps_targets import dense_model_ground_state_mps
from .polar import POLAR_NUMERICAL_RANK_RCOND
from .sweeps import MPSDepthSweepResult, run_mps_depth_sweep
from .tenpy_targets import DMRGConfig, dmrg_ground_state_mps
from .torch_mps import estimated_mps_complex_elements


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _package_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _optional_import_status(module: str) -> dict[str, Any]:
    """Report whether one optional runtime can actually be imported.

    Package metadata alone is insufficient for Qiskit: a mixed pre-/post-1.0
    installation can advertise a version yet fail at import time.  Doctor
    output therefore records this runtime check without making any optional
    dependency mandatory.
    """

    try:
        importlib.import_module(module)
    except Exception as error:  # diagnostic must not fail on a broken optional stack
        return {
            "usable": False,
            "error_type": type(error).__name__,
            "error": str(error),
        }
    return {"usable": True, "error_type": None, "error": None}


def _array_sha256(array: np.ndarray) -> str:
    """Hash a state or matrix together with dtype and shape metadata."""

    contiguous = np.ascontiguousarray(array)
    header = json.dumps(
        {"dtype": contiguous.dtype.str, "shape": list(contiguous.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256()
    digest.update(header)
    digest.update(b"\n")
    digest.update(memoryview(contiguous).cast("B"))
    return digest.hexdigest()


def _file_sha256(path: Path) -> str:
    """Return the SHA-256 digest of one already-written provenance artifact."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _output_path_with_parent(path: str | Path) -> Path:
    """Resolve one CLI output and create its parent without weakening library I/O checks."""

    destination = Path(path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    return destination


def _ordered_u4_archive_payload(
    *,
    arguments: argparse.Namespace,
    result: ADQCTrainingResult | ADQCMPSResult,
) -> dict[str, Any] | None:
    """Optionally freeze projected U(4)s to a Qiskit-independent replay archive.

    The archive is intentionally generated after optimization and never feeds
    back into the loss or the optimizer.  Its file SHA protects exact transport
    bytes, while its semantic SHA protects the ordered matrices and all declared
    replay conventions independently of NPZ/ZIP implementation details.
    """

    output = getattr(arguments, "ordered_u4_output", None)
    if output is None:
        return None
    command = getattr(arguments, "command", "library-call")
    archive = ordered_u4_archive_from_staircase(
        result.model,
        user_metadata={
            "producer": "zhou2021-cli",
            "command": command,
            "physical_gate_source": "SVD polar projection of trained latent 4x4 matrices",
        },
    )
    destination = save_ordered_u4_npz(
        _output_path_with_parent(output),
        archive,
    ).resolve()
    return {
        "schema_version": archive.metadata["schema_version"],
        "path": str(destination),
        "file_sha256": _file_sha256(destination),
        "semantic_sha256": archive.semantic_sha256,
        "num_qubits": archive.num_qubits,
        "num_layers": archive.num_layers,
        "gate_count": len(archive.gates),
        "qiskit_dependency": "none",
    }


def _mps_semantic_sha256(mps: OpenBoundaryMPS) -> str:
    """Hash ordered MPS cores together with their convention-relevant shapes."""

    digest = hashlib.sha256()
    digest.update(b"zhou2021-obc-mps-semantic-v1\n")
    for site, tensor in enumerate(mps.tensors):
        contiguous = np.ascontiguousarray(tensor, dtype=np.complex128)
        header = json.dumps(
            {"site": site, "dtype": contiguous.dtype.str, "shape": list(contiguous.shape)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        digest.update(header)
        digest.update(b"\n")
        digest.update(memoryview(contiguous).cast("B"))
    return digest.hexdigest()


def _complex_vector_json(state: np.ndarray) -> list[list[float]]:
    """Serialize a statevector as explicit real/imaginary pairs, never strings."""

    flat = np.asarray(state, dtype=np.complex128).reshape(-1)
    return [[float(value.real), float(value.imag)] for value in flat]


def _json_physical_nlf(value: float) -> float | None:
    """Encode a physical Eq. (1) value without non-standard JSON Infinity."""

    value = float(value)
    if math.isfinite(value):
        return value
    if value == math.inf:
        return None
    raise ArithmeticError("A physical negative-log fidelity is NaN or negative infinity.")


def _physical_nlf_classification(value: float) -> str:
    """Classify the JSON representation of one floor-free physical NLF."""

    value = float(value)
    if math.isfinite(value):
        return "finite"
    if value == math.inf:
        return "positive_infinity"
    raise ArithmeticError("A physical negative-log fidelity is NaN or negative infinity.")


def _json_safe_metric_record(record: Any) -> dict[str, Any]:
    """Convert physical NLF infinities in a dataclass record to JSON-safe fields."""

    payload = asdict(record)
    for key in tuple(payload):
        if (
            "negative_log_fidelity_per_site" in key
            and "training_floored" not in key
        ):
            value = float(payload[key])
            payload[key] = _json_physical_nlf(value)
            payload[f"{key}_classification"] = _physical_nlf_classification(value)
    return payload


def doctor_snapshot() -> dict[str, Any]:
    """Return a read-only environment/convention snapshot suitable for manifests."""

    provenance_path = _project_root() / "provenance" / "paper_source.json"
    paper = json.loads(provenance_path.read_text(encoding="utf-8"))
    return {
        "schema_version": "zhou2021-doctor-v1",
        "implementation_version": "0.1.0",
        "python": sys.version,
        "platform": platform.platform(),
        "dependencies": {
            name: _package_version(name)
            for name in (
                "numpy",
                "scipy",
                "torch",
                "pytest",
                "physics-tenpy",
                "qiskit",
                "qiskit-aer",
                "qiskit-ibm-runtime",
            )
        },
        "optional_runtime_imports": {
            "physics_tenpy": _optional_import_status("tenpy"),
            "qiskit": _optional_import_status("qiskit"),
            "qiskit_aer": _optional_import_status("qiskit_aer"),
            "qiskit_ibm_runtime": _optional_import_status("qiskit_ibm_runtime"),
        },
        "torch": {
            "default_dtype": str(torch.get_default_dtype()),
            "cuda_available": bool(torch.cuda.is_available()),
        },
        "conventions": {
            "statevector_order": "q0-most-significant",
            "two_qubit_local_basis": ["00", "01", "10", "11"],
            "complex_dtype": "complex128",
            "paper_ansatz": "serial nearest-neighbour U(4) staircase; N-1 gates/layer",
            "paper_loss": "-log(abs(<target|prepared>))/N",
            "polar_rank_deficient_policy": "reject_nonunique_u4_completion",
            "polar_numerical_rank_rcond": POLAR_NUMERICAL_RANK_RCOND,
            "paper_models": ["open-boundary Heisenberg", "open-boundary XY"],
            "extensions": ["TFIM", "XXZ", "FakeWashingtonV2 noise"],
        },
        "paper_source": paper,
    }


def _build_hamiltonian(arguments: argparse.Namespace) -> np.ndarray:
    if arguments.model == "heisenberg":
        return heisenberg_hamiltonian(arguments.n_qubits, coupling=arguments.coupling)
    if arguments.model == "xy":
        return xy_hamiltonian(arguments.n_qubits, coupling=arguments.coupling)
    if arguments.model == "xxz":
        return xxz_hamiltonian(
            arguments.n_qubits,
            delta=arguments.delta,
            coupling=arguments.coupling,
        )
    if arguments.model == "tfim":
        return tfim_tndiff_hamiltonian(
            arguments.n_qubits,
            coupling_zz=arguments.coupling_zz,
            field_x=arguments.field_x,
        )
    raise ValueError(f"Unsupported model {arguments.model!r}.")  # pragma: no cover


def _training_config_from_arguments(arguments: argparse.Namespace) -> ADQCTrainingConfig:
    return ADQCTrainingConfig(
        max_layers=arguments.layers,
        epochs_per_stage=arguments.epochs_per_stage,
        learning_rate=arguments.learning_rate,
        learning_rate_decay=arguments.learning_rate_decay,
        adam_betas=(arguments.adam_beta1, arguments.adam_beta2),
        adam_eps=arguments.adam_eps,
        adam_weight_decay=arguments.adam_weight_decay,
        adam_amsgrad=arguments.adam_amsgrad,
        first_layer_initialization=arguments.first_layer_initialization,
        new_layer_perturbation=arguments.new_layer_perturbation,
        seed=arguments.seed,
        loss_floor_squared_fidelity=arguments.loss_floor_squared_fidelity,
        convergence_nlf=arguments.convergence_nlf,
    )


def _mps_training_config_from_arguments(arguments: argparse.Namespace) -> ADQCMPSConfig:
    """Build a fully explicit MPS execution configuration from CLI coordinates."""

    return ADQCMPSConfig(
        training=_training_config_from_arguments(arguments),
        max_bond_dimension=arguments.max_bond_dimension,
        singular_value_cutoff=arguments.singular_value_cutoff,
        unitary_atol=arguments.unitary_atol,
    )


def _model_parameters_from_record(
    record: Mapping[str, Any],
    *,
    num_qubits: int,
) -> MPSModelParameters:
    """Parse one explicit MPS-energy model record without inferred units.

    The archive schema intentionally distinguishes Zhou's spin-normalized
    Heisenberg/XY chains from the Pauli-normalized TN-Diff TFIM extension.
    Unknown names or missing anisotropy coordinates fail loudly instead of
    assigning a plausible-looking but wrong energy convention.
    """

    model = record.get("model")
    if not isinstance(model, str):
        raise ValueError("MPS energy model record must contain a string 'model'.")

    def required_parameter(name: str) -> Any:
        if name not in record or record[name] is None:
            raise ValueError(
                f"MPS energy model record for {model!r} must explicitly declare {name!r}."
            )
        return record[name]

    if model == "heisenberg":
        return XXZParameters(
            num_qubits,
            delta=1.0,
            coupling=required_parameter("coupling"),
        )
    if model == "xy":
        return XXZParameters(
            num_qubits,
            delta=0.0,
            coupling=required_parameter("coupling"),
        )
    if model == "xxz":
        return XXZParameters(
            num_qubits,
            delta=required_parameter("delta"),
            coupling=required_parameter("coupling"),
        )
    if model in {"tfim", "tfim_tndiff"}:
        return TNDiffTFIMParameters(
            num_qubits,
            coupling_zz=required_parameter("coupling_zz"),
            field_x=required_parameter("field_x"),
        )
    raise ValueError(
        "MPS energy model must be heisenberg, xy, xxz, tfim, or tfim_tndiff; "
        f"got {model!r}."
    )


def _finite_energy_reference(value: Any, *, name: str) -> float:
    """Validate a provenance total-energy scalar before reporting an error."""

    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f"{name} must be a real scalar.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def _metadata_mps_energy_spec(
    metadata: Mapping[str, Any],
    *,
    num_qubits: int,
) -> tuple[MPSModelParameters, float | None, str, str] | None:
    """Extract a model/energy reference from known target provenance fields."""

    hamiltonian = metadata.get("hamiltonian")
    if isinstance(hamiltonian, Mapping):
        model = _model_parameters_from_record(hamiltonian, num_qubits=num_qubits)
        declared_model = str(hamiltonian["model"])
        reference = metadata.get("energy_reference")
        if isinstance(reference, Mapping) and "value" in reference:
            role = reference.get("role", "source-declared total-energy reference")
            return (
                model,
                _finite_energy_reference(reference["value"], name="energy_reference"),
                str(role),
                declared_model,
            )
        if "ground_energy" in metadata:
            return model, _finite_energy_reference(
                metadata["ground_energy"], name="ground_energy"
            ), "exact or source-declared ground-state total energy", declared_model
        if "expectation_energy" in metadata:
            return model, _finite_energy_reference(
                metadata["expectation_energy"], name="expectation_energy"
            ), "source target-MPS total-energy expectation", declared_model
        return model, None, "model declared but no total-energy reference supplied", declared_model

    return None


def _mps_energy_spec_from_arguments(
    arguments: argparse.Namespace,
    loaded: LoadedOpenBoundaryMPS,
) -> tuple[MPSModelParameters, float | None, str, str] | None:
    """Prefer explicit CLI energy coordinates, else consume signed target provenance."""

    explicit_model = getattr(arguments, "energy_model", None)
    if explicit_model is not None:
        model = _model_parameters_from_record(
            {
                "model": explicit_model,
                "coupling": arguments.energy_coupling,
                "delta": arguments.energy_delta,
                "coupling_zz": arguments.energy_coupling_zz,
                "field_x": arguments.energy_field_x,
            },
            num_qubits=loaded.mps.num_qubits,
        )
        reference = (
            None
            if arguments.energy_reference is None
            else _finite_energy_reference(arguments.energy_reference, name="energy_reference")
        )
        return model, reference, "explicit CLI energy coordinates", explicit_model
    return _metadata_mps_energy_spec(loaded.metadata, num_qubits=loaded.mps.num_qubits)


def _dense_hamiltonian_from_mps_model(model: MPSModelParameters) -> np.ndarray:
    """Build the q0-MSB dense Hamiltonian matching one declared MPS model.

    This helper is intentionally limited to the small-N noise path.  It
    preserves the MPS provenance convention rather than inferring whether an
    XXZ record was originally named Heisenberg or XY.
    """

    if isinstance(model, XXZParameters):
        return xxz_hamiltonian(
            model.num_qubits,
            delta=model.delta,
            coupling=model.coupling,
        )
    if isinstance(model, TNDiffTFIMParameters):
        return tfim_tndiff_hamiltonian(
            model.num_qubits,
            coupling_zz=model.coupling_zz,
            field_x=model.field_x,
        )
    raise TypeError("model must be XXZParameters or TNDiffTFIMParameters.")


def _mps_physical_metrics(
    *,
    target_mps: OpenBoundaryMPS,
    prepared_mps: OpenBoundaryMPS,
    spec: tuple[MPSModelParameters, float | None, str, str] | None,
) -> dict[str, Any]:
    """Report MPS-only many-body energy metrics or an explicit unavailable state."""

    if spec is None:
        return {
            "available": False,
            "reason": (
                "No explicit energy model was provided and target metadata did not contain "
                "a recognized Hamiltonian record."
            ),
        }
    model, reference_energy, reference_role, declared_model = spec
    target_energy = mps_model_energy(target_mps, model)
    prepared_energy = mps_model_energy(prepared_mps, model)
    payload: dict[str, Any] = {
        "available": True,
        "declared_model": declared_model,
        "model": model.to_record(),
        "target_mps_energy": target_energy,
        "prepared_mps_energy": prepared_energy,
        "reference_energy_role": reference_role,
    }
    if reference_energy is not None:
        payload["reference_total_energy"] = reference_energy
        payload["target_energy_error_per_site"] = mps_energy_error_per_site(
            target_mps,
            model,
            reference_energy,
        )
        payload["prepared_energy_error_per_site"] = mps_energy_error_per_site(
            prepared_mps,
            model,
            reference_energy,
        )
    return payload


def _result_payload(
    *,
    arguments: argparse.Namespace,
    hamiltonian: np.ndarray,
    ground: GroundStateResult,
    result: ADQCTrainingResult,
) -> dict[str, Any]:
    prepared = result.prepared_state.detach().cpu().numpy()
    target = result.target_state.detach().cpu().numpy()
    final_energy = energy_expectation(prepared, hamiltonian)
    payload: dict[str, Any] = {
        "schema_version": "zhou2021-adqc-run-v2",
        "paper_method": {
            "identifier": "zhou-hong-ran-2021-adqc",
            "strict_paper_component": {
                "latent_gate": "unconstrained complex 4x4 -> SVD polar U Vh",
                "topology": "serial 1D nearest-neighbour staircase",
                "loss": "-log(abs(overlap))/N",
                "optimization": "layerwise warm-start Adam",
            },
            "extensions_present": arguments.model in {"tfim", "xxz"},
        },
        "target": {
            "model": arguments.model,
            "n_qubits": arguments.n_qubits,
            "boundary": "open",
            "parameters": {
                "coupling": arguments.coupling,
                "delta": arguments.delta if arguments.model == "xxz" else None,
                "coupling_zz": arguments.coupling_zz if arguments.model == "tfim" else None,
                "field_x": arguments.field_x if arguments.model == "tfim" else None,
            },
            "ground_energy": ground.energy,
            "ground_residual": ground.residual,
            "ground_gap": ground.gap,
            "ground_degeneracy": ground.degeneracy,
            "ground_selection_basis_index": ground.selection_basis_index,
            "target_state_sha256": _array_sha256(target),
            "hamiltonian_sha256": _array_sha256(hamiltonian),
        },
        "algorithm": {
            "config": asdict(result.config),
            "metric_contract": {
                "negative_log_fidelity_per_site": (
                    "physical Zhou Eq. (1), without an optimizer floor"
                ),
                "training_floored_negative_log_fidelity_per_site": (
                    "finite differentiable loss used for updates and convergence_nlf"
                ),
                "positive_infinity_json_encoding": (
                    "null accompanied by a positive_infinity classification"
                ),
            },
            "final_num_layers": result.model.num_layers,
            "gates_per_layer": result.model.gates_per_layer,
            "ordered_u4_gate_count": result.model.gates_per_layer * result.model.num_layers,
            "paper_matrix_element_count": result.model.paper_parameter_count,
            "real_latent_parameter_count": result.model.real_latent_parameter_count,
            "mps_bond_upper_bound": result.model.maximum_mps_bond_bound,
            "converged": result.converged,
            "stopping_reason": result.stopping_reason,
        },
        "ideal_metrics": {
            "squared_state_fidelity": squared_state_fidelity(target, prepared),
            "negative_log_fidelity_per_site": _json_physical_nlf(
                result.final_negative_log_fidelity_per_site
            ),
            "negative_log_fidelity_per_site_classification": (
                _physical_nlf_classification(result.final_negative_log_fidelity_per_site)
            ),
            "physical_negative_log_fidelity_per_site": _json_physical_nlf(
                result.final_physical_negative_log_fidelity_per_site
            ),
            "physical_negative_log_fidelity_per_site_classification": (
                _physical_nlf_classification(
                    result.final_physical_negative_log_fidelity_per_site
                )
            ),
            "training_floored_negative_log_fidelity_per_site": (
                result.final_training_floored_negative_log_fidelity_per_site
            ),
            "training_squared_fidelity_floor": (
                result.config.loss_floor_squared_fidelity
            ),
            "energy": final_energy,
            "energy_error": final_energy - ground.energy,
            "energy_error_per_site": energy_error_per_site(
                prepared,
                hamiltonian,
                ground.energy,
                num_qubits=arguments.n_qubits,
            ),
            "prepared_state_sha256": _array_sha256(prepared),
        },
        "stages": [
            {
                "num_layers": stage.num_layers,
                "initial_negative_log_fidelity_per_site": (
                    _json_physical_nlf(stage.initial_negative_log_fidelity_per_site)
                ),
                "initial_negative_log_fidelity_per_site_classification": (
                    _physical_nlf_classification(
                        stage.initial_negative_log_fidelity_per_site
                    )
                ),
                "final_negative_log_fidelity_per_site": _json_physical_nlf(
                    stage.final_negative_log_fidelity_per_site
                ),
                "final_negative_log_fidelity_per_site_classification": (
                    _physical_nlf_classification(stage.final_negative_log_fidelity_per_site)
                ),
                "initial_training_floored_negative_log_fidelity_per_site": (
                    stage.initial_training_floored_negative_log_fidelity_per_site
                ),
                "final_training_floored_negative_log_fidelity_per_site": (
                    stage.final_training_floored_negative_log_fidelity_per_site
                ),
                "initial_squared_fidelity": stage.initial_squared_fidelity,
                "final_squared_fidelity": stage.final_squared_fidelity,
                "update_count": len(stage.updates),
                "updates": [_json_safe_metric_record(update) for update in stage.updates],
            }
            for stage in result.stages
        ],
        "provenance": {
            "doctor": doctor_snapshot(),
        },
    }
    if arguments.include_statevectors:
        payload["target"]["target_statevector_real_imag"] = _complex_vector_json(target)
        payload["ideal_metrics"]["prepared_statevector_real_imag"] = _complex_vector_json(prepared)
    return payload


def run_smoke(arguments: argparse.Namespace) -> dict[str, Any]:
    """Run one fully recorded dense small-system ADQC ground-state encoding."""

    payload, result, _, _ = _run_dense_training(arguments)
    archive = _ordered_u4_archive_payload(arguments=arguments, result=result)
    if archive is not None:
        payload["ordered_u4_archive"] = archive
    return payload


def _run_dense_training(
    arguments: argparse.Namespace,
) -> tuple[dict[str, Any], ADQCTrainingResult, np.ndarray, GroundStateResult]:
    """Execute the dense oracle once and return its payload plus live objects."""

    hamiltonian = _build_hamiltonian(arguments)
    ground = exact_ground_state(hamiltonian)
    result = train_adqc(
        ground.state,
        num_qubits=arguments.n_qubits,
        config=_training_config_from_arguments(arguments),
        device=arguments.device,
    )
    return (
        _result_payload(
            arguments=arguments,
            hamiltonian=hamiltonian,
            ground=ground,
            result=result,
        ),
        result,
        hamiltonian,
        ground,
    )


def run_noise_smoke(arguments: argparse.Namespace) -> dict[str, Any]:
    """Evaluate one frozen dense ADQC run under FakeWashingtonV2 noise.

    The training is completed before this function calls any Qiskit code.  It
    is therefore a downstream, calibration-snapshot evaluation and cannot
    feed a transpilation or noise gradient back into the Zhou paper method.
    """

    payload, result, hamiltonian, ground = _run_dense_training(arguments)
    from .native_noise_protocol import evaluate_training_result_under_washington
    from .qiskit_adapter import staircase_gate_records

    physical_path = None
    if arguments.physical_path is not None:
        physical_path = tuple(arguments.physical_path)
    evaluation = evaluate_training_result_under_washington(
        result,
        hamiltonian=hamiltonian,
        ground_energy=ground.energy,
        physical_path=physical_path,
        optimization_level=arguments.optimization_level,
        seed_transpiler=arguments.seed_transpiler,
        seed_simulator=arguments.seed_simulator,
        max_density_matrix_qubits=arguments.max_density_matrix_qubits,
    )
    payload["fake_washington_v2_extension"] = {
        "paper_scope": (
            "Not a Zhou et al. (2021) experiment: exact native decomposition and "
            "static FakeWashingtonV2 calibration-noise evaluation after training."
        ),
        "evaluation": evaluation.as_dict(),
        "ordered_u4_gates": [
            {
                "layer": gate.layer,
                "position": gate.position,
                "left_site": gate.left_site,
                "matrix_sha256": _array_sha256(gate.matrix),
            }
            for gate in staircase_gate_records(result.model)
        ],
    }
    archive = _ordered_u4_archive_payload(arguments=arguments, result=result)
    if archive is not None:
        payload["ordered_u4_archive"] = archive
    return payload


def run_ordered_u4_noise(arguments: argparse.Namespace) -> dict[str, Any]:
    """Evaluate a frozen MPS-trained U(4) archive under FakeWashingtonV2.

    The function is a strict downstream hand-off: it loads the target MPS and
    the separately stored physical U(4) archive, validates both, materialises
    the target only below the explicit density-matrix limit, and invokes no
    ADQC optimizer or latent-polar projection.
    """

    loaded_target = load_open_boundary_mps_npz(arguments.target_mps)
    loaded_archive = load_ordered_u4_npz(arguments.ordered_u4_input)
    archive = loaded_archive.archive
    if archive.num_qubits != loaded_target.mps.num_qubits:
        raise ValueError(
            "ordered-U4 archive and target MPS disagree in num_qubits: "
            f"{archive.num_qubits} != {loaded_target.mps.num_qubits}."
        )
    target_state = loaded_target.mps.to_dense(
        max_num_qubits=arguments.max_density_matrix_qubits,
    )
    energy_spec = _mps_energy_spec_from_arguments(arguments, loaded_target)
    hamiltonian = None
    ground_energy = None
    target_energy_payload: dict[str, Any]
    if energy_spec is None:
        target_energy_payload = {
            "available": False,
            "reason": "target metadata has no recognized explicit Hamiltonian record.",
        }
    else:
        model, reference_energy, reference_role, declared_model = energy_spec
        hamiltonian = _dense_hamiltonian_from_mps_model(model)
        ground_energy = reference_energy
        target_energy_payload = {
            "available": True,
            "declared_model": declared_model,
            "model": model.to_record(),
            "target_mps_energy": mps_model_energy(loaded_target.mps, model),
            "reference_energy_role": reference_role,
            "reference_total_energy": reference_energy,
            "noise_energy_available": reference_energy is not None,
        }
        if reference_energy is not None:
            target_energy_payload["target_energy_error_per_site"] = mps_energy_error_per_site(
                loaded_target.mps,
                model,
                reference_energy,
            )
        else:
            # The density evaluator intentionally requires a matched reference
            # energy.  Do not pass one-sided Hamiltonian provenance to it.
            hamiltonian = None

    from .native_noise_protocol import evaluate_ordered_u4_archive_under_washington

    physical_path = None
    if arguments.physical_path is not None:
        physical_path = tuple(arguments.physical_path)
    evaluation = evaluate_ordered_u4_archive_under_washington(
        archive,
        target_state,
        hamiltonian=hamiltonian,
        ground_energy=ground_energy,
        physical_path=physical_path,
        optimization_level=arguments.optimization_level,
        seed_transpiler=arguments.seed_transpiler,
        seed_simulator=arguments.seed_simulator,
        max_density_matrix_qubits=arguments.max_density_matrix_qubits,
    )
    replayed = replay_ordered_u4_archive(archive)
    target_source = _mps_source_payload(loaded_target)
    return {
        "schema_version": "zhou2021-frozen-u4-noise-v1",
        "scope": (
            "downstream FakeWashingtonV2 density-matrix evaluation of a saved physical "
            "U(4) archive; no retraining, no polar reprojection, no noise-aware update"
        ),
        "target": {
            **target_source,
            "target_state_sha256": _array_sha256(target_state),
            "num_qubits": loaded_target.mps.num_qubits,
            "physical_energy": target_energy_payload,
        },
        "ordered_u4_archive": {
            "path": str(loaded_archive.path.resolve()),
            "file_sha256": _file_sha256(loaded_archive.path),
            "schema_version": archive.metadata["schema_version"],
            "semantic_sha256": archive.semantic_sha256,
            "num_layers": archive.num_layers,
            "gate_count": len(archive.gates),
            "global_phase_convention": archive.metadata["global_phase_convention"],
        },
        "ideal_archive_target": {
            "squared_state_fidelity": squared_state_fidelity(target_state, replayed),
            "archive_state_sha256": _array_sha256(replayed),
        },
        "fake_washington_v2_extension": evaluation.as_dict(),
        "provenance": {"doctor": doctor_snapshot()},
    }


def _mps_result_payload(
    *,
    result: ADQCMPSResult,
    source: dict[str, Any],
) -> dict[str, Any]:
    """Serialize an MPS-native ADQC result without dense materialization."""

    return {
        "schema_version": "zhou2021-adqc-mps-run-v2",
        "paper_method": {
            "identifier": "zhou-hong-ran-2021-adqc",
            "strict_paper_component": {
                "latent_gate": "unconstrained complex 4x4 -> SVD polar U Vh",
                "topology": "serial 1D nearest-neighbour staircase",
                "loss": "-log(abs(overlap))/N",
                "optimization": "layerwise warm-start Adam",
            },
            "mps_execution": (
                "MPS-native target overlap and serial U(4) execution; any bond cap or "
                "singular-value cutoff is a separately declared approximation coordinate."
            ),
        },
        "target": source,
        "algorithm": {
            "config": asdict(result.config),
            "metric_contract": {
                "negative_log_fidelity_per_site": (
                    "physical Zhou Eq. (1), without an optimizer floor"
                ),
                "training_floored_negative_log_fidelity_per_site": (
                    "finite differentiable loss used for updates and convergence_nlf"
                ),
                "positive_infinity_json_encoding": (
                    "null accompanied by a positive_infinity classification"
                ),
            },
            "final_num_layers": result.model.num_layers,
            "gates_per_layer": result.model.gates_per_layer,
            "ordered_u4_gate_count": result.model.num_layers * result.model.gates_per_layer,
            "paper_matrix_element_count": result.model.paper_parameter_count,
            "real_latent_parameter_count": result.model.real_latent_parameter_count,
            "mps_bond_upper_bound": result.model.maximum_mps_bond_bound,
            "converged": result.converged,
            "stopping_reason": result.stopping_reason,
        },
        "ideal_metrics": {
            "squared_state_fidelity": result.final_squared_fidelity,
            "negative_log_fidelity_per_site": _json_physical_nlf(
                result.final_negative_log_fidelity_per_site
            ),
            "negative_log_fidelity_per_site_classification": (
                _physical_nlf_classification(result.final_negative_log_fidelity_per_site)
            ),
            "physical_negative_log_fidelity_per_site": _json_physical_nlf(
                result.final_physical_negative_log_fidelity_per_site
            ),
            "physical_negative_log_fidelity_per_site_classification": (
                _physical_nlf_classification(
                    result.final_physical_negative_log_fidelity_per_site
                )
            ),
            "training_floored_negative_log_fidelity_per_site": (
                result.final_training_floored_negative_log_fidelity_per_site
            ),
            "training_squared_fidelity_floor": (
                result.config.training.loss_floor_squared_fidelity
            ),
        },
        "prepared_mps": {
            "num_qubits": result.prepared_mps.num_qubits,
            "bond_dimensions": list(result.prepared_mps.bond_dimensions),
            "max_bond_dimension": result.prepared_mps.max_bond_dimension,
            "semantic_sha256": _mps_semantic_sha256(result.prepared_mps),
        },
        "stages": [
            {
                "num_layers": stage.num_layers,
                "initial_negative_log_fidelity_per_site": (
                    _json_physical_nlf(stage.initial_negative_log_fidelity_per_site)
                ),
                "initial_negative_log_fidelity_per_site_classification": (
                    _physical_nlf_classification(
                        stage.initial_negative_log_fidelity_per_site
                    )
                ),
                "final_negative_log_fidelity_per_site": _json_physical_nlf(
                    stage.final_negative_log_fidelity_per_site
                ),
                "final_negative_log_fidelity_per_site_classification": (
                    _physical_nlf_classification(stage.final_negative_log_fidelity_per_site)
                ),
                "initial_training_floored_negative_log_fidelity_per_site": (
                    stage.initial_training_floored_negative_log_fidelity_per_site
                ),
                "final_training_floored_negative_log_fidelity_per_site": (
                    stage.final_training_floored_negative_log_fidelity_per_site
                ),
                "initial_squared_fidelity": stage.initial_squared_fidelity,
                "final_squared_fidelity": stage.final_squared_fidelity,
                "final_prepared_max_bond_dimension": stage.final_prepared_max_bond_dimension,
                "final_discarded_weight": stage.final_discarded_weight,
                "update_count": len(stage.updates),
                "updates": [_json_safe_metric_record(update) for update in stage.updates],
            }
            for stage in result.stages
        ],
        "provenance": {"doctor": doctor_snapshot()},
    }


def run_mps_train(arguments: argparse.Namespace) -> dict[str, Any]:
    """Load one strict MPS artifact and train ADQC without a dense vector."""

    loaded = load_open_boundary_mps_npz(arguments.target_mps)
    mps_config = _mps_training_config_from_arguments(arguments)
    _validate_mps_resource_acknowledgement(
        num_qubits=loaded.mps.num_qubits,
        config=mps_config,
        allow_unbounded_mps=getattr(arguments, "allow_unbounded_mps", False),
    )
    result = train_adqc_mps(
        loaded.mps,
        config=mps_config,
        device=arguments.device,
    )
    source = _mps_source_payload(loaded)
    source["normalized_target_semantic_sha256"] = _mps_semantic_sha256(result.target_mps)
    payload = _mps_result_payload(
        result=result,
        source=source,
    )
    payload["algorithm"]["mps_execution_resource_bound"] = _mps_resource_bound_payload(
        num_qubits=result.target_mps.num_qubits,
        config=mps_config,
    )
    payload["physical_metrics"] = _mps_physical_metrics(
        target_mps=result.target_mps,
        prepared_mps=result.prepared_mps,
        spec=_mps_energy_spec_from_arguments(arguments, loaded),
    )
    archive = _ordered_u4_archive_payload(arguments=arguments, result=result)
    if archive is not None:
        payload["ordered_u4_archive"] = archive
    if arguments.prepared_mps_output is not None:
        prepared_path = save_open_boundary_mps_npz(
            _output_path_with_parent(arguments.prepared_mps_output),
            result.prepared_mps,
            metadata={
                "schema_version": "zhou2021-prepared-mps-metadata-v1",
                "source_target_file_sha256": payload["target"]["file_sha256"],
                "prepared_semantic_sha256": payload["prepared_mps"]["semantic_sha256"],
                "algorithm_config": asdict(result.config),
            },
        ).resolve()
        payload["prepared_mps"]["archive"] = {
            "path": str(prepared_path),
            "file_sha256": _file_sha256(prepared_path),
        }
    return payload


def _mps_source_payload(loaded: LoadedOpenBoundaryMPS) -> dict[str, Any]:
    """Return provenance for a loaded external MPS without changing its state."""

    source_path = loaded.path.resolve()
    return {
        "source_type": "external_open_boundary_mps_npz",
        "path": str(source_path),
        "file_sha256": _file_sha256(source_path),
        "metadata": loaded.metadata,
        "source_num_qubits": loaded.mps.num_qubits,
        "source_bond_dimensions": list(loaded.mps.bond_dimensions),
        "source_norm": loaded.mps.norm(),
        "source_semantic_sha256": _mps_semantic_sha256(loaded.mps),
    }


def _validate_mps_resource_acknowledgement(
    *,
    num_qubits: int,
    config: ADQCMPSConfig,
    allow_unbounded_mps: bool,
) -> None:
    """Reject accidental high-N exact staircase growth at the public CLI."""

    if num_qubits > 16 and config.max_bond_dimension is None and not allow_unbounded_mps:
        raise ValueError(
            "Refusing an unbounded MPS ADQC run above N=16. Pass an explicit "
            "--max-bond-dimension (recorded approximation) or acknowledge the "
            "uncompressed 4**L bound with --allow-unbounded-mps."
        )


def _mps_resource_bound_payload(
    *,
    num_qubits: int,
    config: ADQCMPSConfig,
) -> dict[str, Any]:
    """Report the declared MPS approximation/resource coordinate explicitly."""

    effective_bound = (
        config.max_bond_dimension
        if config.max_bond_dimension is not None
        else 4**config.training.max_layers
    )
    elements = estimated_mps_complex_elements(num_qubits, effective_bound)
    return {
        "declared_max_bond_dimension": config.max_bond_dimension,
        "untruncated_staircase_bond_upper_bound": 4**config.training.max_layers,
        "effective_storage_bound_bond_dimension": effective_bound,
        "estimated_raw_complex128_core_elements": elements,
        "estimated_raw_complex128_core_bytes": 16 * elements,
        "approximation_status": (
            "explicit_bond_cap_or_cutoff"
            if config.max_bond_dimension is not None or config.singular_value_cutoff > 0.0
            else "untruncated"
        ),
    }


def _mps_sweep_payload(
    *,
    sweep: MPSDepthSweepResult,
    source: dict[str, Any],
) -> dict[str, Any]:
    """Serialize a seed-resolved warm-start sweep without omitting failures."""

    return {
        "schema_version": "zhou2021-adqc-mps-depth-sweep-v2",
        "status": "completed",
        "paper_method": {
            "identifier": "zhou-hong-ran-2021-adqc",
            "strict_paper_component": {
                "latent_gate": "unconstrained complex 4x4 -> SVD polar U Vh",
                "topology": "serial 1D nearest-neighbour staircase",
                "loss": "-log(abs(overlap))/N",
                "optimization": "layerwise warm-start Adam",
            },
            "sweep_scope": (
                "Configured seed ensemble and optimizer coordinates; the article does not "
                "provide an author-supplied numerical seed sweep."
            ),
        },
        "target": source,
        "algorithm": {
            "shared_config": asdict(sweep.config),
            "metric_contract": {
                "negative_log_fidelity_per_site": (
                    "physical Zhou Eq. (1), without an optimizer floor"
                ),
                "training_floored_negative_log_fidelity_per_site": (
                    "finite differentiable loss using the declared squared-fidelity floor"
                ),
                "positive_infinity_json_encoding": (
                    "null accompanied by a positive_infinity classification"
                ),
            },
            "seed_policy": {
                "seeds": list(sweep.seeds),
                "per_seed_execution": "one full warm-start trajectory through max_layers",
                "shared_config_seed_field": (
                    "placeholder only; each listed seed replaces training.seed before run"
                ),
            },
            "num_qubits": sweep.num_qubits,
            "target_max_bond_dimension": sweep.target_max_bond_dimension,
            "depths": list(sweep.depths),
            "gates_per_layer": sweep.num_qubits - 1,
            "ordered_u4_gate_count_by_depth": {
                str(depth): (sweep.num_qubits - 1) * depth for depth in sweep.depths
            },
        },
        "runs": [
            {
                "seed": run.seed,
                "converged": run.converged,
                "stopping_reason": run.stopping_reason,
                "depth_records": [
                    _json_safe_metric_record(record) for record in run.depth_records
                ],
            }
            for run in sweep.runs
        ],
        "summaries": [_json_safe_metric_record(summary) for summary in sweep.summaries],
        "provenance": {"doctor": doctor_snapshot()},
    }


def run_mps_sweep(arguments: argparse.Namespace) -> dict[str, Any]:
    """Run one complete warm-start MPS trajectory for every requested seed."""

    loaded = load_open_boundary_mps_npz(arguments.target_mps)
    mps_config = _mps_training_config_from_arguments(arguments)
    _validate_mps_resource_acknowledgement(
        num_qubits=loaded.mps.num_qubits,
        config=mps_config,
        allow_unbounded_mps=getattr(arguments, "allow_unbounded_mps", False),
    )
    sweep = run_mps_depth_sweep(
        loaded.mps,
        seeds=arguments.seeds,
        config=mps_config,
        device=arguments.device,
    )
    source = _mps_source_payload(loaded)
    source["normalized_target_semantic_sha256"] = _mps_semantic_sha256(
        loaded.mps.normalized()
    )
    payload = _mps_sweep_payload(sweep=sweep, source=source)
    payload["algorithm"]["mps_execution_resource_bound"] = _mps_resource_bound_payload(
        num_qubits=sweep.num_qubits,
        config=mps_config,
    )
    if arguments.plot_output is not None:
        from .plotting import plot_mps_depth_sweep

        plot_path = plot_mps_depth_sweep(
            sweep,
            arguments.plot_output,
            include_global_fidelity=arguments.include_global_fidelity,
            dpi=arguments.plot_dpi,
        )
        payload["plot"] = {
            "path": str(plot_path),
            "file_sha256": _file_sha256(plot_path),
            "includes_global_squared_fidelity": arguments.include_global_fidelity,
        }
    return payload


def run_dense_target(arguments: argparse.Namespace) -> dict[str, Any]:
    """Generate a small-N ED/TT-SVD MPS target for cross-validation only."""

    target = dense_model_ground_state_mps(
        "tfim_tndiff" if arguments.model == "tfim" else arguments.model,
        arguments.n_qubits,
        coupling=arguments.coupling,
        delta=arguments.delta,
        coupling_zz=arguments.coupling_zz,
        field_x=arguments.field_x,
        max_dense_qubits=arguments.max_dense_qubits,
        max_bond_dimension=arguments.max_bond_dimension,
        singular_value_cutoff=arguments.singular_value_cutoff,
    )
    destination = save_open_boundary_mps_npz(
        _output_path_with_parent(arguments.output),
        target.mps,
        metadata={
            "schema_version": "zhou2021-dense-ed-mps-target-v1",
            "source_type": "dense_exact_diagonalization_then_tt_svd",
            "model": target.model,
            "num_qubits": target.num_qubits,
            "ground_energy": target.ground_state.energy,
            "ground_residual": target.ground_state.residual,
            "ground_degeneracy": target.ground_state.degeneracy,
            "tt_svd_discarded_weight": target.decomposition.discarded_weight,
            "tt_svd_retained_ranks": list(target.decomposition.retained_ranks),
            "hamiltonian": {
                "model": target.model,
                "coupling": arguments.coupling,
                "delta": arguments.delta if target.model == "xxz" else None,
                "coupling_zz": arguments.coupling_zz
                if target.model == "tfim_tndiff"
                else None,
                "field_x": arguments.field_x if target.model == "tfim_tndiff" else None,
            },
            "energy_reference": {
                "value": target.ground_state.energy,
                "role": "small-N exact-diagonalization ground-state total energy",
            },
        },
    ).resolve()
    return {
        "schema_version": "zhou2021-dense-target-export-v1",
        "status": "completed",
        "scope": "small-N ED/TT-SVD cross-validation target; not the paper-scale DMRG path",
        "output": str(destination),
        "output_sha256": _file_sha256(destination),
        "mps_semantic_sha256": _mps_semantic_sha256(target.mps),
        "model": target.model,
        "n_qubits": target.num_qubits,
        "ground_energy": target.ground_state.energy,
        "ground_residual": target.ground_state.residual,
        "discarded_weight": target.decomposition.discarded_weight,
        "bond_dimensions": list(target.mps.bond_dimensions),
    }


def run_dmrg_target(arguments: argparse.Namespace) -> dict[str, Any]:
    """Generate a paper-scale-capable target with optional TeNPy DMRG."""

    target = dmrg_ground_state_mps(
        arguments.model,
        arguments.n_qubits,
        config=DMRGConfig(
            chi_max=arguments.chi_max,
            max_sweeps=arguments.max_sweeps,
            min_sweeps=arguments.min_sweeps,
            energy_tolerance=arguments.energy_tolerance,
            svd_min=arguments.svd_min,
            mixer=arguments.mixer,
            initial_product_state=arguments.initial_product_state,
        ),
        coupling=arguments.coupling,
        delta=arguments.delta,
        coupling_zz=arguments.coupling_zz,
        field_x=arguments.field_x,
    )
    destination = save_open_boundary_mps_npz(
        _output_path_with_parent(arguments.output),
        target.mps,
        metadata=target.provenance,
    ).resolve()
    return {
        "schema_version": "zhou2021-tenpy-target-export-v1",
        "status": "completed",
        "output": str(destination),
        "output_sha256": _file_sha256(destination),
        "mps_semantic_sha256": _mps_semantic_sha256(target.mps),
        "model": target.model,
        "requested_model": target.requested_model,
        "n_qubits": target.num_qubits,
        "energy": target.energy,
        "engine_energy": target.engine_energy,
        "engine_expectation_difference": target.engine_expectation_difference,
        "target_quality": "single_run_not_independently_budget_certified",
        "bond_dimensions": list(target.mps.bond_dimensions),
        "provenance": target.provenance,
    }


def _write_json(payload: dict[str, Any], output: str | None) -> None:
    encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)
    if output is None:
        print(encoded)
        return
    path = Path(output).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(encoded + "\n", encoding="utf-8")
    print(json.dumps({"output": str(path), "status": "completed"}, sort_keys=True))


def _add_common_run_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", choices=("heisenberg", "xy", "xxz", "tfim"), required=True)
    parser.add_argument("--n-qubits", type=int, required=True)
    parser.add_argument("--coupling", type=float, default=1.0)
    parser.add_argument("--delta", type=float, default=1.0)
    parser.add_argument("--coupling-zz", type=float, default=1.0)
    parser.add_argument("--field-x", type=float, default=1.0)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--epochs-per-stage", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=0.02)
    parser.add_argument("--learning-rate-decay", type=float, default=1.0)
    _add_adam_arguments(parser)
    parser.add_argument(
        "--first-layer-initialization",
        choices=("random", "identity_perturbed"),
        default="random",
    )
    parser.add_argument("--new-layer-perturbation", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--loss-floor-squared-fidelity", type=float, default=1e-30)
    parser.add_argument("--convergence-nlf", type=float)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--include-statevectors", action="store_true")
    parser.add_argument(
        "--ordered-u4-output",
        help=(
            "optional strict .npz export of projected ordered U(4) gates for "
            "Qiskit-independent replay/audit"
        ),
    )
    parser.add_argument("--output")


def _add_noise_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--physical-path",
        nargs="+",
        type=int,
        help="optional ordered FakeWashington physical path; otherwise select deterministically",
    )
    parser.add_argument("--optimization-level", type=int, choices=(0, 1, 2, 3), default=0)
    parser.add_argument("--seed-transpiler", type=int, default=2021)
    parser.add_argument("--seed-simulator", type=int, default=2021)
    parser.add_argument("--max-density-matrix-qubits", type=int, default=12)


def _add_adam_arguments(parser: argparse.ArgumentParser) -> None:
    """Add all Adam coordinates so result manifests contain no hidden defaults."""

    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.999)
    parser.add_argument("--adam-eps", type=float, default=1e-8)
    parser.add_argument("--adam-weight-decay", type=float, default=0.0)
    parser.add_argument("--adam-amsgrad", action="store_true")


def _add_mps_training_arguments(
    parser: argparse.ArgumentParser,
    *,
    include_seed: bool = True,
    include_prepared_mps_output: bool = True,
    include_convergence_nlf: bool = True,
    include_ordered_u4_output: bool = True,
) -> None:
    """Add optimizer plus explicit MPS-truncation coordinates to one parser."""

    parser.add_argument("--target-mps", required=True, help="strict zhou2021 OBC-MPS .npz input")
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--epochs-per-stage", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=0.02)
    parser.add_argument("--learning-rate-decay", type=float, default=1.0)
    _add_adam_arguments(parser)
    parser.add_argument(
        "--first-layer-initialization",
        choices=("random", "identity_perturbed"),
        default="random",
    )
    parser.add_argument("--new-layer-perturbation", type=float, default=1e-3)
    if include_seed:
        parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--loss-floor-squared-fidelity", type=float, default=1e-30)
    if include_convergence_nlf:
        parser.add_argument("--convergence-nlf", type=float)
    parser.add_argument("--max-bond-dimension", type=int)
    parser.add_argument("--singular-value-cutoff", type=float, default=0.0)
    parser.add_argument("--unitary-atol", type=float, default=1e-11)
    parser.add_argument(
        "--allow-unbounded-mps",
        action="store_true",
        help="required acknowledgement for an exact no-cap MPS run above N=16",
    )
    parser.add_argument("--device", default="cpu")
    if include_prepared_mps_output:
        parser.add_argument("--prepared-mps-output")
    if include_ordered_u4_output:
        parser.add_argument(
            "--ordered-u4-output",
            help=(
                "optional strict .npz export of projected ordered U(4) gates for "
                "Qiskit-independent replay/audit"
            ),
        )
    parser.add_argument("--output", help="JSON result manifest path; stdout if omitted")


def _add_mps_energy_evaluation_arguments(parser: argparse.ArgumentParser) -> None:
    """Add optional physical-energy coordinates for a provenance-poor MPS target."""

    parser.add_argument(
        "--energy-model",
        choices=("heisenberg", "xy", "xxz", "tfim"),
        help=(
            "override/declare the MPS energy model; otherwise use recognized target "
            "metadata when available"
        ),
    )
    parser.add_argument("--energy-coupling", type=float, default=1.0)
    parser.add_argument("--energy-delta", type=float, default=1.0)
    parser.add_argument("--energy-coupling-zz", type=float, default=1.0)
    parser.add_argument("--energy-field-x", type=float, default=1.0)
    parser.add_argument(
        "--energy-reference",
        type=float,
        help="optional total-energy reference used for signed energy error per site",
    )


def _add_model_target_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", choices=("heisenberg", "xy", "xxz", "tfim"), required=True)
    parser.add_argument("--n-qubits", type=int, required=True)
    parser.add_argument("--coupling", type=float, default=1.0)
    parser.add_argument("--delta", type=float, default=1.0)
    parser.add_argument("--coupling-zz", type=float, default=1.0)
    parser.add_argument("--field-x", type=float, default=1.0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="zhou2021")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("doctor", help="print an environment/convention snapshot")
    smoke = subparsers.add_parser(
        "smoke",
        help="prepare one exact small-system ground state and train the Zhou ADQC compiler",
    )
    _add_common_run_arguments(smoke)
    noise = subparsers.add_parser(
        "noise-smoke",
        help=(
            "run one small dense ADQC encoding, then evaluate the frozen circuit "
            "under FakeWashingtonV2"
        ),
    )
    _add_common_run_arguments(noise)
    _add_noise_arguments(noise)
    archive_noise = subparsers.add_parser(
        "noise-archive",
        help=(
            "evaluate a saved ordered-U(4) MPS-ADQC circuit under FakeWashingtonV2 "
            "without retraining"
        ),
    )
    archive_noise.add_argument(
        "--target-mps",
        required=True,
        help="strict zhou2021 OBC-MPS .npz target used by the frozen circuit",
    )
    archive_noise.add_argument(
        "--ordered-u4-input",
        required=True,
        help="strict ordered-U(4) .npz archive exported by mps-train or smoke",
    )
    _add_mps_energy_evaluation_arguments(archive_noise)
    _add_noise_arguments(archive_noise)
    archive_noise.add_argument("--output", help="JSON result manifest path; stdout if omitted")
    dense_target = subparsers.add_parser(
        "dense-target",
        help="export a small-N exact-diagonalization/TT-SVD MPS validation target",
    )
    _add_model_target_arguments(dense_target)
    dense_target.add_argument("--max-dense-qubits", type=int, default=12)
    dense_target.add_argument("--max-bond-dimension", type=int)
    dense_target.add_argument("--singular-value-cutoff", type=float, default=0.0)
    dense_target.add_argument("--output", required=True, help="target .npz path")
    dense_target.add_argument("--summary-output")
    dmrg_target = subparsers.add_parser(
        "dmrg-target",
        help="generate an OBC ground-state MPS target through optional TeNPy DMRG",
    )
    _add_model_target_arguments(dmrg_target)
    dmrg_target.add_argument("--chi-max", type=int, default=64)
    dmrg_target.add_argument("--max-sweeps", type=int, default=30)
    dmrg_target.add_argument("--min-sweeps", type=int, default=4)
    dmrg_target.add_argument("--energy-tolerance", type=float, default=1e-10)
    dmrg_target.add_argument("--svd-min", type=float, default=1e-14)
    dmrg_target.add_argument(
        "--initial-product-state",
        choices=("neel", "all_up", "all_down"),
        default="neel",
    )
    dmrg_target.add_argument("--no-mixer", dest="mixer", action="store_false")
    dmrg_target.set_defaults(mixer=True)
    dmrg_target.add_argument("--output", required=True, help="target .npz path")
    dmrg_target.add_argument("--summary-output")
    mps_train = subparsers.add_parser(
        "mps-train",
        help="train the Zhou ADQC staircase directly against a strict OBC-MPS target",
    )
    _add_mps_training_arguments(mps_train)
    _add_mps_energy_evaluation_arguments(mps_train)
    mps_sweep = subparsers.add_parser(
        "mps-sweep",
        help=(
            "run one full layer-growth MPS ADQC trajectory per seed and summarize "
            "all requested depths"
        ),
    )
    _add_mps_training_arguments(
        mps_sweep,
        include_seed=False,
        include_prepared_mps_output=False,
        include_convergence_nlf=False,
        include_ordered_u4_output=False,
    )
    mps_sweep.set_defaults(seed=0, convergence_nlf=None)
    mps_sweep.add_argument(
        "--seeds",
        nargs="+",
        required=True,
        type=int,
        help="unique nonnegative random seeds; each is one full warm-start trajectory",
    )
    mps_sweep.add_argument(
        "--plot-output",
        help="optional .png/.pdf sweep figure; requires the optional matplotlib dependency",
    )
    mps_sweep.add_argument(
        "--include-global-fidelity",
        action="store_true",
        help="include a secondary global squared-state-fidelity axis in --plot-output",
    )
    mps_sweep.add_argument("--plot-dpi", type=int, default=180)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Execute the public CLI and return a conventional process status."""

    parser = build_parser()
    arguments = parser.parse_args(argv)
    if arguments.command == "doctor":
        print(json.dumps(doctor_snapshot(), indent=2, sort_keys=True, allow_nan=False))
        return 0
    if arguments.command == "smoke":
        _write_json(run_smoke(arguments), arguments.output)
        return 0
    if arguments.command == "noise-smoke":
        try:
            payload = run_noise_smoke(arguments)
        except ImportError as exc:
            parser.exit(
                2,
                "zhou2021: error: optional FakeWashingtonV2 evaluation is unavailable: "
                f"{exc}\n",
            )
        _write_json(payload, arguments.output)
        return 0
    if arguments.command == "noise-archive":
        try:
            payload = run_ordered_u4_noise(arguments)
        except ImportError as exc:
            parser.exit(
                2,
                "zhou2021: error: optional FakeWashingtonV2 evaluation is unavailable: "
                f"{exc}\n",
            )
        _write_json(payload, arguments.output)
        return 0
    if arguments.command == "dense-target":
        _write_json(run_dense_target(arguments), arguments.summary_output)
        return 0
    if arguments.command == "dmrg-target":
        try:
            payload = run_dmrg_target(arguments)
        except RuntimeError as exc:
            parser.exit(
                2,
                "zhou2021: error: optional TeNPy DMRG target generation failed: "
                f"{exc}\n",
            )
        _write_json(payload, arguments.summary_output)
        return 0
    if arguments.command == "mps-train":
        _write_json(run_mps_train(arguments), arguments.output)
        return 0
    if arguments.command == "mps-sweep":
        _write_json(run_mps_sweep(arguments), arguments.output)
        return 0
    parser.error(f"Unsupported command {arguments.command!r}.")  # pragma: no cover
    return 2  # pragma: no cover


__all__ = [
    "build_parser",
    "doctor_snapshot",
    "main",
    "run_dense_target",
    "run_dmrg_target",
    "run_mps_sweep",
    "run_mps_train",
    "run_noise_smoke",
    "run_smoke",
]
