"""End-to-end provenance tests for MPS-native physical-energy reporting."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from zhou2021 import cli
from zhou2021.cli import main


def test_doctor_records_optional_runtime_import_health(
    monkeypatch,
) -> None:
    snapshot = cli.doctor_snapshot()

    assert snapshot["schema_version"] == "zhou2021-doctor-v1"
    assert set(snapshot["optional_runtime_imports"]) == {
        "physics_tenpy",
        "qiskit",
        "qiskit_aer",
        "qiskit_ibm_runtime",
    }
    assert json.loads(json.dumps(snapshot))["optional_runtime_imports"]

    def broken_optional_import(_: str) -> None:
        raise ImportError("deliberately broken optional package")

    monkeypatch.setattr(cli.importlib, "import_module", broken_optional_import)
    assert cli._optional_import_status("qiskit") == {
        "usable": False,
        "error_type": "ImportError",
        "error": "deliberately broken optional package",
    }


def test_mps_energy_metadata_refuses_implicit_hamiltonian_parameters() -> None:
    with pytest.raises(ValueError, match="explicitly declare 'coupling'"):
        cli._metadata_mps_energy_spec(
            {"hamiltonian": {"model": "heisenberg"}},
            num_qubits=4,
        )
    assert cli._metadata_mps_energy_spec({}, num_qubits=4) is None


def test_ed_mps_target_train_automatically_reports_energy_per_site(tmp_path: Path) -> None:
    """A standardized ED target supplies enough provenance for automatic energy checks."""

    output_directory = tmp_path / "nested" / "mps_energy"
    target = output_directory / "heisenberg_n3.npz"
    target_summary = output_directory / "heisenberg_n3_target.json"
    result = output_directory / "heisenberg_n3_train.json"

    assert (
        main(
            [
                "dense-target",
                "--model",
                "heisenberg",
                "--n-qubits",
                "3",
                "--output",
                str(target),
                "--summary-output",
                str(target_summary),
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "mps-train",
                "--target-mps",
                str(target),
                "--layers",
                "1",
                "--epochs-per-stage",
                "1",
                "--learning-rate",
                "0.02",
                "--max-bond-dimension",
                "2",
                "--output",
                str(result),
            ]
        )
        == 0
    )

    target_manifest = json.loads(target_summary.read_text(encoding="utf-8"))
    manifest = json.loads(result.read_text(encoding="utf-8"))
    metrics = manifest["physical_metrics"]
    ideal = manifest["ideal_metrics"]

    assert manifest["schema_version"] == "zhou2021-adqc-mps-run-v2"
    assert metrics["available"] is True
    assert metrics["declared_model"] == "heisenberg"
    assert metrics["model"]["spin_convention"] == "S=Pauli/2"
    assert metrics["reference_energy_role"] == (
        "small-N exact-diagonalization ground-state total energy"
    )
    assert metrics["reference_total_energy"] == target_manifest["ground_energy"]
    assert abs(metrics["target_energy_error_per_site"]) < 2e-12
    assert math.isfinite(metrics["prepared_mps_energy"])
    assert math.isfinite(metrics["prepared_energy_error_per_site"])
    assert manifest["algorithm"]["config"]["training"]["adam_betas"] == [0.9, 0.999]
    assert manifest["algorithm"]["config"]["training"]["adam_eps"] == 1e-8
    assert ideal["negative_log_fidelity_per_site_classification"] == "finite"
    assert ideal["negative_log_fidelity_per_site"] == pytest.approx(
        -0.5 * math.log(ideal["squared_state_fidelity"]) / 3.0,
        abs=2e-12,
    )
    assert ideal["physical_negative_log_fidelity_per_site"] == pytest.approx(
        ideal["negative_log_fidelity_per_site"]
    )
    assert math.isfinite(ideal["training_floored_negative_log_fidelity_per_site"])
