"""Regression tests for the strict Qiskit-independent ordered-U(4) archive."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from numpy.testing import assert_allclose

from zhou2021 import native_noise_protocol
from zhou2021.circuit import LatentStaircase
from zhou2021.cli import main
from zhou2021.gate_archive import (
    ORDERED_U4_NPZ_SCHEMA_VERSION,
    OrderedU4Archive,
    OrderedU4Gate,
    build_ordered_u4_archive,
    load_ordered_u4_npz,
    ordered_u4_archive_from_staircase,
    replay_ordered_u4_archive,
    save_ordered_u4_npz,
    sha256_file,
    validate_ordered_u4_archive,
)
from zhou2021.ground_states import exact_ground_state
from zhou2021.models import heisenberg_hamiltonian
from zhou2021.mps import mps_from_dense_statevector, save_open_boundary_mps_npz


def _staircase() -> LatentStaircase:
    model = LatentStaircase(4)
    model.add_layer(initialization="random", seed=71)
    model.add_layer(initialization="identity_perturbed", seed=72, perturbation=1e-3)
    return model


def test_archive_replays_the_projected_staircase_and_round_trips(tmp_path: Path) -> None:
    model = _staircase()
    archive = ordered_u4_archive_from_staircase(
        model,
        user_metadata={"producer": "pytest", "seed": 71},
    )

    assert archive.metadata["schema_version"] == ORDERED_U4_NPZ_SCHEMA_VERSION
    assert archive.num_layers == 2
    assert len(archive.gates) == 2 * (archive.num_qubits - 1)
    assert [(gate.layer, gate.position, gate.left_site) for gate in archive.gates] == [
        (0, 0, 0),
        (0, 1, 1),
        (0, 2, 2),
        (1, 0, 0),
        (1, 1, 1),
        (1, 2, 2),
    ]
    assert_allclose(
        replay_ordered_u4_archive(archive),
        model().detach().numpy(),
        atol=5e-12,
    )

    output = tmp_path / "nested" / "gates.npz"
    output.parent.mkdir()
    saved = save_ordered_u4_npz(output, archive)
    loaded = load_ordered_u4_npz(saved)
    assert loaded.archive.semantic_sha256 == archive.semantic_sha256
    assert sha256_file(saved) == sha256_file(output)
    assert_allclose(
        replay_ordered_u4_archive(loaded.archive),
        model().detach().numpy(),
        atol=5e-12,
    )


def test_archive_rejects_order_and_semantic_payload_tampering(tmp_path: Path) -> None:
    model = _staircase()
    archive = ordered_u4_archive_from_staircase(model)
    bad_order = OrderedU4Archive(
        num_qubits=archive.num_qubits,
        gates=(archive.gates[1], archive.gates[0], *archive.gates[2:]),
        metadata=archive.metadata,
    )
    with pytest.raises(ValueError, match="layer-major serial staircase order"):
        validate_ordered_u4_archive(bad_order)

    changed = list(archive.gates)
    changed[0] = OrderedU4Gate(
        layer=0,
        position=0,
        left_site=0,
        matrix=np.eye(4, dtype=np.complex128),
    )
    tampered = OrderedU4Archive(
        num_qubits=archive.num_qubits,
        gates=tuple(changed),
        metadata=archive.metadata,
    )
    with pytest.raises(ValueError, match="semantic SHA-256 mismatch"):
        validate_ordered_u4_archive(tampered)

    original = tmp_path / "original.npz"
    save_ordered_u4_npz(original, archive)
    with np.load(original, allow_pickle=False) as payload:
        matrices = np.array(payload["matrices"], copy=True)
        layer = np.array(payload["layer"], copy=True)
        position = np.array(payload["position"], copy=True)
        left_site = np.array(payload["left_site"], copy=True)
        metadata = str(payload["metadata_json"].item())
    matrices[0] = np.eye(4, dtype=np.complex128)
    modified = tmp_path / "modified.npz"
    np.savez_compressed(
        modified,
        metadata_json=np.array(metadata),
        matrices=matrices,
        layer=layer,
        position=position,
        left_site=left_site,
    )
    with pytest.raises(ValueError, match="semantic SHA-256 mismatch"):
        load_ordered_u4_npz(modified)


def test_archive_rejects_noncanonical_metadata_and_nonunitary_gates() -> None:
    model = _staircase()
    archive = ordered_u4_archive_from_staircase(model)
    invalid_metadata = dict(archive.metadata)
    invalid_metadata["user_metadata"] = {"bad": float("nan")}
    with pytest.raises(ValueError, match="non-finite float"):
        validate_ordered_u4_archive(
            OrderedU4Archive(
                num_qubits=archive.num_qubits,
                gates=archive.gates,
                metadata=invalid_metadata,
            )
        )

    with pytest.raises(ValueError, match="not unitary"):
        build_ordered_u4_archive(
            2,
            [
                OrderedU4Gate(
                    layer=0,
                    position=0,
                    left_site=0,
                    matrix=2.0 * np.eye(4, dtype=np.complex128),
                )
            ],
        )


def test_archive_uses_strict_canonical_json_metadata(tmp_path: Path) -> None:
    archive = ordered_u4_archive_from_staircase(_staircase(), user_metadata={"z": 1, "a": 2})
    output = tmp_path / "gates.npz"
    save_ordered_u4_npz(output, archive)
    with np.load(output, allow_pickle=False) as payload:
        metadata = str(payload["metadata_json"].item())
    assert metadata == json.dumps(
        archive.metadata,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def test_smoke_cli_writes_replayable_ordered_u4_archive(tmp_path: Path) -> None:
    output = tmp_path / "manifest.json"
    gates = tmp_path / "nested" / "ordered_gates.npz"
    assert (
        main(
            [
                "smoke",
                "--model",
                "xy",
                "--n-qubits",
                "2",
                "--layers",
                "1",
                "--epochs-per-stage",
                "1",
                "--seed",
                "73",
                "--ordered-u4-output",
                str(gates),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    manifest = json.loads(output.read_text(encoding="utf-8"))
    archive_info = manifest["ordered_u4_archive"]
    loaded = load_ordered_u4_npz(gates)
    assert archive_info["path"] == str(gates.resolve())
    assert archive_info["file_sha256"] == sha256_file(gates)
    assert archive_info["semantic_sha256"] == loaded.archive.semantic_sha256
    assert archive_info["gate_count"] == 1


def test_noise_archive_cli_uses_saved_u4_and_target_without_retraining(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target_path = tmp_path / "target.npz"
    gates_path = tmp_path / "gates.npz"
    output = tmp_path / "noise.json"
    hamiltonian = heisenberg_hamiltonian(2)
    ground = exact_ground_state(hamiltonian)
    target = ground.state
    save_open_boundary_mps_npz(
        target_path,
        mps_from_dense_statevector(target),
        metadata={
            "hamiltonian": {"model": "heisenberg", "coupling": 1.0},
            "energy_reference": {"value": ground.energy, "role": "test ED ground energy"},
        },
    )
    model = LatentStaircase(2)
    model.add_layer(initialization="random", seed=74)
    archive = ordered_u4_archive_from_staircase(model)
    save_ordered_u4_npz(gates_path, archive)
    received: dict[str, object] = {}

    def fake_evaluate(loaded_archive, target_state, **kwargs):  # type: ignore[no-untyped-def]
        received["semantic_sha256"] = loaded_archive.semantic_sha256
        received["target_state"] = np.array(target_state, copy=True)
        received["kwargs"] = kwargs
        return SimpleNamespace(as_dict=lambda: {"test_double": "frozen_archive"})

    monkeypatch.setattr(
        native_noise_protocol,
        "evaluate_ordered_u4_archive_under_washington",
        fake_evaluate,
    )
    assert (
        main(
            [
                "noise-archive",
                "--target-mps",
                str(target_path),
                "--ordered-u4-input",
                str(gates_path),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    manifest = json.loads(output.read_text(encoding="utf-8"))
    assert received["semantic_sha256"] == archive.semantic_sha256
    assert_allclose(received["target_state"], target, atol=3e-12)
    kwargs = received["kwargs"]
    assert isinstance(kwargs, dict)
    assert_allclose(kwargs["hamiltonian"], hamiltonian, atol=3e-12)
    assert kwargs["ground_energy"] == pytest.approx(ground.energy, abs=2e-12)
    assert manifest["ordered_u4_archive"]["semantic_sha256"] == archive.semantic_sha256
    assert manifest["ideal_archive_target"]["squared_state_fidelity"] < 1.0
    assert manifest["fake_washington_v2_extension"] == {"test_double": "frozen_archive"}
