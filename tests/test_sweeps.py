"""Small-N tests for deterministic Zhou MPS depth sweep and plotting helpers."""

from __future__ import annotations

import importlib
import json
import math
from dataclasses import replace
from pathlib import Path

import pytest

from zhou2021 import cli
from zhou2021.adqc import ADQCTrainingConfig
from zhou2021.adqc_mps import ADQCMPSConfig
from zhou2021.cli import main
from zhou2021.mps import random_open_boundary_mps, save_open_boundary_mps_npz
from zhou2021.plotting import plot_mps_depth_sweep
from zhou2021.sweeps import MPSDepthSummary, MPSDepthSweepResult, run_mps_depth_sweep


@pytest.fixture()
def tiny_config() -> ADQCMPSConfig:
    return ADQCMPSConfig(
        training=ADQCTrainingConfig(
            max_layers=2,
            epochs_per_stage=1,
            learning_rate=0.02,
            seed=999,
        ),
        max_bond_dimension=2,
    )


def test_mps_depth_sweep_is_seed_resolved_aggregated_and_deterministic(
    tiny_config: ADQCMPSConfig,
) -> None:
    target = random_open_boundary_mps(3, 2, seed=801)

    first = run_mps_depth_sweep(target, seeds=(31, 32), config=tiny_config)
    second = run_mps_depth_sweep(target, seeds=(31, 32), config=tiny_config)

    assert first == second
    assert first.num_qubits == 3
    assert first.target_max_bond_dimension == 2
    assert first.seeds == (31, 32)
    assert first.depths == (1, 2)
    assert len(first.runs) == 2
    assert all(
        tuple(record.depth for record in run.depth_records) == (1, 2)
        for run in first.runs
    )
    assert all(
        record.final_prepared_max_bond_dimension <= 2
        and record.final_discarded_weight >= 0.0
        for run in first.runs
        for record in run.depth_records
    )

    for depth in first.depths:
        summary = first.summary_for_depth(depth)
        values = [run.depth_records[depth - 1].final_squared_fidelity for run in first.runs]
        assert summary.num_seeds == 2
        assert summary.mean_squared_fidelity == pytest.approx(sum(values) / len(values))
        assert summary.std_squared_fidelity >= 0.0


def test_mps_depth_sweep_validates_coordinates(tiny_config: ADQCMPSConfig) -> None:
    target = random_open_boundary_mps(3, 2, seed=802)

    with pytest.raises(ValueError, match="at least one"):
        run_mps_depth_sweep(target, seeds=(), config=tiny_config)
    with pytest.raises(TypeError, match="not text"):
        run_mps_depth_sweep(target, seeds="3", config=tiny_config)
    with pytest.raises(TypeError, match="every seed"):
        run_mps_depth_sweep(target, seeds=(True,), config=tiny_config)
    with pytest.raises(ValueError, match="unique"):
        run_mps_depth_sweep(target, seeds=(3, 3), config=tiny_config)
    with pytest.raises(TypeError, match="ADQCMPSConfig"):
        run_mps_depth_sweep(target, seeds=(3,), config="wrong")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="convergence_nlf=None"):
        run_mps_depth_sweep(
            target,
            seeds=(3,),
            config=replace(
                tiny_config,
                training=replace(tiny_config.training, convergence_nlf=0.0),
            ),
        )
    with pytest.raises(TypeError, match="OpenBoundaryMPS"):
        run_mps_depth_sweep(object(), seeds=(3,), config=tiny_config)  # type: ignore[arg-type]


def test_mps_depth_sweep_plot_returns_saved_path(
    tmp_path: Path,
    tiny_config: ADQCMPSConfig,
) -> None:
    pytest.importorskip("matplotlib")
    target = random_open_boundary_mps(3, 2, seed=803)
    sweep = run_mps_depth_sweep(target, seeds=(41,), config=tiny_config)

    output = tmp_path / "nested" / "mps_depth_sweep.png"
    saved = plot_mps_depth_sweep(sweep, output, include_global_fidelity=True, dpi=90)

    assert saved == output.resolve()
    assert saved.is_file()
    assert saved.stat().st_size > 0
    with pytest.raises(ValueError, match="file suffix"):
        plot_mps_depth_sweep(sweep, tmp_path / "no_suffix")
    with pytest.raises(TypeError, match="bool"):
        plot_mps_depth_sweep(sweep, output, include_global_fidelity=1)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="MPSDepthSweepResult"):
        plot_mps_depth_sweep(object(), output)  # type: ignore[arg-type]


def test_plotting_reports_a_clear_optional_dependency_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    tiny_config: ADQCMPSConfig,
) -> None:
    target = random_open_boundary_mps(3, 2, seed=804)
    sweep = run_mps_depth_sweep(target, seeds=(51,), config=tiny_config)
    plotting = importlib.import_module("zhou2021.plotting")

    def no_matplotlib(_: str) -> None:
        raise ImportError("simulated missing matplotlib")

    monkeypatch.setattr(plotting.importlib, "import_module", no_matplotlib)
    with pytest.raises(RuntimeError, match="optional 'matplotlib' dependency"):
        plot_mps_depth_sweep(sweep, tmp_path / "unavailable.png")


def test_result_depth_lookup_rejects_missing_or_noninteger_depth(
    tiny_config: ADQCMPSConfig,
) -> None:
    target = random_open_boundary_mps(3, 2, seed=805)
    result: MPSDepthSweepResult = run_mps_depth_sweep(target, seeds=(61,), config=tiny_config)

    with pytest.raises(ValueError, match="not present"):
        result.summary_for_depth(3)
    with pytest.raises(TypeError, match="integer"):
        result.summary_for_depth(1.0)  # type: ignore[arg-type]


def test_infinite_physical_sweep_metrics_have_strict_json_encoding() -> None:
    summary = MPSDepthSummary(
        depth=1,
        num_seeds=2,
        mean_squared_fidelity=0.0,
        std_squared_fidelity=0.0,
        mean_negative_log_fidelity_per_site=math.inf,
        std_negative_log_fidelity_per_site=math.inf,
        mean_prepared_max_bond_dimension=1.0,
        std_prepared_max_bond_dimension=0.0,
        mean_discarded_weight=0.0,
        std_discarded_weight=0.0,
        mean_training_floored_negative_log_fidelity_per_site=5.0,
        std_training_floored_negative_log_fidelity_per_site=0.0,
    )

    payload = cli._json_safe_metric_record(summary)
    assert payload["mean_negative_log_fidelity_per_site"] is None
    assert payload["std_negative_log_fidelity_per_site"] is None
    assert (
        payload["mean_negative_log_fidelity_per_site_classification"]
        == "positive_infinity"
    )
    assert (
        payload["std_negative_log_fidelity_per_site_classification"]
        == "positive_infinity"
    )
    json.dumps(payload, allow_nan=False)


def test_mps_sweep_cli_writes_seed_resolved_manifest(tmp_path: Path) -> None:
    target_path = save_open_boundary_mps_npz(
        tmp_path / "target.npz",
        random_open_boundary_mps(3, 2, seed=806),
        metadata={"source": "test"},
    )
    output = tmp_path / "sweep.json"

    status = main(
        [
            "mps-sweep",
            "--target-mps",
            str(target_path),
            "--layers",
            "2",
            "--epochs-per-stage",
            "1",
            "--learning-rate",
            "0.02",
            "--max-bond-dimension",
            "2",
            "--seeds",
            "71",
            "72",
            "--output",
            str(output),
        ]
    )

    assert status == 0
    manifest = json.loads(output.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == "zhou2021-adqc-mps-depth-sweep-v2"
    assert manifest["algorithm"]["seed_policy"]["seeds"] == [71, 72]
    assert manifest["algorithm"]["depths"] == [1, 2]
    assert len(manifest["runs"]) == 2
    assert all(len(run["depth_records"]) == 2 for run in manifest["runs"])
    assert [summary["num_seeds"] for summary in manifest["summaries"]] == [2, 2]
    first_record = manifest["runs"][0]["depth_records"][0]
    assert first_record["final_negative_log_fidelity_per_site_classification"] == "finite"
    assert "final_training_floored_negative_log_fidelity_per_site" in first_record
    assert "mean_training_floored_negative_log_fidelity_per_site" in manifest["summaries"][0]
