"""Deterministic MPS depth sweeps for the Zhou 2021 ADQC reproduction.

Each seed is optimized exactly once with the existing warm-start executor up
to the requested maximum depth.  The resulting stage ledger is then reshaped
into one record per circuit depth.  This is important: invoking a fresh
optimization independently at every depth would *not* reproduce the
layer-growth / warm-start protocol in Zhou, Hong and Ran.

The article specifies the ADQC architecture and qualitative optimization
protocol, but not a unique optimizer schedule or random-seed ensemble.  The
``ADQCMPSConfig`` and explicit seed tuple supplied here are therefore
reproduction coordinates, not author-supplied numerical data.
"""

from __future__ import annotations

import math
import numbers
from collections.abc import Iterable
from dataclasses import dataclass, replace
from statistics import fmean, pstdev

import torch

from .adqc_mps import ADQCMPSConfig, ADQCMPSResult, MPSStageRecord, train_adqc_mps
from .mps import OpenBoundaryMPS
from .torch_mps import TorchOpenBoundaryMPS, torch_mps_to_numpy


@dataclass(frozen=True, slots=True)
class MPSDepthSeedRecord:
    """Metrics retained for one seed after one warm-start depth stage.

    ``final_discarded_weight`` is the MPS split/truncation ledger accumulated
    while preparing the final state for this stage.  It is zero for the exact
    untruncated path and never silently omitted from a sweep summary.
    """

    depth: int
    seed: int
    initial_squared_fidelity: float
    final_squared_fidelity: float
    initial_negative_log_fidelity_per_site: float
    final_negative_log_fidelity_per_site: float
    final_prepared_max_bond_dimension: int
    final_discarded_weight: float
    initial_training_floored_negative_log_fidelity_per_site: float
    final_training_floored_negative_log_fidelity_per_site: float


@dataclass(frozen=True, slots=True)
class MPSDepthSeedRun:
    """All stage records obtained from one independent ADQC seed."""

    seed: int
    depth_records: tuple[MPSDepthSeedRecord, ...]
    converged: bool
    stopping_reason: str


@dataclass(frozen=True, slots=True)
class MPSDepthSummary:
    """Population summary across all supplied seed runs at one depth.

    Standard deviations are population standard deviations (``ddof=0``), so
    the result remains defined and equals zero for a one-seed diagnostic run.
    Individual values remain available in :class:`MPSDepthSeedRun` rather
    than being replaced by the aggregate.
    """

    depth: int
    num_seeds: int
    mean_squared_fidelity: float
    std_squared_fidelity: float
    mean_negative_log_fidelity_per_site: float
    std_negative_log_fidelity_per_site: float
    mean_prepared_max_bond_dimension: float
    std_prepared_max_bond_dimension: float
    mean_discarded_weight: float
    std_discarded_weight: float
    mean_training_floored_negative_log_fidelity_per_site: float
    std_training_floored_negative_log_fidelity_per_site: float


@dataclass(frozen=True, slots=True)
class MPSDepthSweepResult:
    """Complete, seed-resolved MPS sweep and its depth-wise summaries."""

    num_qubits: int
    target_max_bond_dimension: int
    seeds: tuple[int, ...]
    config: ADQCMPSConfig
    runs: tuple[MPSDepthSeedRun, ...]
    summaries: tuple[MPSDepthSummary, ...]

    @property
    def depths(self) -> tuple[int, ...]:
        """Warm-start circuit depths in ascending physical layer order."""

        return tuple(summary.depth for summary in self.summaries)

    def summary_for_depth(self, depth: int) -> MPSDepthSummary:
        """Return one unambiguous aggregate record for ``depth``."""

        if isinstance(depth, bool) or not isinstance(depth, numbers.Integral):
            raise TypeError("depth must be an integer.")
        wanted = int(depth)
        for summary in self.summaries:
            if summary.depth == wanted:
                return summary
        raise ValueError(f"depth {wanted} is not present in this sweep.")


def _validated_seeds(seeds: Iterable[int]) -> tuple[int, ...]:
    if isinstance(seeds, (str, bytes)):
        raise TypeError("seeds must be an iterable of nonnegative integers, not text.")
    try:
        raw_seeds = tuple(seeds)
    except TypeError as error:
        raise TypeError("seeds must be an iterable of nonnegative integers.") from error
    if not raw_seeds:
        raise ValueError("seeds must contain at least one seed.")

    normalized: list[int] = []
    for value in raw_seeds:
        if isinstance(value, bool) or not isinstance(value, numbers.Integral):
            raise TypeError("every seed must be an integer.")
        seed = int(value)
        if seed < 0:
            raise ValueError("every seed must be nonnegative.")
        normalized.append(seed)
    if len(set(normalized)) != len(normalized):
        raise ValueError("seeds must be unique; duplicate runs do not form an ensemble.")
    return tuple(normalized)


def _normalized_target(
    target_mps: OpenBoundaryMPS | TorchOpenBoundaryMPS,
) -> OpenBoundaryMPS:
    if isinstance(target_mps, TorchOpenBoundaryMPS):
        target_mps = torch_mps_to_numpy(target_mps)
    if not isinstance(target_mps, OpenBoundaryMPS):
        raise TypeError("target_mps must be an OpenBoundaryMPS or TorchOpenBoundaryMPS.")
    return target_mps.normalized()


def _validated_device(device: torch.device | str) -> torch.device:
    if not isinstance(device, (str, torch.device)):
        raise TypeError("device must be a torch.device or device string.")
    try:
        return torch.device(device)
    except (RuntimeError, TypeError) as error:
        raise ValueError(f"invalid torch device: {device!r}.") from error


def _seed_config(config: ADQCMPSConfig, *, seed: int) -> ADQCMPSConfig:
    """Return ``config`` with exactly its random coordinate substituted."""

    return replace(config, training=replace(config.training, seed=seed))


def _record_stage(seed: int, stage: MPSStageRecord) -> MPSDepthSeedRecord:
    finite_values = (
        stage.initial_squared_fidelity,
        stage.final_squared_fidelity,
        stage.initial_training_floored_negative_log_fidelity_per_site,
        stage.final_training_floored_negative_log_fidelity_per_site,
        stage.final_discarded_weight,
    )
    if not all(math.isfinite(float(value)) for value in finite_values):
        raise ArithmeticError("ADQC MPS stage contains a non-finite scalar metric.")
    fidelities = (stage.initial_squared_fidelity, stage.final_squared_fidelity)
    if any(float(value) < -2e-12 or float(value) > 1.0 + 2e-10 for value in fidelities):
        raise ArithmeticError("ADQC MPS stage contains an invalid squared fidelity.")
    for physical_nlf, fidelity in (
        (stage.initial_negative_log_fidelity_per_site, stage.initial_squared_fidelity),
        (stage.final_negative_log_fidelity_per_site, stage.final_squared_fidelity),
    ):
        value = float(physical_nlf)
        if math.isnan(value) or value == -math.inf or value < -2e-12:
            raise ArithmeticError("ADQC MPS stage contains an invalid physical NLF.")
        if value == math.inf and float(fidelity) != 0.0:
            raise ArithmeticError("An infinite physical NLF requires zero squared fidelity.")
    if stage.final_prepared_max_bond_dimension < 1:
        raise ArithmeticError("ADQC MPS stage contains an invalid bond dimension.")
    if stage.final_discarded_weight < -1e-14:
        raise ArithmeticError("ADQC MPS stage contains a negative discarded weight.")
    return MPSDepthSeedRecord(
        depth=stage.num_layers,
        seed=seed,
        initial_squared_fidelity=float(stage.initial_squared_fidelity),
        final_squared_fidelity=float(stage.final_squared_fidelity),
        initial_negative_log_fidelity_per_site=float(
            stage.initial_negative_log_fidelity_per_site
        ),
        final_negative_log_fidelity_per_site=float(stage.final_negative_log_fidelity_per_site),
        final_prepared_max_bond_dimension=int(stage.final_prepared_max_bond_dimension),
        final_discarded_weight=max(0.0, float(stage.final_discarded_weight)),
        initial_training_floored_negative_log_fidelity_per_site=float(
            stage.initial_training_floored_negative_log_fidelity_per_site
        ),
        final_training_floored_negative_log_fidelity_per_site=float(
            stage.final_training_floored_negative_log_fidelity_per_site
        ),
    )


def _physical_mean_and_std(values: Iterable[float]) -> tuple[float, float]:
    """Aggregate physical Eq. (1) values without hiding zero overlaps."""

    collected = tuple(float(value) for value in values)
    if not collected:
        raise ValueError("cannot aggregate an empty physical metric collection.")
    infinite_count = sum(value == math.inf for value in collected)
    if infinite_count == len(collected):
        return math.inf, 0.0
    if infinite_count:
        return math.inf, math.inf
    return fmean(collected), pstdev(collected)


def _summary(depth: int, records: tuple[MPSDepthSeedRecord, ...]) -> MPSDepthSummary:
    if not records:
        raise ValueError("cannot summarize an empty depth record collection.")
    if any(record.depth != depth for record in records):
        raise ValueError("all records in a depth aggregate must have the same depth.")
    physical_mean, physical_std = _physical_mean_and_std(
        record.final_negative_log_fidelity_per_site for record in records
    )
    return MPSDepthSummary(
        depth=depth,
        num_seeds=len(records),
        mean_squared_fidelity=fmean(record.final_squared_fidelity for record in records),
        std_squared_fidelity=pstdev(record.final_squared_fidelity for record in records),
        mean_negative_log_fidelity_per_site=physical_mean,
        std_negative_log_fidelity_per_site=physical_std,
        mean_prepared_max_bond_dimension=fmean(
            record.final_prepared_max_bond_dimension for record in records
        ),
        std_prepared_max_bond_dimension=pstdev(
            record.final_prepared_max_bond_dimension for record in records
        ),
        mean_discarded_weight=fmean(record.final_discarded_weight for record in records),
        std_discarded_weight=pstdev(record.final_discarded_weight for record in records),
        mean_training_floored_negative_log_fidelity_per_site=fmean(
            record.final_training_floored_negative_log_fidelity_per_site
            for record in records
        ),
        std_training_floored_negative_log_fidelity_per_site=pstdev(
            record.final_training_floored_negative_log_fidelity_per_site
            for record in records
        ),
    )


def _records_from_result(
    result: ADQCMPSResult,
    *,
    seed: int,
    expected_depth: int,
) -> tuple[MPSDepthSeedRecord, ...]:
    if len(result.stages) != expected_depth:
        raise RuntimeError(
            "MPS depth sweep requires a complete warm-start trajectory through "
            f"depth {expected_depth}, but training returned {len(result.stages)} stages."
        )
    records = tuple(_record_stage(seed, stage) for stage in result.stages)
    observed_depths = tuple(record.depth for record in records)
    expected_depths = tuple(range(1, expected_depth + 1))
    if observed_depths != expected_depths:
        raise RuntimeError(
            "MPS warm-start stages must be ordered consecutively from depth one; "
            f"observed {observed_depths!r}."
        )
    return records


def run_mps_depth_sweep(
    target_mps: OpenBoundaryMPS | TorchOpenBoundaryMPS,
    *,
    seeds: Iterable[int],
    config: ADQCMPSConfig,
    device: torch.device | str = "cpu",
) -> MPSDepthSweepResult:
    """Execute one full MPS warm-start trajectory per requested seed.

    The target is normalized once and shared without mutation.  Each seed
    becomes the ``ADQCTrainingConfig.seed`` for a single call to
    :func:`train_adqc_mps`; it is not reseeded separately at every depth.
    Consequently depth ``L`` is the literal continuation of depth ``L - 1``.

    A convergence threshold is rejected here because a sweep must retain all
    depths through ``config.training.max_layers``.  Use a standalone training
    run when early stopping is the desired experiment.
    """

    if not isinstance(config, ADQCMPSConfig):
        raise TypeError("config must be an ADQCMPSConfig.")
    if config.training.convergence_nlf is not None:
        raise ValueError(
            "depth sweeps require convergence_nlf=None so every requested warm-start "
            "depth is evaluated."
        )
    validated_seeds = _validated_seeds(seeds)
    target = _normalized_target(target_mps)
    validated_device = _validated_device(device)

    runs: list[MPSDepthSeedRun] = []
    for seed in validated_seeds:
        result = train_adqc_mps(
            target,
            config=_seed_config(config, seed=seed),
            device=validated_device,
        )
        records = _records_from_result(
            result,
            seed=seed,
            expected_depth=config.training.max_layers,
        )
        runs.append(
            MPSDepthSeedRun(
                seed=seed,
                depth_records=records,
                converged=result.converged,
                stopping_reason=result.stopping_reason,
            )
        )

    summaries = tuple(
        _summary(
            depth,
            tuple(run.depth_records[depth - 1] for run in runs),
        )
        for depth in range(1, config.training.max_layers + 1)
    )
    return MPSDepthSweepResult(
        num_qubits=target.num_qubits,
        target_max_bond_dimension=target.max_bond_dimension,
        seeds=validated_seeds,
        config=config,
        runs=tuple(runs),
        summaries=summaries,
    )


__all__ = [
    "MPSDepthSeedRecord",
    "MPSDepthSeedRun",
    "MPSDepthSummary",
    "MPSDepthSweepResult",
    "run_mps_depth_sweep",
]
