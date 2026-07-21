"""Optional plotting helpers for explicitly configured Zhou 2021 sweeps.

Matplotlib is intentionally imported only inside the plotting path, keeping
the core MPS training package usable on a minimal paper-scale compute node.
The generated figure describes the supplied reproduction coordinates; it does
not label newly generated numerical values as values reported by the paper.
"""

from __future__ import annotations

import importlib
import math
import numbers
from pathlib import Path
from typing import Any

from .sweeps import MPSDepthSweepResult


def _load_pyplot() -> Any:
    try:
        return importlib.import_module("matplotlib.pyplot")
    except ImportError as error:
        raise RuntimeError(
            "MPS sweep plotting requires the optional 'matplotlib' dependency. "
            "Install it with `python -m pip install -e '.[plot]'`."
        ) from error


def _validated_output_path(output_path: str | Path) -> Path:
    if not isinstance(output_path, (str, Path)):
        raise TypeError("output_path must be a str or pathlib.Path.")
    if isinstance(output_path, str) and not output_path.strip():
        raise ValueError("output_path must not be empty.")
    output = Path(output_path).expanduser()
    if not output.suffix:
        raise ValueError("output_path must include a file suffix such as '.png' or '.pdf'.")
    return output


def _validated_dpi(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise TypeError("dpi must be an integer.")
    dpi = int(value)
    if dpi < 1:
        raise ValueError("dpi must be positive.")
    return dpi


def _validated_sweep(sweep: MPSDepthSweepResult) -> MPSDepthSweepResult:
    if not isinstance(sweep, MPSDepthSweepResult):
        raise TypeError("sweep must be an MPSDepthSweepResult.")
    if not sweep.summaries:
        raise ValueError("cannot plot a sweep without depth summaries.")
    depths = sweep.depths
    if depths != tuple(range(1, len(depths) + 1)):
        raise ValueError("sweep depths must be consecutive and begin at one.")
    for summary in sweep.summaries:
        nlf_values = (
            summary.mean_negative_log_fidelity_per_site,
            summary.std_negative_log_fidelity_per_site,
        )
        fidelity_values = (
            summary.mean_squared_fidelity,
            summary.std_squared_fidelity,
        )
        if any(
            math.isnan(float(value)) or float(value) == -math.inf
            for value in nlf_values
        ):
            raise ValueError("cannot plot a sweep with NaN or negative-infinite metrics.")
        if not all(math.isfinite(float(value)) for value in fidelity_values):
            raise ValueError("cannot plot a sweep with non-finite fidelity metrics.")
        if summary.std_negative_log_fidelity_per_site < 0.0:
            raise ValueError("negative-log-fidelity standard deviation must be nonnegative.")
        if summary.std_squared_fidelity < 0.0:
            raise ValueError("squared-fidelity standard deviation must be nonnegative.")
    return sweep


def plot_mps_depth_sweep(
    sweep: MPSDepthSweepResult,
    output_path: str | Path,
    *,
    include_global_fidelity: bool = False,
    title: str | None = None,
    dpi: int = 180,
) -> Path:
    """Save depth versus Zhou Eq. (1) NLF, optionally with global fidelity.

    The primary curve is the mean negative log fidelity per site over the
    explicitly supplied seeds, with a population-standard-deviation band.  If
    requested, the secondary axis reports the corresponding global
    *squared-state* fidelity.  The function returns the resolved output path
    after saving and closes its figure, so it is safe in headless batch jobs.
    """

    sweep = _validated_sweep(sweep)
    output = _validated_output_path(output_path)
    if not isinstance(include_global_fidelity, bool):
        raise TypeError("include_global_fidelity must be a bool.")
    if title is not None and not isinstance(title, str):
        raise TypeError("title must be a string or None.")
    dpi = _validated_dpi(dpi)
    pyplot = _load_pyplot()

    depths = list(sweep.depths)
    # A zero physical overlap has Eq. (1) value +inf.  Matplotlib receives NaN
    # at that depth so the curve has an honest gap instead of silently drawing
    # the optimizer's finite floor as a physical result.
    nlf_mean = [
        summary.mean_negative_log_fidelity_per_site
        if math.isfinite(summary.mean_negative_log_fidelity_per_site)
        else math.nan
        for summary in sweep.summaries
    ]
    nlf_std = [
        summary.std_negative_log_fidelity_per_site
        if math.isfinite(summary.std_negative_log_fidelity_per_site)
        else math.nan
        for summary in sweep.summaries
    ]
    nlf_lower = [
        max(0.0, mean - spread)
        for mean, spread in zip(nlf_mean, nlf_std, strict=True)
    ]
    nlf_upper = [mean + spread for mean, spread in zip(nlf_mean, nlf_std, strict=True)]

    figure, axis = pyplot.subplots(figsize=(7.0, 4.4))
    axis.plot(depths, nlf_mean, marker="o", color="tab:blue", label="NLF/site mean")
    axis.fill_between(
        depths,
        nlf_lower,
        nlf_upper,
        color="tab:blue",
        alpha=0.18,
        label="+/- population std",
    )
    axis.set_xlabel("Serial staircase depth $L$")
    axis.set_ylabel("Physical negative log fidelity per site (Eq. 1)")
    axis.set_xticks(depths)
    axis.grid(True, alpha=0.25)

    handles, labels = axis.get_legend_handles_labels()
    if include_global_fidelity:
        fidelity_mean = [summary.mean_squared_fidelity for summary in sweep.summaries]
        fidelity_std = [summary.std_squared_fidelity for summary in sweep.summaries]
        fidelity_lower = [
            max(0.0, mean - spread)
            for mean, spread in zip(fidelity_mean, fidelity_std, strict=True)
        ]
        fidelity_upper = [
            min(1.0, mean + spread)
            for mean, spread in zip(fidelity_mean, fidelity_std, strict=True)
        ]
        fidelity_axis = axis.twinx()
        fidelity_axis.plot(
            depths,
            fidelity_mean,
            marker="s",
            color="tab:orange",
            label="global $F_{sq}$ mean",
        )
        fidelity_axis.fill_between(
            depths,
            fidelity_lower,
            fidelity_upper,
            color="tab:orange",
            alpha=0.13,
            label="global $F_{sq}$ +/- population std",
        )
        fidelity_axis.set_ylabel("Global squared-state fidelity")
        fidelity_axis.set_ylim(-0.02, 1.02)
        fidelity_handles, fidelity_labels = fidelity_axis.get_legend_handles_labels()
        handles.extend(fidelity_handles)
        labels.extend(fidelity_labels)

    axis.legend(handles, labels, loc="best")
    axis.set_title(
        title if title is not None else "Zhou 2021 ADQC configured MPS reproduction sweep"
    )
    figure.text(
        0.5,
        0.01,
        "Configured reproduction summary; not author-supplied numerical data.",
        ha="center",
        va="bottom",
        fontsize=8,
    )
    figure.tight_layout(rect=(0.0, 0.045, 1.0, 1.0))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=dpi, bbox_inches="tight")
    pyplot.close(figure)
    return output.resolve()


__all__ = ["plot_mps_depth_sweep"]
