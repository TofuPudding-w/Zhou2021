"""Independent, convention-explicit reproduction of Zhou et al. (2021)."""

from .adqc import ADQCTrainingConfig, ADQCTrainingResult, train_adqc
from .adqc_mps import ADQCMPSConfig, ADQCMPSResult, prepare_staircase_mps, train_adqc_mps
from .circuit import LatentStaircase, apply_adjacent_two_qubit_gate, zero_state
from .gate_archive import (
    LoadedOrderedU4Archive,
    OrderedU4Archive,
    OrderedU4Gate,
    build_ordered_u4_archive,
    load_ordered_u4_npz,
    ordered_u4_archive_from_staircase,
    ordered_u4_semantic_sha256,
    replay_ordered_u4_archive,
    save_ordered_u4_npz,
)
from .ground_states import GroundStateResult, exact_ground_state
from .metrics import (
    negative_log_fidelity_per_site,
    negative_log_fidelity_per_site_from_squared_fidelity,
    squared_state_fidelity,
)
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
    mps_from_dense_statevector,
    mps_squared_fidelity,
    save_open_boundary_mps_npz,
)
from .mps_observables import (
    TNDiffTFIMParameters,
    XXZParameters,
    heisenberg_mps_energy,
    mps_adjacent_product_expectation,
    mps_energy_error_per_site,
    mps_model_energy,
    mps_single_site_expectation,
    tfim_tndiff_mps_energy,
    xxz_mps_energy,
    xy_mps_energy,
)
from .plotting import plot_mps_depth_sweep
from .polar import POLAR_NUMERICAL_RANK_RCOND
from .sweeps import MPSDepthSweepResult, run_mps_depth_sweep
from .tenpy_targets import DMRGConfig, dmrg_ground_state_mps

__version__ = "0.1.0"

__all__ = [
    "ADQCTrainingConfig",
    "ADQCTrainingResult",
    "ADQCMPSConfig",
    "ADQCMPSResult",
    "DMRGConfig",
    "GroundStateResult",
    "LatentStaircase",
    "LoadedOpenBoundaryMPS",
    "LoadedOrderedU4Archive",
    "MPSDepthSweepResult",
    "OpenBoundaryMPS",
    "OrderedU4Archive",
    "OrderedU4Gate",
    "POLAR_NUMERICAL_RANK_RCOND",
    "TNDiffTFIMParameters",
    "XXZParameters",
    "apply_adjacent_two_qubit_gate",
    "build_ordered_u4_archive",
    "exact_ground_state",
    "heisenberg_hamiltonian",
    "heisenberg_mps_energy",
    "negative_log_fidelity_per_site",
    "negative_log_fidelity_per_site_from_squared_fidelity",
    "mps_from_dense_statevector",
    "mps_adjacent_product_expectation",
    "mps_energy_error_per_site",
    "mps_model_energy",
    "mps_single_site_expectation",
    "mps_squared_fidelity",
    "load_open_boundary_mps_npz",
    "load_ordered_u4_npz",
    "ordered_u4_archive_from_staircase",
    "ordered_u4_semantic_sha256",
    "prepare_staircase_mps",
    "plot_mps_depth_sweep",
    "run_mps_depth_sweep",
    "replay_ordered_u4_archive",
    "squared_state_fidelity",
    "save_open_boundary_mps_npz",
    "save_ordered_u4_npz",
    "tfim_tndiff_hamiltonian",
    "tfim_tndiff_mps_energy",
    "train_adqc",
    "train_adqc_mps",
    "dmrg_ground_state_mps",
    "xxz_hamiltonian",
    "xxz_mps_energy",
    "xy_hamiltonian",
    "xy_mps_energy",
    "zero_state",
]
