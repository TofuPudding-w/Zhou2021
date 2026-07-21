# Zhou 2021 逐函数工程审计矩阵

本文件区分三类证据：

- `U`：单元性质或解析 oracle；
- `X`：独立实现/表示之间的交叉验证；
- `I`：端到端集成或外部可选依赖验证。

“已覆盖”不等价于“论文作者公布了该数值设置”。论文未公布 Adam schedule、随机
种子和完整 DMRG 设置的部分，均被记录为本复现的受控工程坐标。

## 当前审计状态

当前 Windows 基础环境的可执行结果为：`152 passed, 11 skipped`。跳过项不是静默
成功：7 项是本机 Qiskit 混装导致的 FakeWashington 测试，4 项是未安装
`physics-tenpy` 的真实 DMRG 集成测试。Qiskit/Aer/FakeWashington 与 TeNPy DMRG
已分别在兼容的补充环境执行通过；仍应在目标 Ubuntu 主机按本文件末尾命令独立复核。
跨基础、TeNPy/Qiskit-Aer 和 FakeWashington runtime 三个互补环境，当前 163 个唯一测试
均至少实际通过一次。

| 模块/API | 验证类型 | 关键 oracle / 不变量 | 测试 |
| --- | --- | --- | --- |
| `conventions.validate_*`, `normalized_statevector`, `phase_aligned_state` | U | dtype、有限性、归一化、q0-MSB 长度、global phase | `test_core_conventions_models.py` |
| `models.single_site_operator`, `nearest_neighbor_operator` | U/X | 独立 Kronecker 嵌入、相邻位序 | `test_core_conventions_models.py` |
| `xxz/heisenberg/xy/tfim_tndiff_hamiltonian` | U/X | N=2 解析 Pauli/`S=σ/2` 系数、Hermiticity | `test_core_conventions_models.py` |
| `validate_dense_hamiltonian`, `exact_ground_state` | U/X | `eigh` residual、简并子空间的确定性投影和相位固定 | `test_core_conventions_models.py` |
| `overlap_amplitude`, `squared_state_fidelity`, `negative_log_fidelity_per_site`, energy metrics | U | 相位不变、零重叠、Eq.(1)、解析能量；显式 N 必须与态维度一致 | `test_core_conventions_models.py`, `test_core_adqc.py` |
| `polar.validate_latent_gate`, `project_latent_gate`, `polar_unitary` | U/X | `U†U=I`、核范数最优性、相位等变、幺正输入幂等 | `test_core_polar_circuit.py` |
| polar autograd | X | real/imag 两个坐标的中心有限差分 | `test_core_polar_circuit.py` |
| `apply_adjacent_two_qubit_gate` | X | q0-MSB 独立 dense Kronecker 回放；非幺正矩阵拒绝 | `test_core_polar_circuit.py` |
| `LatentStaircase` | U/X | `(0,1)…(N-2,N-1)` 的 layer-major 顺序、`16(N-1)L`、`4**L`、seed 可重复性 | `test_core_polar_circuit.py` |
| `ADQCTrainingConfig`, differentiable Eq.(1) loss | U | 正/负边界、`0<floor<=1`、显式零重叠 floor | `test_core_adqc.py` |
| `train_adqc` | I/X | 分层更新数、stage LR、warm start、bitwise repeat、提前停止；物理 NLF 与训练 floored NLF 分离 | `test_core_adqc.py` |
| `OpenBoundaryMPS`, TT-SVD, random ensemble | U/X | OBC bond、dense round trip、截断 norm ledger、可重复 random MPS | `test_mps_core.py` |
| NumPy MPS U(4) 与 canonical truncation | X | dense operator 回放；mixed-canonical 后 `discarded_weight = Δ||ψ||²` | `test_mps_core.py` |
| MPS NPZ I/O | U/X | schema、axis order、q0-MSB、complex128、metadata、`allow_pickle=False` round trip | `test_mps_core.py` |
| `torch_mps` | X | dense/MPS U(4) 回放、可微 polar 梯度、截断账本、顺序层 | `test_torch_mps.py` |
| `train_adqc_mps` | X/I | 同一小系统/seed 与 dense `train_adqc` 保真度和两套 NLF 一致；F² 低于训练 floor 时仍报告真实 F² | `test_adqc_mps.py` |
| `mps_observables`（单点/相邻项、Heisenberg/XY/XXZ/TFIM energy） | X | prefix/suffix transfer contraction 与独立 q0-MSB dense Hamiltonian、精确基态 energy/site error 对照；大 N 禁止 densify | `test_mps_observables.py` |
| `mps_targets.dense_model_ground_state_mps` | X | ED 基态与无截断 TT-SVD 一致；N>阈值拒绝 dense | `test_mps_targets.py` |
| `tenpy_mps_to_open_boundary`, `DMRGConfig` | U/X | fake finite TeNPy tensor 的 `(vL,p,vR)`/norm/位序交叉验证 | `test_tenpy_targets.py` |
| `dmrg_ground_state_mps` | I | Heisenberg/XY/XXZ/TFIM N=4 DMRG energy 对 ED；TeNPy 1.1.0 环境已执行 | `test_tenpy_targets.py`（可选） |
| `run_mps_depth_sweep`, `plot_mps_depth_sweep` | I/X | 每个 seed 只执行一次完整 warm-start；逐深度 seed-resolved ledger、物理/训练 NLF mean/std；零重叠以 +inf/图中缺口表示 | `test_sweeps.py` |
| `mps-train` / `mps-sweep` CLI | I | NPZ provenance、资源上界确认、自动 MPS energy/site error、JSON manifest | `test_cli_mps_physical_metrics.py`, `test_sweeps.py` |
| Qiskit site/qarg mapping、logical U(4)、exact native block synthesis | X/I | PyTorch / logical-Qiskit / native-Qiskit 三方 fidelity，原生 basis | `test_noise_adapter.py` |
| compact FakeWashington、density diagnostics | U/I | path、完整 calibration provenance、`Trρ`、Hermiticity、purity、PSD、非零 Hamiltonian 能量 | `test_noise_density.py` |
| `native_noise_protocol` | I | 无噪声训练与噪声下游评估隔离；仅接受 `rz/sx/x/cx`；编译 ledger/layout fail-closed；live/归档 N=2 FakeWashington 已在 Qiskit 2.5/Aer 0.17/runtime 0.48 环境执行 | `test_noise_density.py`（可选） |
| `gate_archive` | U/X/I | 严格 layer-major `(layer,position,left_site)`、complex128 U(4)、canonical JSON、语义 SHA-256、NPZ 安全加载、NumPy q0-MSB 回放 | `test_gate_archive.py` |

## 已发现并修复的审计问题

1. `np.int64` seed 先被训练配置接受、后被 staircase 拒绝。现在统一接受
   `numbers.Integral`，并有回归测试。
2. 公共 `apply_adjacent_two_qubit_gate` 声称接受 U(4)，但原先未验证幺正性。现在
   在不切断 autograd 的前提下检查 (U^dagger U=I)。
3. `loss_floor_squared_fidelity > 1` 会令 Eq.(1) 训练损失变为非物理负数。现在强制
   (0 < \mathrm{floor}\le1)。
4. 直接对非 mixed-canonical MPS 局部截断时，局部奇异值丢弃量不等于全局 norm loss。
   现在 NumPy/Torch 两条路径均先作 mixed-canonical QR sweep；若调用者显式关闭该
   sweep，结果对象会将账本标记为 local-only，不能当作全局截断误差。
5. PyTorch 对重复奇异值或零奇异值空间的 SVD/QR 反传可产生 NaN。U(4) polar 现保持严格
   `U@Vh` 前向并以 Lyapunov 方程计算稳定 VJP；MPS 无截断路径使用 exact identity gauge，
   截断路径对秩亏但可微的切空间采用解析反传，对数学上不唯一的 hard truncation 在优化器更新前 fail-fast。
6. 仅保存矩阵哈希不足以独立重放训练后的线路。现新增严格 ordered-U(4) NPZ 归档：每个物理
   U(4)、门序、位序、canonical metadata、文件 SHA-256 与语义 SHA-256 均被保存和交叉验证。
7. 显式 `num_qubits` 原先只改变分母，可能与 statevector 维度矛盾。现在不一致即抛错。
8. 训练 floor 原先与物理 Eq. (1) 混用，MPS 甚至会在真实 F² 低于 floor 时把 floor 误报为
   F²。现在所有 result/stage/update/sweep/CLI 均分别保存物理 NLF 与
   `training_floored_negative_log_fidelity_per_site`；零重叠物理值为 +inf，并以严格 JSON 的
   `null + positive_infinity classification` 导出。
9. Qiskit 编译 API 原有 `simplify` 参数没有任何效果，现已删除，避免记录虚假的编译坐标。
10. 噪声 manifest 原先只保留部分 path/hash。现在嵌入完整 `CompactWashingtonDevice` provenance，
    包括 backend class、noise options、目标门集、逐比特和逐指令校准及 snapshot SHA。
11. `evaluate_native_density_matrix` 现在重新核对实际 circuit 与 `NativeCompilation` ledger，拒绝
    opaque `unitary`、SWAP、非 `rz/sx/x/cx` 门、伪造计数/深度/位序以及错误 `num_sites`。
12. transpiler layout 仅在确实不存在时采用 identity；layout 对象存在但不可解析、非排列或非
    identity 时均 fail-fast，不再把“无法审计”伪装成 identity。

## 最终 Ubuntu 审计命令

```bash
cd /home/cg/PycharmProjects/Paper_Reproductions/Zhou_2021
conda activate zhou2021-py311

python -m pytest -q
python -m ruff check .

# MPS 原生能量、CLI manifest 与逐深度 sweep
python -m pytest -q tests/test_mps_observables.py tests/test_cli_mps_physical_metrics.py tests/test_sweeps.py

# TeNPy 的四模型 N=4 独立 ground-energy 对照
python -m pytest -q tests/test_tenpy_targets.py

# FakeWashingtonV2 真正的 native compilation + density-matrix 交叉验证
python -m pytest -q -m qiskit
```

只有前三类测试均通过时，才可把该环境标记为“论文核心 + MPS target +
FakeWashingtonV2 extension 已验证”。即使通过，FakeWashingtonV2 仍是静态校准
模拟，不是实机结果；TFIM/XXZ 仍是与后续多论文比较对齐的扩展，而非 Zhou 原文模型。
