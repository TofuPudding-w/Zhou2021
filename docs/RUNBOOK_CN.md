# Zhou 2021 复现运行手册

## 范围

严格论文部分是：串行最近邻 U(4) staircase、复潜变量的 SVD 极分解、
Eq. (1) 的每站点负对数振幅保真度，以及逐层 warm-start Adam。论文模型为
OBC、`S=Pauli/2` 归一化下的 Heisenberg 与 XY。TFIM、XXZ、TeNPy 的具体
DMRG 超参数和 FakeWashingtonV2 噪声均是显式标注的扩展，不能表述为原论文
作者报告的实验。

## Ubuntu 环境

建议使用一个干净的 Python 3.11 或 3.12 conda 环境，避免 Qiskit 1.0 前后包
混装：

```bash
cd /home/cg/PycharmProjects/Paper_Reproductions/Zhou_2021
conda create -n zhou2021-py311 python=3.11 -y
conda activate zhou2021-py311
python -m pip install --upgrade pip
python -m pip install -e '.[test,dev,plot,mps,noise]'

python -m zhou2021 doctor
python -m pytest -q
python -m ruff check .
```

如果 `doctor` 中 Qiskit、Aer 或 `qiskit-ibm-runtime` 为 `null`，或 import
报出 mixed pre-/post-1.0 environment，不要在原环境中强行修补；新建上述干净
环境后重新安装 `.[noise]`。

## Z0--Z2：小系统独立闭环

先运行严格论文模型的 dense oracle。它使用 ED 仅作为小系统目标态 oracle，训练
仍然是论文的 supervised ADQC，不是 VQE。

```bash
python -m zhou2021 smoke \
  --model heisenberg --n-qubits 4 \
  --layers 3 --epochs-per-stage 50 \
  --learning-rate 0.01 --learning-rate-decay 0.5 \
  --seed 2021 \
  --output artifacts/smoke/heisenberg_n4.json

python -m zhou2021 smoke \
  --model xy --n-qubits 4 \
  --layers 3 --epochs-per-stage 50 \
  --learning-rate 0.01 --learning-rate-decay 0.5 \
  --seed 2021 \
  --output artifacts/smoke/xy_n4.json
```

检查 JSON 中：`ground_residual`、每层 `initial/final_squared_fidelity`、
物理 `negative_log_fidelity_per_site`、显式
`training_floored_negative_log_fidelity_per_site`、门数
`ordered_u4_gate_count=(N-1)L`，以及
`mps_bond_upper_bound=4**L`。优化轨迹不保证每一次 Adam step 单调；应以记录的
最终物理保真度和 NLF 判断，而不能只根据优化器是否完成迭代。

两种 NLF 不得混用：前者严格是论文 Eq. (1)，零重叠为正无穷；后者仅是为了反向传播而
使用配置中 squared-fidelity floor 的有限训练标量。JSON schema v2 将物理正无穷编码为
`null`，并同时写入 `positive_infinity` classification，绝不以训练 floor 冒充物理结果。

## Z3：ED--MPS 交叉验证

先将小系统 ED 基态转换成严格 MPS NPZ，再使用 MPS-native 后端训练。两个后端
在同一 seed/小系统下应给出相同的物理结果（pytest 已对此交叉验证）。

```bash
mkdir -p artifacts/mps_smoke

python -m zhou2021 dense-target \
  --model heisenberg --n-qubits 6 \
  --output artifacts/mps_smoke/heisenberg_n6_target.npz \
  --summary-output artifacts/mps_smoke/heisenberg_n6_target.json

python -m zhou2021 mps-train \
  --target-mps artifacts/mps_smoke/heisenberg_n6_target.npz \
  --layers 3 --epochs-per-stage 50 \
  --learning-rate 0.01 --learning-rate-decay 0.5 --seed 2021 \
  --prepared-mps-output artifacts/mps_smoke/heisenberg_n6_prepared.npz \
  --ordered-u4-output artifacts/mps_smoke/heisenberg_n6_ordered_u4.npz \
  --output artifacts/mps_smoke/heisenberg_n6_run.json
```

`mps-train` 的 `physical_metrics` 会在 target archive 含有标准 Hamiltonian provenance 时，
直接以 MPS transfer contraction 输出 target/prepared energy、reference total energy 和
`prepared_energy_error_per_site`。这一路径不会展开大系统 statevector。对于没有模型
provenance 的外部 MPS，结果会明确写为 `available: false`；可用
`--energy-model --energy-... --energy-reference` 显式补充坐标，不能靠文件名猜测。

NPZ schema 固定为 `zhou2021-obc-mps-npz-v1`：axis order 为
`(chi_left,physical,chi_right)`，站点 0 是 q0（MSB），数据类型为 complex128，
且加载时禁用 pickle。不要向项目提供未声明位序或 dtype 的第三方 MPS 文件。

## Z4：论文尺度的 DMRG 目标与 MPS-ADQC

`dmrg-target` 是论文 N=48、`chi=64` 目标的可执行路径。论文未公开完整 DMRG
和 ADQC 超参数，故输出的 config/provenance 必须与数据一起保留，且结果应称为
“算法/趋势复现”，而不是逐点数值复刻。

```bash
mkdir -p artifacts/paper_scale

python -m zhou2021 dmrg-target \
  --model heisenberg --n-qubits 48 \
  --chi-max 64 --max-sweeps 30 --min-sweeps 4 \
  --energy-tolerance 1e-10 --svd-min 1e-14 \
  --output artifacts/paper_scale/heisenberg_n48_chi64_target.npz \
  --summary-output artifacts/paper_scale/heisenberg_n48_chi64_target.json

python -m zhou2021 mps-train \
  --target-mps artifacts/paper_scale/heisenberg_n48_chi64_target.npz \
  --layers 5 --epochs-per-stage 100 \
  --learning-rate 0.01 --learning-rate-decay 0.5 --seed 2021 \
  --max-bond-dimension 64 \
  --prepared-mps-output artifacts/paper_scale/heisenberg_n48_l5_prepared.npz \
  --output artifacts/paper_scale/heisenberg_n48_l5.json
```

在启动上述完整预算前，先做一个真实机器上的单层/单 epoch profile；capped MPS 的每个
两比特门都会经历 canonicalization/SVD，不能仅依据 raw MPS core 的大小估计训练时间或峰值内存。

```bash
/usr/bin/time -v python -m zhou2021 mps-train \
  --target-mps artifacts/paper_scale/heisenberg_n48_chi64_target.npz \
  --layers 1 --epochs-per-stage 1 \
  --learning-rate 0.01 --max-bond-dimension 64 --seed 2021 \
  --output artifacts/paper_scale/profile_n48_l1_e1.json
```

`dmrg-target` 的 `completed` 仅表示单次 DMRG target 生成完成；manifest 中会明确标为
`single_run_not_independently_budget_certified`。在把它作为 N=48 基态结论前，至少比较
`chi_max=64` 与更高 chi、延长 sweeps/svd_min 后的总能量、局域能量和目标 MPS overlap，并保存
每一组输入与输出 JSON。若这些量没有达到预先声明的容差，就不能声称基态 target 已收敛。

最后一条命令中的 `--max-bond-dimension 64` 是**显式近似**，不是原文无截断线路的
等价实现。无截断 serial staircase 的理论上界是 `chi <= 4**L`；在 N=48、L=5
时，仅 complex128 MPS core 的上界已约 0.92 GiB，未包含 autograd、SVD 和 Adam
工作区。项目拒绝 N>16 的无上界 CLI 运行，除非显式传入
`--allow-unbounded-mps`。

## Z4b：多 seed、逐深度 warm-start sweep

论文描述逐层增长和 warm start，但没有给出可复用的随机种子 ensemble 或 Adam schedule。
因此下面产生的是**配置化算法复现曲线**，不是声称逐点复制原论文图中的数值。每一个 seed
只运行一次从 `L=1` 到最大深度的完整轨迹；不会对每个深度重新随机初始化。

```bash
mkdir -p artifacts/paper_scale/sweep

python -m zhou2021 mps-sweep \
  --target-mps artifacts/paper_scale/heisenberg_n48_chi64_target.npz \
  --layers 5 --epochs-per-stage 100 \
  --learning-rate 0.01 --learning-rate-decay 0.5 \
  --max-bond-dimension 64 \
  --seeds 2021 2022 2023 2024 2025 \
  --plot-output artifacts/paper_scale/sweep/heisenberg_n48_depth.png \
  --include-global-fidelity \
  --output artifacts/paper_scale/sweep/heisenberg_n48_depth.json
```

JSON 同时保留每个 seed、每个深度的初末 squared fidelity、物理 Eq.(1) NLF/site、训练
floored NLF、截断账本和 population mean/std；PNG 主轴为物理 NLF/site，副轴（可选）为
global squared-state fidelity。零物理重叠在图中显示为缺口，不会被训练 floor 替代。
图注会标明其为配置化复现，避免与作者未公开的原始数值混淆。

## Z5：FakeWashingtonV2 下游噪声扩展

噪声只用于已经冻结的无噪声 ADQC 线路；它不参与训练，也不改变 Zhou 2021 的
目标函数。density-matrix 因此限制于小系统，建议 N=6。

```bash
mkdir -p artifacts/noise
python -m zhou2021 noise-smoke \
  --model tfim --n-qubits 6 \
  --field-x 1.0 --coupling-zz 1.0 \
  --layers 3 --epochs-per-stage 50 \
  --learning-rate 0.01 --learning-rate-decay 0.5 --seed 2021 \
  --optimization-level 0 \
  --seed-transpiler 2021 --seed-simulator 2021 \
  --max-density-matrix-qubits 6 \
  --output artifacts/noise/tfim_n6_fake_washington.json
```

上面的省略 `--physical-path` 的命令只适合作为可运行 smoke：它会选择字典序第一条可行连续路径，
并在 manifest 写入 `path_selection_policy`、具体 path 与 calibration snapshot SHA-256。正式比较应先
读取该 manifest 或独立校准审计，再用相同的、显式的连续路径重新运行，例如
`--physical-path p0 p1 p2 p3 p4 p5`（替换为已核验的实际整数路径）；不要把自动路径当作保真度最优路径。

必须核查：

- block-native 与 Qiskit logical replay 的保真度约为 1；
- 原生门集合只含 `rz/sx/x/cx`，且没有 SWAP；
- `noisy_target_fidelity`、`purity`、trace error、Hermiticity residual、PSD 最小本征值；
- 与匹配 Hamiltonian 的 noisy energy error per site；
- compact physical path，以及完整 backend/device provenance：backend class、noise options、
  target operations、逐 qubit/instruction calibration 与 snapshot SHA-256。

评估入口只接受 `rz/sx/x/cx`（以及不参与演化的 barrier）；opaque `unitary`、SWAP、其他门、
不可审计/非 identity layout 或与实际线路不一致的 compilation ledger 都会在进入 Aer 前失败。

FakeWashingtonV2 是静态 fake-backend 校准快照，并非实时 IBM 硬件测量。

若要测试**已完成 MPS-ADQC 训练的原有线路**，不要重新训练。`mps-train` 通过
`--ordered-u4-output` 写出的物理 U(4) archive 可由 `noise-archive` 直接读取；它先验证
layer-major 门序、每个 U(4)、canonical metadata 与 semantic SHA-256，再精确编译、映射和
噪声演化：

```bash
python -m zhou2021 noise-archive \
  --target-mps artifacts/mps_smoke/heisenberg_n6_target.npz \
  --ordered-u4-input artifacts/mps_smoke/heisenberg_n6_ordered_u4.npz \
  --optimization-level 0 \
  --seed-transpiler 2021 --seed-simulator 2021 \
  --max-density-matrix-qubits 6 \
  --output artifacts/noise/heisenberg_n6_frozen_archive_fake_washington.json
```

该 manifest 同时记录输入 MPS/线路文件 SHA-256、线路 semantic SHA-256、理想 archive--target
global squared fidelity、native compile fidelity、FakeWashingtonV2 path/snapshot，以及密度矩阵
下的 noisy global fidelity 和能量误差。它是“已有线路导入 Qiskit”的正式路径；N 超过
`--max-density-matrix-qubits` 时会拒绝 materialise target 或完整 density matrix，而不是悄悄
切换到不等价的近似模拟。
