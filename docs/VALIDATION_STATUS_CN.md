# Zhou 2021 ADQC 复现：实现与验证状态

本文件记录当前代码已经验证到什么程度，也明确哪些结论尚不能作出。它避免将“命令
返回成功”误写成“论文数值已逐点复现”。

## 1. 复现边界

严格对应 Zhou、Hong、Ran (2021) 的部分为：

- 从 `|0...0>` 出发；每层按 `(0,1),(1,2),...,(N-2,N-1)` **串行**执行 `N-1` 个 U(4)；
- 每个无约束复 `4x4` latent matrix 通过 SVD polar `U @ Vh` 投影为 U(4)；
- 损失为论文 Eq. (1) 的每站点负对数振幅保真度，等价于
  `-log(squared_state_fidelity)/(2*N)`；
- Adam 的逐层 warm-start：第一层随机，新增层为 `I + small_random`，旧层保持并继续优化；
- OBC、`S=Pauli/2` 归一化下的 Heisenberg 与 XY 基态编码。

XXZ、TN-Diff 口径 TFIM，以及 FakeWashingtonV2 原生门/噪声评估均是为了后续统一比较
加入的**扩展**；它们不属于 Zhou 原文的硬件噪声实验。

## 2. 当前可复核证据

在 Windows 当前环境，执行

```powershell
$env:PYTHONPATH = 'src'
python -m pytest -q -p no:cacheprovider
```

得到：`152 passed, 11 skipped`；并且

```powershell
& 'D:\pycharm_projects\Paper_Reproductions\Ran_2020\.venv\Scripts\python.exe' `
  -m ruff check --no-cache src tests
```

得到：`All checks passed!`。

11 个 skip 是显式的可选依赖缺失，而不是成功结果：

- 7 个 Qiskit/FakeWashingtonV2 测试：本机发现 Qiskit pre-1.0 与 >=1.0 混装，导入被
  主动拒绝；
- 4 个真实 TeNPy DMRG 集成测试：当前 Windows 环境未安装 `physics-tenpy`。

这些项目没有被计入通过数量。

为补齐基础环境中的可选依赖跳过项，修复版源码还在两个互补环境中执行：

- Python 3.10.19 交叉数值环境（不作为 `pyproject.toml` 声明的正式安装版本）、TeNPy 1.1.0、
  Qiskit 2.4.2、Aer 0.17.2：`160 passed, 3 skipped`；
  四个模型的 TeNPy/ED 对照与其余 Qiskit/Aer 测试均实际通过。三个 skip 中两个只因该环境
  未安装 `qiskit-ibm-runtime`，另一个是“没有 TeNPy 时应报清晰错误”的互斥反向测试。
- 正式支持范围内的 Python 3.11.5、Qiskit 2.5.0、Aer 0.17.2、
  `qiskit-ibm-runtime` 0.48.0：完整回归为 `159 passed, 4 skipped`；仅四个真实 TeNPy
  测试因该环境没有 TeNPy 而跳过，包含 live staircase、frozen ordered-U4 archive 在内的
  全部 Qiskit/FakeWashingtonV2 测试均实际通过。

两个环境均通过 `zhou2021.__file__` 确认加载的是本次修复源码，Ruff 结果为
`All checks passed!`。因此当前公共功能已有可执行的可选依赖证据；Ubuntu 命令仍保留为
正式迁移后的独立复核，而不是为了弥补一个已知函数失败。

合并基础环境与两个补充环境的结果，项目当前 **163/163 个唯一 pytest 用例均至少在一个
适配环境中实际 PASS**。这是覆盖当前测试不变量的工程证据，不是对任意输入、任意平台的
形式化正确性证明；CUDA、论文全尺度长程统计和真实量子硬件运行仍不在这句话的范围内。

## 3. 逐层验证结构

| 层次 | 已验证的独立证据 |
| --- | --- |
| Hamiltonian/ED | OBC Heisenberg、XY、XXZ、TFIM 的解析小系统系数、Hermiticity、`eigh` residual 与确定性简并相位选择。 |
| U(4) 与位序 | polar 幺正性/最优性、q0-MSB 独立 Kronecker 回放、串行门序、全坐标有限差分与重复奇异值 `gradcheck`。 |
| 自动微分稳定性 | 极分解反传使用稳定 Lyapunov VJP；秩亏 MPS 的 exact split 与截断切空间反传均有有限差分测试；不唯一 hard truncation 在更新前显式失败，绝不静默产生 NaN。 |
| MPS 物理量 | MPS overlap、保真度、OBC 单点/两点收缩、四模型能量和 energy/site error 与独立 dense oracle 交叉核验；大 N 路径禁止 densify。 |
| 训练与 sweep | dense/MPS 小系统交叉、逐层 warm-start、不同 seed、早停、配置/Adam 参数 manifest、逐深度全候选记录；物理 Eq. (1) 与带 floor 的训练损失分字段保存，真实保真度不再由 floored loss 反推。 |
| 线路交付 | 严格 ordered-U(4) NPZ：`complex128` 门、层/位置/左站点、q0-MSB 回放、canonical JSON、文件 SHA-256、语义 SHA-256、篡改检测。 |
| 噪声扩展 | Qiskit adapter 规定逻辑/物理 qarg 映射；训练后可导出并重新载入 strict ordered-U(4) archive，精确分解至 `rz/sx/x/cx`；raw circuit、编译账本和 layout 均 fail-closed；结果嵌入完整 compact-device 校准 provenance。FakeWashingtonV2 density-matrix 的 live/archive 路径已在兼容环境执行通过；它仍是静态 fake-backend 模拟而非真实硬件。 |

完整函数—测试映射见 [FUNCTION_VALIDATION_MATRIX_CN.md](FUNCTION_VALIDATION_MATRIX_CN.md)。

## 4. 本轮端到端产物

目录 `artifacts/final_validation/` 中保留了小系统回归产物：

- `mps_pipeline/`：N=4 Heisenberg ED target → 两层 MPS-ADQC → prepared MPS、ordered-U(4)
  archive、物理能量 manifest；其 target energy/site error 为零（数值精度内），U(4) archive
  有 2 层 × 3 门，NumPy 回放范数为 1。
- 同目录的 `heisenberg_n4_depth_sweep.json`：两个 seed 的一条完整三层 warm-start 轨迹；
  配置为每层 3 epochs、`chi_max=4`，仅用于验证链路而非论文收敛点。其 F² 平均值随深度
  由 0.6185、0.8850、0.9513 增长，未发生截断丢失。
- `model_matrix/`：N=4 的 Heisenberg、XY（原文模型）和 XXZ、TFIM（扩展模型）都完成
  `ED target → MPS-ADQC → energy/site` 流程；每个 target 的 ED residual 均小于 `3e-15`。

这些 run 的 epoch 很少，目的是端到端回归而不是报告最终保真度。它们不能替代 N=12/N=48 的
收敛研究。

## 5. 仍需在 Ubuntu 的正式核验

必须在新的、未混装的 Python 3.11 环境中执行：

```bash
cd /home/cg/PycharmProjects/Paper_Reproductions/Zhou_2021
conda create -n zhou2021-py311 python=3.11 -y
conda activate zhou2021-py311
python -m pip install --upgrade pip
python -m pip install -e '.[test,dev,plot,mps,noise]'

python -m zhou2021 doctor
python -m pytest -q
python -m ruff check .
python -m pytest -q tests/test_tenpy_targets.py
python -m pytest -q -m qiskit
```

其中：

1. TeNPy N=4 对 ED 是 DMRG 接口的独立正确性检查；N=12/N=48 还应比较至少两个
   `chi_max`/sweep 预算，确认能量、截断误差与 MPS overlap 稳定后，才能称为收敛的基态 target。
2. FakeWashingtonV2 应指定并记录物理连续路径，保存 backend calibration snapshot 和 archive
   的 SHA；它是静态 fake-backend 噪声模拟，不是实机结果。
3. 噪声只在冻结的理想 ADQC 线路后施加；不得把噪声保真度混入无噪声优化目标，也不得声称为
   noise-aware training。
