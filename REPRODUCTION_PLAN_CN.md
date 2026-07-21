# Zhou 2021 ADQC 复现计划与证据边界

## 论文方案的冻结定义

本项目实现本地论文副本（SHA-256 见 `provenance/paper_source.json`）的以下内容：

1. 初态为 `|0...0>`；每层是按 `(0,1),(1,2),...,(N-2,N-1)` **串行**执行的 staircase，只有
   `N-1` 个 U(4) 门，没有 Ran 2020 的末端 U(2) 门；
2. 每个 latent gate `G_tilde∈C^(4×4)` 不受约束；前向门由 SVD
   `G_tilde=UΣV†` 投影为 `G=UV†`；
3. 损失是论文 Eq. (1) `F=-ln(|<psi_tar|psi(theta)>|)/N`，等价于
   `-ln(squared_fidelity)/(2N)`；
4. 使用 Adam；从一层开始，新增层以 `I+small_random` 初始化，保留旧层并逐层反向传播优化；
5. 论文主模型是 OBC `S=σ/2` Heisenberg（Eq. 4）与 XY（Eq. 5）。

论文未提供源码、随机种子、Adam 超参数、学习率 schedule、DMRG/MPS 截断或原始图表数据。因此：

- 会严格复现**算法、线路拓扑、损失与模型定义**；
- 会对图 2--4 做趋势/量级核对，而不会声称无证据的逐点数值相同；
- 所有自定义优化超参数会写入 manifest，且多随机种子报告均值/分布。

## 分阶段实施

| 阶段 | 内容 | 正确性证据 |
| --- | --- | --- |
| Z0 | 约定、模型与小系统精确基态 | Hermiticity、ED residual、解析极限 |
| Z1 | latent SVD→unitary、staircase 前向 | 酉性/极分解最优性、独立 Kronecker 回放 |
| Z2 | 自动微分与逐层 Adam | complex finite difference、下降方向、warm-start |
| Z3 | 多体基态编码 | 论文 XY/Heisenberg；扩展 TFIM/XXZ；fidelity/energy/entropy |
| Z4 | MPS 大系统目标 | MPS overlap 与小系统 dense 交叉验证；`chi≤4^L` |
| Z5 | Qiskit/FakeWashingtonV2 | strict ordered-U(4) NPZ 回放、原生门、density-matrix 噪声与 SHA-256 |
| Z6 | 逐函数审计 | pytest、Ruff、coverage、跨环境回归、工件回放 |

## 模型口径

- `heisenberg`、`xy`、`xxz`：OBC，`S=σ/2`；前两者严格对应论文。
- `tfim_tndiff`：OBC Pauli 口径 `-J Σ ZZ - h_x Σ X`，用于和 TN-Diff N=6 七个 `h_x` 目标对齐；
  这是本项目扩展而非 Zhou 原文模型。
- 任何能量比较必须同时记录模型、算符归一化、边界与参数。

## 噪声边界

FakeWashingtonV2 仅在无噪声 ADQC 线路已验证之后接入。它使用紧凑连续物理路径和静态 fake-backend
校准，输出 mixed-state global fidelity、purity、trace/Hermiticity/PSD、能量与每站误差。它不是实时硬件
结果；N=48 不能使用完整 density matrix，噪声实验将限制在小系统（优先 N=6）。训练后导出的
ordered-U(4) NPZ 可由 `noise-archive` 直接读取并进入 Qiskit，不重建 latent 参数、不重新训练。
