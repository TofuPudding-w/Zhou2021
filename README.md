# Zhou 2021 ADQC Reproduction

本项目独立复现 Zhou、Hong 与 Ran 在 2021 年提出的 automatically differentiable quantum circuit
(ADQC) 多比特态制备方案。它的核心是：对每个无约束复 `4×4` latent gate 取 SVD 极分解
`G = U @ V†` 得到最近酉两比特门，并通过自动微分最小化论文 Eq. (1) 的每站点负对数振幅保真度。

严格论文复现对象是 OBC、自旋 `S=σ/2` 归一化下的 Heisenberg 和 XY 链。TFIM 与 XXZ 是为了和
统一多体基态编码任务对齐而加入的扩展，FakeWashingtonV2 原生门/噪声评估同样是后续扩展，均不应
误写为论文原始实验。

详见 [REPRODUCTION_PLAN_CN.md](REPRODUCTION_PLAN_CN.md)、
[docs/CONVENTION_LEDGER_CN.md](docs/CONVENTION_LEDGER_CN.md) 与
[docs/VALIDATION_STATUS_CN.md](docs/VALIDATION_STATUS_CN.md)。

## 可执行闭环

- `zhou2021 dense-target` / `dmrg-target`：生成带位序、模型和能量 provenance 的 OBC MPS 基态目标；
- `zhou2021 mps-train`：以 MPS 原生 overlap 训练 ADQC，并在 provenance 完整时报告 MPS 原生
  energy 与 energy error/site；
- `zhou2021 mps-sweep`：每个 seed 只跑一次完整的逐层 warm-start 轨迹，保存逐深度分布并可生成
  配置化 sweep 图；
- `zhou2021 noise-smoke`：将冻结的 U(4) staircase 精确分解为 `rz/sx/x/cx`，再以 compact
  FakeWashingtonV2 校准噪声作 density-matrix 评估。
- `zhou2021 noise-archive`：读取 `mps-train --ordered-u4-output` 的严格 physical-U(4) archive，
  不重新训练、不重建 latent 参数，直接完成原生门编译与 FakeWashingtonV2 噪声评估。

当前 v2 结果 schema 明确区分物理 Eq. (1) NLF 与带 squared-fidelity floor 的训练 NLF；
FakeWashington 结果同时保存完整 compact-device 校准 provenance，并对原生门、编译账本和
layout 采用 fail-closed 验证。

Ubuntu 安装、严格验证命令和 paper-scale 资源边界见
[docs/RUNBOOK_CN.md](docs/RUNBOOK_CN.md)；逐函数验证证据见
[docs/FUNCTION_VALIDATION_MATRIX_CN.md](docs/FUNCTION_VALIDATION_MATRIX_CN.md)，
最终可复核状态见 [docs/VALIDATION_STATUS_CN.md](docs/VALIDATION_STATUS_CN.md)。
