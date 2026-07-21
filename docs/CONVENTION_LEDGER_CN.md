# Zhou 2021 复现数值约定

| 项目 | 冻结约定 |
| --- | --- |
| dtype | 训练与线路为 Torch `complex128` / `float64`；导出为 NumPy `complex128` |
| 位序 | 项目 dense statevector 中 q0 是最高有效张量轴（q0-MSB） |
| U(4) 基 | `|00>,|01>,|10>,|11>`，轴顺序 `(out_left,out_right,in_left,in_right)` |
| 每层门序 | `(0,1) → (1,2) → ... → (N-2,N-1)`，串行 staircase |
| 初态 | `|0...0>` |
| 论文损失 | `F=-ln(|<target|prepared>|)/N`；全局相位不影响该量 |
| 论文自旋 | Heisenberg/XY 使用 `S=σ/2`，OBC |
| TFIM 扩展 | `-J Σ ZZ-hx Σ X`，明确标记为 TN-Diff-compatible extension |
| gate projection | `latent=UΣV† -> U V†`；不使用 KAK 参数化替代 |
| 噪声 | Qiskit little-endian 映射在单独 adapter 中显式验证；无测量状态保真度不引入 readout error |
