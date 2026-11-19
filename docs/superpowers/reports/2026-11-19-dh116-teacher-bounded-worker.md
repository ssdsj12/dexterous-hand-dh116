# DH116 Pinch Teacher 有界 Worker 接线

Ubuntu 工作树和 GitHub `11-17` 基线均为 `5dab1f5c3a346171bd8657305df5a701c9c87acd`。Antigravity 使用只读 SSH 审查启动、数值 Reset 和 R06 回调。Codex 独立确认 R06 的 `acquire(sim, owners, trace, cleanup)` 在 `run_owned_call` 中运行。更新和 Collector 清理都留在该资源范围内，R06 仅收到可序列化的结果。关闭继续使用原有的资源读回和清理顺序。

## 实现

- `run_a` 在导入 Isaac 或新建输出目录前重验 Worker 租约、fresh 角色、进程配置与审批意图的输出路径。输出目录必须为新目录。
- `_run_isaac` 核验环境设备映射、隔离 DeformX 二进制和 Kit experience，再使用原 headless App、双手场景参数及 R06；`_acquire` 复用 C04 的原生场景、数值 Reset、单份同步账本、原 Collector 和原 PPO。
- `_prepare_source_pack` 核对首次 Reset 的原生杆实例、左右手设备和物理时钟。C04 验证实际 Reset 和原始状态。C05 在检查点封存前独立重建已记录的模型输入，并核对审批绑定的源码、固定材料节点、Reset、奖励及训练语义。
- 按原配方只实现 1 环境、1 个 fresh episode、32 个控制步和 1 次更新的无相机诊断入口，2 个初始化 SDK 步加 256 个测量步。真实完成仍需当次原生输出核验。
- R06 保留独立 `lifecycle/` 目录。仅 R06 返回正常关闭记录才写 `owned_update.json`，其中 `process_exit_verified`、`native_qualification`、`teacher_admission` 均为 false。外层进程退出和正式产物准入由调度器独立核验，本包没有生成对应通过回执。

## 验证

最初 10 项 RED 复现固定入口缺失。新增两项故障注入进一步复现写入未获正常关闭的内部摘要，随后纳入最终回归。联合 CPU 套件在固定 Ubuntu 工作树的绝对 `PYTHONPATH` 后 283 项通过。最终源码再运行 Worker、R06、力闭合、Tripod 和 Power 相关套件，149 项通过，其中新增 14 项。两套测试有重叠。首次联合运行因子进程在临时测试仓库内把相对 `src` 解析到测试占位源码而有 20 项失败。保留原日志，固定 `PYTHONPATH` 后复跑通过。

- 联合日志 `worker-cpu-pinned-imports.log`，SHA-256 `0f5c9fc7f6b9a8805f757098fd44bfb53d8621b323ba78ad1bc65d1000a1ebbf`。
- 最终日志 `worker-final-style-regression.log`，SHA-256 `485db33ac6ada9bb394372742d991615e4da872b2b819d51745a3280943df018`。
- 完整测试命令与环境见 readiness。新增 Isaac 实例、原生 PPO 更新、抓取合格回合均为 0。

## 三抓法后继

| 抓法 | 当前证据 | 下一个验收点 |
|---|---|---|
| Pinch | 来源检查、固定 Worker 与关闭边界通过 CPU 回归 | 接通调度器派生账本与独立外层产物审计。冻结实际配方后，验证第一次原生 PPO 更新与自然退出 |
| Tripod | 独立三指方向与对向闭合回归通过 | 先修正右手初态桌面净空，再测双手拇指分别与食指、中指的同钟反力，并接入独立 Reset |
| Power | 五指闭合约束与表面来源检查通过 CPU 回归 | 分别排除两手初态桌面碰撞，验证五指包覆与掌面或近端的实际承载，再接入独立 Reset |

本次读取时 WebRTC 仍持有单实例 Isaac 锁。正式训练和评估继续分别要求净抬升至少 30 mm、连续保持至少 5 s、固定锚相对滑移不超过 0.5 mm。
