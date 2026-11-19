# DH116 Pinch 建场与首动作的 Reset 证据

Antigravity 对 `e857f0435106a9f2dfeff8f5130ad393e6a762cb` 源码完成四轮只读复核。本包采用其区分 SDK 建场和杆求解器时钟的建议。Codex 对照已封存的原生 Pinch 数值 Reset 原始字段，确认 `before_clock` 和 `after_actual_clocks` 包含实际 SDK 步、时间、步长以及杆时钟。`after_clock` 只保留 Reset 后杆端点记录。原始实测 SDK 时间与标称 480 Hz 时间存在浮点表示差异，分别按照既有验证器检查，保留来源。

`_audit_fresh_setup` 复用原 `_Sequence` 和 `_no_step`。它从冻结的原始账本读回两个建场记录、一个成功 Reset 和首条执行前记录，核验建场步骤为 0→2，Reset 不增加 SDK 或 solver 步，并确保首条动作的 run、环境、episode、Reset 和杆端点与该 Reset 完全相同。错误顺序、重复 Reset、伪装成整数的布尔步、隐藏推进和跨 episode 证据均会拒绝。实际 A 产物审计在后续物理／checkpoint 校验前运行新审计。旧入口默认语义保留。

本轮联合 CPU 回归 265 项通过，含 6 项新测试。首个实现使用错误模块路径并在回归中暴露，修正后新测试 6/6，通过最终联合回归。完整日志 SHA 为 `b1e8f02a194b5cb5891616d76f1ff1288f65b2eb203cbfb19ac2b1920c2feb46`。原生共计 0 次新 App、0 次 PPO 更新。读取的已封存历史 Reset 仅用于确认读取字段，无新增接触数据。三抓法抬腕报告使用离线候选的 41 个输入节点，其活动范围属于固定几何证据。

Pinch 进入原生 PPO 前仍需独立验证 episode cap、实际父截止与原生回执。当前已有 Isaac 单实例锁被占用。
