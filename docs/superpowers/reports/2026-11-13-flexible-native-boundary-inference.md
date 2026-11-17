# 柔性 Teacher 的实际预动作状态离线推理

双手 Pinch v4 的原生步骤为 2 初始化加 8 测量。外层与 GDB inferior 自然退出 0，68.494 秒。原生资格只覆盖同一根自由杆 Reset 和一个零动作控制窗。8 行共有 3,936 条 DeformX 杆与台面配对，手与杆配对为 0。实际台面 collider 为 `/World/DeformXDualHand/Table/geometry/mesh`，PhysX 手台面逐接触仍为 NA。原始账本 SHA256 `ead0dbf2c15b83864a1c440848ed9cf159b86c88a6ade0e4275208efebcf778c`，独立 post-exit 审计 `c59e629c4dba249215a810a790a79719a56172d8312fc86c783d8ad1f99ac837`，回执 `aa1a6877c29b460e8f12589674d4ba09381faf8aea200400a7cd0e71d1a5a1ec`。

`NativeBoundaryDiagnosticTeacher.pack_boundary` 重新核对实际安装的挂载关节数据、原 371D 字段和同一物理边界的杆快照。由原动作前账本重算 66 个可用通道，其余 305 个保留 NaN 和 false mask。41 个材料节点的 13 项特征沿用既有模型。独立模型绑定 pair-enabled DeformX 实际二进制 SHA256 `92a50ba3747192b72fd607741ff1465c17874f7c8f8fb0cbed82d99d5985d8c0`，不修改旧模型。二指、三指和握式的派生模型身份与命名空间不同，目前只有二指有这份原生观测。

原生 JSONL 读回使用 NumPy 数组，内存快照使用 Torch；首次测试暴露类型差异，适配器显式持有转换后的浮点输入。测试验证 24D 有界 Tanh 动作和有限 critic 输出，拒绝跨时间、跨 Reset、跨杆、跨 run、错误后端、伪造准入和填补未知字段。**35 项 CPU 测试通过**，日志 SHA256 `bbca492e2be8f2623430303ecb9cc3e4de96e7703672faab1c59bec1ab4b98b0`。输入是原生采集，推理在 CPU 回放，未向 Isaac 发送此模型的动作，未训练或写入 checkpoint，原生 PPO 更新 0。旧 checkpoint writer 仍只接受自己的模型身份。

正式柔性 PPO 仍需分抓法的实际承托、材料位移、桌面接触、R0 奖励和有界训练配方。现有完整 371D Teacher 合同继续独立；此诊断模型只消耗已测子集与杆节点。
