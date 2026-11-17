# 刚性代理与柔性 Cosserat 原生排期

基线 `3e50850a1f72ca7bd37b344aab5f27f8a9fc2fb4`。旧登记表同时把柔性 Solver 设为 Student 训练的间接依赖和最终验收的直接依赖，刚性代理的独立验收也会因此等待柔性后端。

- 刚性代理状态和训练按实际启用字段、独立初态、双目资格及成功 Teacher 门控。柔性完整状态资格继续需要真实形变和接触来源。
- 刚性与柔性原生验收各有工作包。项目最终验收需两条各自通过。柔性工作包继续需要真实求解器、同任务 Teacher/Student 和独立物理证据。
- R0 的排期依赖合格 MPC 同步数据和独立训练批准。PhysX 力学资格继续约束冻结 Teacher 的正式验收和 R1/R2。所有新原生工作包均为 gated，CPU 调度器无法运行它们。
- 每任务的二指、三指各 100 独立初态、30 mm Lift、5 s Hold、0.5 mm Slip、双手有效接触、无外部支撑与独立执行条件继续适用。

Ubuntu CPU 命令 `python -m pytest -q -p no:cacheprovider tests/test_dh116_work_packages.py`。修复前两条依赖回归失败，修复后 19 项通过，退出码 0。日志 SHA256 `d00e324e0060dea4be197c3de3cba9f0954fe018ea7fbf6879d30e2d0e0ad449`。本工作包变更排期和验证，原生资格仍为待证。
