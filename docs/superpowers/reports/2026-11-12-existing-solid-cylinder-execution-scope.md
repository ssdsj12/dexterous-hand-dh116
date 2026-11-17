# 现有刚性任务完整闭环（复用原环境，不重新建模）

基线为 `75d337e8c2c5c7a21d60e96992b41df2ef7cc185`。现有 `wp6.rigid-native-final` 的依赖闭包已经与 CUDA 杆求解器分离。本次把该入口设为默认调度目标，复用原 Registry 和调度器。

- `execution_scope()` 校验目标、延期原因和依赖闭包。若将真实依赖写成延期项，拒绝该登记表。默认计划和 CPU 执行使用真实闭包。
- `--plan --scope all` 保留总验收和未来柔性工作包的完整视图。延期项沿用原状态和历史记录，CUDA 杆工作包不会被标为通过。
- 同步双目与 MPC 原生采集依赖关闭和重启验证，执行仍须各自批准。离线代码和回归可独立完成。
- 现有 `solid_cylinder` 任务配置、USD、圆柱资产、物理参数、371D 状态和 24D 动作合同沿用原版本。Teacher 成功、BC、DAgger 和 Student 独立验收在同一环境内完成。柔性后端可用后再新增 Cosserat 任务并迁移学习架构。

Ubuntu 运行 `python -m pytest -q -p no:cacheprovider tests/test_dh116_work_packages.py`。修复前五条新增用例失败，修复后 27 项通过，退出码 0。日志 SHA256 为 `3dde445e3315b9973455a4af9761ea7ddd90a85122a8f4f8606f5eea013ed025`。本工作包是 CPU 软件验证，原生关闭、同步采集及 Teacher/Student 成功仍需各自证据。
