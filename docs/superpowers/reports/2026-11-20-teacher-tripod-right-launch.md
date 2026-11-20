# Tripod 右手有界原生启动入口

Antigravity 核对 Ubuntu 分支 `11-17` 和审查基线 `58d0d80872a8dac8722e3302752ed01c8ef800f9`。采用已有 R06 单实例锁、原生模块 SHA 和正常关闭。独立外层进程核对启动指纹、排他回执、日志、自然退出码、资源释放、缓存 SDK 边界和完整物理账本。已知静置或几何保护拒绝保留实际步数和主错误，按同一实际步尝试一次缓存清理。未知原生异常走原故障处理。旧 Reset 模式继续使用第 2 个 SDK 步的缓存合同。

本轮冻结右手计划 SHA `e419ed68ba6cff54e5ecdf0b679a2b72550851519d918a785c293677c496b112`。GPU 2，单实例单环境，2 初始化＋176 测量 SDK 步、22 个控制窗、2112 以内求解器子步、0 Teacher 查询、0 PPO。180 秒软上限，留出 10 秒供清理，整窗边界届时停止，不强制结束，不重复尝试。杆初始 Z 接近 0.305 m。独立复核的历史 Pinch 同高首窗已验证 96/96 桌面接触和最大节点速度约 0.0383 mm/s。Tripod 按自身当前实例读取原静置门槛。原物理资产、状态和动作合同保留。

CPU 136 通过，命令 `python -m pytest -q -p no:cacheprovider --tb=short tests/test_dh116_tripod_right_tracking_launch.py tests/test_dh116_tripod_right_tracking_native.py tests/test_dh116_tripod_right_tracking_plan.py tests/test_tripod_exchange_admission.py tests/test_tripod_quiet_gate.py tests/test_tripod_native_evidence.py tests/test_fault_safe_lifecycle_v3.py tests/test_fresh_teacher_reset_evidence.py`。日志 SHA `33b44f5723fe7c2aa4f6d7b12869d1a2eadc94e0b6db991fde8c7308b06da8d9`。原生启动与跟踪结果从独立回执另行发布。当前阶段没有抓取和 Teacher 资格。
