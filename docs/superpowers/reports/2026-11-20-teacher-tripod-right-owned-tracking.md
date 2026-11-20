# Tripod 右手独立跟踪执行边界

冻结 CPU 参考 SHA `e419ed68ba6cff54e5ecdf0b679a2b72550851519d918a785c293677c496b112`。独立调用入口 `run_owned_tracking()` 接收既有 SDK 生命周期的鲜活上下文，复用真实 DH116 articulation、两侧原碰撞资产、同一根 DeformX 自由杆和原 2.1 mm 手腕净空保护。当前提交仅接通被调用入口。外层原生配方和自然退出回执后续独立绑定。

执行顺序为 2 SDK 初始化步、8 静置步、160 右手移动步、8 末端静置步。无 Teacher 或 PPO 调用。逐物理步核对同一 run、episode、reset、rod、SDK tick 和时间，保存双侧 17 个 q/qd、SDK 目标、实际六轴挂载基座姿态和每个求解器子步的接触。非原计划通道、实际手杆接触、关节/净空保护触发或静置检查失败即停止。原始终止行或最后可用行保留在现有 JSONL 中，异常关闭路径 fsync 失败前缀。未结算窗口保留原状态。计划和目标轨迹并不提供实际跟踪误差或抓取验收结果。

CPU 回归 68 通过。命令 `python -m pytest -q -p no:cacheprovider tests/test_dh116_tripod_right_tracking_native.py tests/test_dh116_tripod_right_tracking_plan.py tests/test_tripod_quiet_gate.py tests/test_tripod_native_evidence.py`，日志 SHA `295f1a37f3869167023fa6bdf3697d1bc235e4f98d78445a6a32578cf9952bad`。增加旧左手交换准入测试后结果为 68 通过、8 失败，日志 SHA `d5fbea4074aa748553955d4cc356c13bba3eb603090bf84aebaa9f8475458fab`。旧测试的固定 home 路径经过符号链接，拒绝发生在 `read_locked()` 的 canonical 路径检查。实际文件 SHA 仍匹配 `996bdb33ed2d6f9abddb9171245142b5147af222b976fc799b5f89711c4f6a37`。该旧入口保留。历史 Pinch v12 原生账本 SHA `0e9c70159eebb6cc7343a83982d1ca4eac7a6179aacd19c546451ff0969aab85` 的首个物理行已复核，两侧都提供 17 个关节、目标 1×17、挂载姿态 1×7。该日志仅用于核对读回形状。

原生运行 0 次，PPO 更新 0 次。新配方需绑定实际启动闭包、单实例资源锁、自然退出和独立跟踪账本回读，再进行右手接近验证。
