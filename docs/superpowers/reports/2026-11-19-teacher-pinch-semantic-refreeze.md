# Pinch Teacher 配方语义重新冻结

基线 `3bbac3cbe28cf8087825f184a660faed1ef17152`。v3 已使用私有会话目录通过授权握手，启动 Isaac，完成 2 个 SDK 初始化步和一次字节一致的数值 Reset。账本 10 行，尚未签发第一个策略动作。物理测量 0 步，PPO 更新 0。外层退出 1，GDB 未记录信号；清理走故障路径，资源释放未资格化。保留回执 fb067670d0ac8e266d79d3a09e8d547d5a7fe061684903e937042c7510f8a7e3、原生账本与独立日志。

错误是 `live Teacher semantic identity differs from the frozen recipe`。早先刷新 source closure 时保留了旧 semantic SHA，尽管语义本身包含源码闭包。CPU 独立复现同一拒绝，避免把它归因于 CUDA、Reset 物理或关节。

v4 复用 `prepare_dh116_pinch_ppo_draft.freeze_recipe` 和原 `_static_pins`，重新计算完整配方和语义，保留原训练 JSON、后端、物理、奖励、reset 与 target。CPU `_cpu_reference` 以实际源码读取全部语义并通过。v4 指纹 `33147dbd7cc7eadfad0da524fa60d666fd53ee18ff84cc1d546bfe424c44ddb9`。原 GPU3、单实例、32 控制步、258 SDK 步、3072 求解器子步、1 PPO 更新以及 180/240 秒上限保持。

118 项 CPU 回归通过。v4 仅冻结和预检通过，原生启动结果独立报告。
