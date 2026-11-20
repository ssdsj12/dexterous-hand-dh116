# Pinch v13 原生 Reset 验证

使用发布的 Ubuntu `11-17` HEAD `f1303cf1eac878a97497d6d3a950bce4e454d897`，GPU 2 单实例、单环境。命令为 `python -B -m scripts.record_dh116_teacher_numeric_reset`，采用经哈希核验的成对提案 `933acc2b669b3bef249e206752e6364ea30ae4263afce734e22c998054eba67d` 和冻结指纹 `86b4440ac6d80a23f76b11d53eeebf5c4bb0f9792cbdd6074facf4595f0fbd42`。完成 2 个 SDK 初始化步与 1 次 Reset。控制、杆求解器和 PPO 更新均为 0。

原生读取双手当前 link 变换，左/右源凸包台面最低净空 **2.337 / 2.653 mm**，锁定指腹凸包最低净空 **3.411 / 4.339 mm**。两侧原初始几何均通过。运行时 stored cooking 字段仍未覆盖全部碰撞体，源/锁定碰撞包络读回的资格保持原定义。

资源释放、缓存校验及 App 正常关闭均通过。外层自然退出 0，耗时 43.75 s，未重试或强制结束。独立退出回执 SHA `d8014a7464e43976e7c4e05dae5ca45cd8dbced3dd26092ca97ae636adff38f2`；日志 SHA `93f33315e55c67d02ca35e8f5df44bc2c2afa46393458487cfafa443575e6a36`。生产 Reset 校验器重读实际状态、身份与自由杆节点，通过后封存清单 SHA `fce817f5bebc8c4a9368bae1d9169ac702f40c4540847e94db17c520ae6672b8`。原始日志和模型保留在 Ubuntu 证据盘。

下一次使用新清单冻结独立有界 Teacher 诊断，并核验采集器源闭包和墙钟预算。新 Reset 尚未执行动作，接触、抬升、保持和 Teacher 资格均待验证。
