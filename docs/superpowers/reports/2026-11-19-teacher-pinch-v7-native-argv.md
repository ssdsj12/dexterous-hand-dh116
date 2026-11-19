# Pinch v7 冻结子进程签名失败

按用户批准的 `380a08af61036143ae4d39784a31e1a140aaef1bb4408994cb69584d0cf1ca9f` 登记一次。训练配置、静态预检、实际 Reset、原代码闭包、GPU 3、单实例锁、私有目录与磁盘预算均通过启动前检查。控制目录由运行用户持有且为 0700，批量输出在数据盘。

启动调用在解释器与调度器路径之间多加了 `-B`。`_verify_scheduler_parent` 读取 `/proc/<pid>/cmdline` 时按冻结合同拒绝。失败回执 SHA `294bb0630d0dd029d739acf9b95d2ce9788cb28d7ef0392f73a76acdee7b9e52`。Isaac 外层退出码 NA，SDK/求解/PPO 计数均为 0。Registry 标为 blocked。

真实 CPU 子进程回归复现了额外 `-B` 导致授权回复不生成。精确参数顺序与 `PYTHONDONTWRITEBYTECODE=1` 组合通过直接和 GDB 两条握手，其中 GDB 的 CPU inferior 自然退出 0。后续使用现有 `Repository.start_native` 生成参数。原 v7 尝试已经登记，新的冻结任务仍待原生批准。

第一次 CPU 组合检查在几何计算并行时有一项 3 秒夹具超时，保留日志。CPU 数值库限为单线程后复查，4 项通过、32 项未选中。该线程调整仅用于这一组 CPU 回归。
