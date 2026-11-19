# DH116 Pinch Teacher 诊断回执复验

Antigravity 就联合搜索与 Pinch 回执完成三轮只读审查。独立核对确认，`read_launch_binding()` 验证活动父进程、内核锁与套接字，已经退出的作业需要单独的读回接口。通用 `validate_native_artifacts()` 面向另一种证据合同，实际 Pinch 使用现有 `audit_actual_fresh_outputs()`。

实际 A-fresh worker 握手完成后，调度器把 intent、父绑定、grant 及独立外层退出文件的 SHA-256 写入诊断回执。读回接口只从已登记包、当前配方指纹与外部提供的精确回执 SHA 查找历史记录，逐项核验 canonical 绑定、grant 中的配置和资源引用、外层自然退出、共享锁、GDB 状态、日志和原始输出审计。工作树上的无关后续提交不影响仍使用同一配方的旧记录。审计保留 `native_qualification=false` 和 `teacher_admission=false`，依赖晋级接口继续关闭。

与运行环境一致的只读命令如下。首次执行需要新的实际调度产物，给出原始 scheduler receipt 的 SHA-256。

```bash
scripts/python.sh scripts/run_dh116_work_packages.py --verify-actual-diagnostic PACKAGE --receipt-sha256 SHA256 --evidence-dir PRIVATE_DIRECTORY
```

196 项 CPU 回归通过。新增 10 项使用真实 CPU 父子进程与 GDB 自然退出，夹具出口和输出审计使用测试数据。覆盖后续 HEAD、回执 SHA、intent、父绑定、grant、进程配置、退出记录、日志、输出审计值变化和尝试修改原生资格标记。未运行 Isaac 或实际 PPO，未获得实际训练回执。下一步为冻结当前计划，在单实例锁释放且对应预算生效后采集真实 Pinch 输出。
