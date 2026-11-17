# DH116 受限原生调度与恢复

调度器保留现有 CPU 检查和 Windows GitHub 中继。Registry v1 和 v2 均可读取。原 `gated` 包继续按其登记的门槛处理。新的 `native_bounded` 包使用明确命令文件、作用范围和当前批次授权。

## 配方和批准

原生 Registry 包使用 `requires_approval: true`、`resource_class: "isaac_single_gpu"`、精确 `evidence_scope` 和 `retry_policy: {"max_attempts_per_code_config": 1, "retry_requires_change": true}`。需要范围鉴定的依赖通过 `dependency_scopes` 显式声明，如 `{"flex.f3-isaac-rigid":{"scope":"isolated_single_rigid_bridge_240_step_lifecycle_only","native":true}}`。原生依赖必须有该范围、实际执行和外层退出 0。较早证据失效时会重新计算后继就绪性。

`command_spec` 必须指向 `configs/dh116_execution/*.json`。版本 1 配方字段如下。

- `entrypoint` 是受 Git 管理的 `scripts/*.py`。 `argv` 是参数列表，只有唯一的 `--output {output}` 可替换。
- `source_files` 声明执行脚本之外依赖的源码和配置。 `inputs` 可绑定绝对路径外部文件到实际 SHA256。脚本、包装器、调度器、关联测试、命令配方和源码摘要绑定批准。
- `budget` 包含 `gpu_index`、`timeout_seconds`、`min_gpu_free_mib`、`min_disk_free_bytes`、`physics_steps`、`environments`、`updates`。本轮限制是最多 240 个物理步、2 个环境、1 次更新。配方仍须通过原生命令参数落实明确步数或更新预算。零物理步适用于纯渲染。 `environment` 只允许明确的独立库路径与数值线程数。
- `validator` 声明 `result_file`、`equals`、`artifacts`、`line_counts`。执行后独立重开文件，对指定结果字段作类型与值比较，逐行检查完整的 JSON 对象、精确计数和真实文件 SHA256。物理任务须绑定相应日志行数。配方应要求本领域检查结果，例如力与反力或关闭回执，而不是只要求 producer 的通用完成标记。

`--plan --scope all` 输出候选包的两个 SHA256。收到对应范围授权后，可用以下私有 JSON 记录该批次。输出根必须与源码分开。

```json
{
  "version": 1,
  "batch_id": "approved-contact",
  "output_root": "/absolute/private/native-output",
  "packages": {
    "package-id": {
      "command_spec_sha256": "<exact 64 hexadecimal digits>",
      "code_config_sha256": "<exact 64 hexadecimal digits>"
    }
  }
}
```

```bash
python scripts/run_dh116_work_packages.py --plan --scope all --native-approval "$private_batch"
python scripts/run_dh116_work_packages.py --run --scope all --limit 2 --native-approval "$private_batch"
```

## 执行、失败和恢复

`native_ready`、`blockers` 和 `native_results` 分别显示就绪、阻断原因与私有待发布回执。资源受限时 CPU 仍可执行。每个私有 intent 单独加锁，调度器独立预约锁串行化原生 worker。启动前检查既有全局 Isaac 锁、存活进程、目标 GPU 与磁盘。实际原生入口按项目约定持有既有 `native_instance_lock`，worker 从 Linux `/proc/locks` 验证锁所有者属于本次命令进程组。调度器不抢占该锁，入口可以正常获取，也不依赖继承描述符。

启动使用固定 `bash scripts/python.sh`、明确 argv 和指定 GPU。私有 intent、running、outer-exit 和独立 validator 结果先于发布。外层退出 139 时即使内部文件完整也没有原生资格。同代码与配置的失败回执禁止自动重试。普通资源竞争未启动原生，不消耗一次执行。上游调度或发布中断后读取既有回执和本地事务，不重跑已完成物理。墙钟耗尽时写 overdue 并结束调度等待，保留实际子进程及其预约直至自然退出。调度器不发送终止信号，超时回执不具备资格。待发布收据、GPU 路径与完整原始输出留在私有目录，Git 只记录范围、哈希和判定结果。

## CPU 验证

`bash scripts/python.sh -m pytest -q -p no:cacheprovider tests/test_dh116_work_packages.py`，39 项通过。新增测试实际启动 CPU 子进程模拟退出 139 与超预算，不计为原生。Linux flock 所有者测试使用真实子进程。覆盖前置范围变动、授权、资源冲突、独立 JSONL 校验、相同代码不可重复、CPU 并行进度、发布失败和恢复。此工作包未启动 Isaac 或训练，真正原生包须另行登记已审计配方。
