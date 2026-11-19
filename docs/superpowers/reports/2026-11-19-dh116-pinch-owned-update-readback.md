# DH116 Pinch 更新副本复核

Antigravity 对 `11-17` 上的 `be7f9c66cb298ebbf8086f790761ffc04a225cd6` 完成两轮只读审查。Codex 对照 worker 与 artifact auditor，确认 `owned_update.json` 是 `run_a` 最后单独写入的诊断副本。目前 scheduler 已复核原 R06 生命周期与全部派生输出。该副本此前不参与其产物验证。

新增独立只读入口 `scripts/audit_dh116_pinch_owned_update.py`。它先使用已有 `verify_actual_diagnostic()` 重新核对确切外层回执，再以同一回执固定的生命周期结果 SHA-256 核对 `owned_update.json`。缺文件、不同源哈希、训练字段篡改及将布尔值冒充计数均拒绝。该入口不改变现有原生资格或续跑接口。

CPU 独立回归 **89 项**全部通过，新增六项覆盖正例、四个篡改维度、带回执顺序的读取与缺文件。没有改动冻结的 Pinch A 配方或其源码闭包。仍未在检查的证据目录找到首次真实 PPO 尝试。原生计划为 `waiting_approval`，Isaac 单实例锁仍被占用。本轮无 Isaac 或 PPO 运行。

原生首诊有回执后，按实际私有 evidence-dir 调用 `scripts/python.sh scripts/audit_dh116_pinch_owned_update.py --evidence-dir DIRECTORY --receipt-sha256 SHA256`。产出的复验仍为范围受限的诊断证据。
