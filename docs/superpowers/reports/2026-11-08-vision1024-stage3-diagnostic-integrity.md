# 视觉 Student 第三阶段诊断完整性报告

## 完整性修复

480Hz `PhysicsActionDiagnosticWriter` 现在：

- 保存 `env_id/episode_id/episode_uid/policy_step/physics_step/timestamp_s`。
- 对每个物理子步检查 `physics_step` 和时间戳单调性，乱序直接拒绝。
- terminal样本立即写入，后续同episode no-op只标记 `post_terminal_noop`。
- 外部完整 reset 开始新的 `reset_epoch`，生成新的 `episode_uid`，不会复用旧terminal键。

## 验证

- `tests/test_vision_action_diagnostics.py`：**10 passed**。
- 覆盖terminal样本优先级、后续no-op标记、重复reset唯一 `episode_uid`、非单调物理时间/步数拒绝。
- 短原生 M0/M2 各产生 262144 条物理记录。复核原始 JSONL 后更正计数：M0 无终止，M2 有两个三指终止，环境 932 在物理步 252，环境 799 在物理步 256。前者的后续四个子步保留 no-op 标记。
- 相关模块远程 `py_compile` 通过。

原始训练、奖励、动作合同、物理参数和成功标准均未修改。
