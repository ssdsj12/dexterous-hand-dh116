# MPC 371D/24D 同步采集 v01

基线 `b2f5d84ff5e3f9d87d5cd24ba55a3f4edb6f8ca6`。原生采集于 2026-11-12 04:35:34–04:35:58 UTC 运行，外层退出码为 2。在首次物理 reset 后，采集器调用 `torch.isclose` 时遇到 `RuntimeError: Float did not match Double`。尚未执行首个 MPC 策略动作，本轮产生 0 条已连接的 371D/24D 样本，0 条合格专家轨迹。

失败输出保存在 `/mnt/mydisk/xk/dh116/runs/wp3_mpc_privileged_sync32_v01`。外层证据目录为 `/tmp/dh116-mpc-sync32-execution-20261112-1I06th`，`launcher.log` SHA-256 为 `877b4d6b6de954e72423b3ba022801923077389ac82e1d0a1dde93c6732e5589`。

`PreActionCapture.observe` 按原公式以物理测量精度归一化 q/qd，然后转换到导出的观测 dtype，再比较观测的前 24 个字段。原 371D 观测、动作、控制器、有效性容差和物理测量均保持原值。CPU 回归同时覆盖真实的 float64 测量与 float32 观测，以及数值不一致时仅标记对应字段无效。

CPU 复现及回归使用 Ubuntu 的 `/mnt/mydisk/xk/envs/dh116/bin/python`、`PYTHONPATH=src` 和 `PYTHONDONTWRITEBYTECODE=1`：

- 修复前运行 `-m pytest -q -p no:cacheprovider tests/test_mpc_privileged_capture.py -k float64`，1 项失败，复现原生报错。日志 SHA-256 为 `23ab7272cc79cd22df84ca9a5a410917ea59d52baddcad6392319832c493b299`。
- 修复后运行 `-m pytest -q -p no:cacheprovider tests/test_mpc_privileged_capture.py tests/test_dh116_privileged_action_audit.py`，30 项通过。日志 SHA-256 为 `9787269f1ac461ca27c5356fbd54098718ed6c500ef9edddf80c01bbfeb7bbca`。

下一步是在独立输出目录复验 2 环境 × 32 策略步，核验实际下发动作与采集记录逐值一致、物理时钟连续、字段有效性以及外层退出状态。此短采集仅用于数据链路诊断，完整原生成功证明仍是 Teacher Actor-only BC 的准入条件。
