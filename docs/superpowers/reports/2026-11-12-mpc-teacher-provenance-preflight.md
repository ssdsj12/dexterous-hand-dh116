# 原 T5 MPC 到 Teacher BC 的证据预检

新增只读 Ubuntu CPU 入口，检查原生运行的三种首回合分区、同步账本实际 371D 有效位、现有原始日志的首条格式、外层退出记录和独立原生证明。可传入真实逐回合 proof 和目标 Teacher identity 进行现有 promotion 门控，随后检查成功同步账本也由同一外层进程记录，并重做 480 Hz 下发动作、时间和身份的对齐。入口只生成审计回执，专家装配和训练保持独立。

```bash
bash scripts/python.sh scripts/audit_dh116_mpc_teacher_proof.py \
  --run "$native_mpc_run" --observer "$external_run_logs" --output "$new_audit"
```

## 原产物与正式资格之间的缺口

1. T5 当前进程为原 constrained MPC。实际逐物理步观测来自 legacy MPC 记录器的 `_ObserverCapture.observe`。它产生 gzip `states_480hz.jsonl.gz` 和传感账本，原样标注 `legacy_teacher_mpc`。`teacher_native_physics.jsonl` 的记录是每策略步聚合，不提供完整同回合逐 480 Hz hold replay 原始字段。原任务成功和专家分区可用于诊断。
2. 当前正式原生验证器要求真实外层 `dh116-external-native-completion-v1`、退出码 0、完整子进程命令和 stdout/stderr、原生执行期间冻结源码，以及在退出后核验的原始文件与配置产物 SHA。现有 T5 supervisor 的单独 exit_code 是原外层观测，不能替代正式完成收据。现有正式验证器仅认可其既有 runner 源，独立导入 MPC runner 需单独审核源和实际调用关系。
3. 当前正式非 stereo episode producer 仅认可 `tabletop_stereo_native.py` 的 `NativeTensorBridge._capture`，并要求运行期间真实产出配置绑定 `resolved_kwargs` 480 Hz／60 Hz 和当前 action/state SHA。需要真实共享原生物理回调或独立审核的新 MPC producer 版本。当前归档不能通过修改 owner、重命名或重新包装日志获得该证据。
4. 每个成功原生首回合需要实际 reset readback、完整物理时钟、每个原始子步的标准化 MPC owner、实际 issued 24D、终止原始行与五秒 hold replay 源。独立 force proof 仍须逐条对齐同 run、env、episode、reset、physics step、时间与逐指对象侧法向/摩擦力，证明外部支撑、外加力、附着以及 partner 归属。仍使用 30 mm、5 s、0.5 mm 门槛。接触校准或附近回合的样本无法补齐当前回合。
5. 对准后，同一外层进程归档的完整成功 371D/24D 同步源才可进入专家装配；原 371D Actor 无 mask，因此所有输入字段均需可核验。独立真实 force/support proof 尚未取得，正式专家成功数仍待验证。

带真实拟议证明时，为每个 episode 提供 `--episode-proof PATH SHA256` 并提供 `--model-identity`、`--model-identity-sha256`。只读预检沿用既有 native verifier，任何不支持的 producer、外层收据、力学来源或终止证据都会留下失败理由。

本次真实运行只读审计在 T5 v01 的 780 控制步时读取到三种日志的第一条记录，原生任务成功数当时为 0，外层退出和完整专家分区尚未产生，0 回合可装配。私有审计 SHA256 `0336fa744f61179778152be329c05da203d75fa135d65336bcb2fc6d96ee70df`。Ubuntu CPU 回归 41 通过，0 失败，日志 SHA256 `dcc0fd8c7bc04fe5db689ff15449cb4a6a3ef13b34f6109b460ec377c1338f44`。本包只读取运行证据，未启动 Isaac。
