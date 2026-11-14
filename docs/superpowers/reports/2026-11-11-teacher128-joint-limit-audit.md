# Teacher128 关节越限离线审计与逐子步采集

以远端 `240be7ccef37f2cbce705475692716de6f10ef5f` 的 Teacher128 历史证据为基线。此次重读原始数据并运行 CPU 测试。新增 480 Hz 全关节诊断尚待独立 Isaac 原生复验。

## 实际越限

在保存的 **2,030 条存活子步、256 条策略转移**中，五次 `joint_limit` 终止均可见右手 `finger12` 跨过下界。下表使用同 run 的 PhysX 场景读回，`finger12` 下界为 0.007912622765 rad、上界为 1.029999971390 rad，硬保护容差为 0.020 rad，对应下保护线 −0.012087377235 rad。场景摘要保存每手第一个实例的限位，历史子步只保存主动关节，故这里只确认可见的主动关节越线，未排除未保存的被动关节或腕部同步越线。

| 抓法 / env / reset | Policy tick / 物理 step | 仿真时间 s | 前一子步 q rad | 越限 q rad | SDK 目标 rad | qd rad/s | 实发动作 [7] |
|---|---:|---:|---:|---:|---:|---:|---:|
| 二指 / 0 / 1 | 31 / 250 | 0.558333362 | −0.011715761 | −0.012467229 | 0.007942013 | +0.031741 | +0.023513 |
| 三指 / 1 / 1 | 37 / 301 | 0.664583368 | −0.011568955 | −0.012149413 | 0.027141208 | −0.394778 | +0.014323 |
| 二指 / 0 / 2 | 67 / 540 | 1.162500061 | −0.011954968 | −0.012343497 | 0.007912622 | +0.135775 | −0.081871 |
| 三指 / 1 / 2 | 97 / 784 | 1.670833420 | −0.011815133 | −0.012659940 | 0.032001730 | −0.616744 | +0.048209 |
| 二指 / 0 / 3 | 104 / 835 | 1.777083426 | −0.012017779 | −0.012223782 | 0.007964174 | +0.155173 | +0.027495 |

每条 `episode_uid` 为 `c9744657-702d-450f-8c37-d282ca98e569:{env}:{reset}`。每次右 `finger12` 的首次可见硬保护越线恰好是终止子步。其动作索引是 `right_fingers[1]`，即 24D 物理合同索引 7。四次动作 [7] 为正、一次为负。SDK 目标与控制器目标的最大绝对差约 1.80e-9 rad，第 540 步的 SDK 目标比名义下界低 9.31e-10 rad，为浮点边界差。所有五个终止的掌心投影前后目标一致，投影原因和压力限制均未触发。两次三指终止的 `finger12` 速度超过了该场景的 0.3 rad/s 主动执行器配置。当前有限速度保护按既有合同记录并惩罚超速，实际位置跨界触发终止。

这将已观察到的失败定位到合法目标下的物理位置跟踪越界。历史证据无法确认被动耦合、接触冲量、腕部联合运动或其他力学因素的因果贡献。原生成功仍为 0。

## 代码路径

- `tabletop_privileged_cli.py::train_teacher` 沿用现有 `--record-diagnostic-rollout` 有界开关，在采样后、原始动作送入物理前绑定现有 24D latent mean、std 与 sampled latent。开关仍限 2 环境、32 或 128 步、1 update。复用已采样张量，不额外抽 RNG。
- `NativeTensorBridge.set_teacher_action_diagnostics` 核对 Tanh Gaussian 一致性，沿用原 checkpoint、calibration、frame、episode、policy tick 和 issued-action 对齐。其 `_capture` 通过 `EvaluationTelemetry` 在每个存活的 480 Hz 子步、reset 前保存全手数据。已有终止保存与后续 no-op 排除路径不变。
- `tabletop_privileged_action_audit.py::TeacherJointActionAudit.capture` 保存双手各 articulation 关节 SDK q、qd、运行时保护边界、位置与速度余量、可验证的 SDK 目标和主动控制目标。米与弧度单独标注，未驱动的被动关节不填目标。手指按实际 active IDs 绑定索引，开启自由腕时按真实 mount 的 `angular_ids` 绑定内禀 XYZ 索引与控制器角度。腕部平移分别保存控制器世界坐标目标向量和局部 SDK 关节目标，局部平移关节仅绑定语义组。速度上限由首个审计子步的 `get_dof_max_velocities` 逐手读回并缓存，缺失保持空值。
- `classify_joint_evidence` 区分硬位置越界、可见速度越限、目标超界、掌心控制投影与未知 `joint_limit`。没有可归因的被动接触或奇异性来源时保留空值。速度、目标、投影可作为同子步证据；只有对应真实保护触发时才可作为终止原因。没有改动原关节范围、噪声、控制器、物理、奖励或成功标准。

## 复核与证据来源

执行 `python -m pytest -q -p no:cacheprovider tests/test_dh116_privileged_action_audit.py tests/test_dh116_privileged_teacher.py tests/test_dh116_privileged_cli.py tests/test_dh116_stereo_runtime_teacher_identity.py tests/test_dh116_teacher_wp3.py`，**87 项通过，6.02 秒，退出码 0**。新增测试复现后覆盖位置与速度归因分离、目标及投影的并发证据、跨 reset 的首个事件、历史策略身份与实际动作核对、CPU bridge 的 8 子步完整采集与腕部动作映射。测试 fixture 不计入原生结果。

使用下列命令仅复读上一轮原生证据，输出路径必须是新文件。

```text
python -m go2_pvcnn.control.dh116.tabletop_privileged_action_audit --raw <run>/native/native_physics.jsonl --scene-summary <run>/native/scene/summary.json --rollout <run>/diagnostic_rollout.pt --output <new-private-output>.json
```

| 输入 | SHA256 |
|---|---|
| `native_physics.jsonl` | `470e8c5d0f24b2cf97c1664ea26dbd566aa58969677d0b3f64660ff22e2d6ffc` |
| `scene/summary.json` | `9584dc3a28a64134fad30da2f24f3388c8c940f3c405eaf6f47a5f5d180427a2` |
| `diagnostic_rollout.pt` | `7de6e75586a8a8f4d673062f0f2c795cad451ddee24fa76781bd81dff971a4ab` |

## MPC 同步重采设计

历史 `all_executed_supervision.pt` 与 `successful_visual_supervision.pt` 分别是 10,559 × 3844 与 4,698 × 3844 观测，动作均为 24D。历史 `successful_sequence_context.pt` 的 q/qd/root/target/反馈仍不含同一 pre-action 的完整 371D 与逐维有效性，不能直接初始化特权 Actor。数据与成功路径保留原样。

1. 冻结既有成功 Teacher + MPC 的 checkpoint、控制器、reset、物理配置和 action/state 合同 SHA，使用单 Isaac 实例和独立输出。先取得该原生采集的独立批准。
2. 在既有采集器的 `env.step(issued.clone())` 前扩展独立旁路采集。先读取真实 reset 后状态和统一 pre-action 时钟，然后记录现有 `task.observe()` 的完整 371D 及逐字段 validity 来源。读取同步双目时只 render，核对前后物理步、episode 和 reset 不变。记录 checkpoint/标定/状态合同/动作合同 SHA、frame、初态 SHA、policy tick 和 physics tick。
3. 在原 MPC 计算及调度完成后保存**实际发出**的 canonical 24D，同时区分参考动作与 MPC 修改后的动作。动作送入旧控制器后，使用 pre-reset 480 Hz 证据核对同一物理转移与 terminal。拒绝缺维、错钟、恢复、超时、帧/动作身份错配和无完整物理验证的样本。
4. 将不完整与失败轨迹留在诊断池。只有单独完成原生 30 mm / 5 s / 0.5 mm、对向接触和无外部承托审核的整段成功轨迹才进入现有成功池校验。二指回归与三指结果分别记录。以物理初态身份分组做后续验证。

本轮正式 BC、DAgger 与 Student PPO 仍处于 blocked。下一步是在获得独立批准后复验 480 Hz 全关节采集和同一右指终止路径。
