# Teacher128 同状态动作与跟踪越限复盘

基线 `98ba1fb872636459c13212ae97425fbb8b7c56c5`。本轮只读既有原生轨迹和 Checkpoint，运行 CPU 回归及推理。Teacher 原生成功记录仍为 0。原关节保护、物理、动作分布、Actor-only BC 和成功标准沿用现有配置。

## 同状态对照

扩展 `tabletop_privileged_action_audit.py`。读取冻结的 step-major 371D rollout 后，逐个对齐同一 env、episode/reset、frame、pre-action 物理时钟、标定、初态、任务、Checkpoint 和 policy 版本。每个策略状态至少有一条对应的 480 Hz 记录，随机采样和实际发出的 24D 必须与账本一致，终止标记必须双向一致。使用 Checkpoint SHA 核对采集时的权重，在 CPU 重新计算 deterministic mean 与 latent std，保留原采样，不重新抽噪声。诊断入口沿用原 `--record-diagnostic-rollout`，未修改 `tabletop_privileged_teacher.py` 或训练 CLI，因此旧 Checkpoint 的源文件身份可继续使用。

离线复盘对齐 **2,030 个有效物理子步、256 个策略状态、2 组初态**，其中 5 个状态以 `joint_limit` 结束。Windows CPU 重新推理与记录的 latent mean 最大绝对差 `2.682e-7`，latent std 最大差 `7.451e-9`。服务器 CPU 分别为 `3.576e-7`、`7.451e-9`。

| 样本范围 | 24D sample–mean MAE | 24D P95 | 右 `finger12` sample–mean MAE |
|---|---:|---:|---:|
| 全部 256 个策略状态 | 0.077135 | 0.188580 | 0.074779 |
| 五次越限对应状态 | 0.071274 | 0.178465 | 0.040028 |

| 物理 tick / policy tick | 右 `finger12` sample [7] | 同状态 mean [7] | sample latent z-score | q−SDK target rad | qd rad/s |
|---|---:|---:|---:|---:|---:|
| 250 / 31 | +0.023513 | −0.079146 | +1.02829 | −0.02040924 | +0.031741 |
| 301 / 37 | +0.014323 | −0.001863 | +0.16186 | −0.03929062 | −0.394778 |
| 540 / 67 | −0.081871 | −0.071595 | −0.10337 | −0.02025612 | +0.135775 |
| 784 / 97 | +0.048209 | +0.065731 | −0.17580 | −0.04466167 | −0.616744 |
| 835 / 104 | +0.027495 | −0.026001 | +0.53508 | −0.02018796 | +0.155173 |

五个终止的该维 latent std 都为 `0.1`。三次 sample 和 mean 异号，终止样本没有一致的噪声方向，单维偏差的绝对值至多 1.029 个 std。所有五次的归一化 applied-action 向量都未记录，报告保留 NA，并用实测 q、qd、控制器命令和 SDK target 表示实际执行。mean 是同状态的另一项输出，未提交给物理，所以反事实终止、成功与速度均为 NA。

## 可确认的失效与待验证因果

逐关节摘要保留第一次 guard 越线、同关节上一条读数和终止读数，避免把另一只手较早的越线当作终止关节的首次越线。记录限位侧、保护余量、`q − SDK target`、`SDK target − controller target`，并区分主动、被动和腕部关节。被动关节的位置越线独立记录，其接触归因仍要求合格物理证据。

五次右 `finger12` 的首次可见硬越线都发生在终止 tick。下保护线约 `−0.012087377 rad`，上一物理子步均在保护线以上。SDK 与控制器目标差最大约 `1.8e-9 rad`，目标本身在名义限位的浮点容差内。两次三指终止伴随速度超出该场景的 `0.3 rad/s` 主动执行器配置，三次二指终止未超此速度。历史数据支持**合法目标下实测位置跟踪越界**这一直接原因。旧账本只保存主动关节，无法复原被动关节或腕部联合状态。接触来源未资格化，不能把接触冲量、被动耦合、速度超限或噪声说成五次终止的共同物理根因。

## MPC 371D/24D 核验

服务器从原始文件重新读取并核对 SHA。两个归档的动作语义匹配当前 canonical 24D 合同，只有 3844D RGB-D 历史。成功子集来自完整归档，不作为第二组独立验证。下表动作差仅比较各 MPC 旧轨迹自身的已发动作与同记录 reference。

| 归档 | 观测 / 动作 | 旧轨迹 MPC–reference 24D MAE | 当前 Teacher 同状态配对 |
|---|---|---:|---:|
| `all_executed_supervision.pt` | 10,559 × 3844 / 10,559 × 24 | 0.003199 | 0 |
| `successful_visual_supervision.pt` | 4,698 × 3844 / 4,698 × 24 | 0.000080 | 0 |

`successful_sequence_context.pt` SHA `8cd461914c21fb53460f16f9de635011dd8496be181f844d57e7c7e5d6ca007e`，保存 controller、feedback、command cache 和原始测量，仍缺完整同状态 371D 与逐维有效性。比较器要求候选 MPC 状态的合同、全部有效性和真实 pre-action 身份匹配，且 371D 数值逐项一致。现有资料无可配对样本，MPC 对 Teacher mean/sample 的同状态差与反事实物理结果均为 NA，Actor-only BC 继续阻断。下一步按已写好的 MPC 同步重采方案申请独立原生执行，并用已有有界全 articulation 审计覆盖主动与被动关节、实际限位、控制器目标和 SDK 目标，再评估控制、噪声或 BC 干预。

## 复验

Windows CPU 回归 **94 项通过，0 失败，6.05 秒**。服务器 CPU 复盘读取旧归档，进程退出 0，没有启动 Isaac、采集或训练。新增测试覆盖首次越线关节区分、被动越限与接触归因分开、Checkpoint 字节核对、对齐失败、同状态 CPU mean 重算不扰动 RNG 和 3844D 数据不能通过 MPC 配对。

```text
python -m pytest -q -p no:cacheprovider tests/test_dh116_privileged_action_audit.py tests/test_dh116_privileged_teacher.py tests/test_dh116_privileged_cli.py tests/test_dh116_stereo_runtime_teacher_identity.py tests/test_dh116_teacher_wp3.py
python -m go2_pvcnn.control.dh116.tabletop_privileged_action_audit --raw <run>/native/native_physics.jsonl --scene-summary <run>/native/scene/summary.json --rollout <run>/diagnostic_rollout.pt --teacher-checkpoint <run>/teacher_update_000000.pt --mpc <archive>/all_executed_supervision.pt --mpc <archive>/successful_visual_supervision.pt --output <new-private-output>.json
```

| 证据 | SHA256 |
|---|---|
| 原生 480 Hz JSONL | `470e8c5d0f24b2cf97c1664ea26dbd566aa58969677d0b3f64660ff22e2d6ffc` |
| Scene summary | `9584dc3a28a64134fad30da2f24f3388c8c940f3c405eaf6f47a5f5d180427a2` |
| 371D diagnostic rollout | `7de6e75586a8a8f4d673062f0f2c795cad451ddee24fa76781bd81dff971a4ab` |
| 初始 Teacher Checkpoint | `ec5831ac4de039e5e77e2ee4a3420497a0c103a2a5966a1a73662d4997966511` |
| 全量 MPC 归档 | `b498ae8df9b80af8c5c431f4494233841cb400181de56ee01e05062581cc8cad` |
| 成功 MPC 子集 | `664f30456a86a073a734be97ef56819787cf3c2664dd182ffdf667a32e0ec026` |
| Windows 最终只读复盘摘要 | `e939b1886b9a60ca418ed12087542ce0215c19c704b4a1e15bf6dbc698d5f224` |
