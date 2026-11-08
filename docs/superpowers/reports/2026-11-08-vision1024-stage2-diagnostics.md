# 视觉 Student 右掌偏置第二阶段诊断

## 范围与约束

本阶段没有修改奖励、动作幅度、Actor结构、物理参数、执行器限制或历史 Checkpoint，也没有启动长时间 PPO。所有诊断默认关闭；短原生验证使用独立输出目录。

保留 3844D Actor、371D Critic、24D Tanh Gaussian 动作合同。新增两层诊断：

- 60Hz policy 层：24D action、pre-tanh latent、latent mean、每维 `log_std`、`log_prob`、value、advantage、returns、reward、done、全部 reward terms，并绑定 env/episode/policy_step/timestamp。
- 480Hz physics 层：每个物理子步记录 env/episode/policy_step/physics_step/timestamp、动作切片、请求/投影/最终 SDK 目标、reset-relative 掌位/位移/速度、接触载荷、contact_valid、clearance、safe_z 和投影原因。terminal 子步立即写入，之后的 no-op 只标记 `post_terminal_noop`，不会覆盖 terminal 样本。

## M0/M1/M2 离线镜像消融

使用最终视觉 Checkpoint 的 1024 个初始观测做 CPU 确定性推理：

| 模式 | 处理 | 动作 L2 均值 | L2 P95 | 方向一致率 |
|---|---|---:|---:|---:|
| M0 | 原始输入 | 0.754994 | 0.881472 | 0.562500 |
| M1 | quaternion hemisphere 规范化 | 0.806492 | 0.905909 | 0.604167 |
| M2 | quaternion + 完整镜像 paired canonical frame | 0.296581 | 0.345446 | 0.833333 |

M2 明确标记 `offline_only=true`，没有接入部署路径，也没有改写旧 Checkpoint。镜像变换自身的双应用回环中，四帧历史、局部点云、validity、12D关节、12D速度、掌位、quaternion 和 previous action 的 `max_abs/mean_abs` 全部为0。结果说明单纯 quaternion 符号规范化不足，局部坐标/相对几何规范化是更值得继续核对的方向；它不能单独证明原生抓取成功。

## 短原生双层验证

独立目录：

`/mnt/mydisk/xk/dh116/runs/dh116_gpu3_stage2_dualdiag_1update_20261108T053021Z`

从最终 Checkpoint 做 1 次 PPO 更新，完成退出，原生 successes 为0：

- 60Hz `action_diagnostics.jsonl`：32768行（1024环境×32 policy steps）。
- 60Hz `policy_ppo_diagnostics.jsonl`：32768行，包含每维 `log_std`、latent、log probability、value、advantage、returns 和 reward terms。
- 480Hz `physics_action_diagnostics.jsonl`：262144行（1024×32×8物理子步）。
- 物理层完整行数符合每个policy step 8个480Hz子步的预期；本次短回放没有terminal事件，因此terminal保护计数为0，terminal/no-op逻辑由CPU测试覆盖。

本次短回放的策略统计：

- 左掌平移动作均值约 `[0.001860, 0.068902, -0.047112]`。
- 右掌平移动作均值约 `[0.072857, -0.111714, -0.289228]`。
- 全部 `log_std`、value、advantage、returns、reward terms 均已逐环境逐步保存；PPO summary 的均值 advantage 接近0，符合GAE归一化后的批次统计。
- 左右安全投影前后差值和投影比例均为0；控制 frame SDK 目标误差小于 `6e-16 m`。

## 结论与未决项

当前证据继续排除“安全投影压低右掌”作为该样本的解释。M2 明显降低镜像动作误差，但由于 M2 是离线 paired canonical frame，尚不能直接改部署预处理或旧 Checkpoint。下一步应使用这些完整双层数据做同初态反事实分析：固定左手输入、只替换右手镜像规范化，再比较右手动作、接触载荷和任务结果；仍不得拆分 Actor 或直接调右手奖励。

历史第513轮 Checkpoint不存在，不能补造历史峰值的同环境轨迹。原始 30mm/5s/0.5mm成功标准保持不变。
