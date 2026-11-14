# DH116 TASK 4 Teacher PPO 稳定性

本轮基线为 `085f6b8d96891b7c975c6382412e629d18e896b3`。诊断使用 R0 及原 371D 状态、24D tanh 动作。原训练配置仍沿用固定 minibatch 预算。新增两个独立诊断配置，均保留原始学习率 `3e-4`，目标 KL 为 `0.02`。

每个 minibatch 记录与冻结采样策略的 exact KL、采样 log-ratio 估计、熵、PPO probability-ratio clipping 比例、梯度范数及实际样本数。原生收集另记录 24D 分组 std、tanh saturation 和归一化动作 clipping。已发动作与有界 tanh 样本相等，超出区间会被拒绝。原始 480Hz 行保存控制器关节命令、SDK 目标、pressure-limited 掩码及同一物理时钟的手掌安全投影前后值。关节控制器没有同一子步的未约束目标读数，`joint_safety_projection` 记为 null。

Teacher 使用任务结束信号屏蔽 GAE bootstrap。任务超时属于此环境的终止原因，终止位置后的 reset 观测不参与 bootstrap。连续 PPO 更新换策略时，`teacher_policy_epochs.jsonl` 追加每个实际回合、每个策略版本首次物理样本的 checkpoint SHA 和 pre-action identity。原有 episode_start 及物理 reset ID 保持其原始意义。CPU 测试覆盖同一回合内两次策略绑定及 checkpoint policy、Adam、计数和 RNG 的恢复。

## 首次有界原生诊断

配置 `privileged_teacher_ppo_kl_diagnostic_v1.json` 采集 2 环境 × 32 个控制步并尝试 1 次更新。64 行 on-policy 样本对应 506 条有效物理记录。二指环境在 physics 250、policy tick 31 以 `joint_limit` 终止。其余 6 个物理子步为 no-op，未计入。三指环境有 256 条物理记录。冻结策略最大 log probability 误差约 `7.63e-6`。

第一次 Adam 更新的 minibatch exact KL 约 5.806，全 64 行 exact KL 约 5.845。early stopping 使剩余 39 个 minibatch 未执行，但已接受首次过大的步长。该结果触发后续回滚保护。

## 保存原生 rollout 的优化器对照

只在 RTX 4090 上运行优化器，使用同一个初始 checkpoint 的 policy 和 Adam，并在每个实验开始时恢复同一 pre-update RNG。64 行全量实验均检查保存的 GAE returns 和 advantages。所有行均重新验证冻结分布、动作与 log probability。每组最终 KL 都在同一 64 行原生状态上评价。

| 优化数据 | 学习率 | 实际 Adam 步数 | 最终 KL |
|---|---:|---:|---:|
| 前 8 行 | 3e-4 | 40 | 14.193817 |
| 前 8 行 | 3e-5 | 40 | 2.914333 |
| 全 64 行 | 3e-4 | 40 | 23.140474 |
| 全 64 行 | 3e-5 | 40 | 0.615673 |
| 全 64 行，minibatch 为单样本 | 3e-4 | 320 | 16.683876 |
| 全 64 行，首次越过 0.02 后停止 | 3e-4 | 1 | 5.845236 |

在全 64 行、原学习率对照中，KL 的均值变化项为 23.137651，std 变化项为 0.002823。首次越过阈值的对照也由均值变化主导。单靠增加本批次的 rollout 样本数未消除 KL 异常。全 64 行固定样本时降低学习率可明显降低 KL，但 40 步后的 KL 仍高于目标。前缀会改变 GAE 计算范围，单样本 minibatch 对照会同时改变每轮 Adam 步数，因此不能据这些组单独归因 minibatch 大小。冻结检查支持采样与重算一致，不能据此评价任何其他任务的 log probability。

## 回滚保护

`privileged_teacher_ppo_kl_guarded_diagnostic_v1.json` 保持目标 0.02。每个候选 Adam 步在全部冻结 rollout 上核对 exact KL。候选越界就恢复 policy 和完整 Adam 状态，学习率乘 0.1 后重试，最多三次回退。每次候选及最终接受的学习率都留在 trace。所有候选越界则结束该更新余下 minibatch，若没有任何接受步，保留初始 checkpoint 和版本计数并返回 `teacher_optimizer_update_rejected`。

CPU 回归先复现缺少目标、trace、epoch 与诊断产物的 10 项失败，再验证优化、GAE、身份及 indexed Student PPO。回滚用例另先复现 3 项失败。新增用例验证已有 Adam moments 和 step 计数在拒绝后逐张量相等。原生、CPU 和保存原生 rollout 的 CUDA 重放互相独立。

独立的第二次 2 环境 × 32 步 × 1 更新原生诊断退出码为 0。首个 minibatch 的三次候选全 rollout KL 分别为 5.845236、0.0588464、0.000590264，接受的步长是 `3e-6`。总计接受 21 个 minibatch、优化 168 个样本次，其中 16 步使用 `3e-6`，5 步使用 `3e-7`。第 22 个 minibatch 的四个候选全部越界，末候选 KL 为 0.02002028，回滚后结束该更新。最终冻结 64 行状态上的 exact KL 为 **0.01997227**。

checkpoint `3392e30c7ccfd16153c919219f68303de03c19fa23c09804e7e51e4e9970b63d` 回读时 policy、Adam、计数与策略版本均一致。终止记录 1 条，bootstrap 计数为 0。R0 回报差值为 0，归一化动作 clipping 比例为 0，抓取成功 0。验证覆盖一次有界更新。跨更新策略事件由 CPU 同回合两次绑定测试验证。

最终 TASK 4 CPU 测试执行 `python -m pytest -q -p no:cacheprovider tests/test_dh116_privileged_teacher.py tests/test_dh116_privileged_cli.py tests/test_dh116_stereo_runtime_teacher_identity.py`，73 项通过。旧 checkpoint 的严格源码身份检查仍生效，旧文件由其对应源码版本读取。新源码的保存与恢复使用本轮的 checkpoint。
