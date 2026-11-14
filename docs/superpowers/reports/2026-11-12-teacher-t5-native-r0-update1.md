# 原 T5 合同上的特权 PPO Teacher 单更新原生验证

2026-11-12 06:25:18–06:26:15 UTC，在 Ubuntu GPU 3 单实例执行冻结基线 `aa36490ed228a52a3fd13a3e798759bf20c4b8a8`。现有 `solid_cylinder` 场景、T5 reset、传感设置和物理合同继续使用。策略为独立的 371D 特权 Teacher，随机 Tanh Gaussian 原生执行 24D，未调用 MPC 或加载专家数据。

| 指标 | 原生实测 |
|---|---:|
| 环境／控制步／完整 PPO 更新 | 2／32／1 |
| PPO transitions／有效 480 Hz 环境级物理样本 | 64／512 |
| Exact KL / target | 0.0085039584 / 0.02 |
| minibatch updates | 20 |
| KL Guard | early_stopped=true，最终 update accepted |
| R0 奖励闭合最大绝对误差 | 0.0，owner 为 native_env_once |
| 完整回合／原生成功 | 0／0 |
| 外层进程退出 | 0 |

第 1 次更新保存后，Checkpoint 的 policy、Adam、counter 和 policy version 与读回逐项相等。Callback 在应用关闭前写入并 fsync completion 和原生证据，外层退出另由独立 supervisor 核验。此短诊断授予 T5 合同上的 R0 执行证据。原生首回合成功、BC 专家资格及独立冻结评估需要另外完成。

| 产物 | SHA256 |
|---|---|
| 派生 native_task.json | `56cef9f2f811918a070fefd9dc766ca9626410ee02b4a118e49773fba43e4aba` |
| 实际 native_task_contract.json | `3b48a1a05cd2f86d9876b8d6c75d3d018d25f4bd4ea4656b42c6b8e168fcfd51` |
| 第 1 次更新 Checkpoint | `71ea209f6710db9edc9457e5ed3bfde8cecc9daede4e0dd5eff0167e905e6638` |
| iterations.jsonl | `b7834bb5f3fb4b21bfbbc715c09e66ac81b27596b2ebd044c094ef50b61c8312` |
| 480 Hz 原始物理账本 | `23355ad952df228c41904c503e3121ac0fdc47d6ebe0ceb27ad133e0ec923f4d` |
| Callback completion.json | `ab060d21db81eac67d1b73103b0d44a0b80bab638f5a693f3fdb1bc729f6de00` |

命令为 `bash scripts/python.sh scripts/train_dh116_privileged_teacher.py --config configs/dh116_training/privileged_teacher_ppo_guarded_v2.json --native-task-config "$derived" --num-envs 2 --rollout-steps 32 --updates 1 --checkpoint-interval 1 --record-diagnostic-rollout --output "$new_run" --enable-experiment --execute-native`。运行模型、日志和原始数据留在 Ubuntu 私有目录。CPU 回归与外层发布结果另见工作包清单。正式抓取保持 30 mm、5 s、0.5 mm 原生条件。
