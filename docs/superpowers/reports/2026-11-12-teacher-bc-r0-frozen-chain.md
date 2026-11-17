# 特权 Teacher 的 Actor-only BC、R0 PPO 与冻结评估

## 可执行入口

`scripts/warm_start_dh116_privileged_teacher.py` 在 Ubuntu CPU 上读取有 SHA256 绑定的原生任务合同和专家数据。沿用现有完整首回合原生成功验证器，重新打开每个 episode 的同步源账本，逐值比对 371D pre-action state 和 24D issued action，检查 policy tick 连续性与逐字段有效性。现有 Actor 直接使用全部 371 维，因此缺任意输入字段的样本无法通过。更新只使用 Actor 参数，BC Adam 单独归档，critic、log_std、PPO Adam 和更新计数保持初始状态。生成的 `teacher_bc_actor_warm_start.pt` 使用原 PPO Checkpoint 合同，模型为 update 0。

`scripts/train_dh116_privileged_teacher.py` 沿用现有 `--checkpoint` 加载这个 Checkpoint，执行 KL-guarded R0 和原单实例入口。实际场景、源文件和训练配置仍须通过原严格身份比对。相同 rollout 和环境数用于严格恢复。2 环境 ×32/128 的原生诊断入口保留。

`scripts/prepare_dh116_privileged_teacher_evaluation.py` 验证指定 Checkpoint 的 SHA、PPO 更新数、完整 rollout 计数、Adam、模型拓扑和真实原生任务合同，随后复用 cold reset 模板生成至少 100 个二指和 100 个三指的测试初态。这个 CPU 预处理只生成待运行清单。通过 `evaluate_dh116_stereo_trials.py --trained-checkpoint-sha256` 核对清单的 checkpoint、test split 和标定 SHA，进入原生回调后再次核对实际物理合同，再以冻结策略进行原评估。BC-only Checkpoint 或缺失 Adam 的伪更新计数在此路径会被拒绝。

## 复现顺序

以下变量引用私有已核验产物。先通过 `--check-config` 核对 BC 输入，合格后用新目录和显式步数执行拟合。

```bash
config=configs/dh116_training/privileged_teacher_ppo_guarded_v2.json
bash scripts/python.sh scripts/warm_start_dh116_privileged_teacher.py \
  --config "$config" --native-contract "$native_contract" \
  --native-contract-sha256 "$native_contract_sha" --expert-dataset "$qualified_expert" \
  --expert-dataset-sha256 "$expert_sha" --num-envs 2 --check-config
bash scripts/python.sh scripts/warm_start_dh116_privileged_teacher.py \
  --config "$config" --native-contract "$native_contract" \
  --native-contract-sha256 "$native_contract_sha" --expert-dataset "$qualified_expert" \
  --expert-dataset-sha256 "$expert_sha" --num-envs 2 --steps "$bc_steps" --output "$new_bc_run"
bash scripts/python.sh scripts/train_dh116_privileged_teacher.py \
  --config "$config" --checkpoint "$new_bc_run/teacher_bc_actor_warm_start.pt" \
  --num-envs 2 --updates "$approved_r0_updates" --output "$new_r0_run" \
  --enable-experiment --execute-native
bash scripts/python.sh scripts/prepare_dh116_privileged_teacher_evaluation.py \
  --config "$config" --checkpoint "$frozen_r0_checkpoint" --checkpoint-sha256 "$r0_sha" \
  --native-contract "$native_contract" --native-contract-sha256 "$native_contract_sha" \
  --primary-initialization configs/dh116_training/grasp_initialization_below_axis_2mm_v1.json \
  --secondary-initialization configs/dh116_training/stereo_tripod_reset_20261105_v1.json \
  --camera-config configs/dh116_training/stereo_camera_640x480_v2.json \
  --seed "$heldout_seed" --output "$new_cohort"
bash scripts/python.sh scripts/evaluate_dh116_stereo_trials.py \
  --teacher-config "$config" --checkpoint "$frozen_r0_checkpoint" \
  --trained-checkpoint-sha256 "$r0_sha" --manifest "$new_cohort" \
  --owner teacher --check-config
```

真实冻结评估在同一个命令中用 `--enable-experiment --execute-native --output "$new_eval_run"` 替换 `--check-config`，经同一单实例和资源预算控制。运行前应确保测试初态与训练及 BC 初态不重叠。原始采集、外层退出、逐步力／支撑证据和真实抓取结果仍由原 promotion 验证器判定。

## 证据与待完成条件

基线 `9c7058fd4874c5b34098f32713e4afa3a5f76df3`。Ubuntu `test_dh116_teacher_warm_start.py`、`test_dh116_stereo_trials.py`、`test_dh116_privileged_cli.py` 和 `test_dh116_privileged_teacher.py` 共 **75 通过，0 失败**。回归日志 SHA256 `73769bde24d2ed1161078f66df29b1a3628e8834d1cbf6165df554599434e0ef`。历史 CPU 诊断的 terminal return 恰好相等断言在原基线可因 FP32 减后加发生舍入差，回归按单步浮点舍入量比较且仍直接核对 terminal bootstrap mask。

现有 MPC 2×32 私有账本在真实 `--check-config` 被专家数据版本门控拒绝，失败日志 SHA256 `e89afe058f4a12df4f62a68c137a34a031631fba2a8b7638f86ae44ed50f4013`。本包未启动 Isaac，正式 BC、R0 训练和原生成功均无新增证据。完整 T5 MPC 同步采集须得到完整首回合、可重放的 371D/24D 对齐及逐字段有效性，还须满足原始外部力与支撑来源证明，方可进入专家池。MPC、BC 和短 PPO 诊断各自产物保留原资格，正式冻结评估仍采用原生 30 mm／5 s／0.5 mm。
