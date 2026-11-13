# 柔性 Teacher 的实测字段派生模型

原 371D 合同有刚性物体四元数、刚体根角速度和圆柱 SDF。自由 Cosserat 杆没有这些同义字段。旧模型要求 371 项全有效，当前物理入口只读回其中 66 项。把剩余 305 项补零或用杆均值替代会改变状态语义。

新增独立 CPU 模型 `MeasuredCommonTeacher`。它接收原索引的 371 项账本和布尔 mask，检查未提供项为 NaN，仅选择 B00/B01/B15/B34/B35/B36 的 66 项进入网络。源字段定义、动作 24D、原 Tanh、杆节点编码、PPO/GAE/KL 路径均沿用现有实现。派生网络首层输入大小为 66，使用独立模型标识和分抓法 namespace，旧模型权重、原 Checkpoint 格式均被拒绝。该模型当前绑定原 CPU backend 合同，原生扩展和派生持久化需要再独立验收。

现有 `DeformablePPOTrainer` 可以直接使用拥有副本的带 mask 张量树。CPU 夹具完成 4 条模拟样本的一次 optimizer 更新，保留账本的 NaN 并验证旧 checkpoint writer 在写盘前拒绝。夹具奖励不是实际 R0，无原生更新、无正式训练。

## 验证

新增 7 项，相关 **44 passed**。实现前先复现模块缺失。覆盖原字段索引、未观测值、非法 mask、策略输出、各抓法身份、旧权重拒绝和原 PPO 兼容。Ubuntu 使用 `DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 bash scripts/python.sh -m pytest -q -p no:cacheprovider`，测试为 `test_dh116_flexible_measured_common_teacher`、`test_dh116_deformable_privileged_teacher`、`test_dh116_deformable_privileged_ppo`、`test_dh116_deformable_rollout_admission`、`test_dh116_flexible_episode_common`。私有日志 SHA256 `8ff93c39b1790f6994759715efc4e785273d7006c07b931474e11e11c26a6216`。

原生准入仍为 false。这个独立选择方案不改原 371D 的语义，也不将缩减输入称为已满足原全量模型合同。未来需要明确验证 CPU/原生来源、R0 及身份，才能训练和持久化新的策略。
