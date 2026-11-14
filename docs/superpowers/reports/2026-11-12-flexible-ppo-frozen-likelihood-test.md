# DH116 柔性 PPO 冻结似然回归的 FP32 容差

基线 `d7fdecaa980eecdc8d9de331d57577d242373788` 的联合 CPU 回归为 356 通过、1 失败。失败位于结构化 Critic 完整 update 测试，测试私有断言要求 `max_abs_log_probability_error < 1e-6`，实际为 `1.9073486328125e-6`。原日志 SHA256 `290d7b78aaf8610c212d6fe43c09d982037e33970503477be9ac5f839985ef14`，保留在 Ubuntu 私有证据目录 `flex.continuation-p0-p6-batch-closeout-5ab75r3u`。

使用 Ubuntu PyTorch 2.7.0+cu128 的 CPU、2 线程和合成 fixture 重现。收集阶段每次处理两个样本，同样分块重算时 mean 和 log probability 均逐元素完全一致。冻结审计一次处理四个样本，mean 仍完全一致，log probability 出现 `1.9073486328125e-6` 差异。差异可在固定同一批 latent、mean 和 std 后重现，来自 FP32 似然在不同批次形状下的计算。原 `PPOTrainer._check_frozen_policy()` 逐元素比较的合同是 `atol=2e-5, rtol=2e-5`，实测通过。复现脚本用固定 seed 4 定位此数值，正式测试未固定随机种子。

测试现单独重算保存 latent 的 log probability，按原冻结合同逐元素比较，并验证 update 返回原审计的误差值。加入两项拒绝检查：对 Actor bias 增加 0.01 时触发 frozen mean 拒绝，对独立账本副本中的 log probability 增加 0.01 时触发 frozen likelihood 拒绝。正常 rollout 仍完成一个 optimizer update。原 PPO 源码 SHA256 保持 `f0a96f4f8c9548b5153d004013bef9d77b97250aa0bc43272626cffb15416854`，实际比较阈值未修改。

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 /mnt/mydisk/xk/envs/dh116/bin/python -m pytest -q -p no:cacheprovider tests/test_dh116_deformable_privileged_ppo.py
```

9 项专项 CPU 测试通过。已另行验证同一个曾重现差异的 seed 4 样本通过修订测试，并核验 528 个保护文件和 2 个历史 Checkpoint。联合 357 项回归由集成工作树统一运行。未启动 Isaac 或真实训练。
