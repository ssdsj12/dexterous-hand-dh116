# DH116 柔性特权 Teacher 的独立结构化 PPO CPU 接入

两个历史 Teacher Checkpoint 都绑定了原 PPO 源码 SHA。原地改动会破坏其恢复身份，因此采用独立 CPU Trainer。独立 CPU Trainer 已实现结构化接入，原 `tabletop_privileged_ppo.py` 保持 SHA256 `f0a96f4f8c9548b5153d004013bef9d77b97250aa0bc43272626cffb15416854`，原 Checkpoint 模块和 CLI 均保持原字节。

## 复用路径

`DeformablePPOTrainer` 在构造时核对旧 PPO 文件 SHA。仅独立覆盖 `collect_step()` 和 rollout 工厂，复用原 24D Tanh 采样和 `record_step()`。采样时把 common371、rod nodes 和 node mask 的实际张量树复制进 rollout，确保外部后续修改不会污染采样记录。继承的 rollout 负责批次拼接和 GAE。独立 `StructuredCriticBatch` 是显式字典容器，使用原 tree helper 对实际记录的各字段同时索引和转移设备。原 `update()`、frozen policy audit、价值损失、KL、Adam 回滚直接继承，未复制或修改。

独立 Checkpoint 版本绑定模型/抓法、R0 配置对应的任务身份、PPO 配置、旧 PPO/验证源码、独立适配和保存源码。新保存入口要求干净 rollout，使用排他发布避免覆盖。CPU PyTorch、NumPy、Python RNG 与 Adam 一同保存。恢复前校验身份、模型拓扑、Adam 参数顺序、全部参数组设置、完整 rollout 的计数及 RNG。仅被 KL 拒绝的完整批次可以增加环境步数，更新计数保持不变。新 loader 使用 `weights_only=True`。两个受保护的真实旧 Checkpoint 均在只读测试中被新入口拒绝。

## 实际 CPU 回归

| 验证 | 结果 |
| --- | --- |
| 2 环境 × 2 步真实输入树合成样本 | 完成一次 CPU update，Actor/Critic 两个 rod encoder 均发生参数更新。Critic 树中 common、节点和 mask 均正确索引，返回的批次副本可独立修改。 |
| 真实终止与截断 GAE | 真终止丢弃 next value，截断按显式 terminal bootstrap 值计算，两者均截断后续 carry。 |
| 旧 Tensor 与结构化 common-only 等价 fixture | 同随机状态下采样、latent、log probability、value、GAE、PPO update 指标、policy 权重和 Adam 状态完全一致。 |
| KL 回滚 | 已有 Adam moments 时强制拒绝两个候选。模型、Adam、Torch/NumPy/Python RNG 恢复到事务起点。 |
| 独立 Checkpoint | 恢复后随机采样及下一次 update 与原进程完全一致，拒绝跨 grasp 和旧版本，重复保存不能覆盖。 |
| Checkpoint 损坏回归 | 拒绝改动的 AMSGrad、Adam betas 和不可能的 rollout 环境步数。完整收集后被 KL 全拒绝的合法 Checkpoint 可以恢复。 |

先观察到缺少独立入口及四个篡改 Checkpoint 被误接受的测试失败，再运行：

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 /mnt/mydisk/xk/envs/dh116/bin/python -m pytest -q -p no:cacheprovider tests/test_dh116_deformable_privileged_ppo.py tests/test_dh116_deformable_privileged_teacher.py tests/test_dh116_deformable_grasp_task.py tests/test_dh116_privileged_teacher.py tests/test_dh116_stereo_runtime_rod_identity.py tests/test_stereo_action_contract_audit.py tests/test_stereo_privileged_state_contract_audit.py
```

170 项 CPU 测试通过，包括 9 项结构化 PPO 端到端与恢复回归。528 个保护文件和两个历史 Checkpoint 核验通过。合成 fixture 只验证训练器算法及身份。本次未启动 Isaac 或真实任务训练，三种抓法配置的 `native_ready` 仍为 false。真实材料节点、371D 公共字段、Reset、R0 奖励来源、双手同一自由杆和独立物理预算仍需原生资格。现有模型 `pack()` 会检查 common 与 rod 的身份，返回张量时不保留元数据。实际原生入口仍须在 Actor/Critic 收集和动作执行间绑定同一 episode、reset、rod、physics tick、solver tick 及 action owner，并锁定原生任务、R0 和 Reset 源码 SHA。未增加原生 CLI 路由。
