# R01：BC / DAgger 复用 Actor 输入模式

基线：`d1fa6630f76420177a93cda7c24de25d97bf20af`。实现分支：`codex/repair-r01-actor-20261115`。本包仅包含两处生产源码、新增测试和本报告；公共 README、CHANGELOG、readiness 与 Registry 由集成任务维护。Ubuntu 本地提交后交由主任务审查及 Windows 发布，本报告不宣称已推送。

## 问题和修复

`StereoBCTrainer` 原来直接将 `output['latent']` 传入动作网络，绕过 `StereoStudentActor.latent_mean` 中已有的 `TaskStateAdapter`。32D latent 在 `estimated_task_state` 的 3D、`hybrid` 的 35D 动作输入上分别触发 `1x32 / 3x128`、`1x32 / 35x128` 矩阵维度错误；DAgger 继承同一训练路径。

新增 `features_from_estimates(estimates, history)`，复用原有 Actor 特征构造逻辑：latent 直接返回 latent；非 latent 调用既有 adapter，排除 `world_auxiliary`；hybrid 拼接 latent 与已门控的 task features。`latent_mean` 和 BC 均使用该方法，BC 的辅助损失继续使用同一次 estimator 输出。TinyStudent / 无 estimator 的原有 fallback 保留。

非 latent 监督训练由维度异常变为复用其既有推理特征并正常更新，这是本包批准的 bugfix；旧 latent 动作数学保持逐位一致。没有新增网络参数、改变布局或修改 canonical24 / TanhGaussian 数学；没有新增 GT 或 Critic Actor 输入。BC 的独立 Adam、PPO clean-boundary 检查、PPO storage 隔离及 `log_std` 排除逻辑保持原有路径。

## 实际 CPU 验证

统一命令前缀（项目根目录运行）：

```bash
env DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' bash scripts/python.sh -B -m pytest -q -p no:cacheprovider
```

1. 修改生产源码前，运行以下 6 个旧测试文件：**59 passed in 26.51s**。
2. 先写 `tests/test_dh116_stereo_bc_actor_inputs.py` 并运行 RED：**5 failed, 2 passed in 6.09s**，pytest exit 1。四个非 latent 训练组合均复现矩阵维度错误；精确 latent 测试因缺少共享方法而失败；两个 latent 训练组合通过。
3. 最小补丁后，运行新测试加以下 6 个旧测试文件：**66 passed in 27.56s**，pytest 与外层命令 exit 0。

```text
tests/test_dh116_stereo_bc_mode.py
tests/test_dh116_stereo_runtime_actor_modes.py
tests/test_dh116_stereo_collection.py
tests/test_dh116_stereo_dagger.py
tests/test_dh116_stereo_student_ppo.py
tests/test_dh116_stereo_runtime_rollout_storage.py
```

新测试实际执行 BC/DAgger × 三种模式的各一次 CPU optimizer step，`lambda_state=0`，验证 action 权重更新、action gradient norm > 0、estimator forward 恰好一次、每个 module 的混合 train/eval 状态精确恢复、`log_std` 不变且无梯度、`steps=1`、`native_qualified=False`、`ppo_storage_modified=False`。两种非 latent 模式的预测头第 116 行梯度非零，禁止区间 `180:200` 梯度为零。latent 重构的动作数学按 `rtol=0, atol=0` 验证。

额外只读兼容性检查通过 `git show` 加载基线类，与新类比较 world/palms × 三种模式：state_dict 键、shape、初始化张量完全相同（world 124 tensors，palms 127 tensors）；相同 history 上的 mu24 与基线逐位一致；新 `latent_mean` 每次调用 estimator 一次；palms 非 latent adapter 的实际输入不含 `world_auxiliary`。该检查为额外诊断，不计入 66 项 pytest 数量。

基线 pytest 实际退出 0；初次 SSH 脚本尾行的 CRLF 曾使外层 shell 的 `exit` 解析失败。之后显式发送 LF 脚本，RED、GREEN、兼容性和最终保护核验均确认外层返回码；该问题没有改变 pytest 结果或源码。

## 证据与保护

原始日志仅保存在服务器私有目录 `/home/xk/.local/share/dh116-autonomous-3c80166/evidence/`，未纳入 Git：

| 文件 | SHA256 |
| --- | --- |
| `r01-baseline-20261115.log` | `0d616b4ae2378f1acf4703d5f02cdb9ab4e359e72e33c4489287257a1874d69b` |
| `r01-red-20261115.log` | `feb746bc0fcdf805200fc66d7b1de5125087192c87a5309ce9e35d84a8f4b123` |
| `r01-green-20261115.log` | `5b42d21abf27bb7c39ffc3cf24a5e0640d173437f6e3b5b3ae5c173913f47612` |
| `r01-compatibility-20261115.py` | `4cd8199478ef22cb11c7e7496dce8bae4de097ed472c2c60284f6e3cfa252ea7` |
| `r01-compatibility-20261115.log` | `f670f6e46ead4026350671712c3e5a3a09d8c375255eb1effec370359c4400d5` |
| `r01-protection-before-20261115.log` | `1c14d484ca08e06e918c57a205cb950abb8f7b3153a7f0cf7bf548055ae8b448` |
| `r01-protection-after-20261115.log` | `9b08a5602072ba703f4f23d5b4b7019cbbda225ceed69015b5efbddd95fafa84` |

前后调用 `scripts.run_dh116_work_packages.Repository.verify_protection()` 均通过；后验明确核对 **528 个受保护源码/脚本/配置文件及 2 个历史 Checkpoint**。inventory SHA256 为 `fb6b58cfd2fe86408eb498a4390ecda35df2221d7198ca3a01ff59f86dea05ec`，两 Checkpoint 哈希分别为 `ec5831ac4de039e5e77e2ee4a3420497a0c103a2a5966a1a73662d4997966511`、`c1735e5f112b7f10ff4dacf00b2ac547728cea82a08d5b051e5b78cca84a9999`。没有修改、迁移或覆盖旧 Checkpoint；未安装环境依赖，未启动 GPU、Isaac 或原生训练作业。`git diff --check` 通过。

上述实际 optimizer steps 使用 schema-only 合成 fixture；它们不构成原生物理、实际抓取、成功数据池准入或模型精度证据。后续由主任务执行规格/质量审查、集成保护复核和 Windows 同提交发布。
