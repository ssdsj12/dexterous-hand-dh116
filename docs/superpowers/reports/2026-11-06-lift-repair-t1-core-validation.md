# T1 Squashed Gaussian 核心验证

## 范围与版本

分支 `codex/dh116-lift-repair-20261106`；基线 `11-5` 的 `b42f1aa43edd6e7ef94131e47d49a75f30eda585`。本阶段只新增三个概率/策略/PPO 模块及三个测试文件，没有更改旧生产模块或旧测试。训练入口及完整 v3 恢复合同另行集成。

数学定义为 `z ~ Normal(mu,std)`、`a=tanh(z)`，log probability 用保存的 `z` 和稳定的 `2*(log(2)-z-softplus(-2*z))` 计算，不对动作做 inverse-tanh。旧 Gaussian 原采样加固定环境 clip 可以解释为 latent policy，不能仅由 clip 推断 PPO 比率无效；新版本改变策略分布和熵的语义，不构成已修复 Lift 退化的证据。

新 actor 移除最终输出的无参数 Tanh，保留隐藏层、PointNet 与权重布局；确定性推理使用 `tanh(mu)`。Actor3844D、Critic371D、动作24D、物理安全投影和原执行器约束保持。环境实际执行动作与保存的有界动作相同将在入口集成测试核验。

## 实际执行

在隔离副本 `/mnt/mydisk/xk/projects/dh116-lift-repair-20261106` 使用现有 Python 环境，隐藏 CUDA，未启动 Isaac：

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src PYTEST_DEBUG_TEMPROOT=/tmp \
 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_NUM_THREADS=2 \
 /mnt/mydisk/xk/envs/dh116/bin/python -m pytest -p no:cacheprovider \
 tests/test_squashed_gaussian.py tests/test_vision_squashed_policy.py \
 tests/test_vision_squashed_ppo.py tests/test_vision_ppo.py \
 tests/test_vision_student.py tests/test_dh116_policy_noise.py \
 tests/test_surface_policy_noise.py -q --tb=short
```

实现代理最终运行：107 passed，11.90s；根代理独立复跑：107 passed，10.40s。存在一条原有零元素初始化 warning，无失败。新增 35 项测试包含独立 PyTorch 变换分布对照、changed-mu/std 概率比和优化器数值 oracle；72 项相关旧测试保留。最初新接口缺失产生 18 个 RED 失败；独立 oracle 又捕获 sample-std 归一化差异、单样本 NaN 与默认 minibatch 预算错误，修正后转绿。

独立规格审查 PASS；独立质量审查 PASS。质量审查另跑总样本数1/3/9的3项 CPU 边界检查通过。它们不能替代下述原生门槛。

## 验证要点

- float32 CPU storage 实际分配并复制 `latent_actions` / 有界 `actions`，transition clear/reuse 不泄漏 latent；保存旧 transformed logp、latent mean/std。
- 全批和微批共享 permutation 与当前 entropy epsilon；尾批完整覆盖，均值按样本数加权，每个 logical batch 一次 clip / LR / Adam。正常1024环境保持5 epochs ×8 minibatches。
- finite guards 在 Adam 前检查输入、logp、ratio、loss、gradient 与 gradient norm；极端 latent 的 logp 和梯度保持有限。
- 首次 Adam 前全 rollout 检查 frozen old/new logp；更新不重新采样 PPO 动作。当前策略 reparameterized bounded entropy 与 latent Gaussian entropy 分开记录。
- 诊断从 `last_update_diagnostics` 获取，不改变 `{value_function,surrogate,entropy}` 三项 loss API；包含经验/解析 KL、clip fraction、元素和环境级饱和、24维有界动作统计和 latent 参数统计。

## 全仓库收集与尚未完成

CPU collect-only（USD 库路径加到 PYTHONPATH/LD_LIBRARY_PATH，CUDA隐藏）收集4831项，11个旧测试文件因 `co_sim`、`elastica` 或 `open3d` 缺失发生 collection error。原始输出保留在服务器 `/tmp/dh116_lift_repair_collection_20261106.txt`。尚未宣称全仓库测试通过；后续集成后继续完整 CPU 回归并区分旧依赖缺口。

当前 GPU3 原训练未停止或修改；没有额外 Isaac 实例。尚无修复版原生动作执行、配对初态评估或原生成功。269/482/1328历史权重缺失，不能由日志还原；已冻结实际1439轮权重与原始日志用于后续独立回归。固定模型验证、验证集选模、奖励单因素消融和后续重新训练仍待各自门槛通过。
