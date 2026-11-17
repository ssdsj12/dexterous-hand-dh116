# 实测字段柔性 Teacher 的独立 Checkpoint

`tabletop_flexible_measured_checkpoint` 在干净 PPO 更新边界独占写盘。保存动作策略、Adam、计数和 Python/NumPy/Torch CPU RNG，绑定模型版本、抓法、原字段选择、PPO 参数、动作合同及源代码 SHA。恢复前验证格式、网络形状、计数、Adam 与 RNG，先装载到临时模型以拒绝不兼容状态。

原柔性 Checkpoint 的身份包含其实现文件的整个 SHA。为保持已保存模型可读，本包保留其字节并复用独立的校验工具，在新格式实现写入和恢复。旧训练和模型路径继续使用原合同。当前派生格式只用于 CPU 验证，没有开启原生训练权限。

Ubuntu CPU **44 passed**。新增三项覆盖合成 rollout 更新后的精确续训、覆盖拒绝、跨抓法和旧模型拒绝、损坏计数拒绝、待处理动作时拒绝保存。先写测试复现缺模块。两次合成更新后的权重、Adam、RNG、计数和 update metrics 精确比对。实际物理步骤和原生 PPO 更新为 0。

命令为 `DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 bash scripts/python.sh -m pytest -q -p no:cacheprovider`，测试包含 measured checkpoint、measured Teacher、旧柔性 PPO/Teacher 和 rollout admission 五个文件。日志 SHA256 `f4fe6f7aec0e5026b6bf6b8fe3c29d827679ec7bc63112c7ca8ba34eb585818b`。
