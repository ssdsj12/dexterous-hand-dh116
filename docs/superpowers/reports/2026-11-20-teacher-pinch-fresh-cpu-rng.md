# Pinch fresh Worker 的随机性来源

v9 和 v10 使用同一份已封存的 Reset，首次动作前却记录了不同的模型权重摘要。工厂测试用同一个模型合同、两个外部 CPU 随机状态复现了 `actor.0.weight` 差异。原模型构造使用环境 CPU 生成器，原 PPO 的采样和 minibatch 顺序也使用同一个进程内 CPU 生成器。

经 Antigravity 对 `5df07c54d3a4c070101350eb22f79df41eb9b9c7` 的只读复查，将新诊断的随机性绑定在两个最窄入口。模型构造使用固定 CPU seed `2026112001`，首次 rollout、原 PPO update 和 checkpoint 封存使用另一 seed `2026112002`。两处作用域均保存和恢复外部 CPU RNG。使用 `torch.default_generator.manual_seed` 与 `fork_rng(devices=[])`，CUDA 生成器不被重新设种。Python、NumPy、物理 Reset、原控制器和 PPO 算法沿用原实现。

实际 Worker 在第一条 decision 前 fsync `fresh_policy_rng`，记录初始化权重摘要、两枚种子和当时实际 CPU RNG 摘要。绑定由新配方的源码闭包覆盖。已有 checkpoint 加载和恢复路径仍使用各自保存的 RNG。复现范围是同代码和同输入下的 CPU 初始化及探索序列，模型输入中的真实剩余墙钟和实际物理运动需独立比较。两枚种子均是本次新 fresh 诊断的固定值。

工厂测试复现旧版失败并验证新版本在不同外部 RNG 下权重逐参数相同。Provider 的正常路径与动作、关闭异常路径均测试了首次动作前的种子与摘要，以及外部 CPU RNG 恢复。联合 CPU 回归 214 项通过。原始 v9、v10 运行保留各自真实的随机状态与结论。

本包完成 0 个新 Isaac 物理步和 0 次 PPO 更新。
