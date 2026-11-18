# DH116 Teacher PPO 价值估计来源账本

## 审查基线

Ubuntu worktree 与 Antigravity 的只读 SSH reviewer 均读到 `11-17` 的 `81a1c661fba275ca3db72388ec63018da024310a`。`tabletop_native_teacher_session.py` 文件 SHA-256 为 `b38b7273f56d0a2ce470455f88d8e1c38ca1a1438cc554e5b33711dc5b35486b`。原 reviewer 无服务器读取工具，先审查了冻结源码摘录。具有只读 SSH 工具的 reviewer 随后直接核对了 Collector、模型打包和力闭合代码。

复核确认同一个 Collector owner 同时负责动作前观测和动作后价值估计观测，模型打包已经检查 SDK 步、rod tick、episode、Reset 和 context 身份。旧账本保存 `next_value`，缺少它接收的观测。新增 `settlement.bootstrap_source` 保存实际输入、输入 SHA-256、动作 token、Reset 前的物理端点和该端点原 pre-action 包的 SHA-256。截断保留该记录，真正终止时值为 null。相机读取后预算不足的分支仍发生在策略采样之前，不生成结算。

## 验证

首次运行两项端点断言都因缺少 `bootstrap_source` 失败。修复后 Ubuntu 训练与力闭合 CPU 回归 129 项通过，调度器 Registry CPU 回归 55 项通过，合计 184 项通过、0 失败。其中包括真终止、截断、动作 token、实际 critic 输入 SHA 和现有 PPO 训练夹具。

```sh
/mnt/mydisk/xk/envs/dh116/bin/python -B -m pytest -q -p no:cacheprovider \
  tests/test_native_teacher_actual_launch.py tests/test_native_teacher_actual_factories.py \
  tests/test_native_teacher_collector.py tests/test_native_teacher_training.py \
  tests/test_native_teacher_worker_artifacts.py tests/test_teacher_force_closure.py
```

CUDA 设备在 CPU 测试中隐藏，沿用已有 PyTorch、USD 和 DeformX 夹具环境。本包 Isaac 启动数 0，原生 PPO 更新数 0，原生抓取成功数未测。

## 三抓法剩余证据

- **Pinch** 已有同钟数值 Reset 和 Approach smoke 来源。首次真实 PPO 更新仍缺 `run_a`、每次打包的原生 raw41 来源见证、实际产物审计与有界 worker 租约。本次账本可审计实际 bootstrap 输入，完整原生来源见证仍待接入。
- **Tripod** 的力闭合演示已有拇指、食指、中指载荷和对向性判据。在源码中调用方是 `tabletop_teacher_force_runtime.py`。当前 PPO 会话仍拒绝非 Pinch，需要独立双手三指测距、同钟接触和闭合来源。Antigravity 把演示判据称为已接入训练，这与调用链不符。
- **Power** 需要真实掌面或近端接触以及五指闭合包住线缆的证据。现有 13 槽数据有指腹和外部承托，`power_palm_proximal_source_missing` 仍是显式拒绝原因。

每种抓法的原生验收仍要求线缆净抬升 30 mm，连续稳定保持 5 s，固定锚相对滑移不超过 0.5 mm。
