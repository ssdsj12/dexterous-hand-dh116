# DH116 Teacher 首次 PPO 更新与检查点封存

Antigravity 使用只读 SSH 在 `11-17` 的 `343c4de458be2d5ec74abed7babb953314763c22` 上完成三轮复审。核对的原训练代码 SHA-256 为 `2a3d31ed352398fb6460cc4a66ab7a90d8654e3cb07f49f20f47336f243773d0`，原检查点代码 SHA-256 为 `2b26625a62bb8cad74280d2c0ca07f29af418d3b0d4a3b20d67e4f564444c021`。Codex 另外追踪了 Collector 到物理 episode 和 optimizer 的调用。三抓法不能靠更改工厂中的 style 白名单获得合格物理来源。现有 Pinch 已验证的是原生数值 Reset 和该实验自然退出。Tripod 需要独立双手三指对向载荷及状态来源，Power 需要真实掌面或近端载荷及五指包覆。

## 当前实现

新增 `tabletop_native_teacher_actual_update.run_fresh_update`，供现有 C04 Collector 后续接入受租约原生 Worker。它要求单环境、无相机、fresh 策略、空 rollout、时钟 SDK2/rod0、同账本 scope 及新输出文件。使用原 `run_updates` 执行一个 32 控制窗口，途中短回合终止时保留失败账本。完整更新要求同一 run、episode、Reset 和杆，32 条转移，256 个物理步，末端 SDK258/rod3072。

检查点先通过原 C05 writer 保存到私有目录，再由原审计器检查不可变字节、fresh 来源、Adam、RNG、版本、计数和真实策略权重。独占输出使用已经审计过的字节。原策略、奖励、控制与碰撞参数保留。检查点的 SHA 与四个首尾时钟记录到原有 fsync 训练账本。该函数没有 App 或独立执行命令，结果仍是 unqualified source。后续 R06 与外层原生产物审计决定试验是否可接受。

## CPU 证据

新增及改造的 9 个回归在实现前复现缺少模块。实现后实际原 PPO 在 CPU 替身场景完成 32 次采样并接受更新。账本保留 64 条动作/价值输入来源、256 条物理事件以及 checkpoint 封存。测试还验证了已有 checkpoint、已用策略、错 scope 和错时钟在移动前拒绝，短回合拒绝保存，审计拒绝时无最终文件，以及审计期间替换私有暂存文件不改变最终字节。

本轮 Isaac 与 GPU solver 运行均为 0，原生 PPO 更新为 0。无新的抓取、保持或抬升资格。旧 WebRTC 进程 `1633110` 在检查时仍持有单实例内核锁。

| 抓法 | 下一项原生证据 |
|---|---|
| Pinch | 将本更新函数接入实际 `run_a`，补齐原生 source capability、后验物理账本和 R06 外层自然退出复核，冻结有界新配方。首次单更新通过之后再做真实抓取与保持。 |
| Tripod | 用独立初态消除现有右手桌面相交，再核同钟左右三指反力与同一根杆的接触；接通独立状态读取器。 |
| Power | 用独立初态消除手指桌面相交，再测真实掌面或近端承载及五指包覆；接通独立状态读取器。 |

完整抓取标准仍为净抬升 ≥30 mm、连续保持 ≥5 s、固定锚滑移 ≤0.5 mm。

## 联合回归结果

采用当前 approach_wiring_smoke_v1 的 PPO 配置，KL 目标 0.02，允许原算法的 3 次回退。完整 CPU fixture 账本含 932 行，接受 1 个 minibatch，参数 L2 变化约 0.00421635，完整 rollout exact KL 约 0.00135907。此数值仅反映本次 CPU 采样窗口。

正式测试命令在 Ubuntu DH116 worktree 下运行，`PYTHONPATH=src:.` 并沿用既有模块解析环境。

- `/mnt/mydisk/xk/envs/dh116/bin/python -B -m pytest -q -p no:cacheprovider -s tests/test_native_teacher_actual_update.py tests/test_native_teacher_training.py`，18 项通过。
- `/mnt/mydisk/xk/envs/dh116/bin/python -B -m pytest -q -p no:cacheprovider tests/test_native_teacher_checkpoint.py tests/test_native_teacher_actual_launch.py tests/test_native_teacher_actual_factories.py tests/test_native_teacher_worker_artifacts.py tests/test_teacher_force_closure.py tests/test_teacher_force_runtime.py tests/test_dh116_work_packages.py`，267 项通过。
- 三种现有静态预检均保持禁止直接执行。Tripod 和 Power 均报告各自的同钟距离读取器尚缺。日志及 SHA 详见 readiness。
- Registry 更新后单独复跑 55 项，全部通过。
