# 真实 dual Pinch v4 的 2+8 R0 离线诊断重放

基线 `d31b88c6f71e168a7b8f5f7ba6afd616ccdb3535`。本包消费一次真实原生运行的
不可变 ledger，不使用 CPU SDK/solver fixture 替代物理样本，不重跑原生。
新增独立 CLI、9 项测试、本报告和机器可读结果摘要；原 R0 task、reducer、replay、
legacy 配置、物理和 checkpoint 均未修改。未启动 GPU、Isaac、训练，未推送。

## 真实输入与执行结果

输入：
`/home/xk/.local/share/dh116-autonomous-3c80166/evidence/approved-dual-pinch-zero-v4-20261113-utbAE0/native-runs/flex.f7-dual-pinch-zero-interval-v4-native/attempt-fz50kogx/native/physical.jsonl`

SHA256：
`ead0dbf2c15b83864a1c440848ed9cf159b86c88a6ade0e4275208efebcf778c`。

16 行严格包含 header、2 个 setup 边界、reset、左右 initial geometry、begin、
8 个 physical、end。已有 `inspect_evidence` 先核验完整顺序和实际时钟，
然后原 `PhysicalR0Diagnostic` 仅消费 reset/begin/8 physical/end。
setup 与初始几何不会再次计作物理步。保存后的 bundle 又经现有
`replay_diagnostic_bundle` 和原 CLI 的真实磁盘读取重放，完整 R0 report 一致。

- env 0，episode `dual-pinch-d144d4f9c1374708af3a8ff54c49004b:one`，reset 1。
- reset SDK 2 / rod 0；物理 SDK 3–10，rod tick 12、24、…、96。
- 实际输入动作是原记录的全零 24D；按原来源核验双手 pre/post 17 DOF q/qd/target、
  具名 6 finger + 6 mount target 索引、实例路径、同钟 rod/provider 重建和原 guard。
- 1 个完整控制窗口，8 次原 R0 evaluation，每次重放仅结算一次。
  测试拦截原 `FlexibleGraspEpisode.step` 证明每一遍调用序列恰为 3–10。
- 0 个新物理步、0 次 optimizer update，未补造更多 hold/approach 样本。
- 本包未发现需要修改旧 replay/reducer 的真实数据兼容性 bug。

## 已知值与 NA

真实 raw pair 中手部样本 0，杆—桌面 SDF 样本 3936（8 × 12 × 41）。
这提供了“不满足至少两个 pad 的必要条件”，而不是推定一个未知的 slip 值。
PhysX hand-table source 在原 ledger 中明确 unavailable，不能由杆—桌面 SDF 替代。

| 原 R0 项 | 8 样本中的有效数 | raw sum | 依据或限制 |
|---|---:|---:|---|
| approach_progress | 0 | NA | 没有绑定的逐步 approach-distance 源 |
| required_contact | 8 | 0 | 完整 pair capture 内双手均无 pad 样本 |
| contact_persistence | 8 | 0 | 未满足双手接触必要条件 |
| grasp_geometry | 8 | 0 | 必要对向条件为 false |
| lift_progress | 0 | NA | 原生 radius 读回未记录，不用配置值替代 |
| hold_progress | 0 | NA | radius/stable 等来源未齐 |
| material_anchor_slip | 0 | NA | 无连续材料锚或表面 marker，不能以空接触置零 |
| joint_limit | 8 | 0 | 实际 q 经原 joint guard 校验 |
| instability | 0 | NA | stable 没有绑定测量源 |
| external_support_violation | 8 | 8 | 原 R0 定义包含真实杆—桌面 SDF 接触 |

最后一行的 raw=1 是原 task 已有 reward 语义，不表示本包发现新外部施力。
原逻辑在 approach 阶段允许桌面接触，不因此设置 prohibited_support_latched；
8 行均未 latch、未终止，且 phase 始终为 approach。本包未重新定义该项。
显式 auxiliary 字段 external_support/authored_force/attachment/stable/approach_distance
全部保持 NA；没有假定“无支撑”或“稳定”。

从真实 rod nodes 计算的 median/min-node 净高度变化均为
`-8.940696716308594e-08 m`。这只是 readback 差值，不能作为 lift 成功。
材料锚、tangential slip、radius 原生读回、每步 actual pad pose、完整 371D、
native R0 reset 顺序以及 Teacher 准入仍未成立。

## 诊断系数和预算的明确范围

原 task 的十个系数依然全为 None。仅使用
`tests/test_dh116_deformable_control_step.py:16–22` 的既有 CPU fixture：
按 R0_TERMS 顺序为 1…10，并对 fixture、task 与预算模板做源码 SHA 核验。
因此单项 contribution 只属于该诊断 fixture；例如桌面项为 -80 不是真实标定奖励。
完整 `diagnostic_reward` 和 `training_reward` 均为 None。

原 reducer 要求 frozen_eval 模板包含至少 300 个 hold control，本包不改此合同。
复用原 360-control CPU 容量模板、setup=2、solver stride=12；
摘要明确 `existing_CPU_reducer_capacity_not_native_run_authorization`。
真正消费和实际账本预算仍严格为 2+8，总 10 个历史 SDK tick。没有新 native 配方或许可。

backend artifact 按实际 ledger 的 SHA
`92a50ba3747192b72fd607741ff1465c17874f7c8f8fb0cbed82d99d5985d8c0`
只读核验。原 task 的不同 backend pin 仅在已有诊断 copy 中适配，未写回旧 task。

## 复现与 CPU 验证

```text
DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 bash scripts/python.sh -m pytest -q -p no:cacheprovider tests/test_dh116_dual_v4_r0_native_replay.py tests/test_deformable_r0_replay.py tests/test_dh116_deformable_control_step.py
bash scripts/python.sh scripts/replay_dh116_dual_v4_r0.py --output <new-native-v4-r0.json> --bundle-output <new-native-v4-r0-bundle.json>
bash scripts/python.sh scripts/replay_dh116_flexible_r0.py --input <native-v4-r0-bundle.json> --output <new-native-v4-r0-disk-replay.json>
```

真实执行统一使用上述 DH116_PYTHON、2 CPU threads、空 CUDA_VISIBLE_DEVICES。
**62 passed / 34.83 s**：9 新增、53 旧回归。
新模块缺失时先出现 5 failed / 4 fixture errors，日志保留。
最终两个 CLI 自然退出 0，native_imports=[]，磁盘 report 完全相同。
已有输出和符号链接在重放前拒绝，写入使用独占创建，旧 evidence 不覆盖。

wireseg agent 只读审查新 CLI 的顺序、NA、fixture 和输出保护，未发现新增阻断；
未替本包运行测试，也未声称审过旧 R0 全部分支。

证据目录：
`/home/xk/.local/share/dh116-autonomous-3c80166/evidence/dual-v4-r0-replay-cpu-gmhrfazr`。

| 证据 | SHA256 |
|---|---|
| 最终完整 native-v4-r0.json | `1489d19c86f7eb19731f7b59c09fd1e6d01e721219fdb8be91bff80941ced5b8` |
| native-v4-r0-bundle.json | `0d9dd875826a589e8f6d64dca1ffbed322d33b1fe403bb48515726d0c6002f17` |
| 原 CLI 磁盘重放回执 | `419bb8485624ac35779216b077b6638b2c8a466bc56dd1526ba59bd2d6f44e37` |
| green.log | `a2ae09bf6311e1eadc5134326e4b36df634b1c871dd1d740407cb95afd5b6352` |
| green-command.json | `ec5b0bd80e5f2b1d500674766e65daa1f3b611b54ddb210378643bbd646fa087` |
| final-cli-0-command.json | `3134f88c3477e171dea41765d1044d4e99b1a3c0642f78542a8f86ef24476f38` |
| final-cli-1-command.json | `b376a7c07c5d902657c2edd664f86415fdb7b12702ed1931ff65d33dff6d9efa` |
| 已提交机器摘要 | `7573a695150f2c7337e197a091b7c0d2e0798b693d2bcef8327ded22c7e9a85d` |
| 新 CLI | `f8bca051e538c2825b61a70f6b1fb5921b2deb3e01339a7097243a683173d0bb` |
| 新 tests | `1ce66c21c4e2971d707acf19a771c2d528836906f5cee299eb8bc7952019d714` |

## 下一最小无相机依赖（只读定位，未扩展本包）

现有 `DualHandGeometryReader._read` 已在每次 SDK 后通过
`root_physx_view.get_link_transforms` 读取真实 23-link poses，并生成十个 pad poses。
原 `FlexiblePhysicalExchange._step` 第 147–150 行消费这些同钟 poses 推进 rod，
但返回 ledger 未保存它们。本次 raw 只有 reset pad poses、初始实际 link poses和
每步 q/qd，不能把离线 FK 结果冒称每步实测 pad pose。

下一小块应在后继的现有 after-SDK geometry 诊断链保留这些已读取的 owned poses、
真实 link 身份及 `sdk_complete_rod_pending` 阶段/前一 rod clock，再在 rod 完成后
绑定同一物理行。先完成 CPU 身份/时钟与序列化测试，真正的新实测仍需独立 native。
这能支持后续明确的 approach-distance 定义，不解决材料 slip、hand-table 正向读回、
奖励系数或 Teacher 资格。本包未重复建设 reader、控制器、几何扫描或 callback subsystem。
