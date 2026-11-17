# PhysX 历史轮询接触来源审计

基线 `5a7eeb41c3d2f4efed98541ad76c375f9a1ee1d2`。本工作包只读已归档标定，未启动 Isaac 或改变 reward、物理、现有观察合同或 Checkpoint。

`tabletop_stereo_contact_calibration.py` 对完整 poll 的 header/contact/friction 三数组核查整数 offset/count、边界、重复归属及无归属记录。旧版 residual 原先只计数错误 collider 后仍会累加其 impulse，现在错误 collider 被排除。物体另有未分类的 partner 时，残差不归因于配置内单对；仅观察到额外接触也不足以判断是否有承托。

严格输出逐侧保存 Body A/B 的 actor 和 collider、stage、SDK step、物理时间和 dt。对配置中同侧 actor/collider 一致且数组归属完整的 pair，报告原始冲量，以及分别假设冲量作用于 actor0、actor1 时物体上的世界坐标平均力。换算用该行物理 dt。分开报告 contact、friction 和可能重复计数的联合假设；不同 actor 顺序分别比较重力校正后的动量残差，不选择最佳符号冒充物体实测力。多个 stage 的数据不合并。旧展平记录、现代空 poll 和非零 Tensor 传感器合力分别统计。每步直接保留未分类物体 partner；经验证的 on-object force、外部支撑排除和附着排除仍为 `null`。R1、R2、Force Proxy 未获资格。

## Ubuntu CPU 复核

命令：

```text
PYTHONPATH=src:. /mnt/mydisk/xk/envs/dh116/bin/python -m pytest -q -p no:cacheprovider tests/test_dh116_physx_polled_contact_audit.py tests/test_dh116_stereo_contact_calibration.py tests/test_dh116_stereo_shutdown_validation.py
PYTHONPATH=src:. /mnt/mydisk/xk/envs/dh116/bin/python scripts/audit_dh116_physx_contact_sources.py --ledger <raw.jsonl> --ledger-sha256 <pin> --inventory <inventory.json> --inventory-sha256 <pin> --stage <calibration_scene.usda> --object-actor /World/CalibrationObject --partner-actor /World/Finger14 --output <new-report.json>
```

9 项针对错侧、条件符号、range 重叠/缺口/越界、未知支撑及文件身份的测试先失败。加入实现后相关回归 **50 项通过**。两个旧 run 的 ledger 与 inventory 分别按独立 SHA 校验，stage SHA 对照各自 inventory。归档不重写，输出只允许新文件。

| 归档 | 物理步 | 非零 Tensor 步 | 旧展平步 | 结构化空 poll 步 | actor/collider 身份配对 |
|---|---:|---:|---:|---:|---:|
| v02 | 240 | 30 | 240 | 0 | 0 |
| v04 | 240 | 30 | 0 | 240 | 0 |

两份 archive 的 sensor-side 动量最大残差仍为 `4.348754785594155e-9 N·s`。旧 v02 的展平格式不能证明原始三数组为空，v04 则保存了空结构化 poll。均缺实测 body/collider pair 和逐 episode 的外部支撑/附着证据。这些样本来自固定左拇指网格的单刚体标定，不能推断右手、活动关节或双手成功。原生自然关闭与重启复验仍是下一步的独立批准门槛。

| 证据 | SHA256 |
|---|---|
| v02 原始 ledger | `981a40e256a9bb810799ebdbd30c8bbdeed432b6d1e2ef1be75f4c1954903877` |
| v02 inventory | `42743cc2c943d395456a15623b5d9d695a7f8492f446a9875535f5c22df2ae7d` |
| v04 原始 ledger | `f70a7d8fec08093a2c6a11e634b2b117162af5ec2bc61c2eed7ecd37b3b23064` |
| v04 inventory | `477563ee3674fb1ce6f1000a29e8f7165214083d4298451340e7948e5d6ea18c` |
| 两份 stage | `3f88dec8e88596ad0c3ab5249a56c3a7679a4bc60b685fb252927982a5442c57` |
| v02 CPU 审计输出 | `b3a5de6f1fbaa0e0062b1e5b9ab4138ced3500204f92cabbd4a984199123954c` |
| v04 CPU 审计输出 | `f410fdc12b060149a53099f89e5d81db4a7a17017674020b229dbab914c6daaf` |

私有 Ubuntu 输出：`/tmp/dh116-wp4-polled-archive-HwNheu`，初始失败日志：`/tmp/dh116-wp4-polled-red-wqRQ4S.log`。新字段名称在最后一次测试前明确了平均力针对 object，两个历史归档没有命中 pair，因此以上复放输出不含被重命名的字段。
