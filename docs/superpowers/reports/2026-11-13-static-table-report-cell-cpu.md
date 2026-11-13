# Static Table full-report cell：离线来源和身份验证

本包提供最小离线验证单元，不提供新 native launcher、callback owner 或 recipe。
未启动 Isaac/GPU，未改变读回设置、原 Table、既有 v4、控制器或物理配置。
基线 `43abe38b555fe6063cbdf2fac3ad87907c4a5686`，另依赖已有静态子节点后继
`f2da19e236984b083ff1df5323d3ca57d26ac1ad` 的等价内容。父 agent 负责集成/发布。

## 可执行的 CPU 接口

`tabletop_static_table_report_cell.source_identity(stage, *, table_root, moving_actor,
moving_collider, stage_id)` 读取真实 USD：Table 必须保留 Xform/Cube 的
`Table/geometry/mesh` 层级及无 RigidBodyAPI 的完整祖先链；moving body 必须是
启用的 dynamic RigidBody，具备 threshold=0 的 PhysxContactReportAPI，指定 collider
必须属于该 body。原 Table 的尺寸、材料和姿态仍由原 scene preparation 检验，
本模块不把仅路径验证当成全部原物理验证。stage_id 为调用者显式提供，
记录为 `caller_supplied`，没有冒充 native stage cache 读回。

`classify_full_report(raw_report, identity)` 复用
`tabletop_stereo_contact_calibration._report_pairs` 的数组归属/范围及冲量解析。
它只统计 stage、moving actor/collider、实际 table collider 全部匹配的 contact 样本，
同时保存原始 table actor 字段和分开的 contact/friction impulse；不猜静态 actor 路径，
不选受力侧，不把两冲量通道合成已验证总力。支持两种 actor 顺序。
未知 partner 不计入该单元；错误 stage、越界、NaN、非单位 normal、
CONTACT_LOST 附带正向样本等情况拒绝。

结果的 `table_actor_identity_verified` 和 `force_on_moving_body_world_n` 始终为 None；
`absence_verified/clock_qualified/native_cell_qualified/teacher_admission` 始终 false。
空数组仍不能证明无接触。本包没有原生 provenance 验证器，CPU 合成样本只验证解析。
函数不订阅、步进或保留 stage owner，finally 清除 native-bearing 参数和 prim aliases。

## 安装 API 的确切语义

安装文件：
`/mnt/mydisk/xk/envs/dh116/lib/python3.10/site-packages/isaacsim/extsPhysics/omni.physx/omni/physx/bindings/_physx.pyi`
SHA256 `b75b94176c7e6b766bf6f36058d2115016eaabc0d45d8b36a28c64cbf29f4dd9`。

- 第 1380–1390 行 `subscribe_contact_report_events` 接收 header/contact 两数组，
  返回 subscription holder；callback 在 simulation step 后调用。
- 第 1392–1402 行 `subscribe_full_contact_report_events` 另有 friction-anchor 第三数组。
  两种 callback 的 SDK buffer 都只可用于一个 simulation step。
- 第 1292–1303 行 `get_full_contact_report` 返回当前 simulation step 的三数组。
- header actor/collider 是 uint64，应用 `PhysicsSchemaTools.intToSdfPath` 解码。
  header 没有物理 timestamp 字段。enum 为 found=0、lost=1、persist=2。
- 已有 `ContactReportReceiptBuffer` 会在 callback 内立即复制数组，并保存
  `arrival_only_with_step_call_window_not_contact_timestamp`。本包直接复用和测试此行为，
  不把 callback 到达 SDK tick 升级为 native contact timestamp。

最短已有 owner 调用位于 `scripts/calibrate_dh116_stereo_contacts.py:178`：
原 handles.source、callbacks 和 full/ordinary subscription holder，
通过 `begin_step → sim.step → end_step → get_full_contact_report/decode → drain` 保存记录。
关闭路径已有 `_close_calibration_trial`。本包不再建立另一套 owner。

安装 `PhysxContactReportAPI.py:594–683` 的 kinematic–static 测试还要求
`reportKinematicStaticPairs`。它不是当前 GPU articulation capability 的正向证明。
本单元只接受 dynamic reporter，因此不会暗中开启该额外 scene flag。

## 历史证据复放与当前阻断

从原始 archive 按独立 SHA 重新运行现有 `audit_dh116_physx_contact_sources.py`：

| archive | 物理行 / SDK | poll headers | contact samples | ordinary / full callbacks |
|---|---:|---:|---:|---:|
| left | 240 / 3–242 | 37 | 36 | 37 / 37 |
| right | 240 / 3–242 | 43 | 42 | 43 / 43 |

所有 collider 都是 `/World/CalibrationObject` 或 `/World/Finger14/collision`。
测试逐行重读后确认这 480 条真实记录不能成为静态 Table 阳性证据。复放的完整审计输出
SHA 与既有成功审计一致；未重写旧 ledger、inventory、stage 或原回执。

原 `2026-11-12-teacher-task-contact-early-readback-failure.md` 记录：
首次 reset 前设置 `/physics/suppressReadback=false` 导致
`Articulation.reset → wrench_composer.reset` 的
`Index array device (cuda:0 does not match data array device (cpu))`，0 条采集。
原晚开启探针在 reset 后改变设置，记录了 256 个 SDK 步，但 headers 为 0。
所以 thumb 的成功 readback 开关不能直接迁移到当前 mounted GPU owner。
本包不提供声称可原生执行的 dual callback 入口。

下一最小正向 native 单元可在另行冻结/批准后复用原标定 owner，以独立 dynamic sphere
接触保持原静态属性和材料的实际 `Table/geometry/mesh`；需要先准备完整 cell builder，
验证读回设备/时序，并绑定 actual stage/actor/collider 和 raw report。
此测试只能证明该静态 collider 的报告可观测性，不能自动证明 mounted articulation
路径、无接触完整性或 force qualification。当前尚未实现该 builder，也未请求或执行它。

## 验证与证据

新增 19 项测试首先因模块缺失全部失败，日志保留。实现后：

```text
DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 bash scripts/python.sh -m pytest -q -p no:cacheprovider tests/test_dh116_static_table_report_cell.py tests/test_dh116_stereo_contact_calibration.py tests/test_dh116_physx_polled_contact_audit.py tests/test_dh116_bimanual_table_contacts_v2.py
```

**74 passed / 4.49 s**，19 新增、55 相关旧回归。覆盖真实 USD 来源、双向 pair 顺序、
根/mesh 路径区别、错误 actor/stage、非有限值/坏 range、callback buffer 复制、
空值 NA 以及两份实际 archive 不能冒充 Table 阳性。
camera agent 只读审查了本模块、原 range reducer 和 installed enum，未发现新增阻断；
未替本包运行测试或原生。

私有证据：
`/home/xk/.local/share/dh116-autonomous-3c80166/evidence/static-table-report-cell-cpu-19qvx06i`。

| 文件 | SHA256 |
|---|---|
| 新 module | `a9a97269f4d2563ac542d7cf996bf7a9849834ff0cbc95ea30fbaf40d7e13429` |
| 新 tests | `2ab660eefeb32788fb4b0d5f7a6c0fa811779c91887aa8d7b47740640197731a` |
| green.log | `796f6493de2a81d887c48f44f70edeaa1a4ef4fbc0f03b52f205798b7abd2056` |
| green-command.json | `8b5985cf452b9f759b4b01fbbd4d11026705005a361d319e83e75fa8edea6293` |
| archive-commands.json | `59a7fc087ca774f0aaf8bb6faa7e1fd90c8e6b31fe47440de71f1e616eca6749` |
| archive-summary.json | `395df7d0296f311e6328da4515420c13d95c928b51a568891d6aa85d98e30727` |
| left-archive-audit.json | `d7306b360553b410c6e7d162a54ba0af465de11289986b186f1d229cb2cbd34a` |
| right-archive-audit.json | `9f4f02c71721ba3cf1b97f8d7f4ebd4c7798b07a813ee374ae4168aa66a9be2b` |
