# Supported flexible Teacher Reset/readback bridge

基线：4dd7bb9cc91d2a3c0f354e9285ac3dfac256b864，Ubuntu 11-17。
本包只补真实支架柔性场景到 Teacher 所需的 Reset/读回前置，不开放尚无来源的 Tripod PPO 门禁。0 新原生、0 PPO；不授予抓取资格。

## 已复现与修复

1. DualHandResetTransaction 原来只提取每手5个 distal pad。实际17-mesh场景是14个手部link（每手5 distal＋base_link＋finger52_link）、table、2 support。传10个pose给PowerSharedRodRuntime.reset会在手已写入后失败。新增CPU测试先复现该失败；现在 _hands 在任何写入前核验完整、对称的5/6/7-link布局、side/body/index/实际prim路径，run在no-step kinematic refresh之后给全部14绑定传实际姿态，不使用旧缓存或 authored FK。
2. LiveGeometryReader 增加显式 contact_bindings，保留原10-pad默认入口。额外接触必须是完整对称布局，不能覆盖原 inspected pad身份，全部位置和COM来自当前SDK getter。净空仍使用原 inspected 的全部碰撞形状；测试特设一个非接触body为最低点，防止将全手保护缩成仅14个solver link。该入口为后继supported collector准备，本包尚未将其写成已完成的collector。
3. Reset没有新增SDK/solver步，保持同一RopeBatch、全部17slot与静态支架。Reset后contact/wrench仍是None，无测量不能补零。保留原关节、mimic、桌面、源几何和成功标准。

## 验证与失败证据

命令环境：现有 /mnt/mydisk/xk/envs/dh116/bin/python，使用Registry原CPU环境；没有启动Isaac。
- 新Reset测试红灯3项；新增reader测试红灯2项，均在生产修复前保存。
- 最终窄回归：27 passed，0 failed，3.61s（新增5项＋原Reset20项＋原reader2项）。
- 20文件扩展回归：281 passed，1 failed，317.45s。失败为 test_reset_and_post_source_remove_all_approach_missing_samples：旧封存记录与当前MeasuredClosureGeometry来源SHA不符。
- 独立A/B：仅把本包两份生产源码临时恢复至基线HEAD原文后，同一失败仍复现（1 failed，7.53s），随后按字节恢复本包改动；没有git reset或用户文件覆盖。
- 逐字段证据：旧记录中的 solve_dh116_tripod_sdf_constrained_direction.py 与 tabletop_deformable_physical_exchange.py SHA不同于基线HEAD，而这两份文件本包均未改。保留旧失败、严格SHA与原数据；不把它包装成通过，也不为该记录重标来源。
- 运行日志在服务器私有 ref902/supported-*.log，精确SHA见readiness。日志/原始数组不上传GitHub。

## 独立审查

按 antigravity-desktop-review，经readonly MCP完成Teacher-Tripod-Smoke R1：双方HEAD和session/sources/policy/source-preflight SHA匹配。其最小三指来源→fresh owner→复用PPO方案可采用；其建议中的“8路”已由Codex以实际bindings与native记录纠正为7路。R2补丁复审发出后立即因外部账户quota失败，未执行，不能称此补丁已获对方复审；未启用付费overages。Codex独立源码及红绿回归已执行。

## 首次求解的真实源码约束

锁定DeformX csrc/bindings.cpp与csrc/sdf.cu证实：set_mesh_poses的teleport写入target/flag，stageMotionK在begin_step才消费；再写teleport=False会覆盖待处理flag。不是多等若干空帧就能证明的同步问题。
已存在 tabletop_mesh_reset_order.RebuiltMeshReset 用原clear/add实现零步commit：add_mesh经commitMeshK提交自身slot；MeasuredMeshReset随后读回current/target/solver/pending。
因此下一包应复用已验证算法，建立独立17-mesh来源plan/readback并全部校验，而不是删掌面/支架/近端网格、伪称第一PPO动作stationary、偷偷推进或修改CUDA后端。现有11-mesh Pinch源码资格不自动迁移给Tripod。

## 下一包与尚未通过

- 独立Tripod每手3-pad（双手6-pad）SDF距离＋同钟Reset，完整7路wrench身份。
- 复用原支持场景的fresh owner，唯一Reset由C04负责；不能把SDK240静置快照当SDK2新Reset。
- 完整17-mesh零步Reset与fresh exchange通过后，复用现有66＋真实rod＋task_context PPO，做独立update1/resume2有界接线。
- Pinch既有原生update1/resume2保留；不重做。Power全握仍需真实掌面/五指力闭合资格。
- 本包不证明三指接触、5秒Hold、Lift或Teacher成功。R2关闭，两台固定640×480相机配置不变。

## 可复现命令

在项目根目录，复用Registry的现有CPU环境变量（不更改Python/SDK）：

```bash
/mnt/mydisk/xk/envs/dh116/bin/python -m pytest -q tests/test_native_teacher_supported_reset.py tests/test_dual_hand_reset.py tests/test_dh116_flexible_live_geometry.py
/mnt/mydisk/xk/envs/dh116/bin/python -m pytest -q tests/test_native_teacher_supported_reset.py tests/test_dual_hand_reset.py tests/test_dh116_flexible_live_geometry.py tests/test_dh116_flexible_physical_exchange.py tests/test_native_teacher_policy_pinch_distance.py tests/test_dh116_dual_hand_episode.py tests/test_native_teacher_session.py tests/test_native_teacher_actual_artifacts.py tests/test_native_teacher_actual_factories.py tests/test_native_teacher_actual_inputs.py tests/test_native_teacher_actual_launch.py tests/test_native_teacher_actual_launch_process.py tests/test_native_teacher_actual_provider.py tests/test_native_teacher_actual_resume_artifacts.py tests/test_native_teacher_actual_resume_budget.py tests/test_native_teacher_actual_resume_launch.py tests/test_native_teacher_actual_resume_origin.py tests/test_native_teacher_actual_resume_update.py tests/test_native_teacher_actual_update.py tests/test_dh116_r0_reset_approach.py
/mnt/mydisk/xk/envs/dh116/bin/python -m pytest -q tests/test_dh116_work_packages.py
```

Registry回归60 passed，0 failed，16.04s。扩展回归的1项历史失败仍保留；Registry passed仅指本包列出的CPU Reset/readback合同，不等于Tripod collector、原生或Teacher通过。
