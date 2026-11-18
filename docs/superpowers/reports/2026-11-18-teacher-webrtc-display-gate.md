# DH116 Teacher WebRTC 显示门槛与物理循环解耦

## 原生历史证据

核验 `teacher_visual_copy_20261117T043644Z` 的已封存回执。三环境完成 480 个控制步，首帧和末帧各核对 102 个 link，最大位置误差为 0，旋转误差分别为 2.80×10⁻⁷ 和 2.19×10⁻⁷ rad。外层进程最终 SIGSEGV（−11）。这段回放的姿态断言没有中断推理。原入口达到控制预算后保持末帧，抓法按钮只切镜头。查看现场现象需检查 `control_step` 和进程回执。

## 实现

`view_dh116_teacher_grasps_webrtc.py` 的 `_publish_audited_view` 继续使用 0.1 mm 和 0.001 rad 门槛。超差或非有限的数值审计结果会隐藏 Viewport，把控制步、SDK 时钟和最大误差写到 `display_rejections.jsonl` 并 fsync。物理循环按原预算继续，下一合格帧恢复画面。首次合格才生成 `ready.json`，`viewer_complete.json` 单独记录完成控制步和画面状态。来自物理控制或 `display.sync` 的实际异常仍由原入口保留完整回执并传播。

核对了服务器安装的 `PhysicsContext.set_physx_update_transformations_settings`。参数 `update_to_usd` 和 `update_velocities_to_usd` 分别设置 `/physics/updateToUsd` 和 `/physics/updateVelocitiesToUsd`，作用是 SDK 向原 USD 回写的选择。当前显示副本直接读 SDK 位姿，继续使用原来的隔离 session layer。Viewer 新增 `--checkpoint-root`，工作树可以读取原项目的冻结模型，逐份 SHA 核验。

## 本次验证

- CPU 定向回归 44 项通过，包括注入位置超差、旋转超差、NaN 和 Inf，随后用合格帧恢复显示。
- 原生申请操作范围为一次短重启。GPU 2、单 Isaac、三个既有姿态环境，各运行 60 步。SDK 共到 step 482，零 PPO。首帧、末帧核对的最大位置误差均为 0，最大旋转误差分别为 2.79570565e-07 和 2.61514318e-07 rad。本次未触发姿态拒绝。服务端完成短回放，未验证客户端 WebRTC 接收。
- 已记录 `TEACHER_WEBRTC_FINISHED` 和 `viewer_complete.json`，外层仍为 **−11（SIGSEGV）**。墙钟 67.82 秒。保留失败证据，无自动重跑或强制结束。崩溃根因未确定。
- 60 步短预览只测试动作推进与画面校验，Teacher 对向接触、30 mm Lift、5 秒 Hold、0.5 mm Slip 和 Student 的监督准入均须独立验证。

原生执行命令由 [readiness](../../contracts/teacher_webrtc_display_gate_v1.json) 记录。实验冻结 SHA `fd6c111f6435c857f7556b550a34d788188a6ea8f28ac0f0afe3e43936d3fc41`。服务器证据目录 `/home/xk/.local/share/dh116-autonomous-3c80166/evidence/teacher-webrtc-display-gate-20261118/native-viewer-short`。
