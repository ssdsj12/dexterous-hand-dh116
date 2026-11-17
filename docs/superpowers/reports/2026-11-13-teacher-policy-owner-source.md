# Teacher B40/B41：原控制器所有者的只读来源

基线 bf7d8e1054d17d1ce28080f0dfb3affcd7e278b1。新增独立模块、CPU 测试、本报告；不修改原 TensorTask、TensorWorld、动作解码、配置、冻结模块或已有来源 gate。实际 native physics steps=0，optimizer updates=0，未创建 policy/optimizer、未启动 Isaac/GPU/solver。

本包落实上一 mounted 报告中已确认的最小设计：只提供原 371D 的 **36 个字段**，另外 **335 项仍未知**。它没有合并 mounted/named source，没有把有效数宣称为 102，也没有接入或放宽 rollout admission。所有输出 native/Teacher qualification 均为 false。它是控制器内存与生命周期的一致性检查，不是 native reset、实际动作执行、完整物理区间或 R0 奖励证明。

## 原语义和实施边界

| block | 半开索引 | 实际来源 | 原归一化 |
| --- | --- | --- | --- |
| B40 current_joint_target | [203,215) | 原 TensorTask.command，float64，L/R active joint 顺序 | 2*(command-low)/(high-low)-1 |
| B41 last_submitted_action24 | [215,239) | 原 TensorTask.actions，在下一 begin_step 前读取 | 原 accepted 24D [-1,1] 值不变 |

command 不来自 SDK/exchange target；不把 float32 目标回读转换为原 float64 command。actions 不来自尚未提交的 actor 输出、previous_actions 或外部任意字典。原 continuous begin_step() 的 clipping 由原方法完成，本模块只复制 accepted 结果；invalid/terminal selected row 被拒绝。本模块不修改 controller buffer、不调用 begin_step/reset/end_step，不发送目标或推进任何 clock；这些方法仍由实际 owner 执行。

实现计划与执行顺序：先建立缺模块失败测试，再实现只读 owner 状态机和独立重建 validator；用原 begin_step/observe 对照精度和 action 语义；用原 reset/end_step 验证 buffer 副作用；增加错误 reset/step/replay 的失败测试，最后生成独立 CPU fixture、运行相关回归和提交。

PolicyOwnerSource 绑定实际原 TensorTask 类型、原 reset/begin_step/end_step 方法、surface action24 flags、原 active limits，以及显式 buffer row 与独立 env_id。实际 task.command/actions 必须是完整 float64 buffer；不接受 SimpleNamespace 冒充 owner、固定默认值、26D 模式或重新解释 SDK target。测试使用 TensorTask.__new__ 和已有 CPU controller fixture，不构造原生环境。

只读调用顺序：

1. 调用方完成原 reset 后调用 record_reset()。选择 actual task._per_env_reset_seed 的对应 row，或必填 task._seed；分别核对 command/actions，初始 action24 必须确实为原 reset seed 的零值。缺 seed 时拒绝，不自动生成零。record_reset 保存实际 run/env/episode/reset/rod 身份；下一 reset 必须显式新 episode/reset。
2. 在 begin_step() 之前调用 capture_pre_action()。只有同一 pre-action shared clock、substep=0 和已记录的 source 状态可被导出。
3. 原 begin_step() 接纳 action 后、control/target write 前调用 record_submission()。核对 original old_command=刚才的 command、previous_actions=刚才的 actions、last_raw_action24=当前 accepted actions，且 command 尚未改变。重复 submission 或缺 pre-action 捕获均拒绝。
4. 下一 capture 要求同一 run/env/episode/reset/rod，policy_step 恰好 +1、SDK physics_step 恰好 +8、rod_tick/solver_time 单调增加。核对 task.actions 仍为已记录 accepted action，以及原 end_step 的 previous=command 副作用。完整来源重建同时检查 SDK tick 与本 reset 起点和 policy_step 的对应关系，拒绝平移或跳过起点。没有记录的中间写入、partial interval、terminal、跨 rod/reset、错 env 或变化中的 clock 均拒绝。

以上 clock 和副作用一致性不能独立证明原方法实际执行或真实完整 8 个物理步骤。相同的静态数值不能充当 native 执行证据；实际 reset/physics/target transport 与 R0 仍须各自外层 owner 的独立回执。本包保留这些限制，不通过任意 receipt、valid mask 或 finite placeholder 授予资格。对于没有真实 TensorTask owner 的新 deformable runtime，本入口会拒绝；不能把 reset 请求或手部 exchange packet 当 controller source。

packet 为独立 dh116-teacher-policy-owner-source-v1，含 owned reset snapshot、最近一次 accepted submission、当前 owned command/actions、精确 clock、原 action contract SHA 和相关实现 SHA。validate_policy_owner() 从 packet 重建 371D mask、values、sources、contract 和零执行计数。未知字段为 NaN / false / null，篡改输出、版本、源码、action contract、计数或 mask 被拒绝。保存的是本包只读捕获发生前已有的控制器状态；new_physics_steps=0 表示适配器本身没有推进物理，不能抹去外部历史执行。

B02/B03（previous/reference）、B16（valid consume residual）、B37–39（requested/submitted wrist/palm）、B70 区间平均仍未实现。也未把刚体 cylinder/object 字段强行映射到 flexible rod。

## 验证

**30 项新增 CPU 测试；相关总计 153 passed / 3.68s，退出码 0。** git diff --check 通过。首轮测试因新模块不存在失败；后续特意将 submission/current tick 同时平移、保持 policy_step 不变，观察到应拒绝但未拒绝的失败，再增加 reset 起点核对，最后回归通过。

测试包括原 begin_step/observe 严格逐值 parity；原 reset/end_step 的实际 buffer 写入（仅完整 observation 物化 stub）；float64 与 float32 target 的区别；accepted clipping；per-env seed；缺 seed、26D、错误 limits/dtype、非法或终止动作；pre/submission 顺序和恰好一次记录；跨 env/reset/rod、partial/skipped control、改变 clock、缺 end_step 副作用；owned copies、Torch RNG 不变、重放和 mask/计数篡改。声明的 CPU clock 前进不计作真实物理步。

~~~sh
PYTHONPATH=src:.:tests OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' \
/mnt/mydisk/xk/envs/dh116/bin/python -m pytest -q -p no:cacheprovider \
  tests/test_dh116_teacher_policy_owner_source.py \
  tests/test_dh116_teacher_mounted_wrist_source.py \
  tests/test_dh116_teacher_preaction_readback.py \
  tests/test_dh116_deformable_rollout_admission.py \
  tests/test_stereo_action_contract_audit.py
~~~

私有证据目录：
/home/xk/.local/share/dh116-autonomous-3c80166/evidence/teacher-policy-owner-source-20261113

- controller-fixture.json SHA256 c2e0f477061a9e1948e44da0c5f005de05d9bac19b9570d21bff13ca0469379a。
- replay-proof.json SHA256 f7784cf22aa5cd90467f2db8b5b83cdc50dce002a34daebe4a811bdb82ee05de。

fixture 明确标记 CPU_CONTROLLER_FIXTURE_not_native_capture；原 reset、begin_step、end_step 各执行一次，完整观察物化被 stub，SDK/rod clock 为 CPU 声明值。reset 和一个 completed-boundary packet 从 JSON 重放后各得到 36 项有效、335 项未知，actual_native_physics_steps=0 / optimizer_updates=0。原始文件独占创建，不修改旧证据。

后续若整合 mounted 与 controller 来源，必须在新的独立复合 validator 中重放两个当前版本，核对同一完整 clock、reset 和来源身份，再计算不重叠 mask；本包没有实施该整合，也没有请求任何追加训练或原生预算。
