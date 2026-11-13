# Flexible Teacher rollout：模型构造前 CPU 准入审计

本包基于 210d865fbe55e1d838b80d8913e154f4fb482439，只新增 CPU 审计模块、CLI、测试和本报告。原 Pinch/Tripod/Power Teacher 配置、模型、PPO/GAE、task、R0、预算、物理 worker 和 checkpoint 均未改。没有构造 policy、optimizer 或 solver；新增物理步 0，优化更新 0。

现有 Teacher loader 已拒绝将 native_ready 改为 true；现有 PPO 实现检查结构、on-policy token 和 GAE，但没有把当前来源证据连接到模型构造前的逐抓法检查。新增入口补充该报告边界，没有训练或执行开关，也没有新的资格授予器。

## 当前三种抓法均拒绝

模块复用原 load_deformable_teacher_config、require_teacher_common、plan_flexible_experiment 和 frozen base PPO SHA 检查。三个 checkpoint namespace 继续独立。缺少可用的完整前动作 371D 时，实际 source gate 拒绝；伪造 371 个 finite zero、valid=true 和来源字符串，也不能越过当前只实现部分通道的来源适配器。

Reset、实际 issued action24、同钟 rod、同一根杆的双手物理、native R0 与明确批准的原生预算，均需要独立原生来源验证。当前没有完整 deformable rollout qualification verifier，因此本审计**没有正向放行路径**。这是当前缺口的明确状态，不能把 admitted=false 改为 true 或添加自我声明的 approval 字段来解决。它不是已完成的原生 rollout launcher。

| 抓法 | 本次绑定的既有 scoped receipts | 结论 |
| --- | --- | --- |
| Pinch | 已存 1 行原始左手来源 inventory；CPU R0 fixture 结算 | 原始账本重放仍 0/371 admitted；fixture 不授予 R0 原生资格 |
| Tripod | 独立 CPU 几何准备 | 几何范围不等于双手同杆接触、Reset/371D 或 R0 |
| Power | 独立 CPU 几何准备 | 独立 namespace 保留，几何范围不授予原生资格 |

每份 receipt 先核验 SHA。来源 inventory 的原输入再核验 SHA，并直接用核验过的原 ledger 字节调用原 audit_legacy_pinch；不采用 receipt 自称的 native qualification。R0、pose、单杆平面 pair calibration 仅作为范围证据。API 会拒绝未知 role、跨抓法 R0 引用和请求中的自我批准字段。

## Smoke 预算与 GAE 边界

沿用原 v2 Teacher smoke 示例，而不是把控制步当成物理步：

- 2 environments × 32 control steps = 64 transitions。
- 每 control step 8 个物理样本：256 measured physics steps/env，总 512 行。
- setup 及 solver substep 数仍为 null；总物理预算不能当作 512 已确定。
- 示例 planned updates=1、wall_seconds=300；本审计 actual updates=0，未授权或执行该计划。

新增 gae_boundary() 只给出 mask 语义，不计算 reward/value，也不授予准入：

| 情况 | done | bootstrap_mask | GAE carry |
| --- | --- | --- | --- |
| 持续 transition | false | true | true |
| 真 terminal | true | false | false |
| 时间/预算 truncation | true | true | false |

bootstrap 要求 value 来自相同 env/episode/reset/rod 和精确物理端点，显式处于 post_action_pre_reset；reset 后的 observation/value 不可替代。terminated 与 truncated 同时为 true 或非布尔值会被拒绝。测试直接复用原 OnPolicyRollout.returns_and_advantages；没有复制或改写 GAE 公式，没有构造 policy/optimizer。

## 验证与重放

先写测试并观察缺模块失败；再先写“SHA 核验后文件变化”的失败测试，修正为重放同一份已核验字节。新增 **18 项 CPU 测试**，相关 **91 passed / 5.41s**；将 CLI 源码 SHA 也加入输出绑定后，专项再次 **18 passed / 3.04s**。拒绝路径通过 monkeypatch 禁止构造 DeformablePrivilegedTeacher/Adam，并核对 Torch RNG 不变。

~~~sh
PYTHONPATH=src:.:tests OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' \
/mnt/mydisk/xk/envs/dh116/bin/python -m pytest -q -p no:cacheprovider \
  tests/test_dh116_deformable_rollout_admission.py \
  tests/test_dh116_teacher_preaction_readback.py \
  tests/test_dh116_deformable_control_step.py \
  tests/test_dh116_deformable_task_budget.py
~~~

私有证据目录：
/home/xk/.local/share/dh116-autonomous-3c80166/evidence/teacher-rollout-admission-20261113

- request.json SHA256：911c1deea0a02ff4123672ad5b28e10a5d18321c3a6542b3be490f58a03bacc9
- audit.json 与 replay.json 均为：282b82e21e967193296a54cfe758d0d95c85434bfd5c8d160ce55cd6509a9a90
- 两次 CLI 退出码均为 **3**，表示审计已完成但 rollout admission denied。cmp 确认输出逐字节一致，原 receipts 未改。

~~~sh
PYTHONPATH=src:. OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' \
/mnt/mydisk/xk/envs/dh116/bin/python scripts/audit_dh116_deformable_rollout_admission.py \
  --request /home/xk/.local/share/dh116-autonomous-3c80166/evidence/teacher-rollout-admission-20261113/request.json \
  --request-sha256 911c1deea0a02ff4123672ad5b28e10a5d18321c3a6542b3be490f58a03bacc9 \
  --output /absolute/new/private/audit.json
~~~

输出必须不存在，禁止覆盖。模型/任务/源码和每份输入的完整 SHA 均写入审计结果。可重放性限于相同源码和已绑定输入；不等同于原生采集或训练可复现。
