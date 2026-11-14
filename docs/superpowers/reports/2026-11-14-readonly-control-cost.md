# 历史物理账本控制代价的只读核算

基线：8e2dbd5884b8e735ac385a505a0cde7d550fdb11。
独立分支：codex/readonly-control-cost-20261114。
本包只新增 evaluator、诊断配置、CLI、CPU 测试和本报告；不修改原 task、原十项 null
系数、原三抓法配置、checkpoint、物理参数或旧 evaluator。新配置不是训练 profile，
不提供 PPO/native 准入开关，不调用 optimizer、SDK step 或 solver step。

## 已有实现复用与来源

tabletop_readonly_control_cost.evaluate_bytes 对同一份 bytes 先校验 SHA 再解析，
复用原 strict owned-array codec 与 tabletop_deformable_dual_evidence._Sequence。
原序列检查负责 setup/reset/common/SDK/rod/episode/控制窗顺序与 terminal settlement；
新增检查负责左右独立 instance/view、完整命名 17DOF、同一 articulation path/row、
pre/post hand 连续、首行与 pre-action q/qd 相等、原 action contract、
requested/submitted/actual backend targets 对应。每行 actual request.action24
必须等于该控制步 begin.action。历史 owner ID 不证明 live owner 当前存在。

诊断配置固定 14 个原源码/合同 SHA，包含公式、原完整 world.qd 来源、parser/
mounted source/clock/name/action 合同。配置自身 SHA 在新模块中固定：

- config SHA：5f77d8ad37d1d3edbf98b6e2401b8b95a595647cbea69d16a93713400316bbd7
- evaluator SHA：dd9f37e18770329e8635d9493153b65e6c0916bcb06f0e89c1bceb6b8b881616
- CLI SHA：c558423d58174c56f3daf04c5547b8ec02b6dc9f00caa65a8b699e8e28201380

所有输出 training_reward=null、optimizer_updates/new_physics_steps=0、
native_qualification/teacher_admission=false。原 R0 不被修改或结算。

## 原公式与未知量

每个真实 480 Hz 行使用 dt=1/480，控制窗仅累加实际记录的行：

| 诊断项 | 原实现与数值 |
|---|---|
| time | TensorTask.consume：-0.1 * dt |
| action_rate | surface_exploration.reward_rates 的 dex_action_rate：-0.5 * mean(clip((a-a_previous)/2,-1,1)^2) * dt |
| velocity_excess | TensorEnv.record_velocity/step：-0.1 * min((max(0,max_abs_qd-3)/3)^2,25) * dt |

velocity 保留原双手各 17DOF 的完整 numeric reduction。腕部平移为 m/s，
腕部旋转与手指为 rad/s，原实现直接共同取 max；本包明确标注
max_abs_full_17DOF_per_hand_numeric_mixed_m_per_s_and_rad_per_s，
不把该数字重命名成 rad/s，不改原阈值或移除腕部 DOF。极端有限输入若平方无法用
float64 表达，明确报数值审计错误，不截断或当成可信有限源。

上一动作来自同 episode 上一已结算控制步的 begin.action，且上一窗口各实际 request
已核对。历史首 reset 没有 TensorTask previous_actions 回执，controller 被记录为
not_present，因此首个控制窗 action_rate 全部为 NA。原 TensorTask reset 置零不能
当作这些历史手工控制器的实测事实。未知不填零。

这些是记录行上的三条公式分量，不是重建完整 TensorTask failure/scoring mask 或最终
native reward。Tripod 自定义 geometry guard 不被擅自映射为原 failure=-25。
未加入正向几何、接触、成功、材料滑移或支撑奖励。单独优化这些成本会偏向静止，
不能作为完整抓取目标，也不能据此启动长训练。

## 两个真实历史账本

Pinch：
/home/xk/.local/share/dh116-autonomous-3c80166/evidence/approved-approach-r0-20261114-jfd3i7wj/native-runs/flex.f7-approach-r0-native/attempt-wy27jh8g/native/physical.jsonl

SHA：4e9b42b1fc96119888cce34d4977d2fd8720b4385409dda45a10f59fcbb8b83b

Tripod：
/mnt/mydisk/xk/projects/dh116-autonomous-3c80166/approved-tripod-owned-20261114-2xl75l3a/native-runs/flex.f8-tripod-owned-contact-native/attempt-r_cnsg84/native/physical.jsonl

SHA：fc30831476a2247df49c875e185f13106168ffa117b696491ba0b25d94748f33

| style | physical 行 | 控制窗 | time sum | action_rate 已知行 observed_sum | action_rate 完整 contribution_sum | velocity_excess sum |
|---|---:|---|---:|---:|---|---:|
| Pinch | 232 | 29×8 | -0.04833333333333334 | -0.00009621746570074074，224/232 有效 | NA，首 8 行未知 | 0 |
| Tripod | 197 | 24×8 + 5 | -0.04104166666666667 | -0.00012787752263806782，189/197 有效 | NA，首 8 行未知 | 0 |
| Power | NA | 无对应固定账本 | NA | NA | NA | NA |

Tripod SDK 199 的 closure_geometry_guard 与真实最后 5 行保留，不补齐、不支付失败奖励、
不把失败样本变成成功。Pinch outer exit0 不证明 mesh Reset 插值时序已通过；
现有结果不提供本包的 mesh_reset/solver_health/Teacher 资格。

最终输出目录：
/home/xk/.local/share/dh116-autonomous-3c80166/evidence/readonly-control-cost-20261114

| 文件 | SHA256 |
|---|---|
| pinch-final.json | b89fe2db600e56e115ece2e6de7eb7c8463bcacb4f162c0a17d175cf2858e860 |
| tripod-final.json | 05ce3ed7eef50d8844b689db3525f5fe11f05dd1badbb63e2ecfe9f5a1741d51 |
| power-final.json | d3a1b1bf3d88f9ecf369324b8bb8092873b4ff07ce7d9b087e235bce60b67df1 |

三次 CLI 均 exit0，native_imports=[]。首次输出也保留，最终报告使用独立 final 文件。
大型原数据和输出不进入 Git。

## 验证

新功能先红：28 failed，evaluator/CLI 尚未实现。实现后修正真实 dof_types 的 (1,17)
行形状与测试 prefix 索引，28 passed。追加来源、数值、实际账本测试后，root 审查
提出的两项先单独复现：helper 重复 side 被字典覆盖、1e308 平方 OverflowError；
2 failed 后分别增加严格两侧唯一检查和明确数值错误。原序列入口本已有重复 side
拒绝；新 helper 独立调用也拒绝。physics agent 忙于主线，未计入审查。

最终 43 项新测试 + 46 项既有测试：89 passed in 22.07s。
没有 optimizer/update 测试；CPU doubles 只用于既有物理 owner 的字典账本夹具。

Ubuntu 原 argv（在本 worktree，环境 PYTHONDONTWRITEBYTECODE=1，
PYTHONPATH=src:.:tests，OMP_NUM_THREADS=2，MKL_NUM_THREADS=2，CUDA_VISIBLE_DEVICES=''）：

~~~
/mnt/mydisk/xk/envs/dh116/bin/python -m pytest -q tests/test_dh116_readonly_control_cost.py tests/test_dh116_dual_hand_evidence_journal.py tests/test_dh116_flexible_solver_clock.py tests/test_privileged_teacher_reward_audit.py
~~~

实际 CLI 使用同一环境，分别以 --grasp pinch/tripod/power 和上述输出目录下的独占
pinch-final.json/tripod-final.json/power-final.json 执行：

~~~
/mnt/mydisk/xk/envs/dh116/bin/python scripts/evaluate_dh116_readonly_control_cost.py --grasp pinch --output /home/xk/.local/share/dh116-autonomous-3c80166/evidence/readonly-control-cost-20261114/pinch-final.json
~~~

| 日志 | SHA256 |
|---|---|
| red.log | 415d731682faa0e50d57572cd1acffc03e8346b8fdbc9fe9e7c902abb2cf0b6b |
| review-red.log | 5bd099082fb4fbe54605e61766b4b020f8848c2cae9c8368e2af5d77c90d479f |
| regression-final.log | 3dc0ec57447eb91bc438c787f4ba0abcba58a0002245217f7f2c5f503989213c |

正式训练仍需独立训练奖励合同、实际 Reset/health/source 准入与执行预算。
本包只提供可重放的控制代价证据，不添加训练入口或奖励晋级门。
