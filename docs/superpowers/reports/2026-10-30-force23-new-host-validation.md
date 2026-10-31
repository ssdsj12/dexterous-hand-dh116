# 新服务器：每手总法向力 2–3 N、多指分担与稳定抓取

## 本轮要求与实现

用户确认 2–3 N 指每只手的总夹持法向力，不是逐指目标；摩擦力不计入该法向力。允许任意多指承力，原两指相差不超过50%的要求按实际承力指最大/最小值不超过1.5执行。内部初始目标2.5 N/手，恢复时仅在2.1–2.9 N内微调。

新增分配器保留所有已承力指，平摊扣除手掌负载后的手指预算；未接触指继续寻触，不稀释已有分配。监测包含手掌的实际手总力，每个连续144物理步（480 Hz下0.3 s）均须位于2–3 N；均值仅用于控制，不能代替原始稳定性。每手至少两个持续承力指，原始逐指CV≤25%、相对法向、相对杆三点漂移≤0.5 mm及最大漂移速度≤1 mm/s条件保留。

发生滑移时撤销已排队的上升目标并继续补力；普通有限偏差不停止或reset。真实提升目标仍为杆底部离桌30 mm且连续无滑移保持5 s。高度事件不直接导出教师。原关节角度、0.3 rad/s主动速度及0.2 Nm驱动限幅保持，没有杆外力或附着，没有PPO更新。

## 新服务器与恢复

- Host：`a21693232428085248691341`，项目`/root/hand`，Conda环境`/opt/conda/envs/hand`。
- GitHub `10-30`基线：`2b21413ed9ed950f507aaf5a299992e6a33a23ca`。本轮2–3 N修改另行部署，不能把该基线SHA当作包含本轮修改的证明。
- 迁移前代码归档SHA256：`ff2074df4db013c1cb789ef6e549a9f77bda993fc443cbbe7557922599177d3f`；另存stash `6f6e168ad91ada883783507f7075ddedd0cfd9ab`，运行记录未删除。
- Isaac Sim4.5.0.0、IsaacLab0.54.4、Torch2.7.0+cu128、RTX4090；现有环境可用，`pip check`通过。资产仅修正本机路径，原USD内容哈希核验一致。
- 专用Supervisor恢复TensorBoard，服务仅监听`127.0.0.1:6006`；本地SSH转发`127.0.0.1:16006`，HTTP200。替换历史硬编码PID重启脚本，使用项目专用socket管理。

迁移凭据、备份、源码冻结及实际数据均保存在gitignored `runs/deployment/new_host_20261030_a/`，本地另存`hand-deployment-10-30/new-host-a21693232428085248691341/`。不将运行数据或密钥提交Git。

## 实际Isaac试验A：未稳定，未进入提升

```bash
DH116_NUM_THREADS=4 bash scripts/python.sh scripts/run_dh116_multifinger_lift.py \
  --candidates configs/dh116_training/multifinger_pregrasp_v1.json \
  --output runs/multifinger_force23_20261030_a \
  --seconds 18 --wall-seconds 1200 --maximum-poses 4 \
  --reference seed --closure-margins 0 --thumb-spreads 0 --hand-force-range 2 3
```

实际4140步、8.625 s后人工请求结束并完整保存，并非保护停止。四环境全程phase0，reset0、成功0、教师0。末1 s的手总力（均值±总体标准差，N）：

| 环境 | 左手 | 右手 | 全程双手同时连续2–3 N最长时长 |
|---|---|---|---|
| 0 | 3.494±2.424 | 2.534±1.752 | 2.08 ms |
| 1 | 3.109±1.397 | 2.480±2.457 | 0 ms |
| 2 | 3.105±1.646 | 2.518±1.767 | 2.08 ms |
| 3 | 3.984±1.887 | 2.477±1.086 | 122.92 ms |

因此均值接近目标不代表稳定。右手90.9%–98.0%方差发生在每个8步控制区间内部；逐指CV、持续承力及滑移速度也未通过。手掌负载为0，无法用手掌计入分配来解释本轮脉冲。

独立采用实测连杆姿态、源USD关节轴与原mimic折叠重新计算触点法向Jacobian，和运行记录原生COM移点结果最大误差`1.6903540261958838e-7 m/rad`；抽查24个已承力非退让纠偏样本没有反向指令。已排除该点映射及纠偏符号错误，尚未证明力脉冲的唯一根因。

证据：`raw_physics.pt`、`multifinger_trace.pt`、`contact_models.pt`、`native_jacobian_audit.json`；A实际运行源码冻结在`runs/deployment/new_host_20261030_a/force23_a_sources/`。

## 求解器数值对照

B仅请求`--solve-articulation-contact-last`，在创建场景后的步进前被读回校验拒绝：当前4.5不支持该属性；已装IsaacLab的版本说明明确该设置要求Isaac Sim5.1。未绕过校验，B没有物理步，不能当作已执行对照或改善证据。

C显式`--velocity-iterations 64`，默认仍为16；其余A控制器源码、480 Hz、PGS64位置迭代、mimic400、质量、重力、材料和限幅保持。输出`runs/multifinger_force23_20261030_c_velocity64/`，命令其余同A，但`--seconds 8 --wall-seconds 900`。实际3840步、8 s正常结束，reset0、稳定窗口0、提升0、成功0。三个控制模块hash与A相同。

相同7–8 s区间，A/C手总力对比（均值±总体标准差，N）：

| 环境 | A左 | C左 | A右 | C右 |
|---|---|---|---|---|
| 0 | 3.509±2.452 | 3.529±1.102 | 2.529±1.721 | 2.323±2.563 |
| 1 | 3.056±1.596 | 2.680±2.214 | 2.499±2.448 | 2.402±1.933 |
| 2 | 3.218±2.070 | 3.606±0.647 | 2.524±1.835 | 2.005±1.312 |
| 3 | 4.010±2.153 | 5.710±3.423 | 2.523±0.826 | 3.011±1.585 |

共同前8 s，双手同时处于2–3 N的最长连续步数，A为`[1,0,1,59]`，C为`[0,0,1,106]`，均不足144步。增加速度迭代没有形成一致改善，故不修改默认16。

原始动力学审计发现不同指有80–240 Hz附近或宽频的快速振荡，没有统一2/4/8步周期。A环境0右食指的从动速度残差RMS0.627 rad/s，与指力同步相关−0.984，而与补力命令速度相关仅0.029；这是耦合响应证据，尚不能判定从动约束是激励源还是冲击响应。右拇指、左中指的目标领先量都低于0.008 rad，不能仅沿用历史thumb lead cap当成本轮根因修复。

D使用独立辅助harness，在5 s后固定所有手指关节目标、禁止上升，物理设置回到A的16速度迭代。实际3840步、8 s正常结束，reset0、合格保持0。冻结事件记录为tick2400、age5.002083 s，固定目标原值和harness哈希另存。它用于区分补力闭环和接触/联动自身振荡，不是新抓取控制器或提升成功测试。

独立CPU确认5.01 s以后所有主动关节`q_target`的span及逐步变化均为0 rad，上升目标也为0，快速力波动仍未消失。冻结后7–8 s环境0左/右总力分别为`3.021±1.317 / 2.360±1.771 N`，环境1为`3.627±1.317 / 2.888±2.803 N`，环境2为`3.374±0.954 / 2.140±1.264 N`，环境3为`5.557±3.203 / 1.879±0.901 N`。该1 s内四环境双手同时在band的最长驻留均为0步；从动速度残差也持续存在。

因此持续更新补力目标不是快速脉冲存在的必要条件；接触、联动和恒定隐式位置驱动下仍可维持波动。冻结未移除位置驱动或此前形成的动态状态，仅观察3 s，不能宣称已确定唯一激励源或长期自持机制。完整审计及图为`force23_A_C_D_freeze_cpu.json`、`force23_D_case0_before_after_freeze.png`。

## 离线验证与未完成项

迁移基线214项通过；力分配、原始窗口及滑移队列撤销在实际A前375项通过。新增诊断CLI先红后绿，速度迭代及求解顺序相关86项通过（2.62 s）。局部修复：手间平衡使用实际手总力，已承力但几何eligibility变更的手指仍进入控制，多指入口正确声明力范围参与验收。历史A/C/D元数据曾保留旧入口的false标记，仅为声明错误；实际monitor始终实施原始力范围验收，历史记录保留不改写。

三项审查修复均在D完全退出后部署。最终定向验证 **483 passed in 6.20 s**，shell语法及Python编译通过。最终代码相对实际A/C/D另有这些审查修复，尚未重开最终版本的Isaac，不能称483项离线回归为真实抓取通过。

```bash
DH116_NUM_THREADS=2 bash scripts/python.sh -m pytest -q \
  tests/test_multifinger_control.py tests/test_multifinger_monitor.py \
  tests/test_multifinger_motion.py tests/test_multifinger_entry.py \
  tests/test_pinch_reference.py tests/test_grip_lift.py \
  tests/test_dh116_solver_iterations.py tests/test_manual_joint_contact_control.py --tb=short
bash -n deploy/hand_services.sh scripts/switch_current_tensorboard.sh
```

尚未完成稳定2–3 N双手夹持及真实30 mm/5 s提升。不能据此启动长PPO或发布合格教师；下一步依据原始波动的因果证据选择控制修正，继续验证。
