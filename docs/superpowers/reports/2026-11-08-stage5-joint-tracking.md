# 第五阶段右 finger12 原生跟踪审计

本次 1024 环境、32 个策略步的独立 M2 原动作重放再次复现两次三指 `joint_limit`。两个环境均由右 `finger12` 实测角度越过原下界保护线触发。508 个存活物理子步的主动关节名称映射全部匹配，指令目标均在原名义限位内。终止时模拟隐式执行器的正向力矩估算均为 +0.200000003 Nm，实测关节仍向下运动。该结果定位了合法目标与实际运动之间的跟踪失败。接触反力、被动联动与腕部运动各自的动力学贡献仍需干预实验识别。

## 原生范围与来源

实验目录为服务器 `/mnt/mydisk/xk/dh116/runs/stage5_physical_mapping_20261108T0840Z/t1_joint_audit02`，冻结源码位于同级 `t1_joint_audit02_source`，进程退出码为 0。重放的 Actor 原始动作来自第三阶段 M2 的已保存 32 步采样。物理初态合并 SHA-256 为 `f41f42a5ca0f95c539a6dcc275adf2b9cbda69861bbe08119f7937dfd65975d4`，与第四阶段原动作回放一致。

`joint_tracking_480hz.jsonl` SHA-256 为 `691a7a5f5bc2748093306cc5aae0f27dcbfbcc0b2fe03643716335c7feed470a`。其中 env 799 有 256 条，env 932 有 252 条。两个回合各保留一条最终 terminal，记录绑定原始 env、episode_uid、policy_step、physics_step 和仿真时间。env 932 的后续 4 个 no-op 位于独立 `joint_replay.jsonl`，审计写入器保留最终有效样本，跳过关闭回合的 no-op。两个启动目录中的较早 `t1_joint_audit` 只遇到复制时间戳权限错误，未启动原生物理，验证次数为 0。

## 最终有效子步

下界为 0.007912623 rad，容差 0.020000 rad，下界保护线为 −0.012087377 rad。物理指令顺序为原 24D 合同的右 `finger12` 索引 7。实际 articulation DOF 通过真实 joint_names 定位为 11。

| env / episode_uid | policy step | physics step | 时间 s | 原始 Actor 索引 1 / 物理索引 7 | 提交 float32 目标 rad | 实测 q rad | 实测 qdot rad/s | 目标−实测 rad |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 932 / `1:932:0` | 32799 | 252 | 0.562500029 | −0.254340 | 0.039643493 | −0.012651443 | −0.300013900 | 0.052294935 |
| 799 / `1:799:0` | 32799 | 256 | 0.570833363 | −0.267491 | 0.042210873 | −0.013072879 | −0.300022990 | 0.055283752 |

逆变换前后完整 24D、每手全部 q/qd、完整主动目标、被动联动残差、腕部指令和模拟估算力矩均保留于同一行。Actor 索引 1 经 M2 交换后进入右手物理索引 7，方向保持原合同正负号。env 799 原始 Actor 索引 11 的 −1.065073 经裁剪后进入左手物理索引 5，和右 `finger12` 属于不同维度。

这次采集使用审计 schema v1，字段 `sdk_target_rad` 实际保存的是积分器 float64 的 `world.commands`。对独立回放日志中同一提交表达式产生的 float32 目标逐行核对，508/508 条均严格满足 `float32(world.commands)==提交目标`，最大差为 1.85780814e−9 rad。表中提交值采用 float32，误差分位数使用原始双精度目标。后续 schema v2 直接记录 `hand.data.joint_pos_target`，同时保留积分目标及来源标记。若接口缺失，仅将 float32 重构值标注为重构，不能冒充 native readback。原始 v1 证据与冻结源码均保留。

## 同环境完整轨迹统计

| 环境 | qdot 绝对值 P50 / P90 / P95 / max rad/s | 跟踪误差绝对值 P50 / P90 / P95 / max rad | 拇指 follower 残差绝对值 P50 / P90 / P95 / max rad |
|---:|---|---|---|
| 932 | 0.090204 / 0.300010 / 0.391827 / 0.500841 | 0.010176 / 0.040161 / 0.044981 / 0.052295 | 0.005396 / 0.013733 / 0.014327 / 0.018133 |
| 799 | 0.110386 / 0.295051 / 0.300009 / 0.488142 | 0.019335 / 0.049250 / 0.052791 / 0.055284 | 0.004946 / 0.011968 / 0.014509 / 0.018505 |

follower 指真实运行时 mimic 配置中的 `finger13`，按 `q_follower−ratio*q_master−offset` 计算。终止前的残差均低于原 0.02 rad mimic 阈值，无法把它解释为已证明的独立终止原因。完整 JSON 分析保存两个回合的末 8 个子步和各峰值对应的同环境样本。

| 环境与终止物理步 | 右 thumb 指腹有效载荷 N | 该步接受的法向冲量 Ns | 右 finger12 隐式执行器力矩估算 Nm | 拇指 follower 残差 rad |
|---|---:|---:|---:|---:|
| 932 @252 | 2.432588 | 0.005067892 | +0.200000003 | 0.018133115 |
| 799 @256 | 6.415095 | 0.013364782 | +0.200000003 | 0.018504590 |

接触冲量由该物理步原生 `get_contact_data(dt=physics_dt)` 的指腹接受载荷乘回同一 1/480 s 得到，保留了公式和来源。全接触原始 solver impulse 未额外读取，不能把接受指腹冲量当作全手总冲量。力矩来自 Isaac `hand.data.applied_torque` 的隐式执行器模型估算。真实硬件力矩和实际 solver drive torque 均未知。

env 799 物理步 255 到 256 的右拇指载荷由 0.086257 N 升至 6.415095 N，q 从 −0.011577938 rad 降到 −0.013072879 rad。env 932 在步 245–252 始终有约 2.39–2.76 N 的右拇指载荷，q 持续下行，不能只凭一次载荷尖峰解释所有越限。两个环境最后一步均在正向驱动力矩估算已达约 +0.2 Nm 时继续下行。该观测支持继续检查接触与腕部驱动构成的运动冲突，尚不能定量分离接触反力和被动机构作用。

## 可控的下一步单因素实验

从相同冷重置重放原 24D 指令至第 240 个物理步，保持此后左手全部动作、右手手指与平移逐项一致，仅将右手内禀旋转速度 21–23 置零，保持已经到达的腕部旋转目标。对照组继续原动作。比较同一子步的 SDK 目标、q/qdot、接触冲量、mimic 残差与估算力矩。若这种单因素干预消除原事件且手指目标仍逐项一致，可把右腕旋转作为该动作段的可复现贡献因素。之后可另行中和右腕平移，不能同时中和平移与旋转再宣称唯一根因。所有组沿用原物理和执行器限制。

## 验证命令与当前状态

CPU 命令为 `PYTHONPATH=src /mnt/mydisk/xk/envs/dh116/bin/python -m pytest -p no:cacheprovider -q tests/test_bimanual_joint_tracking_audit.py tests/test_bimanual_contact_moment_audit.py`，13 项通过。新增测试覆盖真实 joint_names 重排、非法映射、非有限读数、terminal/no-op、跨 reset 时钟和 UID 唯一性、native SDK target 优先级及显式 float32 重构、实测 base pose 和接触点来源。

原生分析命令为 `python tools/audit_bimanual_joint_tracking.py <run>/joint_tracking_480hz.jsonl --replay <run>/joint_replay.jsonl --output <experiment>/t1_analysis02.json`。输出位于上述实验根目录，输入完整性和 508 条 SDK 目标交叉核对均通过。本次原生动作回放属于故障重现，Student 成功为 0，不能计作抓取能力验证。新物理动作适配器的原生验证和 Teacher 回归由独立阶段报告。

## 接触力矩离线重建接口

后续采集增加同子步实测 `world.roots` base_link pose 和 `effective_pad_points_w` 指腹接触点，各自保存有效性。`tools/audit_bimanual_joint_tracking.py --asset <冻结USD> --asset-sha256 <实际SHA>` 以该回合实测全关节 q 和 base pose 调用真实右手 USD FK，获得 finger12 的世界轴和轴原点，再计算轴向投影 `axis·((point−origin)×force_on_hand)`。只有位于 finger12 下游的实际接受指腹参与求和。

该值是逐指腹法向力加权接触中心与法向合力所构成的近似力矩。指腹内法向变化可以产生未恢复的局部力偶。摩擦、被拒接触、被动联动约束反力和实际 solver/电机力矩均保留未知，不能把重建值直接当成总外力矩。原 T1 v1 数据缺少实测 base pose 与接触中心，接口对此返回未记录，禁止回算为零。三个纯数组测试仅验证符号、祖先筛选和缺失证据处理，原生力矩结论须由后续独立回放产生。
