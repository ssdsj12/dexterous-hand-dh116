# 双 DH116 实际关节帧、Jacobian 与原生微扰验证

左右手各自使用其 USD 的完整关节树后，三指指腹位置与 SDK 实测刚体位姿吻合，最大位置误差分别为 0.000354 mm 和 0.000399 mm。53 组原生短微扰均未触发终止。当前 24D SDK 动作顺序的正负方向得到验证，右 finger12 小幅正负运动均有正确响应。该结果覆盖局部运动映射，右 finger12 在接触与腕部共同运动下的跟踪失效仍由独立 T1 物理证据诊断。

## 实现与模型合同

新增 `tabletop_bimanual_physical_action_adapter.py`。`HandPhysicalKinematics` 分别读取左右 USD 原始关节轴、关节前后连接帧、固定关节和三种指腹 CAD 区域。`fk_full` 使用实测十一关节状态，包括五个独立读取的被动关节；`fk_active` 另算既有 affine mimic 的预期几何。位置和法向 Jacobian 使用关节轴叉积解析计算，再通过实际已有的 named coupling matrix 折叠到六个主动自由度。

`BimanualPhysicalActionAdapter` 按 `left/right` 分别解释主动关节命令、world XYZ 平移和 seed intrinsic XYZ 旋转。指关节依实际 SDK joint name 映射，不通过理想镜像改写轴或速度符号。原 24D 语义已经使用上述 SDK 坐标，`decode → to_sdk` 保留原命令，CPU 随机往返误差低于 2e-16。掌局部平移和世界角速度单独输出，实际 base 的跟踪滞后不会被反算成 authored seed。非零内禀角诊断必须显式传入真实 authored wrist seed。

Checkpoint 中新增 adapter 时必须包含 `physical_action_adapter` 的精确版本、左右资产 SHA、代码 SHA、动作语义和 affine mimic 系数。旧 Checkpoint 缺少该字段时 `adapt_for_checkpoint` 明确拒绝接入。当前既有模型和默认 M0 继续使用原路径。源码把该 adapter 标记为独立原生验证候选，未赋予硬件校准资格。

原生刚体列表没有独立 `palm_link`，诊断中的 `palm_link_name` 因而明确记录为 `base_link`。测量是实际 DH116 base/wrist 参考，不代表经过硬件标定的人体学掌心。左右实际 base→palm 变换独立读取，没有臆造新的掌心偏置。

## 证据目录与身份

独立实验目录为服务器：

`/mnt/mydisk/xk/dh116/runs/stage5_physical_mapping_20261108T0840Z/t2_mapping_probe`

冻结源码位于同级 `t2_mapping_probe_source`，启动清单 `t2_mapping_probe_launch.json` 记录逐 Python 文件 SHA。使用原生单实例锁，8 个物理环境、480 Hz 子步、60 Hz policy step。各 case 先实际 reset，按同一既有初态模板运行四个策略步。case 包括零动作、24 维分别 ±0.03、左右手各自手指与腕部联合 ±0.03，共 53 组。每次脉冲持续 32 个物理子步、约 66.67 ms。原控制路径、保护、物理、执行器和成功标准沿用现有配置。

`mapping_probe.jsonl` 共 1,696 条完整 JSON 行，为 848 个 reset 与 848 个 after 的逐手快照。每条绑定环境、抓手、episode_uid、物理步和仿真时刻，并保存完整实测十一关节、实际刚体位置、法向及 FK/FD 误差。JSONL SHA-256 为 `3500353fad2fdadf9a22550528da7c1440785d11a74bace471da976521e2b1a2`。`mapping_probe_summary.json` 报告退出成功、848 个 after 快照和空 failures 列表。统计保存在同级上层 `t2_mapping_analysis.json`。

左右 USD 资产 SHA 分别为：

- 左 `7c6b319074281775c45bc83ac8f434b9a4b7e7e7ba18815e345577d068eb9e6c`。
- 右 `37ff2a3a724778696f96732f713c0b55c01cc96de0f1c35d4c7e8a7fd7532f71`。

## 实测几何与耦合误差

下表对全部 reset/after、各环境及 thumb/index/middle 指腹参考点合并统计，位置单位为 mm。几何点是 CAD 接触区域的参考，不冒充原生接触。

| 手 | 实测 q→SDK 位置误差 P50 | P90 | P95 | max | affine q→SDK 位置误差 max |
|---|---:|---:|---:|---:|---:|
| 左 | 0.000081523 | 0.000260390 | 0.000303143 | 0.000354279 | 0.000353616 |
| 右 | 0.000078280 | 0.000187219 | 0.000212376 | 0.000398542 | 0.000398518 |

| 手 | 法向角误差 P50 rad | P90 | P95 | max | 被动耦合残差绝对值 max rad |
|---|---:|---:|---:|---:|---:|
| 左 | 4.849e-7 | 7.219e-7 | 7.388e-7 | 1.256e-6 | 1.606e-7 |
| 右 | 3.749e-7 | 8.044e-7 | 8.044e-7 | 9.398e-7 | 9.117e-8 |

各行使用源关节帧和实际 root quaternion；没有把右手 source joint12 用左手反射轴替代。原生微扰中的被动耦合残差很小，不能据此否定强接触阶段可能存在的跟踪或耦合偏差。

## 解析与有限差分 Jacobian

实际原生快照上采用中心差分 ε=1e-6 rad。以下 P50/P90/P95 是各快照自身三指全部分量最大误差的分布。

| 手 | 位置 Jacobian 误差 P50 m/rad | P90 | P95 | max | 法向 Jacobian 误差 max /rad |
|---|---:|---:|---:|---:|---:|
| 左 | 1.321e-10 | 2.433e-10 | 2.433e-10 | 3.675e-10 | 2.366e-10 |
| 右 | 1.322e-10 | 2.911e-10 | 2.911e-10 | 4.652e-10 | 2.135e-10 |

另外以 CPU 读取两份真实 USD，各取 64 个相同的有效随机主动关节状态及相同 root 位姿，独立验证解析导数。`tools/audit_bimanual_mapping_cpu.py --output ../t2_source_jacobian_cpu.json` 产出固定 seed=1165 的独立文件。左/右位置导数欧氏误差 max 分别 1.358e-10、2.110e-10 m/rad，法向导数 max 分别 3.534e-10、2.968e-10 /rad。CPU 统计包含与对应指腹无祖先关系的零 Jacobian 列。该 CPU 文件明确记录 `native_physics_episodes=0`。

## 正负动作与联合动作响应

每个单维脉冲按同 env、同 hand 的零动作轨迹扣除自然漂移。384 次单维/正负/环境方向检查中，380 次朝请求方向运动，4 次无可辨别增量，0 次反向。4 次静止均为左手中指主动关节 `finger31` 的负向脉冲，位于原有下界限制的一组环境；对应正向脉冲均可移动。详表和逐环境增量保存在 `t2_mapping_analysis.json`。

| 原 SDK 动作通道 | 左手响应中位数范围 | 右手响应中位数范围 | 单位 |
|---|---:|---:|---|
| 六主动关节正向 | 0.000391～0.000427 | 0.000393～0.000427 | rad |
| 六主动关节负向 | −0.000427～−0.000195 | −0.000427～−0.000393 | rad |
| world XYZ 正向 | 0.016928～0.017032 | 0.016928～0.017017 | mm |
| world XYZ 负向 | −0.016972～−0.016958 | −0.017107～−0.016958 | mm |
| 内禀 XYZ 正向 | 0.000268～0.000274 | 0.000268～0.000274 | rad |
| 内禀 XYZ 负向 | −0.000274～−0.000268 | −0.000274～−0.000268 | rad |

右 finger12 正/负响应中位数为 +0.000427092 和 −0.000427093 rad，八环境方向均正确。四组左右手/正负的指腕联合微扰也全部完成，没有触发 joint_limit。四步内的测得位移小于速度请求积分，与驱动滞后和现有保护的存在相容。当前微扰文件没有另存每个 SDK 目标，表中响应不能替代“实际目标减实测角”的跟踪误差；对应逐子步目标误差由 T1 专用审计记录。

## 执行命令与限制

CPU 合同回归命令为 `PYTHONPATH=src /mnt/mydisk/xk/envs/dh116/bin/python -m pytest tests/test_bimanual_physical_action_adapter.py -q -p no:cacheprovider`，13 项通过。覆盖左右独立轴、全十一关节及耦合六关节解析导数、法向、非理想被动联动、24D 往返、实际 SDK names 排列、wxyz/内禀旋转、旧 Checkpoint 拒绝和原生快照分层。

原生入口为 `launch_native.py --root /mnt/mydisk/xk/dh116/runs/stage5_physical_mapping_20261108T0840Z --name t2_mapping_probe --mode probe`。实际命令、环境和源码 SHA 以启动清单为准。独立的 CPU USD 审计只加载 packaged USD 库，没有创建 Kit 或物理实例。

本实验验证源 FK、实际 SDK 帧和小幅动作方向。持续受载、强接触下 joint12 跟踪、Teacher 4/4 回归和 Student 抓取成功有独立验收项；此处没有产生可计入 Teacher 或 Student 的成功回合。当前 adapter 保留精确 SDK 语义，原有关节保护没有放宽。

## 完整重复探针的目标跟踪与身份校验

首次探针将 `case` 单列存储，但 episode_uid 跨 case 重复使用。独立重复 `t2_mapping_probe_complete` 已将 case 纳入 episode_uid，同时补录 active SDK 提交目标、root 提交目标和内禀角目标。原数据保留。重复实验仍为 53 case × 8 环境、1,696 条完整 JSON 行，848 个 after 快照，424 个物理 episode_uid 和 848 个 episode/hand 组合。三指位置/法向误差的最大值与首次探针相同，未出现新增终止。

新 JSONL SHA-256 为 `78c5ac596605d5d65243d39c67ca93127b467d5390911c2b236a52b83eedea9a`，统计文件为实验总目录下 `t2_mapping_complete_analysis.json`。逐 reset/after 配对检查全部保持相同 case、env、hand 和 episode_uid；每对间隔 32 个物理子步，时间差均为 0.06666667014360428 s。全文件的物理步和时间戳单调非递减，after 时间均严格晚于对应 reset。字段 `active_q_rad` 与独立按十一关节名字取出的实测主动 q 完全相同。

此重复探针的 `sdk_active_target_rad` 来自 `world.commands.float()`，与控制器给 `set_joint_position_target` 的 float32 表达式相同。这里明确将它记录为**重建的 SDK 提交值**，没有声称它是 PhysX后端目标的独立读回。SDK 缓存读回与重建值的来源区分由 T1 schema v2 逐子步审计另外保存。下表统计所有 after 快照中的绝对“提交目标减实测角”误差，包含未扰动维度。

| 手/关节 | P50 rad | P90 rad | P95 rad | max rad |
|---|---:|---:|---:|---:|
| 左全部主动关节 | 0 | 9.835e-7 | 0.000172734 | 0.000222325 |
| 右全部主动关节 | 0 | 1.353e-6 | 0.000172690 | 0.000217915 |
| 左 finger12 | 0 | 9.239e-7 | 0.000172883 | 0.000172943 |
| 右 finger12 | 3.725e-9 | 9.034e-7 | 0.000171108 | 0.000172907 |

| 手 | root 平移提交跟踪误差 P50 mm | P90 mm | P95 mm | max mm | root 姿态角误差 max rad |
|---|---:|---:|---:|---:|---:|
| 左 | 0 | 0.002988 | 0.003100 | 0.005249 | 0.001265 |
| 右 | 0.000030 | 0.002981 | 0.003099 | 0.005232 | 0.001264 |

root 姿态角误差来自实际 base quaternion 与 root 提交 quaternion 的相对旋转。两手小幅动作下存在相近驱动滞后，不能从该四步微扰外推强接触时的 finger12 最大误差或长期安全性。这些短探针采用同一物理初态模板重复 reset，424 个身份是独立物理 episode 身份，不能代替 Z0～Z3 的每抓法 100 个独立初态评估。
