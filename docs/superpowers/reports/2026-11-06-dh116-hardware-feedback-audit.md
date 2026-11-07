# DH116 硬件反馈能力审计 — 2026-11-06

## 结论与证据范围

当前仓库及训练服务器未提供可验证的真实 DH116 采样通道。不能确认任何指尖触觉、腕部六维力传感器或独立从动关节编码器已经安装且接通；也不能仅凭未找到驱动就断言实机没有这些器件。当前运行可读的 joint_pos/joint_vel、手掌FK、接触和受力均来自 Isaac/PhysX，必须与实机数据区分。

本报告审查当前 USD/URDF、ROS launch、Python 控制与反馈路径、依赖和设备接口；并参考发行者的[SDK 20260921 手册](https://github.com/sorrowfeng/LHandProLib-Releases/releases/download/20260921/LHandProLib_SDK_Manual.md)。SDK 接口存在只证明软件能力，不证明用户实机型号、固件、传感器覆盖、校准或接线状态。

用户补充约束优先：P0继续推进有界动作合同；Actor保持3844D、Critic保持371D。20D仅作为可选传感接口设计，任何实机未证实的数据不得追加到真实 Student 输入。

## 四类状态

| 分类 | 当前可确认的结论 |
|---|---|
| 已有硬件且当前可读取 | 当前项目和训练服务器范围内，尚未证实任何真实DH116传感通道处于此状态。Isaac数值不能列入本类。 |
| 已有硬件但驱动尚未接通 | 尚无实物安装清单、型号/固件、设备读数和校准记录，不能将SDK支持的通道直接列为“已安装”。下表统一标为未确认/未接入。 |
| 需要额外安装传感器 | 若实机没有三维触觉/压力或腕F/T，相应承托/剪切反馈需要真实器件与驱动。是否需要加装须先核实实物，不能由USD中的接触设置代替。 |
| 只能在仿真获得的特权信息 | PhysX接触manifold的法向/摩擦力、穿透深度、object-only接触归属、原生物体真值和质量、成功判定器状态等。保留给Teacher/Critic/奖励/诊断。 |

“未确认”是证据状态，不表示器件已存在或不存在。

## 通道清单

| 通道 | 模型/SDK证据 | 当前项目实际读取来源 | 实机状态及接入条件 |
|---|---|---|---|
| 主动关节位置 | 每手URDF有6个独立关节；SDK有角度读取能力 | Isaac articulation joint_pos | 未确认/未接入；应读取设备返回的实际角度，明确°→rad，不得用position target代替 |
| 主动关节速度 | SDK有角速度读取能力 | Isaac joint_vel | 未确认/未接入；区分实测角速度与差分估计，记录真实采样时间 |
| 从动关节角 | 每手5个mimic follower表达机械耦合 | Isaac各关节状态 | 未确认独立实机测量；由主动角计算的mimic值不是从动编码器读数，不能据此判断耦合异常 |
| 电机电流 | SDK有电流读数，单位为驱动器比例量而非A/N·m | 没有实机电流采样路径 | 未确认/未接入；须确认实际反馈与指令电流的区别、比例标定和电机常数 |
| 估算力矩 | 可在驱动器确有该接口或电流标定后估算 | PhysX关节力与驱动模型 | 未确认；必须标为driver_estimated或calibrated_estimate，不能把仿真applied_torque当实测 |
| 指尖法向/切向触觉 | SDK手册包含法向、切向、方向及taxel位置能力；压力默认格式可能为无量纲通道 | native contact sensor和CAD接受掩码 | 实际器件型号/覆盖/固件未确认；无器件时不得提供“真实触觉”值，不能把无量纲压力当N |
| 腕部六维力/力矩 | 机械USD/URDF不证明已装传感器 | 当前无真实F/T驱动 | 型号、安装、驱动、重力/惯性补偿和时间同步均未确认 |
| 温度、状态、故障码 | 需要对应设备固件与SDK定义 | 当前无DH116设备读取实现 | 未确认/未接入；模拟器控制状态、配置的电流上限不能冒充实际设备状态 |
| 手掌实际位移/FK | 可由真实承载机构编码器与标定得到 | Isaac body_link_pose_w及模拟carriage | 当前是仿真测量；外部carriage不是DH116内置腕部编码器 |
| 有效承托、对向接触 | 原生接受接触+法向/摩擦+仿真物重 | tabletop_surface_feedback / native contact filters | 当前为仿真特权；实机需标定接触力、方向、归属与可信载荷来源 |
| 普通碰撞比例、接触深度 | PhysX接触分离距离/过滤器 | native manifold | 当前为仿真特权；部分指尖传感不能代表全手碰撞，视觉间隙不能直接冒充穿透深度 |
| 固定锚漂移、相对滑移 | 当前实现读取物体真值位姿/速度 | native object_relative/object_quat | 当前为仿真特权；实机应使用有对应关系的视觉/触觉跟踪与不确定度，不能比较每帧重采样的同编号点 |
| Hold时间、稳定窗 | 当前SurfaceFeedback引用成功监视器 | surface_lift_success.duration/window_good | 现实现间接含30mm目标和GT；可部署版本须另建仅由真实传感驱动、与成功/奖励无关的稳定监视器 |

## USD、URDF、ROS 与运行接口

- 厂家左右URDF各11个revolute joint，其中6 active、5 mimic follower。文件位于 `resources/manufacturer/DH116V2.0_URDF_MJCF_Files/`，机械描述与mimic公式不等于硬件驱动或独立编码器支持。
- 历史fixed_dual_panda资产含Panda；当前视觉训练使用双DH116和外部模拟carriage，不应把历史Panda自由度或carriage自由度算作DH116反馈能力。
- 厂家display.launch启动joint_state_publisher_gui、robot_state_publisher和RViz；Gazebo入口为模型生成和模拟校准。这些不是读取实机的ROS节点。
- 当前 src/scripts/configs 和依赖未找到DH116 SDK/CAN/RS485/串口采样驱动、ros2_control硬件接口或设备时间同步实现。已有仿真接触传感器不代表实物安装。
- 本次服务器只读检查：ros2/rostopic不在PATH；未发现 `/dev/ttyUSB*`、`ttyACM*`、`serial/by-id/*` 和can*接口；lsusb有5个设备，未匹配DH116/LHand名称。网络设备、未识别USB或其它控制主机仍可能存在，不能由此证明没有实机硬件。
- `tabletop_tensor_world.py` 的q/qd来自Isaac joint_pos/joint_vel，FK来自body_link_pose_w；源码明确说明implicit actuator的估算applied_torque不等于实测力矩。
- 现有SupportServo是teacher诊断wrapper，依赖native Jacobian、接触和物体真值。当前24D视觉student没有安装该servo，不能把它当已部署双手力控。

## 可选20D接口的边界

接口只定义数据容器，不接入当前3844D actor。每字段至少携带名称、来源ID、来源类别（measured/driver_estimated/commanded/simulation_privileged）、单位、标称采样频率、校准版本、采样时间、对应控制区间、episode/reset epoch和有效性。数值未知时必须有效位为false，不能用0冒充真实0；命令值与实测值分别命名。

仅在实物确认、驱动接通、单位标定、时间对齐和reset/断触测试通过后，才允许进入独立版本的Student输入。PhysX接触、物体GT、奖励和成功监视器不得进入该通路。20个数值并不包含掩码或age；未来观测维度需将元数据显式纳入新合同，不能挪用现有image/object有效位。

### 20个候选值的逐字段来源设计

下表展开的是可选接口的值列，不是新的 Actor 观测。左右列各自拥有元数据，不能只共享一个“20D有效”标志。当前硬件来源、采样频率和采集时间全部未确认；仿真480Hz采样/60Hz窗口不能写成硬件已经达到的速率。

| 值列 | 名称 | 单位 | 当前实现来源及类别 | 将来实机接入所需证据 |
|---|---|---|---|---|
| 0,1 | left/right_support_n | N | accepted pad normal + attributable PhysX friction，simulation_privileged | 已标定接触力/方向/归属，或可信腕F/T加完整重力惯性补偿；无器件时无效 |
| 2 | support_margin | weight_fraction | 仿真支撑减仿真物重，simulation_privileged | 可信载荷来源和实际支撑估计，明确 estimated，禁止物体GT |
| 3 | support_imbalance | ratio | 仿真左右支撑计算，simulation_privileged | 两侧同一时间窗的有效支撑估计 |
| 4,5 | left/right_paired | boolean | native 对向指腹判定，simulation_privileged | 真实接触位置/方向及接触几何，不由命令闭合推断 |
| 6 | ordinary_fraction | ratio | rejected/accepted native load，simulation_privileged | 全手必要覆盖的接触传感，局部触觉不能证明无普通碰撞 |
| 7 | pad_depth | m | PhysX manifold depth，simulation_privileged | 当前无实机等价通道；视觉间隙或触觉压强不得冒充穿透深度 |
| 8 | anchor_drift | m | native 固定锚+物体真值，simulation_privileged | 稳定建立的视觉/触觉对应锚与不确定度、断触重建规则 |
| 9 | slip_speed | m/s | 物体真值在手掌系的速度，simulation_privileged | 有时间对应的真实目标跟踪/触觉剪切估计 |
| 10 | hold_duration | s | native success tracker，simulation_privileged | 独立真实传感稳定监视器，不能依赖奖励、成功标志或GT |
| 11 | window_good | boolean | native success tracker，simulation_privileged | 同上，必须单独版本化其判定条件 |
| 12,13 | left/right_pressure_limited | boolean | 现有仿真控制器限压标志，simulation_privileged | 真实驱动限流/限压状态的读取与协议定义；指令上限不是已触发状态 |
| 14,15 | left/right_mimic_blocked | boolean | native blocking + 仿真从动角误差，simulation_privileged | 真实可观测的从动角或驱动故障证据；机械mimic公式不能充当测量 |
| 16,17 | left/right_wrist_z_request | m/s | 提交动作14/17乘现有速度尺度，commanded | 控制器实际发出且带时间戳的指令，只能标 commanded |
| 18,19 | left/right_wrist_z_increment | m | 实测模拟body-link窗口末减初，simulation_privileged | 外部承载机构真实编码器+标定/FK；DH116自身不能凭空提供世界位移 |

候选接口版本暂称 `dh116-optional-feedback-packet-v1-design`，只用于未来独立传感适配器的设计。单个通道记录至少包含以下信息（缺少真实采集证据时保持示例的无效状态）：

```json
{
  "name": "left_support_n", "value": null, "unit": "N",
  "source_id": null, "source_kind": "unconfirmed", "hardware_verified": false,
  "driver_version": null, "calibration_sha256": null,
  "nominal_sample_hz": null, "measured_sample_hz": null,
  "capture_timestamp_ns": null, "receive_timestamp_ns": null, "clock_domain": null,
  "interval_start_ns": null, "interval_end_ns": null, "reset_epoch": null,
  "valid": false, "invalid_reason": "hardware_and_driver_not_verified"
}
```

采集时间和主机收到时间不能混用；只有 receive 时间时须明确来源并判定延迟/对齐能否接受。重置、过期、掉线、断触或单位/校准不明都会使对应通道无效。未来进入 Student 的字段必须通过单独白名单和新观测版本，当前没有把此 packet、20个值或模拟器反馈写入3844D输入。

## 接入验收证据

每个将来启用的真实通道均需提交：左右手型号/序列号与固件、器件安装及覆盖清单、驱动/API版本、真实设备读数（与命令对照）、单位与标定、采样率/延迟/单调时间戳、掉线/过期处理、reset/断触时序，以及驱动估算与实测的区分。没有这些证据时维持未确认，不阻塞当前P0动作合同工作。

## SDK 状态、温度和时序的补充核实

发行者手册明确有 `get_now_alarm`、`get_now_status`、`get_enable`、`get_position_reached`。未找到专用温度getter；通用SDO参数读取接口不代表温度寄存器已经确定。协议过温告警不等于连续温度读数。getter未返回设备采集时间戳，实际采样率、总线延迟和同步尚未实测；配置单位1ms不能写成已经验证1kHz。

SDK的转矩协议单位文字存在量纲不清项，不能未经设备与标定确认就转换为N·m。压力格式、驱动电流比例量、力估算、实际触觉力和控制指令必须分别记录。审计所读20260921手册SHA256为 `7b82a4a75a191b8d0880ab41e25d929f257b8e54d60e4a86e3b3a207333bcd05`；原始证据JSON保存在本机artifacts，不进入Git。所有硬件接通状态仍为未确认。
