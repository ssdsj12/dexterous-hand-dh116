# Multifinger Nonslip Lift Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** 多指较均衡地持续承力，在无持续相对滑移时提升圆杆并保持。

**Architecture:** 纯Torch稳定性/相对位移监测与原生多指Jacobian位置驱动分开。新显式诊断入口接入现有物理执行/故障隔离与原始记录，保留普通PPO入口。初始几何候选与真实接触、真实提升分开验收。

**Tech Stack:** Python、Torch、源USD几何、Isaac 4.5/IsaacLab、原生PhysX contact/Jacobian。

## Task 1: 原始数据及运行时接触检查（主执行者）

- [ ] 读取I原始数据，确定失联与杆相对双手位移，检查native collider变换/法向与Jacobian方向。只在一个Isaac进程内读取，不改变物理参数。
- [ ] 保存运行时body视图/源几何与接触证明，不能将约1mm差异直接当作固定偏置扣除。

## Task 2: 纯稳定性与防滑监测（独立worker）

Create `src/go2_pvcnn/control/dh116/tabletop_multifinger_monitor.py` and `tests/test_multifinger_monitor.py`.

- [ ] 先测试共同刚体移动、单手相对位移、零力假稳定、冲击均值假稳定、任意多指组合、载荷分配、partial reset。
- [ ] `rod_points_in_hands(object_pose[N,7], hand_pose[N,2,7]) -> [N,2,3,3]`：杆三点变到各自手坐标；wxyz，保留dtype/device。
- [ ] `GraspStabilityMonitor(n,device,window_steps=144)`：`update(loads[N,2,5], relative_points[N,2,3,3], normals[N,2,5,3], dt, valid[N], reset[N])` 返回`ready[N]`、`loaded[N,2,5]`、`force_cv`、`relative_drift_m[N]`、`slip_rate_m_s[N]`。ready要求完整有效窗口、每手至少任意两指>0.05N持续承力、原始CV≤.25、各指mean的0.5..1.5范围、双手总负载比≤1.5、存在loaded法向dot≤−.25的对指、窗口内相对位移≤.0005m、速率≤.001m/s。每次update验证数值和形状；坏行/selected reset仅清该行。
- [ ] `latch(ids,relative_points)`/`anchor_error(relative_points)`保持参考不随时间漂移；reset清相应anchor。
- [ ] 运行 `DH116_NUM_THREADS=2 bash scripts/python.sh -m pytest -q tests/test_multifinger_monitor.py`，先确认测试红，再实现、全绿。

## Task 3: 多指力分担与原生控制（主执行者）

Create `tabletop_multifinger_control.py`, tests and explicit `scripts/run_dh116_multifinger_lift.py`.

- [ ] 先测试Jacobian COM移点、mimic折叠、均衡分配与限幅、非指定指组合、力变高退让、坏数据。
- [ ] 从所有五指节取实际contact数据；接触点/法向以实际manifold优先。逐指关联body及folded Jacobian。
- [ ] 采用原驱动的阻抗关系缓慢补力，监测原始波动；所有12个主动关节可参与，保留实际角度/速度/扭矩限制，记录所选内部力目标。
- [ ] 接入稳定性监测后锁定杆相对双手参考；稳定后计划上升，滑移时撤销超前目标并继续纠偏；达到30mm后保持并监测5s。
- [ ] 原PPO、旧诊断默认行为不变；共同记录原始全指力和实际状态，不复用只有thumb/index的聚合来代表多指。

## Task 4: 候选与实际验收

- [ ] 选择或生成多指可达的合法预抓姿，源几何0.3mm初始容限/桌面净空0.3mm复核；标记geometry-only。
- [ ] 冻结运行源文件和参数；单Isaac进程先小批次短测，记录稳定/滑移/力分配/上升原因。
- [ ] 执行真实物理提升保持，独立检查物体底面、连续保持、防滑、原始各指CV/分担及源表面穿入。
- [ ] 更新README、CHANGELOG及报告，保存实际失败或成功，校验本地备份；不以测试绿宣称抓取成功。
