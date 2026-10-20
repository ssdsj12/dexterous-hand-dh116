# DH116 Forward Top Grasp Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans, inline in this session. No subagents.

**Goal:** 在固定点不变的条件下求解前伸、掌心朝下的双手姿态，最终验证真实夹持与抬升。

**Architecture:** 将方向验收做成独立纯几何组件，读取已标定 palm_frames；在离线候选搜索中加入方向约束，再复用现有路径、接触和 lift 运行链路。禁止通过更改模型耦合、Jacobian 或相机代替姿态调整。

**Tech Stack:** NumPy、SciPy、USD FK、现有 PhysX/DeformX 运行器、pytest。

## Global Constraints

- Isaac 当前由用户要求关闭；离线工作不自动启动 GUI 或 headless Isaac。实际运行前取得重新启动确认。
- 保留固定点和 USD 安装关系；CPU affinity 0，数值线程 1。
- 手指方向 +X、掌心法向 -Z，各自误差不超过 15 度；腕部 Panda link8 位于对应抓取中心后方。
- 求解边界内缩至少 0.005 rad；保留原碰撞、指腹、滑移及接触载荷判据。
- 镜像仅限当前抓取/lift；后续操作仍可非对称。
- 不覆盖 v1/v2 或 runs 证据，不伪造状态兼容指纹，不称仿真驱动参数为额定参数。

## Task 1: 前伸方向的独立几何验收

**Files:** Create `src/go2_pvcnn/control/dh116/forward_grasp.py`; create `tests/test_dh116_forward_grasp.py`.

**Interface:** `forward_pose_report(palm_world, wrist_world, grasp_world)` 接受 (4,4)、(3,)、(3,) 有限数组，返回 `finger_error_rad`, `palm_error_rad`, `wrist_rear_margin_m`, `passed`。

- [ ] 写失败测试：以 `R=[[0,1,0],[1,0,0],[0,0,-1]]` 构造合法掌心矩阵、腕部 x=0.3、抓点 x=0.43，应通过；反向手指、朝上掌心、腕在前方分别不通过；NaN、反射矩阵、非法形状抛 ValueError。
- [ ] 运行 `bash scripts/python.sh -m pytest tests/test_dh116_forward_grasp.py -q`，确认缺少模块导致失败。
- [ ] 实现以下计算，先检查有限值、正交矩阵和 det=+1，再返回布尔验收：

```python
finger_error = np.arccos(np.clip(palm_world[:3, 1] @ np.array([1., 0., 0.]), -1., 1.))
palm_error = np.arccos(np.clip(palm_world[:3, 2] @ np.array([0., 0., -1.]), -1., 1.))
margin = float(grasp_world[0] - wrist_world[0])
passed = bool(max(finger_error, palm_error) <= np.deg2rad(15.) and margin > 0.)
```

- [ ] 重跑上述测试及 `tests/test_dh116_symmetric_geometry.py`，确认正反向和左右标定语义。
- [ ] 只提交本任务模块和测试；不得混入工作区其他未提交改动。

## Task 2: 离线联合求解与路径资格

执行记录：Task 1 已实现并验证，先观察缺少模块失败，再运行新方向测试与镜像几何测试，19 passed in 0.12s。现有 feasibility8 的实际 USD 掌心方向检查失败：左/右手指误差 2.54750/2.50848 rad，掌心朝下误差 0.48813/0.50220 rad，腕后方余量 -0.068506/-0.067448 m。说明旧姿态确实回折，不能仅旋转视角解决。尚未求出新候选。

**Inputs:** Task 1 的验收函数；`configs/surface_references.json` 的 palm_frames；现有 `runs/symmetric_feasibility8/summary.json` 仅作种子，不当作新姿态通过证据。

**Existing diagnostic sources:** `/tmp/dh116_refine_symmetric.py`, `/tmp/dh116_symmetric_approach_probe.py`, `/tmp/dh116_symmetric_jacobian_probe.py`；使用前完整阅读，若不存在不得猜测运行结果。

- [ ] 用实际 FK 的 `base_link` 乘标定 `palm_frames[side]`，记录现有姿态方向误差和 link8 后方余量，证明当前姿态不满足新方向要求。
- [ ] 将已有优化诊断复制为新的、独立版本后再编辑；新增方向残差 `R[:,1]-[1,0,0]` 与 `R[:,2]-[0,0,-1]`，保留指腹圆心、法向和镜像残差。搜索范围内缩 0.005 rad；多种子顺序执行，不并行占核。
- [ ] 对每个候选调用 Task 1 验收及原始 source/指腹区域检查；任何一项失败都输出失败，不发布配置。
- [ ] 只从合格候选生成张开及抬高预姿态；逐姿态验证方向和腕后方条件，执行连续全臂路径检查和点 Jacobian 有限差分。
- [ ] 输出新命名的 runs 报告，记录输入 SHA、方向误差、关节余量、指腹残差、全部路径检查；若无可行候选，展示受限条件并再讨论安装位置，不能擅自换桌边。

## Task 3: 新配置、运行前审核与真实验收

Task 2 诊断进度：forward_feasibility1/2 可前伸但碰撞失败；3 加入桌面残差后仅剩拇指非指腹与杆间隙失败，已生成用户查看的 pose_preview.png。5 改为固定世界采样以避免优化差分中 ceil 采样数量跳变，间隙下界转正但不足 1 mm。6 全部 source 静态检查通过（最小 1.181 mm），但指腹径向法向误差 33–42 度，不能发布。7 增大指腹权重后方向超过 15 度；8 显式约束求解仍未通过方向/碰撞复核。上述均为诊断候选，不证明无可行解，也不构成抓取成功；继续检查多种子与接触几何兼容性。Isaac 未启动。

**Files:** 新的 `configs/forward_top_grasp_v1/`；`README.md`、`CHANGELOG.md`；必要时 `scripts/run_current.py` 的显式配置选项及对应测试。

- [ ] 由 Task 2 合格证据生成独立配置、所有 SHA 和 validation；`physics_validated` 初始必须为 false。
- [ ] 配置入口测试先失败，再加入明确选择项，确保使用原 `--symmetric-lift` 阶段控制，不改变通用分配器。
- [ ] 运行 `bash scripts/python.sh -m pytest tests -q` 并检查完整输出，更新实际结果而非沿用 634 项历史数字。
- [ ] 向用户展示离线可行性和新姿态方向，取得重新启动 Isaac 的确认；在此之前停止于运行准备，不自动开仿真。
- [ ] 单实例实测落桌、接近、闭指、夹持及 lift；两侧真实材料站均升高 >=30 mm 保持 1 秒，高度差 <=1 mm，所有旧判据通过才标记 lift 成功。
- [ ] 原始轨迹/阶段状态保留；更新文档区分离线与实际验收，失败则定位失败阶段，不绑定线缆补成功。

## Review

范围覆盖方向、关节边界、接触、路径、真实 lift 和阶段限定镜像。移动固定点仅为待讨论备选；Isaac 关闭要求优先于自动运行。暂无新的物理成功证据。
