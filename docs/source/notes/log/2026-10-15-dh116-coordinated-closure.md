# DH116 闭指切向漂移与臂手协调对照

父节点：[T500 D4 coordinated closure](../todo/T500-m1-dual-panda-o6-bimanual-mpc.md)。Baseline/Candidate `5e3bcc4` + dirty，外部backend dirty；Inline/core0，无硬件、无提交。

## 已结束的 pilot4

`assets/dh116_v2/closure_pilot_trial4_convex_sign` 已结束，不再poll旧session40602。
9807完整物理步、.9807s，四点最长连续有效接触.4013s；末载荷LT/LI/RT/RI为.25449/.25300/.52456/.39783N，右拇指normal separation speed门不满足。最大固定材料滑移1.68433mm；没有四点同时到目标夹力，没有保持/抬升。

停止原因是左拇指局部非前区wire间隙.999935mm，低于原1mm保护；不是旧13N符号错误，也不是已证明实际相交。完整源其它部位保守下界1.00206mm；局部前区最大膨胀穿透.45205mm。保护在本步机器人反力发布前拒绝。`summary.json`/`trajectory.jsonl`/`aggregation.json`保存；聚合源码SHA与快照匹配。

## 测量与候选控制

pilot3末完整帧实际q/qd和已验证USD点J计算，LT/LI/RT/RI切向速度分别约.821/1.597/.556/.503mm/s。固定掌心仅闭指会拖动接触位置，但这不是所有失败的唯一原因证明。

[实施计划](../../docs/superpowers/plans/2026-10-15-dh116-coordinated-closure-core.md)：每侧Panda7+finger11/12/21三主动，两个点3D速度只沿各法向，掌心角速度为零；9约束10控制最小范数，统一缩放限速臂.05/手.02rad/s。其它手指不动，不改目标力/物理摩擦，不提供碰撞资格。

核心先RED缺少模块，后新10项与原闭指/坐标/点J共56通过。实际左右名义USD的9行约束满秩，法向速度正、切向和掌心角速度残差≤1e-10。运行时再先RED缺flag，添加显式默认关闭`--coordinated-closure`、要求closure-pilot、从实际PhysX掌心角J按命名臂列取值，双臂+手目标累计。专项65 passed（`/tmp/dh116-coordinated-runtime-green.log`）。仅接线/数值验证，不是实际抓握。

另保存每步候选rod_state/mesh_end_poses/impulses到保护之前，`reaction_published`明确是否提交反力；原异常帧丢失候选物理状态的问题由AST先失败测试覆盖。11项运行诊断测试通过。这项诊断改变未进入pilot4，已进入pilot5。

## 当前运行

17:55启动pilot5，session91087，输出`assets/dh116_v2/closure_pilot_trial5_coordinated`，日志`/tmp/dh116-closure-pilot-trial5-coordinated.log`。原名义切触姿态/4力目标/100µs/c_v1000/16subsamples/全部安全门不变，仅协调控制启用；30000步上限。前100ms仍落定。源码快照在输出目录`source_snapshot`，runner另存；禁止热更加载代码或重复启动。

继续观察实际源保护、四通道接触、滑移和夹力，只有通过才推进保持/抬升。仍需连续接近最终门、真实负载保持、30mm抬升1s及3次重复，持续目标未完成。

## 接线后回归与几何可承载初筛

完整DH116+moving-contact回归 **341 passed in45.88s**，日志`/tmp/dh116-coordinated-full-regression-v2.log`；此前第一条全量命令误用了不存在的test_moving_contact通配符，未运行测试，已纠正且不算通过。

只读源几何初筛`/tmp/dh116_pad_load_clearance_screen.py`，结果`assets/dh116_v2/pad_load_clearance_static_v1.json`：四个pad分别沿原法向假想独立平移0–1mm，面对未变形直杆、原k1e4与16积分点，无速度阻尼。原区域1mm保护仍通过的采样最大法向载荷LT/LI/RT/RI为3.024/3.885/4.125/2.195N，都覆盖目标1.05/1.35/.92/.76N。只能排除“局部区域门本身必然容不下目标力”这一简单假设；忽略杆变形、机器人和全臂约束，不能下发这组独立平移或声称实际握力已达标。无需由此放宽保护或修改刚度。

## D4-grasp-cross-section 只读姿态分支

pilot5仍运行，至7201步载荷约.161/.213/.256/.264N，左食指分离速度门间歇不满足，最大材料滑移1.045mm，没有夹持资格。pilot4离线复核：四通道载荷/间隙/积分接触时长同时满足最长.864s，但右拇指4613步只因分离速度门不满足而无资格；不能误称这些帧已经完全脱离，也不据此移除原速度门。

当前每只手两个接触点轴向错开44.6/25.4mm，会向柔性杆施加弯曲，而设计要求截面两侧对捏。旧10mm间隔候选即使将抓取区移到端部并翻转轴向，808个候选仍被中节/掌心拒绝（`endpoint_pinch_screen_v1.json`）。

新增只读求解直接令两个`point + radius*normal`圆柱中心一致；共同轴由两法向叉积确定，不使用要求中心间距非零的旧align方法。左162组解出105组，128个桌面/摩擦方向初筛候选；展开端部/翻转后1024个中96个通过手部非指腹筛选（`left_zero_axial_pinch_candidates_v1.json`、`zero_axial_endpoint_screen_v1.json`）。右162组解出52组、98个初筛候选，后续筛选中。

左手前8候选进一步通过实际contact-hull的原6mm/30°patch及12mm/60°外cap、额外.1mm半径裕量、Panda有界IK和完整双臂静态源检查（右手仍原姿态）；静力力矩平衡可行。报告`zero_axial_left_ik_screen_v1.json`，局部patch非前区约2.10–2.18mm，完整源保守下界1.035–1.438mm。仍未生成动作资产/通过扫掠/执行物理；不是抓取成功。

下一步右手同截面筛选→成对完整源/指腹/静力与点J→路径，再独立有载试验。材料观察允许同一手两指观察同一材料截面；现有未接线`GraspEvidenceGate`要求四个互异node ID会错误排除此物理合理抓法，未来接线前须TDD改为左右抓取区互不重叠，而不要求同手两指材料ID不同。不得通过每步换最近节点掩盖滑移。
