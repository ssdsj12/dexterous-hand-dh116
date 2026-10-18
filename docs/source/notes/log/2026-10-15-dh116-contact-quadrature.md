# DH116 D4 接触积分与闭指诊断

父节点：[T500 D4](../todo/T500-m1-dual-panda-o6-bimanual-mpc.md)。Baseline/Candidate：`5e3bcc4` + dirty worktree；另有外部 PyElastica-Mesh 修改，未提交。Inline、core0 单线程，不是真实硬件资格。

## 已完成的运行

- `grasp_approach_trial1` 实际960步完成，逐步保护全通过、最小源几何下界1.0012554mm，28个点/J探针通过。但最终summary因 NumPy bool JSON序列化失败，最终实际点位比较丢失，**最终approach门未验收**。已从保存的trajectory/point_probe生成 `recovered_evidence.json`，不冒充缺失的测量。新 `evidence_json.py` 严格支持NumPy且拒绝NaN/Inf，2项回归通过。
- `closure_pilot_trial1` 完成4000步/0.4s，诊断从独立名义切触初态开始，不含连续approach。四通道最后法向载荷 `[.1043723,.0860827,0,0] N`；四点同时接触步数0。132个实际点/J探针通过；最大J误差2.756e-7m/rad。未夹持/未抬升。
- 合成K10/D.2、总arm60/hand.2Nm、100µs耦合/10µs后端、全50体及指腹非前区1mm保护保持。wire无绑定。

## 漏采样根因与修正

旧40单元只在单元中点查询，轴向间距12.5mm。`contact_resolution_static_v1.json`：名义右食指连续线缆已切触，但旧中点gap=1.193mm；640/1280轴向查询接近零。不能通过放宽穿透保护或人为移动线缆轴向位置规避。

外部 `elastica/contact_forces.py`、`_contact_functions.py` 新增可选单元内部复合中点积分：每点力乘1/N；同一重心权重用于节点力、点速度及偏心力矩。原40动力学单元、总刚度、摩擦系数不变，默认1保留旧路径。桌面与四个移动网格统一传递 `contact_subsamples`。

`MovingContactRodEngine.contact_geometry` 在同样积分点返回gap/相对法向速度，包含杆件自旋与当前网格COM twist，并返回原单元ID。DH116观测区分采样序号和材料单元，避免错误滑移索引。

TDD：内核10项先因缺参数失败后通过；引擎5项先失败后通过；观测/CLI两项先失败后通过。最终内核36项通过（含非单位director、自旋功率/刚体共运动、总力/矩/刚度）；项目接线/后端80项通过，有2个Numba非连续数组性能警告，无断言失败。日志 `/tmp/dh116-quadrature-{kernel-v2,wiring-green}.log`。

当前全身保护优化只缓存分离方向，每步仍重新投影全部当前顶点；不足时重算完整方向，绝不复用旧间隙值。20项源几何/区间回归通过，局部5次检查4.280→1.101s；不是Play帧率。

## 当前运行与后续

16:21启动 `closure_pilot_trial2_quadrature16`（6000步、GPU0、session28148），仅改16倍接触积分及其对应观测；力目标1.05/1.35/.92/.76N和所有原保护不变。脚本/外部源码已在输出目录快照。日志 `/tmp/dh116-closure-pilot-trial2-quadrature16.log`。未见最终结果前不得称通过。

D4-contact-resolution 子节点接线DONE，实际验证OPEN。D4-hold-friction 子节点OPEN：后端切向为 `min(c_v |v|, μFn)` 的速度正则化摩擦，当前 c_v=10，不是静摩擦记忆模型。静力摩擦锥可行不能证明该动力学能保持，需实际载荷/滑移证据；不盲目调大阻尼。

仍需：四点闭指成功、固定材料锚点1s保持、双区域30mm抬升保持1s、连续approach最终门及3次重复。静态9个lift IK候选只作准备，不能当运动证据。goal active。
