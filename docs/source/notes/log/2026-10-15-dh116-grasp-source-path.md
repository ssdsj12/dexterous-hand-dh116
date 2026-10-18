# DH116 抓取源几何、路径和新姿态重力复验

父节点：[T500 D4](../todo/T500-m1-dual-panda-o6-bimanual-mpc.md)。Baseline/Candidate：5e3bcc4 + dirty worktree；未提交，源USD未改。阶段：静态候选→路径预检→实际预抓取重力，不是抓握报告。

## 输入与方法

固定双Panda+DH116 V2线性仿真模型，50cm/8mm自由Cosserat wire，桌面z=.3m。源指腹三角面/重心/朝外法向独立选取，同圆柱接触允许不同轴向材料坐标。新增源FK有界臂IK、凸包分离下界、全50碰撞刚体场景和保守关节区间运动界；仅直接相邻刚体对排除自碰，两个空固定法兰明确检查，不豁免普通手部。

所有数值任务core0/单线程；GPU模拟顺序执行。主仓库assets/dh116_v2存证，脚本不覆盖已有结果。

## 结果

- 早期同圆柱切触候选背部穿桌约4mm，已拒绝，不再沿用。
- 左2/右5个单手候选通过；抓取中心y=±.15m时双臂0/10通过（臂间干涉）。同一wire改为y=±.20m后10/10静态组合通过。
- `grasp_source_screen_v3/summary.json`由正式脚本`verify_dh116_grasp_source_screen.py`重算，10/10通过，最小下界1.042349mm。四目标指腹仍须独立前表面接触门，非整刚体豁免。
- `grasp_source_sweep_v1/summary.json`：选择[0,2]组合，home/open→上方30mm→接触高度/open→名义切触手形，359次区间检查、181个覆盖区间，最小下界1.005630mm。只覆盖固定直线wire和名义源几何路径，不含动态wire/实际跟踪/cooked资格。
- `grasp_prepose_v1.json`：固定上方30mm初始姿态，SHA `268d281a286f37e6ce65578a87786321ecc645b0cb802f65b3a7202b042a164a`。只在初始化使用，不允许用状态写入冒充抓取运动。
- 预姿态gravity trial1/2在step0拒绝：左Panda joint2需要44.935455Nm，超过原40Nm；中指主动轴折叠重力.132518Nm，超过原.1Nm。此前home重力通过不代表扩展姿态通过。
- trial3显式合成仿真arm60Nm/hand.2Nm，PD5/.2及原速度/净力门不变，GPU0已完成480steps/2s且summary PASS。最大g44.935455Nm，左右最终臂误差2.84e-8/5.94e-8rad、手跟踪7.86e-5/1.04e-4rad；wire表面误差2.7287µm、最大速度1.35e-6m/s，四通道力均0、无固定约束。没有实际全几何逐步保护，故仍非完整运动资格。该轮脚本备份在输出目录script_snapshot.py，SHA 7c12e004653d102907ea48ce5aa56a79185ba310e4076aea1323f62237566f36。
- 源URDF与USD均含中指finger32异常较大质量：左.1569719kg、右.1624528kg，显著高于其他指对应连杆。保留源值，不擅自改质量，不称实物标定。

## 测试与后续

此前混合专项报告200；本次明确只运行`tests/test_dh116_*.py`，初轮196 passed，补接近/异常落盘接线后198 passed（/tmp/dh116-regression-approach.log）。后续连通前区分类新增4项通过，最新全量待复验。不同选集不横比计数。

现已新增measured_body_poses逐帧50体检查、source-only证书SHA校验、0.5s落定+3s限速三次插值接近；每物理步前后检查实际位姿，失败保存步/已执行轨迹。`grasp_approach_trial1`正在GPU0/core0，session45112，960steps/4s，日志/tmp/dh116-approach-trial1.log。接近期间四指腹也完整保留1mm无接触保护；不会直接闭指。

新抓取点文件grasp_point_references_v1.json SHA f928490c4c0bf6e5c8e1d9722fe0d4787ed83a7a6f907a5f450b4304c5be4d5b，四点来自source_screen_v3已核验的三角面；approach还检查实际Jacobian与终点位置2mm/法向5°。

## 前表面分类诊断（尚未接入闭指）

仅最近表面查询会漏掉其他接近wire的面，不能作为完整非目标保护。v1只代表最近查询PASS；v2保留的passed字段也仅是该查询，不能用于资格。v3加入所有非前面距离检查后四通道全部拒绝，最小保守余量.37–.77mm。限制三角面法向与中心约30.2–30.8°、顶点距点3.5–5.8mm，属于指腹圆弧过渡，不是背板穿透。

v4显式前弧候选（12mm内、朝前法向60°以内）后的全剩余表面下界左拇指1.663mm/左食指2.261mm/右拇指2.441mm/右食指1.500mm。保留背面、侧面与远端壳体，未放宽1mm间隙；6mm/30°内区仍用于实际接触位置核验。v5加入仅选择含标定点的连通分量，并输出可视证据，待复验。全部静态，不是实际接触/抓握通过。

D3 50µs左食指800步/40ms也通过：最大速度.100372rad/s、峰值.361650N。同物理采样时与100µs最大q差9.62019e-5rad，力RMS差.00187053N；只是该局部接触的一致性证据。

OPEN：实际位姿/动态wire全臂保护，前指腹局部接触门，实际approach→close→hold→lift，1s保持/固定材料坐标滑移/30mm抬升和3次重复。`motion_qualified`、`grasp_qualified`均不能据本报告置true。目标持续active。
