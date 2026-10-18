# DH116 同截面对捏源路径与有载试验

父节点：[D4-grasp-cross-section](../todo/T500-m1-dual-panda-o6-bimanual-mpc.md)。Baseline/Candidate `5e3bcc4` + dirty；外部 backend dirty。Inline/core0，单 Isaac 进程，不提交。

## pilot5 已结束

`assets/dh116_v2/closure_pilot_trial5_coordinated`：12347 完整物理步，step12347 左拇指非前区间隙保护拒绝。最长四点有效接触 .103s，最大材料滑移 2.082067mm，末法向载荷 .27063/.28317/.41942/.42336N，没有四点达到目标力。完整源其它部位下界 1.01602mm，不能将保护拒绝称为已证明实际穿透。

候选 rod/mesh/impulses 已保留；`aggregation.json` 为 `/tmp/dh116_summarize_closure.py` 逐行汇总，碰撞模块快照 SHA 与本次 pin 一致。无保持、抬升或完整连续抓取。

## 新几何与路径

左右两指分别在同一轴向截面对捏，左右抓取区仍约 y±.2m。右784个候选中152个通过手部非指腹源几何；进一步前8个通过四个原局部patch/外cap、完整双臂源几何与静力支持。选出的静力法向载荷约 .75157/.66791/.76070/.80076N；这不是动态握力证据。

原预抓取位姿左腕 link5/link7 的保守下界为负，未下发。仅修改冗余臂预姿态后，通过原间隙门和连续源路径391个区间检查。报告：`assets/dh116_v2/zero_axial_source_sweep_v1/summary.json`，SHA `596c09ce661a42ff502a86ddec62fabd23179376d216cffd3f3648d6c1228e54`。这是静态直杆和源几何路径，不是实际 cooked 碰撞或真实抓取验收。

`/tmp/dh116_prepare_zero_axial_artifacts.py` 核对各层 SHA、原路径 coverage/速度限制、4个实际接触凸包局部保护（额外 .1mm 半径裕量）及预姿态/切触姿态完整静态场景后，生成独立资产，未覆盖旧输入：

- `zero_axial_point_references_v1.json` SHA `4dc373beee35d7bba97caa29a5290152c3274df478849b8e03d575d56eb34c0b`。
- `zero_axial_touch_v1.json` SHA `9b9b0fadac1d0a62c79e2b319d0a909aa31930a87cdc2546459f3cb9c724c264`。
- `zero_axial_prepose_v1.json` SHA `9c3017537670b347b9db15df8e8e0e7ba5b775ed5a5d88aecb5893a47f603dd8`。

## 材料观察合同

同手两指允许共享同一材料 ID，左右集合仍须不相交；同手同 ID 必须返回一致掌心坐标。4通道接触、5mN、.1s接触、1s保持、2mm滑移、30mm抬升1s全部不变。观察器先RED（3失败），后与材料坐标17项通过；完整 DH116/moving-contact **347 passed in46.19s**（`/tmp/dh116-shared-material-full-regression.log`）。它尚未接入物理抬升，不可声称门已实际通过。

## pilot6 运行中

**本节运行状态已过期：** pilot6在step2483结束，2483完整步。四点最长有效接触.1331s，最大固定材料滑移.066962mm，末载荷.17541/.23540/.19995/.22308N，四路均有效。左食指外cap之外三角面间隙从初态1.07733mm降到.999928mm，保护停止；局部小patch本身通过。未保持/抬升，未达到目标预载。`aggregation.json`保留结果。下一步针对全部候选筛选独立法向压入后的局部余量，再做完整源/路径/真实有载验证，不放宽原1mm/区域门。

18:33 启动 `closure_pilot_trial6_same_section`，session2972/PID71666，日志 `/tmp/dh116-closure-pilot-trial6-same-section.log`。GPU0/core0，30000×100µs，实际点J协调闭指、原物理和保护全部保留；使用新点/姿态，显式仿真力目标 .87/.77/.88/.93N（约新静力支持的1.15倍）。固定材料ID `[36,36,4,4]`，无wire绑定。

这是独立初始切触诊断，不替代从预抓取起的连续接近。脚本/DH116/外部 backend 快照在输出目录保存。后续先实测保持，仍须完整接近、实际抬升、全wire离桌及3重复；目标 ACTIVE。
