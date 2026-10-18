# DH116 右拇指虚假内部距离与力突跳

父节点：[T500 D4-hold-friction/contact-sign](../todo/T500-m1-dual-panda-o6-bimanual-mpc.md)。Baseline/Candidate `5e3bcc4` + dirty，外部backend dirty；Inline/core0，无硬件、无提交。

## 实际失败

pilot3 PID56943/session未知已于17:12结束，不能继续报运行中。`closure_pilot_trial3_cv1000/summary.json`：failure_step3619，`force gate exceeded before sink submission`，右拇指峰值13.458725N，超限wrench未交给机器人。前一完整步四点载荷约.1502/.1185/.2007/.1812N、最大wire速度.00410m/s，右拇指峰值仅.22498N。最长四点有效接触.2452s，固定材料滑移最大.307991mm，预测/实测接触COM差最大.228µm。没有达到目标夹力/保持/抬升。

完整失败证据和3619条已完成行保留，`aggregation.json`已聚合。源码快照与原记录pin相符。

## 根因证据

从最后完整杆状态及关节读回重建的100步近似回放未复现；此回放并非完整PhysX状态恢复，且诊断桌面box尺寸不同，不能由它否定原失败。

进一步对右拇指相同网格/相近查询施加±.5µm扰动，第800组发现可重复的符号错误：材料坐标
`[-.0163660437423104, .0004152284341645757, -.013940692021180976]`
被原单射线parity判为距离 **-.01369227452m**；独立凸包支持平面证明其在外 **+.01369227449m**。直接以该精确点查询原算法仍复现，不依赖GPU。与约13N虚假穿透弹簧峰值量级一致；原失败帧微步查询未保存，不能声称逐位恢复了该帧。

证据主仓库`assets/dh116_v2/contact_sign_probe_v1.json`，查询程序`/tmp/dh116_query_sign_probe.py`。四个实际指腹Mesh加载后的所有顶点对所有外法向支持平面最大越界仅3.5e-18～2.1e-17m，网格本身是凸的，未修改/凸化资产。

## 修复与验证

`MeshRigidBody`只在闭合网格全部支持平面验证通过时采用确定性半空间内外判定，最近表面/距离大小仍由原网格提供。非凸网格保留原路径；环面孔洞回归防止把凹几何换成凸包。原表面1µm约定保留。DH116 `PadRegionGuard`采用同样凸性证明/符号规则，非前区1mm和穿透门不变。

外部3项先RED后GREEN，包含真实反例、变换后的内部点、非凸环面；mesh/contact共 **52 passed**。指腹guard反例先RED后GREEN。项目DH116+moving-contact整体 **329 passed in22.89s**，日志`/tmp/dh116-convex-sign-full-regression.log`。runner记录四网格签名方法，backend SHA覆盖mesh_rigid_body.py。

## 下一轮实际验证

17:24启动pilot4：session40602/PID62055，日志`/tmp/dh116-closure-pilot-trial4-convex-sign.log`，输出`assets/dh116_v2/closure_pilot_trial4_convex_sign`。原切触姿态/目标夹力/c_v1000/100µs/力矩速度安全门不变，新增确定性凸符号及显式宽阶段，30000步上限；每120步实时打印固定材料滑移。runner/全部DH116模块/后端源文件已经快照，禁止热更或重复启动。

尚未证明实际夹稳或抬升，D4连续接近最终门、保持、抬升和3次成功重复均OPEN。目标保持active。
