# DH116 逐阶段最终资格报告

阶段D4，父节点 [DH116真实抓握](../todo/T500-m1-dual-panda-o6-bimanual-mpc.md)。Baseline/Candidate `5e3bcc4` + dirty，Inline/core0。

验收设计要求summary逐阶段PASS/FAIL/NOT_RUN；此前新连续模式只有stage和总passed，不能充分区分未执行与失败。新增`TabletopGraspSequence.stage_summary(final_passed=...)`，从实际落定基线、接近门创建、预载/保持转移与lift开始记录推导阶段资格，不根据时间或命令假设已通过。未运行阶段不晋级；最终lift PASS还需要runner全部物理/J探针通过。

运行脚本在finally完成点/J最终校验之后写`stage_status`。独立pilot不调用此连续资格报告；运行中的pilot8使用旧快照，不热更。

TDD：阶段方法RED3失败/11通过；最终接线RED1失败/26通过。实现后完整DH116/moving-contact **424 passed in50.52s**，日志`/tmp/dh116-stage-summary-full.log`。这验证报告合同，不构成实际抓取成功。

同轮物理核查：pilot8 PID75598/session59075仍在运行，最新读取4943步通过，载荷.700/.643/.506/.611N，右食指valid=False；四路最长.2758s、材料最大滑移.59791mm、非目标源下界1.03603mm。没有保持/抬升结果。磁盘剩余440GB，当前单核约93%，没有第二个Isaac仿真。

下一步继续跟进pilot8的真实目标夹力与接触持续性；通过后再启动新连续模式，后者虽已420/424回归，仍未做实际运行验收。固定预位姿路径见[新接近准备](2026-10-15-dh116-new-prepose-approach.md)。
