# DH116 实际位置迭代短时对照

父节点：[D3-position-rate-consistency](../todo/T500-m1-dual-panda-o6-bimanual-mpc.md)。Baseline/Candidate `5e3bcc4` + dirty；Inline/core0。

启动串行16/4/1对照，exec session46250，首个Python PID77392（taskset读回CPU0）。输出主仓assets/dh116_v2/iteration_comparison_v1_n16、n4、n1。各1200×100µs；同postpilot7_touch_v1固定SHA姿态、coordinated_closure、loads .87/.77/.88/.93、16接触采样、切向阻尼1000、gravity、60/.2Nm、hand10/.2、逐迭代外力、一阶预测、全部保护不变。只有位置迭代参数变化。

资产目录iteration_comparison_v1_evidence保存启动manifest、runner_snapshot.py（SHA c742330fb85f8886abef64eb5b03000aea36ab71865b7e01e05f45d82fc37e5f）、compare_snapshot.py。诊断生产脚本/tmp/dh116_compare_iteration_runs_v1.py读取全部结束后的summary和轨迹，核对其余run_config一致并比较最后400帧双臂q差分与报告速度、非零速度下位置冻结样本数。不产生抓取成功声明。

当前仅启动和实时轨迹验证；16次基线已到step298，不能视为完成。不得修改运行脚本使后续串行实例不同版本，不因观察超时重启。0.12s短试验只覆盖早期接触载荷，不能验证1s夹持或抬升。完成后应核查是否足以复现pilot8后期问题，必要时独立有载初态对照。

## 完成第一组，开始有载初态对照

session46250已退出0，三份summary和各1200行完整，局部诊断均passed=True但grasp/hold/lift始终False。主仓assets/dh116_v2/iteration_comparison_v1.json核对除迭代数外run_config完全一致。最后400帧10ms窗口关节速度平均绝对误差(rad/s)：16次左4.39e-5/右9.21e-5，4次左4.23e-5/右1.16e-4，1次左1.53e-5/右8.31e-5。1次改善左侧，但不能据此证明高载荷一致性已修复；末夹力仅.01–.024N。

独立诊断脚本/tmp/dh116_loaded_pose_preflight_v1.py读取pilot8最后完整step10105的实际主动q，以原线性耦合重建静态双手，并对直线wire进行原全臂源与四指腹保护检查，全部通过。没有替换USD或放宽1mm等保护。生成loaded_iteration_pose_v1.json，SHA694163499ef88e8924a4f36acfbef327b40d9165c2b7b77b8c615ec0b260142d；preflight报告同根目录。它不是恢复完整rod状态/速度，也不是连续抓取证明。

新session14734串行loaded_iteration_comparison_v1_n16/n4/n1，各400步100µs；其余物理、载荷、保护与第一组一致，只将共同初始姿态换为上述已筛查姿态。日志/tmp/dh116-loaded-iteration-v1-n16.log等。正在运行，尚无结论。不要修改runner或并发启动Isaac。

## 有载初态结果与渐进加载

session14734已结束。16次完成400步、12点J探针通过，但实际位移/速度仍不一致（10ms均误差左.001206/右.000571rad/s）。4次/1次仅1步，因右手速度.312156/.321285rad/s超过原.31阈值而正确停止；16次同初态首步.306610rad/s。无异常error字段不代表成功，应读last记录的hand.passed=False。全部几何保护通过，不能放宽速度门；不足两帧的组不计算速度一致性。报告loaded_iteration_comparison_v1.json保存不足样本，不宣称比较通过。

后续改为从原安全触线姿态逐渐增载，避免直接有载初始化冲击。新session89681运行closure_pilot_trial9_iteration1，12000步×100µs/solver位置1，其他与原pilot8相同。log /tmp/dh116-closure-pilot-trial9-iteration1.log。主仓assets/dh116_v2/closure_pilot_trial9_iteration1_launch.json固定runner SHA和现存快照。仍独立闭指诊断，不授予保持/抬升资格。应跟踪高载荷位置差分/速度及原保护，再决定是否进入连续流程。
