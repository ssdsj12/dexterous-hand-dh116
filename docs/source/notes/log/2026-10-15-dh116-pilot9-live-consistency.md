# Pilot9 渐进增载中的位移速度复核

父节点：[D3-position-rate-consistency / D4](../todo/T500-m1-dual-panda-o6-bimanual-mpc.md)。Baseline/Candidate `5e3bcc4` + dirty，Inline/core0；无控制热修改。

实际session89681/PID78902持续运行。step2789：四路有效，载荷.2913/.2487/.2598/.2482N；最长四路连续.1648s，最大首次有效接触后材料滑移.10275mm。尚未达到预载或持续夹持资格，也未抬升。最近200帧19.9ms双臂q差分/报告速度平均绝对误差左.0002275、右.0000726rad/s（平均速度.0018095/.0015058rad/s）。这是观测，不是新阈值或放宽门。

独立复用pilot8审计方法，/tmp/dh116_pilot9_contact_velocity_audit_v1.py在CPU0读取连续50帧重建右食指mesh速度，输出主仓assets/dh116_v2/pilot9_contact_velocity_audit_v1.json，49比较帧结束step2433。均值：报告分离−.636365mm/s，几何gap变化−.567715mm/s，mesh COM报告法向速度+.553805mm/s、位姿差分+.497194mm/s；方向一致，较pilot8曾观察到的相反方向有所改善。当前载荷与pilot8后期不同，不能据此完成高载荷修复认证；也不能把平均一致性当每步都一致。

只读观察器/tmp/dh116_watch_pilot9.py只解析完整连续末200行，不修改数据、不授予资格。两脚本快照保存在iteration_comparison_v1_evidence。后续继续同一进程到原12000步或保护终止；高载荷一致性、保持、连续接近/抬升仍OPEN。

## 后续连续观察

同session89681连续轮询确认仍活，step3839/4310/4785分别完成有效轨迹。step4785四路载荷.649/.601/.569/.567N，全部valid，连续最长.3644s（超过pilot8的.2758s），最大滑移.27853mm；记录passed=True，未发生保护终止。尚未到所有目标90%预载，也未满1s保持，更未抬升。不可据此授予抓取资格；继续同实例，不重启。

更高载荷复核：pilot9_contact_velocity_audit_v2.json连续49帧结束5186，报告分离−.353811mm/s、gap变化−.309703mm/s，mesh COM速度+.483434mm/s、差分+.495331mm/s。此时四路约.62–.72N，方向与平均量级继续一致；生产脚本快照已保存。

step6395四路.816/.750/.800/.823N全部valid，连续最长.5254s，首次接触起算最大滑移.45224mm。右食指仍低于其.837N的90%预载门，不能判保持；当前无抬升。session89681/PID78902仍运行原12000步，未重启。
