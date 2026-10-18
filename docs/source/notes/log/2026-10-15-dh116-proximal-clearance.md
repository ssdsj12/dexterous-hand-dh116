# DH116 pilot7 近端指节余量

阶段D4，父节点 [DH116抓取](../todo/T500-m1-dual-panda-o6-bimanual-mpc.md)。Baseline/Candidate `5e3bcc4` + dirty；Inline/core0，未改物理参数或安全门。

## 实际失败与证据

pilot7 PID73847已结束，`closure_pilot_trial7_clipped_front/summary.json`与trajectory已落盘。5061完整步，failure_step5061，实际读回后左`finger13_link`与wire保守下界 **.999657305mm** 触发原1mm保护。四指腹区域全部通过；左拇指非前区下界1.046612mm。不是再次修改指腹裁剪的理由，更不是放宽保护的理由。

`aggregation.json`由`/tmp/dh116_summarize_closure.py`流式汇总：最长四路连续接触.1566s，所有四路同时到目标夹力的帧数0，最大材料滑移.596611mm；末载荷.593650/.618797/.529621/.640080N。碰撞源码快照SHA与运行pin一致。168次点/J探针通过，最大点位误差6.04e-7m、J误差5.29e-7m/rad。无保持/抬升资格。

## 根因假设与只读筛查

当前左手原候选非指腹初态余量仅1.438259mm，闭指使近端指节继续靠近wire；候选筛查此前在旧整面前区判定下跳过更大余量解。保留右手和wire物理不变，重新筛左手非指腹余量≥2mm、同截面y=.2的4个候选。

`/tmp/dh116_postpilot7_margin_candidates.py` 输出 `assets/dh116_v2/postpilot7_higher_clearance_left_v1.json`：候选88/89名义余量2.347550mm，80/81为2.172149mm；4个都通过实际USD IK、完整源静态检查、静力支持可行性以及独立指腹法向压入0/.2/.4/.6mm局部门。仅静态筛查，不是闭指路径或物理证书。

后续`/tmp/dh116_postpilot7_closure_preview.py`按实际USD点J/掌心角J做名义协调闭指采样，日志`/tmp/dh116-postpilot7-closure-preview.log`，输出`postpilot7_closure_preview_v1.json`。这是零载最大法向速度的离散预览，明确不是连续区间证书；通过后仍需区间/初态载荷/真实闭指，不能直接晋级抬升。

结果：4/4候选全部通过0.2s名义闭指采样。`/tmp/dh116_postpilot7_closure_certificate.py`对候选88做20段joint-box源碰撞覆盖，指腹距离按关节运动半径保守膨胀；20检查通过，输出`postpilot7_closure_certificate_v1.json`。仅固定直线wire的名义闭指路径，非动态wire/完整接近证书。

## pilot8 新独立物理试验

`/tmp/dh116_prepare_postpilot7.py`生成新左候选88/原右手参考：

- `postpilot7_touch_v1.json` SHA `7b9cbd0b22f3646516ae2508047b2755658fb359df5f8d5282ed9a8bb036e4bc`
- `postpilot7_point_references_v1.json` SHA `9874291a3b2740369e4cf8d1933b08f44303f4f0a34ac764837c8fe34a27c9c6`

19:13启动session59075/PID75598，输出`closure_pilot_trial8_proximal_margin`，日志`/tmp/dh116-closure-pilot-trial8-proximal-margin.log`。原.87/.77/.88/.93N目标、100µs/30000步、GPU0/core0、全部物理和保护不变；左姿态/reference改变，不是完整approach。新脚本使用流式证据；脚本/模块/后端快照已保存。结果待实际观测，不提前授予夹持或抬升资格。

## 下一步

D4-proximal-clearance是D4真实闭指的子问题，当前OPEN。先看预览实际结果，选择有余量且完整路径可达的候选，固定新reference/pose SHA后再运行受保护的实际闭指。连续approach→hold→lift接线和3次重复仍OPEN，394代码回归不改变此结论。
