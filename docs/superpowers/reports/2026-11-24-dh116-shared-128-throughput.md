# Actual 128-lane native throughput

128根真实共享DeformX柔性杆、256个实际DH116 Articulation、GPU3单实例，26SDK步
（2初始化+8场景检查+16测量），两个实际on-policy控制窗，0次优化。自然exit0，无强制退出。
保留原完整24D、480Hz/60Hz、12 solver substeps；相机关闭，无视觉采集。
原关节/MIMIC/30N/2.1mm桌面及完整非目标几何保护保持。

## 实测
- 两窗41.46975s / 44.65229s；整个启动/构建/测量/关闭457.963s。
- 整卡显存采样峰值2860MiB，约2.79GiB；不是连续追踪的严格瞬时峰值。
- 峰值RSS4,830,344KiB，约4.61GiB。
- 128路首态真正测得8448个有效common字段（128×66），完整N=128参数来源，非复制N=1。
- 8个无杆场景物理步与回读仅0.4203s，不能将整个457.963s算成物理求解耗时。
- 两控制窗未覆盖128路局部Reset/完整rollout/PPO，不能给这些项目授予通过。
- 输入有效不意味着接触或抓取成功；本次是吞吐探针。

首窗墙钟分项（嵌套source/sample不是独立可相加项）：

|分项|秒|
|---|---:|
|action projection/reaction submission|17.9984|
|PhysX step dispatch|0.07209|
|SDK readback/guards/mesh targets|0.96951|
|shared solver/compact|0.17155|
|action post-readback|2.21521|
|geometry guards/R0|18.65618|
|source/sample（包含于前项）|1.26470|

这些是调用边界墙钟计时，不是CUDA Event核时间。没有为计时新增每SDK同步。
现状瓶颈在Python控制/回读与几何奖励，不能声称全CUDA、零拷贝或已达到秒级128路控制窗。
按两窗简单外推32窗约22–24分钟只属粗估，未测完整128路update/Reset，因此不能作为训练计时结论。

## 代码
train_dh116_shared_native_teacher新增独立--probe-control-windows 2。
probe仍用真实策略/原32窗PPO writer，但不更新；仅输出shared_ppo_probe.json，
绝不写shared_ppo_result.json或训练Checkpoint。CPU回归显式验证这一隔离。
coupled新增墙钟分项，原默认诊断/完整训练入口保留。
每窗一次nvidia-smi采样与RSS读数；逐SDK仍不全量落盘。
12项对应CPU回归通过，固定argv、代码SHA、原生回执和原始日志保存在Ubuntu证据目录。

## 后续独立实施
Antigravity R8认可一次SDK快照多消费者和按窗/失败留证方向；其MCP仍旧树，未当作当前源码实读。
独立审查发现：仅检查时钟不能替代submit前的实际q/qd/target比较，因为无物理步也能发生外部目标写入。
因此不会直接照抄“省掉submit capture、只验时钟”；先取得真实调用树，再实现保留漂移/外写拒绝的快照接口。
将按实测热点优先减少重复metadata/deepcopy/hash与GPU小张量往返，几何批量化必须和旧CPU奖励/守卫/目标比较。
Tripod左finger33仍受原接触保护阻断；Pinch/Power已完成N2两update，不冒充三抓法128训练完成。

## 原生调用树与最终回归
首个外层cProfile只覆盖了启动，实际coupled/begin/after_step调用未出现在stats中；
因此没有用它推断运行时热点。增加--profile-first-sdk，在实际拥有资源的回调内
分别包裹begin/submit/geometry，默认关闭、不增加物理或优化步。
独立N2、26SDK原生诊断自然exit0；三个stats真实覆盖对应函数，原始二进制stats未上传。
首SDK两路geometry合计0.044s，其中_bytes/encode_owned约0.012s、rod_support_lower约0.009s、
姿态矩阵约0.005s、额外_capture约0.004s。begin0.006s、submit0.011s；
submit包括_apply_actuator_model和实际write_data_to_sim，不可简单删除。
该微观剖析是N2的首SDK，并不能直接当作N128全程的占比。
保留submit前零时钟外写检测；后续将先消除每子步几何JSON/deepcopy和冗余矩阵转换，
再验证分路GPU批量几何与控制写入。未降低任何检查阈值。
最终对应CPU/Registry联合回归73 passed，16.00s；代码/配置冻结与528保护文件检查通过。
