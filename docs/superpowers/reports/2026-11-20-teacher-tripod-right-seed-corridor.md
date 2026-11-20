# Tripod 右手较低初姿态的离线动作核验

用右手原生 SDK178 的杆节点、关节和实际腕位姿筛选末端。候选三指到杆的保守间隙上界为 0.498 / 0.488 / 0.490 mm，拇指分别与食指、中指通过原几何对向门槛。原起点重置后的目标腕高度偏移为 −7.243 mm，超出原 −4.000 mm 合同。

提出低 3.500 mm 的独立右手初姿态。相对这个初姿态，末端腕目标只需下降 3.043 mm。原控制器编译出 22 窗、176 SDK 步的离线路径。在 23 个控制窗边界，最低桌面间隙为 2.568 mm。末端阻断仍是近端拇指 `source:finger13_link`，名义杆间隙下界 30.837 µm，小于同一连杆在历史右手原生账本的顶点跟踪偏差峰值 53.518 µm，压力检查余量为 −22.680 µm。

该差值来自历史观测的标量压力检查，无法证明该路径一定会相交。此候选需要重新分配近端拇指净空，并补充独立滞后区间与新 Reset 的原生证据。未启动新的 Isaac 实例。原执行器、关节阈值和杆状态均来自已有合同。

运行 `python -m scripts.audit_dh116_tripod_right_seed_corridor --baseline <verified-sdk178-run> --endpoint <screened-right-endpoint> --tracking <same-link-tracking-audit> --output <new-result.json>`。回执和样本见 readiness。

119 项针对性 CPU 回归通过。全 Tripod 扩展回归 665 项通过、5 项失败。失败均为旧 `test_tripod_worker.py` 的原生回执路径校验：回执 SHA 一致，但 `/home/xk/.local/share` 是指向数据盘的目录别名，旧入口要求字面路径等于真实路径。单独解析第一份回执到真实路径后，历史第二份回执仍被同类检查拒绝。旧回执保留，当前右手候选审计独立通过。
