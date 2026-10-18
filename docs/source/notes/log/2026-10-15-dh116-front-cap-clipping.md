# DH116 跨边界粗三角面保护修复

父节点 [D4-grasp-cross-section](../todo/T500-m1-dual-panda-o6-bimanual-mpc.md)。Baseline/Candidate `5e3bcc4` + dirty，Inline/core0、无提交/推送。依据[修复计划](../../docs/superpowers/plans/2026-10-15-dh116-front-cap-clipping.md)。

## 根因证据

pilot6在2483步停止的left_index face168，三个顶点距指腹参考4.057/4.205/32.386mm，法向夹角27.18°。最近点在距参考约4.2mm处，旧分类却因远端顶点超12mm而将整个面算作非前区。`/tmp/dh116_limiting_pad_face.py` / `.log`读取实际失败现场；不是凭想象扩大接触区域。

原全部248候选的独立法向余量筛选：左96个均不能多压入.2mm，右152个中64个最多通过测试到.2mm。报告`assets/dh116_v2/same_section_pad_margin_screen_v1.json`。这种静态筛选不是机器人动作或动态负载证明。

## 修复范围

新增`front_partition.py`，用严格内接12mm球的三级icosphere凸多面体裁剪源三角面，仅法向dot≥.5、与参考点沿实际被裁边连通的内侧片为前区。所有外侧碎片仍在remainder；不豁免整块长面或整个末节。分区保留每片源face ID、边界误差上界和SHA；full最近点/符号查询与物理接触网格完全不变。

局部6mm/30°载荷patch、1mm非前区间隙、1mm穿透、wire及摩擦/刚度、关节/力矩/速度门均不改。内接近似比12mm球更保守，径向误差<60µm。实际face168近端现在前区、32mm远端仍非前区。

## 验证

- 核心先RED缺模块，10基础测试GREEN；实际源face面积守恒/法向/半径/源归属及同源表面细分不变性补齐。
- 集成先RED 1失败/19通过，明确原近点距remainder为0；修复后20通过，包括旧失败帧重算、远端仍受保护、反面接触拒绝。
- 完整DH116+moving-contact **377 passed in24.16s**，日志`/tmp/dh116-front-partition-full.log`，无跳过实际USD测试。
- `same_section_clipped_cap_preflight_v1.json`：未变形直杆、各指腹独立法向位移0–.6mm，实际接触核k1e4/16积分点、无速度项；局部保护额外.1mm半径裕量。四通道仍通过保护的采样最大法向载荷1.513/2.647/3.136/2.526N，覆盖目标.87/.77/.88/.93N。只说明局部几何可容纳预载，不替代机器人/柔性杆耦合试验。

已按Inline方式复核：逐面面积守恒、分片源面内性、球内性、正常方向、连通限制、局部负对照及默认控制零平移回归均保留。没有独立子代理审查，也没有实际抓握通过声明。

## 正在进行的实际试验

18:53 pilot7，session29342/PID73847，`assets/dh116_v2/closure_pilot_trial7_clipped_front`，日志`/tmp/dh116-closure-pilot-trial7-clipped-front.log`。与pilot6相同姿态/点位/目标力/全部物理参数，仅使用修复后的几何区域划分（以及失败保存点J诊断）。默认平移为零，没有启用抬升，也没有改变loaded controller。

30000×100µs，GPU0/core0，完整源码快照保存；这是独立初始切触闭指诊断。保持、连续接近、抬升及3重复仍OPEN。下一步依据实际结果，不能把旧失败报告重新标PASS。
