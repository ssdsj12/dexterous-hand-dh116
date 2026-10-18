# DH116 接触宽阶段等价性与诊断进度

父节点：[T500 D4-hold-friction](../todo/T500-m1-dual-panda-o6-bimanual-mpc.md)。Baseline/Candidate：`5e3bcc4` + dirty；外部PyElastica-Mesh同样dirty。Inline/core0，非硬件，未提交。

## 原因与范围

对实际四指腹网格的后端做50步预热后profile，累计8.79s墙钟，其中Open3D closest-points约6.72s。单独把查询线程数从automatic改成1没有明显收益，未采用该改动。

在接触力积分前新增显式可选材料坐标AABB宽阶段：网格包围盒按当前每个样本半径和float32数值余量扩张。只对盒内候选执行原精确查询，盒外样本以距离大于半径的哨兵令原力内核跳过；不减少动力学单元或候选接触积分点，不修改刚度/摩擦/反作用力。内部点也保留。`contact_geometry`观测仍走完整精确查询，禁止把哨兵当距离证据；所有碰撞保护不变。

关键文件：外部`elastica/rigidbody/mesh_rigid_body.py`、`elastica/contact_forces.py`、`co_sim/{table_rod,moving_contact}.py`；项目`Go2Pvcnn/scripts/verify_fixed_dual_panda_dh116.py`新增`--contact-broad-phase`，默认关闭，三处后端构造一致传递并记录配置；补充mesh源文件SHA。

## 验证

- 内核7项先RED（缺失API/参数）后GREEN；旋转平移网格、运动接触、不同积分数、内部/表面边界样本、非法半径。合并外部43 passed，日志`/tmp/dh116-broadphase-green.log`。
- 后端/CLI新增5项先RED后GREEN：完整状态、冲量、接触时长及几何观测相同，拒绝非bool参数；接线52 passed（2个已有Numba连续性性能警告），日志`/tmp/dh116-broadphase-wiring-green.log`。
- 实际四个DH116源网格、原wire/桌面、16倍积分、c_v1000，对照101步（首步不计时）：杆位置/速度/director/omega逐元素完全相同，四通道线/角冲量完全相同。100步CPU时间8.6101s→2.6777s，局部后端3.215倍。仅固定指腹微基准，不是机器人完整Play倍速或抓握资格。
- 证据：主仓库`assets/dh116_v2/contact_broadphase_equivalence_v1.json`；程序`/tmp/dh116_broadphase_bench.py`，日志`/tmp/dh116-broadphase-bench.log`。
- DH116+moving-contact整体327 passed in45.99s，`/tmp/dh116-broadphase-full-regression.log`。
- 之后新增每120步`DH116_MATERIAL`实时滑移打印，先RED后GREEN；CLI静态专项10 passed。当前已在运行的pilot3没有热更新，仍到结束后读取其完整材料记录。

## 实际抓取状态

pilot3 PID56943继续执行原实现、原网格、c_v1000对照；17:12约3481/20000，四点有效但仅.11–.19N，未达到目标载荷。优化只供下一轮显式启用，不重启现有物理试验。保持、抬升仍未验证，goal active。

下一步：完成pilot3并检查原始固定材料滑移、连续接触、实际位姿预测差和保护；再决定力控制/几何下一步，不由运行退出0或单位测试判定抓住。
