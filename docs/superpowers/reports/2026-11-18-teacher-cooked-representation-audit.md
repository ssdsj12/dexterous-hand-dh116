# 截图建议核验与cooked读取审计修正

## 范围与结论
客户端日期2026-11-18，基线d4bdb04024653b6743ccda0f6d121835cefb8b9c，Ubuntu工作树/mnt/mydisk/xk/projects/dh116-teacher-display-owner-20261117，分支11-17。仅执行CPU离线代码/测试及已有原生数据复算，没有新的Isaac实例、物理步、求解器步、Teacher查询或PPO。

截图中“先对照原始和cooked数据、检查坐标链”的方向可采用；以下假设不直接采纳：
- 单次正常退出1且无SIGSEGV，不等于所有生命周期已修复；上一轮是初始化失败清理路径的一次对照。
- 0/34的当前直接拒绝来自protected validator的polygon plane/enclosure谓词，并非已经证实的Rigid Body Matrix内存对齐错误。
- measured_closure_geometry不是此次严格hull校验实现。它调用的SciPy Rotation已归一化四元数；此前独立初始化入口的派生矩阵问题也已修复，不重复修改。
- 原始三角网格记录数量/顺序与cooked凸包顶点不同，不能对同一数组索引做np.allclose。

## 实际修改
scripts/audit_dh116_teacher_cooked_readback.py新增opt-in --compare-source-vertices：
1. diagnose_cooked_representation读取当前packet原始points、triangles、orientation、cooked vertices及列向量坐标链。原CLI默认schema/output不变，opt-in使用v2。
2. 用原usd_surface.read_surface字节合同重算保存的geometry SHA，核对source、live及source_collision三处identity。方向、整数三角索引、有限点、单位和矩阵来源缺失时明确失败。
3. 使用去重后的无序顶点集合和分块cKDTree双向最近顶点距离，避免错误的逐索引比较；每批最多65536点，没有构造源点数×目标点数的完整距离矩阵。
4. 独立记录mesh→body→world→mesh往返误差、原始/live mesh-to-body差异及SDK/保存USD矩阵crosscheck。明确列向量、米单位；不自动转置、改比例或修改输入。
5. 所有派生运算受np.errstate保护。两个分别有限的1e308平移矩阵组合溢出时返回available=false/error，避免带NaN结果被写入partial JSON。
6. 重算原plane谓词，并记录1e-4m的plane-only反事实；不改变protected validator，不把它当作完整闭合凸性校验或准入。
7. 修正opt-in审计的时钟读取：取当前真实packet根级sdk_clock_before/after，缺失或不同时钟不伪造同一边界。旧v1输出保持原样。

## 当前34份数据的结果
数据来自上一轮已完成原生捕获，不是本次新采集。完整packet SHA：
c122c87408d681166cf9a437479d0010e1a8dd335dfdf55cb7e7ec0e2610982e。
最终CPU输出representation-audit-v2.json SHA：
20c08a2b0e69965a2a3cae451607db8ae1ae5e4eaa2004b8a41d14e9d5050dbd。

| 项目 | 左手17个owner | 右手17个owner |
|---|---:|---:|
| 原始顶点记录/去重顶点 |790812 /132847|775332 /130292|
| cooked顶点 |562|563|
| cooked→原始去重集合最大最近顶点距离 |0m|0m|
| world/mesh往返最大误差 |2.61846e-16m|2.55570e-16m|
| 原始严格plane谓词失败面 |95|94|
| 1e-4m反事实仍失败面 |15|15|
| 最坏原平面残差 |0.218929mm|0.218441mm|

34/34保存源几何内容重算SHA与三处identity一致，1125个cooked顶点全部来自对应原始源点集合；34份坐标诊断可用。SDK/保存USD矩阵crosscheck最大绝对entry差约6.817e-7/4.869e-7；该值混有无量纲旋转项和米平移项，不能直接写成位移毫米误差。

反向的原始→cooked最近顶点距离最大38.911/43.202mm，是顶点简化后顶点集合稀疏的测量，绝不是表面距离、穿透或线缆误差。所有相关字段显式vertex_distances_are_surface_errors=false。

原189个失败面中，1e-4m会掩盖159个，仍留下30个；17个hull在该阈值下没有plane谓词失败，也不能推导其已通过完整受保护validator。正式完整严格校验仍0/34；现有阈值、plane、索引、网格和变换均未改写。

这些结果不支持“此次0/34由顶点数值读错或矩阵转置造成”的猜测；保存的polygon表示与精确平面谓词的语义差异仍需独立契约处理。内容hash证明保存payload与保存identity一致，不单独证明外部资产、当前live PxShape附着或C++ ABI布局；cooking_abi_alignment_proven/live_shape_attachment_proven/native_qualification均保持false。已有表面拓扑/参考PhysX算法审计见2026-11-17-teacher-cooked-surface-semantics.md，不重复作为新结果计数。

## TDD、审查与验证
- 16项新增API/CLI缺失测试先全部失败，外部red.log保留。
- 独立审查发现派生溢出问题，并指出保存SHA字符串相同不代表内容重算；新1项溢出及5项payload/索引/方向/hash篡改测试先失败，再实现修正。
- 最终新增22项+原有15项专项共37 passed；独立审查也重跑37项通过，未发现剩余阻断。
- 联合CPU回归544 passed、0 failed、0 skipped；Registry回归另记readiness（与联合套件重叠，不累加）。
- 原34份数据最终CPU CLI用时4.746s，仅代表这份保存数据在此机的离线审计耗时，不代表训练、渲染或GPU吞吐。
- 初版representation-audit-v1.json与全部红/绿日志保留；新增内容hash及溢出保护后的v2是最终权威输出。未覆盖初版证据。

精确命令、环境、输入/输出及日志SHA均在readiness；原始264MB量级packet、日志及数组留数据盘，不上传GitHub。代码、测试、报告与状态更新经Ubuntu commit、Windows导入同commit正常push，核对远端SHA/tree/文件；PR2始终Draft。

## 后继与资格
528个保护文件、旧三种模型、USD、SDK、物理/材料、执行器、动作/观测合同和30mm/5s/0.5mm标准不变。截图的“只剩唯一门槛”未被采纳：严格几何是当前读取主线阻断，真实接触/载荷、完整回合及Teacher成功资格仍需分别验证。

下一项应基于已保存的原polygon和实际求解器消费方式，提出独立版本的表示契约及空间/力学误差验收；不能以放宽allclose替代。没有新授权前不启动原生、不重Cook、不把此审计通过写成Teacher或抓取成功。

元数据更新后的Registry回归55 passed、0 failed、0 skipped，和544项联合回归重叠，不累加。原生预算没有增加或消耗。
