# DH116 V2 线性仿真范围实施计划

> 使用 executing-plans，按用户已选 Inline Execution 执行，不创建新 worktree。

Goal：继续用户批准的 URDF 线性仿真近似，补齐 V2.0 手册与模型有效域交集；不等待 Excel 标定、不采用 DH2116 或 MJCF 系数。

Architecture：linear_coupling.py 保持6→11仿射展开与同点Jacobian折叠，新增 active_limits() 返回独立上下界数组。手册主动上限 deg=[80,90,70,70,70,70]；URDF 上限 rad=[1.588,1.030,1.57,1.57,1.57,1.57]。每个从动 q_f=a*q_m+b 再反推主动区间。无夹紧、无驱动力配置、无USD写入。

Tech Stack：已有 NumPy/pytest，core0及数值库单线程。

## Task 1：范围与回归

- [ ] 修改 tests/test_dh116_linear_coupling.py：80度侧摆/70度四指超界应拒绝；active_limits返回下界[0,.00697725445/.88178789,0,0,0,0]、上界min(URDF,手册)；两端可展开且全部从动在限位内；返回数组修改不影响后续调用；边界外拒绝且输入不修改。
- [ ] 运行 `python -m pytest tests/test_dh116_linear_coupling.py -q`，观察RED。
- [ ] 修改 control/dh116/linear_coupling.py：active_limits用全关节上下界初始化，再以 `lo=max(lo,(follower_lo-offset)/ratio)`、`hi=min(hi,(follower_hi-offset)/ratio)` 收紧对应主动域；expand使用该函数。常数元组保存模型限位，不改MIMIC。
- [ ] 重跑线性及真实表格回归；对边界邻域执行有限差分J验证，原有列重排/非法输入测试保留。

## Task 2：证据与交接

- [ ] 记录新版手册SHA、有效域与实际结果；更新todo、T500、log索引与CODEX_RESUME。
- [ ] 旧表格设计注明用户改选URDF近似，Excel坐标标定未完成但不阻塞此独立近似。
- [ ] 检查diff，保留未提交。物理跟随器/PhysX、Panda安装、指腹与抓握仍未完成；不得将单测当作物理成功。
