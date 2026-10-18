# DH116 源资产只读审计

Baseline / Candidate: b5666c2 + dirty worktree；未改运行代码或源资产。

阶段：双 Panda 从 O6 迁移到 DH116 的 USD 前置。

方法：core0 下用 go2 Python xml.etree.ElementTree 解析桌面左右 URDF/MJCF，枚举关节、mimic、equality、actuator、限位、质量并检查 mesh 文件存在。

结果：左右各 17 links、11 revolute、5 fixed、6 主动通道；每侧 34 mesh 引用均存在。URDF 和 MJCF 的耦合公式不一致；拇指零主动角映射到负从动角，违反源限位；全部 URDF effort/velocity 为零。

结论：源资产可进入结构转换，但不能直接作为物理开合或抓取合格资产。未生成 USD、未运行仿真、未训练。下一步确认 URDF 静态结构优先方案并实施独立转换。

设计：[DH116 USD 迁移](../../docs/superpowers/specs/2026-10-15-dh116-usd-design.md)。关联父节点：[T500](../todo/T500-m1-dual-panda-o6-bimanual-mpc.md)。
