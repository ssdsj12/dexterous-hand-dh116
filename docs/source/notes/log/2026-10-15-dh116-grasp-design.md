# DH116 上方对捏设计

Baseline/Candidate：b5666c2 + dirty worktree；本轮仅文档，无新运行测试。

父节点：[T500](../todo/T500-m1-dual-panda-o6-bimanual-mpc.md)。用户确认先双手上方拇指—食指对捏，成功后扩展贴桌滑入。已完成方案比较，形成[书面设计](../../docs/superpowers/specs/2026-10-15-dh116-pad-jacobian-wire-grasp-design.md)，待用户审阅。

D1表面/掌心标定→D2点Jacobian→D3重力与桌面wire→D4接近/闭指/保持/抬升。保持DH116线性仿真近似，不沿用O6指腹、关节映射或驱动参数。Cosserat接触和机器人反作用只消费一次，不用固定绑定模拟成功。

设计区分局部抬升与整根离桌；要求材料坐标滑移和逐对真实接触证据。拟定阈值不是测试结果。逐阶段失败停止；CPU单核/顺序运行。

审阅后下一步写实施计划，首先D1/D2。尚未新增标定文件、未测新Jacobian、未运行抓取。本轮验证限于文档diff/链接及自审，不改变此前52项测试的覆盖范围。
