# 阶段恢复复验门

父节点：[D4](../todo/T500-m1-dual-panda-o6-bimanual-mpc.md)。用户已确认[设计](../../docs/superpowers/specs/2026-10-16-dh116-stage-reset-design.md)，[实施计划](../../docs/superpowers/plans/2026-10-16-dh116-stage-reset.md)按Inline执行。

新增restore_gate.py纯核心，不改变现有runner或活跃物理试验。接近/闭指入口须连续.1s有效测量，保持/抬升入口须新计时1s预载接触。滑移相对保存材料锚点2mm限制；绑定或安全失败锁存abort；预载不足重置计时。没有连续抓取成功输出，工件来源校验属于后续Task2，不能把此门当完整恢复实现。

TDD：18新用例先因模块缺失全部失败；实现后与grasp_evidence/grasp_sequence合计52 passed in1.71s，core0单线程。涵盖四入口时长、失败锁存、预载丢失、接触/法向不满足、时间跳变/NaN和非法入口。

实际trial2 session90016/PID15485继续原快照，elapsed14:44、CPU99.8%，step10808/passedTrue/approach；未闭指或抬升，不热改、不重启。下一步状态工件及杆静止结构恢复，之后才接运行时验证。

Git：既有codex/t500-dual-panda-o6-mpc dirty工作区，未自动提交。产品改动仅新增模块和测试，未沿用硬件假定改变物理参数。
