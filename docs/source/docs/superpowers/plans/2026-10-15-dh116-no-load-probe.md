# DH116 独立空载仿真试验

用户允许使用仅仿真的保守驱动参数；Inline Execution。此计划不涉及实物、双Panda安装或wire。

配置：固定根、无重力、自碰撞开启，dt=1/240秒；6主动PD stiffness=1 N m/rad、damping=.05 N m s/rad、effort=.1 N m、velocity=.3 rad/s；5从动驱动为0，使用USD已有线性mimic。所有参数为本次合成试验值，不能从结果推断实物上限。

输入：SHA固定的linear_limits_verified_v3左右USD。目标初始6主动均.2rad，食指在2–6秒用半余弦往返.2–.3rad，8秒结束。只在初始化写关节状态，循环中仅下发位置目标并实际推进PhysX，不用逐帧teleport。

步骤：先写reference/metrics单测并观察RED，再最小实现；运行CPU回归；单环境GPU0左手试验。每步检查有限状态、主动限位、速度<=.31rad/s、耦合<=.5度、跟踪<=1度，失败立即停止循环并保留JSONL/summary。首试失败不自动改阈值。自碰撞开启不等于完成碰撞间隙/接触传感器验收。

文件：control/dh116/no_load.py、tests/test_dh116_no_load.py、scripts/verify_dh116_no_load.py。输出独立目录，不覆盖资产。试验与异常写notes/log及CODEX_RESUME，报告实际状态。CPU绑定core0，数值库单线程。
