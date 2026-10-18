# DH116 实测轨迹回放

父节点：[T500/D4](../todo/T500-m1-dual-panda-o6-bimanual-mpc.md)。只读显示，不改变动力学/控制合同。

2026-10-16检查：旧PID80300不存在，原/tmp日志也不存在；连续试验最后完整行step21482（2.1483s）、approach/passed，最终summary缺失。不能推断正常完成或具体停止原因。

用户要求演示。新增独立诊断回放脚本位于主仓assets/dh116_v2/recorded_demo_20261016/replay.py。从两个原trajectory.jsonl抽取实测关节位置和rod_position/director，0.5x目标播放速度，显式区分接近记录与独立pilot9闭指。只有render，没有physics step，无新抓取验收。线缆半径8mm/长度50cm，桌边固定双Panda+DH116，非旧M1/O6。

首次启动因未指定默认关节姿态（零值超出Panda4和从动关节限位）被Isaac拒绝；回放初始化改为首帧实测关节姿态，无资产限位修改。第三次启动session12185、log /tmp/dh116-recorded-demo-3.log，显示验证待完成。CPU亲和core0、GPU0，原证据不覆盖。

Baseline/Candidate：现有dirty工作区，产品源码未改。保持与抬升尚未验收。

显示复验：第三次启动打印DEMO_READY，桌面截图确认双Panda/DH116、桌面橙色wire及回放标签，首段已显示step21400。只读回放入口可用，未产生新的物理合格证据。
