# DH116 新姿态接近准备

阶段D4，父节点 [D4真实抓握](../todo/T500-m1-dual-panda-o6-bimanual-mpc.md)。Baseline/Candidate `5e3bcc4` + dirty；Inline/core0，未改运行中的pilot8。

## 验证范围

`/tmp/dh116_postpilot8_approach.py`从candidate88新touch/reference出发，用有界非线性求解使两指各沿法向退开2mm，求掌心高30mm的预抓取IK。左右预位姿候选选择后，检查完整双臂、非目标手部、桌面、静态wire，再用joint-box覆盖接近段。

结果：`assets/dh116_v2/postpilot8_approach_source_v1.json` **116 checks，static_wire_source_sweep_passed=True**，`load_approach`覆盖/速度/pin验证通过。SHA `37ad0f20ceef8b30c4902128da1339aefb83edb9ef55416f5ef562682b73eb4f`。日志`/tmp/dh116-postpilot8-approach.log`。

起点是固定预抓取位，不是home；旧格式leg0使用重复预位姿，leg1为实际接近段。不暗示home到预抓取的路径已验证，不暗示动态接近/闭指/抬升已通过。

`/tmp/dh116_prepare_postpilot8_prepose.py`额外检查预抓取姿态相对wire初始高度.318m和落定高度.308m均通过，生成`postpilot8_prepose_v1.json`，SHA `74895d9ff65b25cc3947c4180dcfc9af40437e6f8473f7670bfcf687b081fdfb`，实际loader身份验证通过。

## 运行与后续

pilot8 PID75598/session59075实际运行持续，未停止/热更。已从其未关闭的trajectory.jsonl读到第852步完整帧，证实新流式记录在真实运行中工作；当时尚未开始充分增载，左侧低于5mN有效接触门，不能称稳定夹持。

连续接线列于 [held-transport Task6](../../docs/superpowers/plans/2026-10-15-dh116-held-transport.md)：先TDD显式模式隔离，再接真实测量、共同双臂分配和阶段门。新预抓取实际重力/动态接近、稳定保持、抬升和3重复仍OPEN；不因源证书通过直接授权抬升。
