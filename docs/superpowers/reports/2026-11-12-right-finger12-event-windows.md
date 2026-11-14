# Teacher 右手 finger12 下限事件窗口

现有 `tabletop_privileged_action_audit` 新增可选的 `--finger12-event-windows`。按 run、env、episode 绑定 canonical 24D 动作索引 7 与实际 q、qd、SDK 目标、控制器目标及原保护线。分别记录首次在下界上方 0.005 rad 内、首次进入原 hard tolerance 区、首次跨过原下保护线的观测样本。各事件保留原 480 Hz 步号 ±4 步内实际存在的记录，缺步和终止后的样本显式缺失。0.005 rad 只用于日志检索。原控制、关节限位、训练与物理参数均沿用现有版本。

只读复盘 Teacher128 v02 的 2,030 条样本和 7 个可观测 episode。5 个原 joint_limit 事件的右 finger12 下保护线首次跨越发生在物理步 250、301、540、784、835。每个事件保留 4 个连续前样本及终止样本，后 4 步均缺失。各次近下界与进入容差区分别为 146/162、241/256、388/433、684/709、653/712。输入是同一历史场景的保存记录与 SDK 范围参照，不能用于证明物理反事实或独立因果。

原始 JSONL SHA256 为 `470e8c5d0f24b2cf97c1664ea26dbd566aa58969677d0b3f64660ff22e2d6ffc`，窗口 JSON SHA256 为 `ab7713a2f7a7809aa07b74b9c353ba757bffc8f22fcc7c82ca47abc42ac757fc`，原件留在 Ubuntu 私有目录。未启动新的 Isaac 实验或训练。

Ubuntu 回归 `python -m pytest -q -p no:cacheprovider tests/test_dh116_privileged_action_audit.py tests/test_mpc_privileged_capture.py` 通过 29 项，退出 0，日志 SHA256 `b231a777b69f85e8fb64489e3c4f13174fe93a35ce24a5ffc8c47a7a7204648c`。两条新增用例在改动前失败。原生根因待同初态、单因素配对和完整关节读回。
