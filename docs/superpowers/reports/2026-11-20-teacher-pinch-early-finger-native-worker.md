# 有界 Pinch 左食指提前闭合原生入口

`run_dh116_pinch_early_finger.prepare()` 核对 41 个边界的只读 CPU 证书 SHA、原账本和原动作表，冻结至多 74 SDK 步。新的实测 SDK34 必须与源初态的同钟双腕、关节和 41 个节点匹配，才进入改动后的第 5 个控制窗。只改变两个左指动作，继续由原版控制器执行现有动作率、保护和每个测量步的 PhysX/杆求解器源检查。已有终止原因优先保留，前缀偏离走现有保护终止与完整写盘路径。

上一次 SDK66 补偿使用同一前缀提取器的原默认参数。新增入口校验只加载冻结 CPU 结果并重读测量，精确核对配方 SHA。真实 Python 子进程已通过 AppLauncher 前的干净导入检查，CPU 未启动 Isaac，联合回归 167 项通过。

用当前项目解释器运行 `python -B -m scripts.run_dh116_pinch_early_finger --sealed-reset <reset> --baseline <baseline> --preflight <geometry_receipt> --output <new_output> --plan <plan>`。原生执行用同一参数和 `--approved-frozen-sha256 <sha>`，GPU2 单 Isaac/1 环境，一次，2 初始化 + 72 测量、864 求解器子步，零 Teacher、PPO 或抬升。不重试、不强制结束。证书只描述历史同钟条件盒，实测抓取需要逐子步独立验证。
