# Power 左右掌原生凸包 CPU/GPU 等价

新入口 `audit_dh116_power_palm_parity.py` 读取两侧原始 USD 的 `base_link` 碰撞分支。两次都使用同一非对称校准体确认 PhysX 返回的是 mesh-local 坐标，读取 collision owner，使用既有 0.5 µm 容差和有向多边形环比较。每次独立 headless 进程只显式执行 2 个 480 Hz SDK 步，采样同为 SDK4。实际物理设备由 SimulationManager 和 Tensor View 的 ordinal 同时确认。CPU 与 GPU 两次外层自然退出均为 0，软截止 180 秒，无重跑和强制结束。

| 原始掌部 | CPU / GPU 顶点 | CPU / GPU 多边形 | 对应顶点最大差 | 有向索引环 |
| --- | --- | --- | --- | --- |
| 左 `base_link` | 36 / 36 | 63 / 63 | 0 m | 一致 |
| 右 `base_link` | 37 / 37 | 62 / 62 | 0 m | 一致 |

两个局部基底的最大矩阵差均为 0。GPU 本轮读回的每侧顶点、索引缓冲和多边形记录与历史 Teacher 场景掌部数据逐项相同。历史严格闭合凸体核验仍因原 SDK 多边形平面不能严格包围顶点而失败，左／右最大越界约 69.5／67.9 µm。当前结果独立验证设备等价，掌部 SDF 尚未烘焙或注册，也没有实际持握数据。下一步针对同一原始表面验证合适的连续距离表示及误差界，再接入 Power 接触路径。几何资产、动力学和安全阈值保持原合同。

33 项 CPU 回归通过。复现实验使用 `python -m scripts.audit_dh116_power_palm_parity --capture-pair <new-dir>`，可用 `--compare <existing-dir>` 重新校验回执与顶点拓扑。实际回执保存在 Ubuntu 证据目录 `ag2171-power-palm-parity-native`。CPU 外层 SHA `f24b231283706510b8a87cf072a96c1fb6f67662ad96b5db01d3a32527f55e17`，GPU 外层 SHA `689ac3fb8cdd39c5ea3aa189c17c1595d7f4cbf379785941e95ec75da8bb93da`，比对 JSON SHA `c296c2301d9c266693ff5c4956889bb1e279a8e7e6afe9cc988650bc87c381cb`。
