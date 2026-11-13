# 柔性 Power 独立初态预检

从既有握式多腕搜索档案读取左右各 16 个负候选。输入档案 SHA256 `e5e040251c218dc90d198721b316b0a2e0bb4fec4c60b1f373de518ba8146c39`，每个候选保留其自身世界根姿态及六个主动关节。没有引用 Pinch 候选姿态。

将这些候选放到当前柔性任务的桌面和初始 5 mm 半径直杆几何中，只计算静态碰撞与几何间距。每侧检查 17 个源 USD 凸包和五个锁定 PhysX cooked 指腹。左右各 16/16 个样本通过静态桌面及杆不相交筛选。左、右组最小全手桌面净空分别为 26.167 和 29.183 mm。各侧最接近候选的最差 cooked 指腹至杆表面间距仍为 58.499 和 58.831 mm。原档案对两侧的握式筛选各 0/16 通过。仍需独立的柔性 Power 接近轨迹、双手可达性、实际 Reset 和接触证据。

只读结果 `power-geometry-first.json` SHA256 `ef59990bf9181feb66e6cf4d9963e2b79c73973401969d37c4a53103d3e11922`。报告包含每个姿态的碰撞列表和逐指腹间距。该批次没有启动 Isaac、训练或生成配方。

~~~sh
DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 bash scripts/python.sh -m scripts.audit_dh116_flexible_power_pose --archive /home/xk/.local/share/dh116-autonomous-3c80166/evidence/teacher-wrap-diverse-search-20261112-Ey7e0j/probe.json --archive-sha256 e5e040251c218dc90d198721b316b0a2e0bb4fec4c60b1f373de518ba8146c39 --output /absolute/private/power-result.json
~~~

档案身份 CPU 测试 7 项通过，保留来源变化、候选身份重复、手侧重复和虚假资格拒绝。其结果作为几何筛选数据，Power 的 Teacher 准入保持关闭。
