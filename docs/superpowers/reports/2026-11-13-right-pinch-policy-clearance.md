# 右手 Pinch 使用原手腕净空的静态复验

前一右手候选 right-targeted-092 的连续间隙 2.039475 mm 通过独立筛查的 1.9 mm 门槛。双手运行入口使用原策略手腕 2.1 mm 净空门。本次额外复验一个定向候选，保持原杆、关节、手腕、碰撞阈值及输入 SHA。

`validate_dh116_right_pinch_runtime_clearance.py` 读取并检查先前提案、相同的 task、USD 和几何源码。新候选以原右手独立坐标为基准，根部 Z 增加 65 µm，拇指初始主动关节增加 0.0001 rad。保留原闭合速度和 161 点连续界核验，仍检查原始 0.05 mm 严格指腹端点与对向角。

- 连续最低桌面间隙 **2.112018 mm**，超过 2.1 mm 约 12.0 µm。
- 所有非目标碰撞包络对杆的连续下界非负，最小约 **38.247 µm**。
- 两个必要指腹的最大绝对端点间隙 **47.659 µm**，几何方向 dot **−0.82141**。
- 单独增加 65 µm 的对照可满足桌面门，端点误差 **52.242 µm** 超过严格阈值。更大的 80–125 µm 平移对照丢失所要求的方向或端点条件。
- 0.305 m 支撑杆提案保持自由端。原生任务当前未采用这套 Reset seed。微小静态余量和缺失的实时 non-pad cooking 几何仍需原生核对。

CPU 回归 **37 passed**，含 5 项新增。首次脚本写盘发现 NumPy bool 序列化失败，保留不完整的私有文件。改为显式 Python 标量后独立写盘成功。Ubuntu 使用空 CUDA_VISIBLE_DEVICES、2 CPU 线程，`bash scripts/python.sh scripts/validate_dh116_right_pinch_runtime_clearance.py --proposal <SHA锁定先前提案> --output <独立新文件>`，回归包括 runtime clearance、targeted、closure screen 和 flexible style pose 四个文件。

完整结果 SHA256 `9e47b1990493cf9ea62f7324e17c5fb596cd5dd337ca4ec8557ea7098136af1d`，测试日志 SHA256 `d20855870170aea55ec2620a97805abef7c4d6eb6a90ffde8c5c30661dd82e43`。原生物理步和 PPO 更新均为 0。没有授予闭环接触、双手 Reset 或 Teacher 资格。
