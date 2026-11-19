# Power 固定预形腕部接近验证

基线 `d92caa343a98e145fffe808cd26a2f272fb3be2b`。`prepare_dh116_power_palm_seed.py --audit-approach` 在选定的固定关节姿态上，沿掌面水平轴以 0.5 mm 档位审计最多 20 mm 平移。每个候选取各原始和 cooked 凸包从起点到终点的连续平移扫掠，再检查 41 节点杆与两端实体支架。水平移动不改变已核验桌面净空。选项只输出证据，不改写种子位置或捕获原生接触。合成回归验证了两个终点均不与支架重叠，但路径中间经过支架时会拒绝。

执行 `PYTHONPATH=src:. <Python> -B scripts/prepare_dh116_power_palm_seed.py --end-supported --aligned-preshape --audit-approach --left-node 33 --right-node 8 --output <独立输出>`。左侧安全到 19.5 mm，支架最小分离轴间隙 35.019 mm，掌面间隙上界 0.2918 mm。右侧安全到 14.5 mm，支架净空 33.314 mm，掌面间隙上界 0.3435 mm。各再推进一档时，原始 `base_link` 无法证明杆净空。最晚有证净空的位置上，五个指腹最大间隙上界仍分别为 27.526 和 27.539 mm。离线命令以 2 退出，记录为未达完整包握几何条件。

53 项 CPU 回归通过。未运行 Power 原生或 PPO，未更改原始物理、安全限制及成功标准。下一步以能够靠近杆的腕部位置为参考，分别验证两手的闭合顺序、指腹与掌面反力、固定手腕保持，再进入完全离架后的净抬升。
