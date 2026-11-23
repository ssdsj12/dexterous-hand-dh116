# Power 右掌换站与多指同步指令

从实测 SDK 6504、rod tick 78024 和已发布的 963 窗口双侧路径终点起步，重新验收起始双手原几何，整个搜索保持左手目标。使用上一轮右掌节点 11、右食指节点 4 的 12D 入口，先将原来超限的右腕 +Y 目标向内投影 0.453287 mm，再联合拟合右腕及六个主动关节。目标优先贴合掌面 11 和食指 4，同时重新核对拇指 8、无名指 8、原碰撞包络和工作区。

六个候选使用同一条实测杆。右掌允许经过节点 9、10、11。拇指允许旧节点 4 和目标 8，无名指允许旧节点 3 和目标 8，食指只允许节点 4。每个节点都满足原材料半区、物理世界 Y 半区和支撑邻接限制。正向指令在 60 Hz 原 24D 合同下精确拼接。每个窗口的八段线性弦以锁定的 SDF 做 480 Hz 材料采样，掌面一旦推进就不能恢复到较早节点。间隔几何证明连续覆盖其他原 source、cooked 护栏、侧别、台面、支撑和原登记掌面 SDF，各已通过的小区间都独立重查。这个流程以固定实测杆和理想指令弦为对象。

| 候选 | 连续认证窗口 | 最早 SDF 采样阻断 |
| --- | ---: | --- |
| 投影后直达 | 0 | 样本 tick 29，掌面 9 超过原穿透限值 |
| 掌面优先重拟合后直达 | 2 | tick 24，掌面 9 超限 |
| 经过中间掌站 | 1 | tick 37，掌面 9 超限。较早的连续检查已发现拇指 source、cooked 裕量为负 |
| 仅按掌面法向先退 | 0 | tick 11，拇指 4 超限。较早的连续检查已发现原无名指 source 裕量为负 |
| 共同松脱方向与多指同步 | 5 | tick 47，掌面 9 超限 |
| 先沿共同方向退再同步 | 0 | tick 3，掌面在许可走廊外触及节点 8 |

最长前缀有 5 个 60 Hz 窗口和 40 个 SDK 采样间隔，5 个窗口均同时改变右腕和至少两个右手主动关节。它来自原掌、拇指和无名指三个锁定 SDF 法向共同求得的松脱方向，世界坐标单位向量为约 (−0.065800, −0.997833, 0)。右腕位移为 (+0.258, −0.410, +0.636) mm。最大归一化关节分量 0.318751，平移向量范数 0.959535，腕转角向量范数 0.330880。独立重查的最小连续裕量为 0.122 µm。前向材料采样始终只看到掌面节点 9，换站尚未发生。结果文件包含五个窗口的完整目标、正向命令及通过相同 24D 速率检查的反序取反命令。逆向几何经过相同目标弦，掌面单调性只用于正向采样。

右掌的 source 凸包、原 cooked 三角扇和实际登记的 SDF 分别绑定历史测量。源凸包相对连续杆胶囊的间隙上界起点 −2.293 mm，前缀终点 −2.405 mm，均有相交。原 cooked 三角扇在杆节点的最小间隙为 −66.49 到 −244.19 µm，登记 SDF 读数为 −50.10 到 −221.94 µm，两种读数都只占据节点 9。原来被选中的掌面用登记 SDF 作为连续杆护栏。source 凸包的杆相交和 cooked 三角扇的节点值是固定目标检查点，不能据此宣称两者全程与杆分离。最终目标虽然都在原工作区内，仍没有通过右掌 11 和三个指定指垫的精确联合验收。本轮没有原生施力、载荷、包握、Isaac 步或 PPO 更新。

## 复现

在 Ubuntu DH116 Python 加载原 Isaac USD 的 Python 和动态库路径。

~~~bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -B \
  -m tools.audit_dh116_power_synchronized_relocation \
  --palm-receipt "$EVIDENCE/ag2404-power-palm-witness-native/outer.json" \
  --continuous-audit "$EVIDENCE/ag_power_thumb_route_continuous_6504_v4.json" \
  --partial-audit "$EVIDENCE/ag_power_partial_opposition_6504_v1.json" \
  --bilateral-detour "$EVIDENCE/power-bilateral-detour-archive-v1.json" \
  --relocated-entrance "$EVIDENCE/sync-power-round5-relocated-palm.3y18Mv/result.json" \
  --right-palm-cooked-record "$EVIDENCE/ag2171-power-palm-parity-native/gpu/native/right_base_link.json" \
  --output "$RESULT"
~~~

回放输出为 sync-power-round6-corridor.Gv0oyf/result.json，SHA-256 为 ebaceeee4f56692937da2e9e5ce704a96e38b5fb21571cd2f7184a97691fe77f。117 项 CPU 回归通过，其中新增 4 项。日志为 sync-power-round6-cpu.YwqsXE/pytest.log，SHA-256 为 c3c427e107258d112ae904f3288052943464664bbd692ca23f28251ee0f3e244。冻结原生配方的输入清单未改动。
