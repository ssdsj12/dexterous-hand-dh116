# Power 右掌换站与大角度倾转

数据来自实测 SDK 6504、rod tick 78024 和发布的 963 窗口双侧路径。左右起点先通过原局部接触几何检查。本轮保持左手目标，将右掌从节点 9 换到 7、11 或 13。右腕用新掌站的材料坐标构造轴，在两条非切向轴的平面内改变倾转方向，倾角限制为 31–63°，另外保留独立绕杆滚转。新掌站允许三个局部平移分量各 ±12 mm。拇指、无名指和食指按实际 SDF 重新分配站位并联合重拟合，其余右手关节也参与整手净空调整。工位选择要求同一实际右侧材料半区和世界 Y 半区，且与支撑不相邻。

## 连续几何排除

由 SDF 低值顶点向相邻插值单元扩展，形成对真实三线性子水平集保守的外包盒。食指全行程经 257 个关节角覆盖，并扣除 0.342 mm 的样间位移储备。所得腕位无关掌面—食指节点间距区间为 0–231.216 mm。该区间过宽，未进一步排除这三种掌站下的任何独立合格食指节点。

对实际搜索方向使用连续腕工作区界。每个方位区间覆盖中心方位 ±0.40 rad、31–63° 倾角和 ±0.42 rad 滚转。每个区间的三维角域分成 12³ 个小区间。解析弧长界覆盖相邻采样中心之间的全部腕根位移，再加局部平移盒和数值储备，与原工作区逐轴比较。12 个方向区间共 20,736 个小区间，其中 11,186 个与原腕工作区完全分离。三个整方向区间被排除。

| 新右掌站 | 被排除的方位中心 | 超出原工作区的保守下界 |
| --- | --- | --- |
| 11 | π | 25.635 mm |
| 13 | −π/2 | 14.856 mm |
| 13 | π | 51.487 mm |

## 精确验收与剩余限制

最终六个联合重拟合均未通过精确端点验收。一项落在原工作区内，但右食指站位仍开离，拇指 cooked 包络同时超出原允许深度。初次宽角度筛查出现一个有区别的右食指入口。右掌指定节点 11，拇指和无名指指定节点 8，食指指定节点 4，实际腕相对倾角 35.740°。右食指在原 locked SDF 上仅节点 4 为负，间隙 −26.825 µm，整右手 source、cooked、桌面和支撑几何通过。独立复核显示腕 +Y 偏移达到 30.451 mm，超过原 +30.000 mm 限值 0.451 mm。掌面和无名指仍各有 +1.542 和 +3.745 mm 间隙，拇指目标节点在其原 SDF 网格之外。这一入口可作为带约束重拟合的实测杆几何反例。

CPU 几何记录未测原生载荷。原 Wrap 定义继续要求两手每个求解子步的五指垫、掌面载荷以及对向证据。世界 Y 侧核对使用同一完成 SDK 行的实测杆分界，原始求解接触逐对子步标签也须有效。未更改这些判据，未使用外部姿态数据。本轮新增 Isaac 运行、求解器步数和 PPO 更新均为 0。

## 复现

使用 Ubuntu DH116 Python，加入原 Isaac 安装中 USD 的 Python 和动态库路径。

~~~bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -B \
  -m tools.audit_dh116_power_relocated_palm \
  --palm-receipt "$EVIDENCE/ag2404-power-palm-witness-native/outer.json" \
  --continuous-audit "$EVIDENCE/ag_power_thumb_route_continuous_6504_v4.json" \
  --partial-audit "$EVIDENCE/ag_power_partial_opposition_6504_v1.json" \
  --bilateral-detour "$EVIDENCE/power-bilateral-detour-archive-v1.json" \
  --right-index-screen "$EVIDENCE/sync-power-round4-right-index.c4q8SL/result.json" \
  --output "$RESULT"
~~~

最终回放退出码为 2。结果为 sync-power-round5-relocated-palm.L1PgQS/result.json，SHA-256 为 612be36c63a06a974492ce3659aed9b0e047daf1b674553dae3ad2b263f12000。初次宽角度候选的 12D 状态位于 sync-power-round5-relocated-palm.3y18Mv/result.json，SHA-256 为 367fe3f2ae5316b93f91ed4604b7aec1a1b13adc303eac7ef6768970b5059b55。调用本工具的 exact_gate 对站位 11/8/8/4 的原状态重新读取全部几何，独立复核保存在 sync-power-round5-exact-check.xfe9qh/exact.log，SHA-256 为 34d13dbbe50f934d70bd910ec9d59fe1abc9ec74ef2e89941220250311e6b5cb。

107 项 CPU 回归通过，其中新增 5 项。日志为 sync-power-round5-cpu.FASzZl/pytest.log，SHA-256 为 3042dc79ece3d321f0ea31d69d53a6d6cad7e9eb84c8e70748bea70ccf59e8b9。
