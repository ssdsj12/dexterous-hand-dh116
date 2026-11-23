# Power 双侧局部终点的剩余指垫筛查

本轮从 963 个控制窗口认证路径的最终双手目标出发，固定历史 SDK 6504、rod tick 78024 的 41 节点原始杆几何。工具核对了双侧路径、左手重拟合、前置三段路径、原始回执与支持场景的 SHA-256，并重新检查两个终点原有掌面、拇指和搭档指垫的站位、对向角与 source、cooked 几何。这个起点是已认证的控制目标，原生局部接触资格仍需后续回执。

每项扫描从冻结终点的当前关节角走到原 DH116 耦合闭合上限，保留双腕、其余关节与对侧手的目标。对六个待加指垫各读取 257 个关节角。杆节点先转换到已对齐的原指垫局部坐标，再读取已登记 SDF。材料站必须在正确的索引半区和世界坐标半区，且不邻接支撑。每对有限 SDF 样本之间的 −6、−30、−80 µm 闭合交点使用括界法精查。候选逐一复核原有站位的几何、source 和 cooked 碰撞护栏，以及原 60 Hz 24D 速率。此前的 OakInk 设计对照也采用局部坐标描述。本轮数据全部来自 DH116 历史实测及发布的重拟合。

| 待加指垫 | 结果 |
| --- | --- |
| 左中指 | 最浅独占试探在节点 35、0.902459 rad。cooked hull 间隙上界 −0.548 mm，低于 −0.150 mm 限值至少 0.398 mm。后续节点同时占据其他节点，其中节点 33 的试探还触发原 source finger32_link，间隙上界 −4.849 mm，要求 +0.250 mm。 |
| 左无名指 | 节点 33、1.017858 rad 的最浅独占试探，cooked hull 间隙上界 −0.671 mm，低于限值至少 0.521 mm。后续节点 32 同时接触节点 33。 |
| 左小指 | 整个闭合区间的连续 cooked 间隙下界 +3.764 mm，超过 0.500 mm 几何见证阈值。 |
| 右食指 | 整个闭合区间的连续 cooked 间隙下界 +12.384 mm，原 SDF 采样中无有效合格节点。 |
| 右中指 | 最接近的合格 SDF 样本在节点 3，间隙 +1.791 mm。其他样本已出现 cooked hull 与杆相交。 |
| 右小指 | 节点 6、1.214902 rad 的最浅独占试探，cooked hull 间隙上界 −0.575 mm，低于限值至少 0.425 mm。 |

18 个端点试探均未通过。连续间隙排除仅覆盖表中两条固定腕位闭合区间。其他四垫的结果来自采样和有限交点，腕位或多个关节共同重拟合仍需单独验证。后续重拟合可把左中指节点 35 和左无名指节点 33 的已查实 cooked 裕量作为约束，保留现有三站位并重新核对整手。原生物理步数、PPO 更新均为 0。

## 复现

运行环境为 Ubuntu DH116 Python，加入原 Isaac 安装内的 USD Python 和动态库路径以读取原几何。

~~~bash
python -B -m tools.audit_dh116_power_remaining_digits \
  --palm-receipt "$EVIDENCE/ag2404-power-palm-witness-native/outer.json" \
  --continuous-audit "$EVIDENCE/ag_power_thumb_route_continuous_6504_v4.json" \
  --partial-audit "$EVIDENCE/ag_power_partial_opposition_6504_v1.json" \
  --bilateral-detour "$EVIDENCE/power-bilateral-detour-archive-v1.json" \
  --left-refit "$EVIDENCE/power-left-contact-refit-v1.json" \
  --output "$NEW_RESULT"
~~~

冻结路径 SHA-256 为 b97cd3577b43bc6bd4de68e1c46e28efdf0b678b924fb126051f500e888f8737。筛查退出码为 2，报告保存在 sync-power-round2-qualified-screen.DImGBP/result.json，SHA-256 为 eff0f021b3dc16cdb77bed0775a5b11ff16a340eb5f8cb4768620eab0bd0b5bf。

相关 CPU 回归通过 91 项，其中新增筛查测试 4 项。日志为 sync-power-round2-cpu.uCXXGz/pytest.log，SHA-256 为 e81109ed0693c9c711d7a628dbcac34455fbb95d9fdf122fac0d2a0569bf895a。离线筛查工具、测试和此报告位于原生执行输入清单之外。
