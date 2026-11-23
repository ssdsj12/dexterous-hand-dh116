# Power 右食指的材料站可达域

使用原 SDK 6504、rod tick 78024 的实测杆，以及已发布 963 窗口路径的双手终点。工具核对冻结路径、原生历史回执和上一轮固定腕筛查，核验所复用的两个离线工具的源码摘要，再重验左右手原有局部接触几何。左手目标全程固定。右手原站位为掌面 9、拇指 4 和无名指 3。

## 轴向平移与绕杆滚转

右食指从 0 到 1.221730 rad 的全部原关节区间读取 257 组 locked cooked 凸包。对耦合 FK 的实际关节链验证独立性，再从每个材料节点的几何距离下界中扣除 0.299071 mm 的相邻样本间点位移储备、2 µm 数值储备、5 mm 杆半径和原 0.5 mm 几何见证阈值。

轴向滑移从垂直杆切向的投影中消失。绕通过掌面节点 9 的测得切向做任意角度滚转时，节点的逆向几何查询在横截面上的位移上界是节点到该轴距离的两倍。由此排除右半侧无支架邻接节点 3–17。对轴向投影再应用原腕工作区，纯轴向位移只允许 −58.668 到 +7.453 mm。节点 18、19 分别需要至少 +172.925 和 +185.360 mm。两个约束联合排除了全部 17 个合格节点 3–19。这个证明覆盖固定杆上的完整食指关节区间及任意轴向滚转，其他方向的腕平移或旋转仍需检查。

| 右食指目标 | 固定腕朝向所需位移下界 | 总平移不超过 6 mm 时的一般转角下界 | 轴向滚转后的最少横向缺口 |
| --- | ---: | ---: | ---: |
| 节点 3，原无名指站位 | 21.857 mm | 12.14° | 9.976 mm |
| 节点 5，最近的未占站位 | 32.053 mm | 30.20° | 12.219 mm |

## 非轴向拟合与右小指

在三轴各 ±12 mm、三个独立转角分量分别 ±0.28、±0.16、±0.32 rad 的提议盒内，使用两个相反的横向和翻转种子。四次右食指拟合分别尝试节点 3、5，均未通过原腕工作区、保持原站位和整手 source、cooked 几何验收的组合条件。

右小指单独闭合到节点 6 时，掌面、拇指和无名指仍保持原独占站位，原拇指与无名指法向余弦为 −0.25956。右小指也只占据节点 6，SDF 间隙为 −6 µm。cooked 小指间隙上界为 −0.575060 mm，超出 −0.150 mm 穿透限值至少 0.425 mm。一个耦合腕方案使四个 SDF 站位都达到约 −12 µm，但超出原腕工作区，且 cooked 无名指间隙上界为 −1.689142 mm。按该姿态刷新分离方向后重新拟合，仍无合格端点。七次拟合总计通过 0 项。包握与原生载荷资格保持待测。

独立几何读取以同一 SDK 行的实际右食指刚体姿态和 41 个实际杆节点为输入，逐字节核验外回执、内嵌冻结计划和相关原始行。在历史 SDK 6504 上实测形状相对杆的 cooked 间隙为 25.873–25.877 mm，最近杆段为靠近支撑的 0 段。该历史读数是旧实测姿态的数据。后续可对已冻结新配方的真实回执读取相同指标，默认取最后两条 SDK 行。几何距离与正载荷资格由不同证据决定。

## 复现

Ubuntu DH116 Python 需加入原 Isaac 安装内的 USD Python 路径和动态库路径。历史筛查使用以下命令，未启动 Isaac。

~~~bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -B \
  -m tools.audit_dh116_power_right_index screen \
  --palm-receipt "$EVIDENCE/ag2404-power-palm-witness-native/outer.json" \
  --continuous-audit "$EVIDENCE/ag_power_thumb_route_continuous_6504_v4.json" \
  --partial-audit "$EVIDENCE/ag_power_partial_opposition_6504_v1.json" \
  --bilateral-detour "$EVIDENCE/power-bilateral-detour-archive-v1.json" \
  --fixed-wrist-screen "$EVIDENCE/sync-power-round2-qualified-screen.DImGBP/result.json" \
  --output "$RESULT"

python -B -m tools.audit_dh116_power_right_index inspect \
  --receipt "$FUTURE_OUTER_JSON" --receipt-sha256 "$INDEPENDENT_OUTER_SHA256" \
  --output "$GEOMETRY_RESULT"
~~~

筛查退出码为 2，证据为 sync-power-round4-right-index.c4q8SL/result.json，SHA-256 为 121cf3dde2c1762b701c79b4cfc086295c15ae4cd3b224394f49454b1d9da381。102 项 CPU 回归通过，其中新增 6 项。日志为 sync-power-round4-cpu.8KxI9z/pytest.log，SHA-256 为 1502c8a6a28ff1cdca7220f90577b2cf0be93703bd45863e62ffccbbb552e033。读取器拒绝了旧配方的真实外回执。新原生运行、求解器步数和 PPO 更新均为 0。
