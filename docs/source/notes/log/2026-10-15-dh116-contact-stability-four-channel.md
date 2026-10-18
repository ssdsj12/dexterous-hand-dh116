# DH116 四通道接触稳定性与抓取几何初筛

父节点：[T500 D3/D4](../todo/T500-m1-dual-panda-o6-bimanual-mpc.md)。Baseline/Candidate Ref：5e3bcc4 + dirty worktree；未提交，未改源USD。

## 实际动力学诊断

v7_dynamics直接记录PhysX当前COM Jacobian、质量矩阵、重力及驱动readback。所有源驱动为force；实际手增益5/.2、effort .1Nm、velocity .3rad/s。食指折叠后的等效惯量约4.24e-5kg·m²。1ms下接触力和速度隔步交替，JᵀF约-.010至-.044Nm，非简单驱动限幅证据。

v8_first_order仅将运动外推改为一阶（保留实测pose/twist，加速度外推为0），仍第17步过速拒绝。v9_100us保持其他条件，仅将机器人/杆耦合dt减为100µs，400步/40ms通过，max speed .100096rad/s、tracking .00321057rad、coupling 2.085e-6rad、峰值力.354801N。支持显式耦合步长导致本例不稳定，不声称全任务稳定；未放宽任何安全门。

## 同配置四通道有/无反力对照

GPU0，taskset core0，数值库单线程，顺序跑；全部自由rod、无端点约束。每通道400步/40ms、初始10µm诊断接触，非桌面抓取初态。当前脚本SHA c5d0a9da4ccbc340e16390303341415183bc03fb2e1b174e30179b8ed6565c7a。

主仓库assets/dh116_v2内contact_pair_{channel}_100us_v1.json固定各自四份summary/trajectory SHA，全部passed=true/grasp_qualified=false：

| 通道 | max q difference rad | max qdot difference rad/s |
| --- | --- | --- |
| left_index | .003210574 | .100107098 |
| left_thumb | 7.6294e-6 | .011976113 |
| right_thumb | 8.32975e-6 | .012971640 |
| right_index | .003085911 | .093555154 |

所以四通道冲量→机器人反力不是只算不施加，但不证明持续夹持/抬升。接着运行left_index_50us_v1，800步同物理40ms，用于步长一致性检查。

## D4只读几何初筛

新增grasp_geometry：两点允许位于不同线缆材料坐标，以两切触中心差与各法向垂直定义同圆柱接触，并用刚体变换对齐指定wire轴/中心/roll。TDD缺模块失败后5测试通过。不是接触求解器或运动资格。

沿同一源指腹前表面射线调整位置，left thumb fraction .2/index .8、right thumb .8/index .8，法向dot=-.8，两侧数值可精确切触。主动前三角分别约[.48774144,.41545716,.68393798]和[.24395483,.02585772,.64647657]。

每侧17个源手部碰撞网格凸包顶点全部纳入桌面平面筛查（线性极值不丢失）。181个roll枚举后最优min z左.296084m、右.296152m，低于桌面.3m；限制部位左finger13_link、右finger22_link。因此候选拒绝，未做物理动作、未修改现有参考点文件。下一步继续点位/朝向筛选，通过后完整臂路径及非目标wire保护，随后真实闭指、夹持、抬升。
