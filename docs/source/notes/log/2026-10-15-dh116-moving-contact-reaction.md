# DH116 运动指腹及反力接线（进行中）

父节点：[T500](../todo/T500-m1-dual-panda-o6-bimanual-mpc.md)，D3-contact；Baseline/Candidate：5e3bcc4 + dirty。前置[重力/桌面wire](2026-10-15-dh116-gravity-wire-integration.md)通过。本轮不改变完整抓握目标，未声称D4完成。

## 实现

DH116 contact_runtime按左右articulation内finger14_link/finger23_link映射四通道，独立加载DH116 contact_hulls_v2，要求外部SHA871edd15420dc26e9aa18823d37bcdc6719cde389190b1ed7cd62be8ab0cbd32。验证源层与派生文件SHA、路径不出bundle、COM有限。采用当前link姿态/物理COM速度，将mesh COM线速度加上角速度叉偏置，并用前后样本估计加速度。

仅复用旧目录内已经测试的通用BodyState、ContactReactionBridge、wire显示及发布前速度/穿透检查；不复用O6 body映射、关节/局部点/增益。反力来自世界原点角冲量换算到当前物理COM，再经PhysX apply_forces_and_torques_at_position每步一次。桥接已有去重、先验全验证、峰值/均值力10N上限、线性/角冲量平衡拒绝门。

verify_fixed_dual_panda_dh116新增contact manifest/SHA与诊断probe、无反力基线选项。模拟仍GPU0、core0+单线程，合成40N·m臂/0.1N·m手，有重力，保留机器人/桌面碰撞。正对照仅初始0.1mm指腹重叠的自由杆，不代表接近或抓握；该探针初始非桌面落定。全臂wire非目标保护尚待D4接线，不能拿诊断初态作抓取路径。

## 失败与修正

- baseline1/reaction1：只读rotation切片被原地修改，未跑物理步。改显式拷贝，后抽取测试覆盖的probe_rod_start。
- baseline2：10步安全但无接触，不合格。原因40杆元素在中点采样，初始把参考对准节点，邻近曲面距离大于半径；修正为元素20中点(.25625m弧长)，不增加穿透。
- 新4项motion/identity/probe测试通过；通用反力桥+DH接口合计22项通过。

## 左拇指真实有/无反力对照 PASS

主仓库assets/dh116_v2/contact_probe_baseline3与contact_probe_reaction2，各10步、dt=1/240s、同源/script SHA/初态/配置、selected=left_thumb，全部物理跟踪/净接触/发布前穿透等门通过。接触峰值1.00001665N；其他三个pad冲量无接触。自由rod仍无固定边界。

新增reaction_pair及verify_dh116_reaction_pair.py要求两份完整逐步序列、同来源/初态/配置、实际q/qd且非零接触、反力开关相反；拒绝只凭计算力宣布反馈成功。7项TDD通过。

实际配对报告contact_pair_left_thumb_v1.json：PASS，max q difference=0.000605255rad，max qdot difference=0.02857555rad/s。报告固定四个summary/trajectory文件SHA，grasp_qualified=false。说明反力真实影响机器人，仅短时单指接触，不证明持续夹持。

## 当前问题与下一步

顺序四通道回归在左食指v1被contact duration>dt门拦截，基线/有反力均未完成，配对文件没有生成。D3-duration诊断：第8步累计0.004166666666668277s，dt=0.004166666666666667s，差1.61e-15s。与已有总积分时长检查统一为atol1e-15+rtol1e-12；增加正舍入容差和1e-9s真实超时仍拒绝测试，24项桥接+DH合同通过，D3-duration CLOSED。只修改通用桥的浮点容差，不改接触/力/穿透门。

新增D3-loaded-stability OPEN：

- 左食指v2：无反力10步通过，反力首步qdot=.540636rad/s，超过.31门停止；接触峰值1.09197N。
- v3诊断初始压入改为显式10µm（不是成功门改变）：无反力10步通过，反力第3步qdot=.808311rad/s停止，峰值.776N，穿透42.3µm。三个时刻qdot=-.288/-.138/-.808，接触力间歇，说明初始压入不是唯一原因。
- v4_1ms：保持原PD，显式耦合dt=.001、40步；无反力通过，反力第13步qdot=.338652rad/s停止，峰值.5515N，跟踪.00266rad、耦合.000251rad。缩短步长改善但不足。
- 已向用户说明不能继续只改初始压入/步长，开始独立有载合成PD验证：stiffness=5/damping=.2，仍effort=.1N·m/velocity=.3rad/s、验收速度<=.31rad/s，dt=.001，40步。v5_loaded_pd正在顺序运行基线与反力；未称通过。这不是厂家实物参数。

更新：v5_loaded_pd基线40步通过，反力第16步qdot=.313324rad/s停止、峰值.617823N。没有通过，不继续扫增益。查本机IsaacLab simulation_cfg.py:123文档，TGS enable_external_forces_every_iteration可改善外力速度更新；v6_tgs仅打开此flag，其他v5条件不变。基线40步通过，反力第19步qdot=.314348rad/s停止。只是延后失败，仍未通过。v5/v6进程均已终止，无训练/仿真在后台继续。

下一步不再盲调：记录接触wrench经实际COM Jacobian映射的关节广义力、驱动/重力力矩、等效惯量与逐步速度，区分驱动能力不足和显式共仿真能量/预测问题；保留.31速度保护和全部失败证据。D3-loaded-stability仍OPEN，非外部权限阻塞。

截至新增数值容差后的DH116全专项+通用桥160 passed（22.77s），真实资料/资产测试无skip。未跑完整抓握。

只读候选几何调查：初始参考点固定时局部两指对向受限，不应强行把两点当同一截面的相对两侧。沿同一源指腹前表面变更射线位置可找到左手候选：thumb局部轴范围fraction=.2/index=.8，q前三主动≈[.89743,.07759,.80783]，两法向dot=-.95、同圆柱切线条件残差1.85e-13、轴向接触间距41.1mm；右手粗5×5样本尚未找到同质量解。仅控制台只读几何实验，未修改参考文件、未做全臂IK/碰撞、未允许运动，不能当D4姿态资格。

Key files：contact_runtime.py、reaction_pair.py、verify_fixed_dual_panda_dh116.py、verify_dh116_reaction_pair.py及tests。未提交或上传。
