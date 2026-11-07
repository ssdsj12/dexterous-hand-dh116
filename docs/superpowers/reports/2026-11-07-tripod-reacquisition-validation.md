# 三指接触再建立：单因素验证记录

本阶段从已发布的 `0fce5e9f6f4d9bd90dd3e13522d8c56e0d806389` 继续，针对三指 Teacher/MPC，未开始长时间 PPO。3844D 纯感知 Actor、371D 特权 Critic、24D 动作、Tanh Gaussian 和原 v2 默认路径保留。所有实验保留原物理、执行器、初态、相机和净抬升30 mm／连续保持5 s／滑移≤0.5 mm的原生标准；没有外力、物体附着或新增支架。

独立源码目录为 `/mnt/mydisk/xk/projects/dh116-tripod-20261107`；本阶段输出为 `/mnt/mydisk/xk/dh116/runs/tripod_contact_20261107T0400Z`。历史 Checkpoint、原始运行和失败证据不覆盖，运行文件及模型不进入 Git，PR #1 保持 Draft。

运行02和03均已完成并通过独立只读证据审计：二指均4/4成功，三指均0/4成功。02三指最高物体最低点净抬升为 **0.006431 mm**；加入几何约束求解的03为 **0.019719 mm**，仍未达到30 mm。03三指两个回合掉落、两个在零参考尾段达到原超时，没有采集截断；真实接触有所变化，但三指稳定抓取和 Lift 尚未修复。

## 原压力控制失败的实际时间线

对上一轮 `pressure_reference_pilot` 的480-Hz物理子步、实际指令和第一个终止事件做只读重放。表中 tick 为零基，最后安全周期与终止后 reset 分离；连续接触不是不同时间片段的累计。

| 三指环境 | 右手最后有效对向 tick | 右手最后有效 pad 承载 tick | 双侧三指最长连续接触 | 掉落控制数 |
|---|---:|---:|---:|---:|
| 4 | 395 | 438 | 6.25 ms | 684 |
| 5 | 409 | 481 | 347.92 ms | 938 |
| 6 | 405 | 462 | 8.33 ms | 783 |
| 7 | 339 | 521 | 289.58 ms | 864 |

四个三指来源从 tick437 起，12个手指速度永久全零，只剩双腕各3 mm/s向上。右手对向接触在此之前已丢失，tick437 时右手均无有效承载 pad。旧压力 MPC 只给已有承载触点正权重；缺失 pad 的回退只是当前来源指速，因此后段不会主动找回接触。这是已经核实的控制缺口，不等于已解释初次接触丢失的全部原因。

末安全周期双腕已上移约15–29 mm，物体却横向漂移18.10–19.67 mm、最低点下降5.31–9.51 mm；环境7主要为单边托举和倾斜。原实际运行代码在物体底部低于初始底部10 mm（本场景0.35 m）时触发 `object_dropped`，终止物理样本明确早于 reset。不得把质心上升或腕上移当作物体净 Lift，也不能将掉落描述为仅差一点高度。

独立时间线：`tripod_pressure_drop_timeline_20261107.json`，SHA256 `502304ba9b16df1444119b2c0e2626d8da67fa54faf54bf361cefd33f5b65cf6`。逐 link 被拒接触的原始位置/法向和物体—支架接触力仍缺，初次失联究竟由腕运动、退压或触点迁移主导，不能唯一归因。

## 本次实验设计与证据边界

先仅增加缺失拇指、食指、中指的有界几何接近目标。已有承载 pad 沿用原压力规则，双腕逐值保持原 Teacher 参考；不同时改变腕时序、奖励或物理。原 Source-CAD atlas 的中央表面点和外法向经实际 body-link pose 转换，目标来自当前有限圆柱侧面，明确为特权几何参考，不写入实测接触位置、法向、有效性或载荷。

几何目标必须面向物体、具有可解释间距且在局部 Jacobian 下能改善距离；错误朝向、负几何间距、无进展、预算耗尽、无效或被拒接触均不能成为继续向内闭合的理由。接近速度最多1 mm/s，每个 pad 的本回合累计请求接近距离最多1 mm，尝试时间最多1 s，持续0.5 s无进展后停止。接触闪烁不能重置预算。缺失 pad 的目标锚定当前物体，不能跟随命令腕位移一起搬走，导致模型把脱离物体抵消掉。

首个原生实验使用原8环境布局，二指作为未改控制，三指检验实际再建立接触。先报告真实 pad 承载和连续对向时间，再独立报告原生 Lift、Hold、滑移和成功。CPU几何误差下降不是原生接触；四个固定来源回放不是100个独立初态，Teacher/MPC成功不是 Student成功。若12个手指自由度不足，再依据证据单独考虑腕部优化或物理就绪阶段控制。

## 已执行 CPU 验证

基线于新隔离目录运行原压力几何/绝对参考/采集测试，90 passed，11.89 s。新 factory/准入测试先实际得到2个预期失败（未知 profile、错误 factory 分支），接入后 `test_vision_teacher_collection.py` 为23 passed，8.93 s。JUnit 分别保存在本阶段输出的 `baseline_cpu.xml`、`factory_red.xml` 和 `factory_green.xml`。这些是 CPU 测试，不计原生抓取次数。

冻结的 helper SHA256 为 `6238c56252b6e035602a176a76ac03536578d033dc8469377ece90bdcb9d0d8b`，adapter 为 `18c5c42a9e77eb5b87164f3834494f04ee5a5e18225b2c2b8e5bb09cac423cb9`。最终相关模块228 passed，18.83 s；`acquisition_review_red.{log,xml}` 与 `acquisition_final_cpu.{log,xml}` 保留真实失败到通过过程。独立复审修正了重解后的剩余目标准入、失效/过期预测标记、原生无效传感器隔离、正间距接触后的进展记账和四元数范数有效性；不改变原正常物理路径。

独立36个旧原生快照 CPU 重放包含 tick0，所有获准动作的编码后首步距离复算均无违规，二指动作和所有双腕动作逐值等于原参考；初态每个三指环境六个所需指腹均可生成局部几何请求。审核文件 `acquisition_adapter_native_snapshots_20261107.json` SHA256 为 `0c86af8baa4a280d2ebee00d5af1b37e046aee34a5e5c1f0087dc0446c662d74`。这36行不是新原生回合，也没有证明任何物理承载。

完整 CPU 使用原范围 `tests --continue-on-collection-errors`、已有 USD libs 与 DeformX 路径、隐藏 CUDA、OMP2 和禁自动插件。最终 `full_cpu_final_04.xml` 对应 **5569 passed、96个原有failed、28个原有errors、49 skipped、91 passing subtests**，304.40 s，exit1。原5670个节点无缺失或状态变化，新增72项全部通过；不能称全仓库全绿。JUnit SHA256 为 `be0765c56e13e466c15667c338ed1d50a3d64eab8939e779b5673f62956575d0`，节点对照为 `full_cpu_final_04_comparison.json`。

此前两次全套尝试因最终边界回归修改而主动中断并保留。第三次全套多出的3个旧测试失败来自新归档环境：旧机器绝对资产路径和三个文本fixture的CRLF。仅在服务器隔离stage恢复此前已测试的资源路径和逐字节相同的LF fixture；3项针对性复查通过后执行上述第四次全套。仓库物理/配置文件没有因此修改，也没有更改正在运行的冻结副本。

## 原生启动与来源核对

第一份源码复制因该文件系统不允许 `copystat` 元数据操作而在启动前退出，保留部分副本和独立证明；只复制内容可以得到完全相同的SHA。改用逐文件内容复制并核对全部文件哈希后，运行01在尚未执行物理步时因归档资源清单中的旧 `/home/...` 路径退出。两次均不计任何抓取回合。

运行02使用相同冻结代码、独立目录 `geometry_acquisition_pilot_02` 和 `geometry_acquisition_pilot_02_code`。服务器资源清单仅4处 `/sources/*/path` 不同，其余JSON内容完全相同；4个USD的原实际源文件、新冻结副本和新旧manifest SHA逐项一致。全部9项冷重置读回和7组物理/材料/solver信息，与旧Teacher和压力控制基线完全相同。初始状态证明SHA256为 `281506f870100a1edfaf2dba17f89bda3d4784092d36d17f7ad781453780643d`，资产证明为 `f5c63d3b883eba3b10975ee422a836fc7abbc84194accebdf9c44ca43226cdf3`。

## 运行02最终原生结果与数据来源

`geometry_acquisition_pilot_02` 自然结束，exit0，运行耗时1644.808 s，共1175个控制周期／原始帧、74个物理记录分块、9113个实际执行样本。二指四个首回合均达到原生成功标准，最高净抬升31.627704 mm，形成4698行合格二指监督数据。三指四个首回合均以 `object_dropped`（failure7）结束，没有合格三指监督行；原生成功观察器的480-Hz子步重放与实际首成功／首终止事件、时间完全一致。

Teacher 来源为 `/mnt/mydisk/xk/datasets/dh116_teacher_multistyle_371_v1/surface_demonstrations.pt`，SHA256 `f4fa715c58ce0d767588a6f2bf5aed248d62d22833d4794a97343baf31b8aaf9`。运行清单 SHA256 为 `e0cde026f2e7c836875ed64a772fad5a39cf358120e8daf55fd2842fcff9fb76`；全执行样本数据 SHA256 为 `bd4d11d0172259626bfe5953552adfc4af2b3192ddb82b1a124869fb125d91c1`。实际9113行来源动作与对应 Teacher 时序误差为0，所有已发出动作的腕通道与来源逐值相等，24维动作有限且有界。该阶段 PPO 更新数为0，也没有训练或验证新 Student。

独立审计核对全部1175帧的动作前物理对齐，逐行来源绝对轨迹重建误差为0；9105行具有前一物理快照的 command／载荷与诊断相等，首8行另由精确冷重置证明绑定。81行感知点重建最大误差为 `2.682209014892578e-7 m`，相机时间误差为0。Source-CAD atlas SHA256 为 `f4df6a78243cce0cd39f706de3eb26deb6d80ca440486a8c56702b3422b2151e`，其提取证明 SHA256 为 `1016d6ee82a8d076f3bdda09a880b663bf3602c40c17bd011daaf7d14058e1fc`。几何参考点、模型预测和原生实测接触分别审核，未把几何相交当作测得承载。

完整只读证据为 `geometry_acquisition_pilot_02_independent_audit.json`，SHA256 `ca2f143d09b84e0ad41469c5f569312ea53483014f5ba2d3cb0ac28ed73fd341`。该审计验证已记录数据的一致性与原成功观察器重放，不补造未记录的运行内部信息；合格数据池校验也不等于重新执行物理。

运行03期间再次核对受保护来源：Teacher 原数据、来源 iter129、基线 iter1439、原 v2 iter1750 四份文件的 SHA 均与本阶段开始前相同。证据 `protected_models_while_native03.json` SHA256 为 `8e973d1fe6fafc89ee5e256abbb01a5afc556fbe73a852e3530273a94bee15b7`；这证明来源未被覆盖，不表示新的训练或泛化结论。

下表 Lift 使用物体最低点相对实际初始最低点的最大增量。为排除终止后 reset 污染，机制统计排除整个终止控制周期；首终止事件另从实际物理记录核对。控制数从1计数，对应零基终止 tick 分别为1066、1115、1103、1127。

| 三指环境 | 旧 pressure 最高 Lift (mm) | 02最高 Lift (mm) | 旧／02掉落控制数 | 02原生成功 | 02合格 Hold |
|---|---:|---:|---:|---:|---:|
| 4 | 0.018010 | 0.004436 | 684／1067 | 否 | 0 s |
| 5 | 0.733126 | 0.006431 | 938／1116 | 否 | 0 s |
| 6 | 0.103573 | 0.005927 | 783／1104 | 否 | 0 s |
| 7 | 1.075333 | 0.004007 | 864／1128 | 否 | 0 s |

四个三指回合均在各自来源轨迹结束前实际掉落，未被来源长度截断；因此增加来源结束后的采集尾段不能解释或挽救这次已结束的失败。三指没有满足离地、对向接触、深度和原生有效性条件的可观测滑移窗口，**slip 为 NA，不能写成0或滑移合格**。二指条件滑移观测与原生成功观察器的保持判据分开记录。所有环境的实际 torque 均未记录，不能从 effort limit、关节误差或接触力推算为0，也不能声称通过实测力矩验收。

## 运行02接触机制与旧压力控制对比

实际承载要求480-Hz记录中的传感器有效、有效 pad 载荷大于0.05 N；对向接触沿用原判据。以下连续时长来自实际子步，不使用 CAD 间距或 MPC 预测替代承载。累计几何请求距离包含后来被拒的候选请求，属于控制预算，**不是实际 pad 运动距离**。

| 三指环境 | 02左手三指最长连续 (ms) | 02右手三指最长连续 (ms) | 旧／02双侧对向最长连续 (ms) | 02右手最后承载 tick：拇／食／中 |
|---|---:|---:|---:|---|
| 4 | 6725.000 | 27.083 | 335.417／175.000 | 426／406／255 |
| 5 | 7968.750 | 152.083 | 1016.667／177.083 | 426／384／384 |
| 6 | 8077.083 | 8.333 | 10.417／172.917 | 430／380／256 |
| 7 | 8056.250 | 6.250 | 372.917／175.000 | 429／380／255 |

旧 pressure 双侧六 pad 同时承载的最长连续时间为6.25、347.92、8.33、289.58 ms；02四个环境均为0 ms。02左手确实持续承载数秒，不能把前60步的零承载样本推广为全程无接触；右手三指重叠很短，且三个 pad 均在 tick437 前丢失。来源手指速度永久归零后，右手没有观察到实际再捕获。

| 三指环境 | 旧／02峰值有效 pad 总载荷 (N) | 旧／02最大有效 pad 深度 (mm) |
|---|---:|---:|
| 4 | 15.943／4.473 | 1.022／0.783 |
| 5 | 22.773／4.482 | 1.327／0.647 |
| 6 | 18.474／4.502 | 1.310／0.662 |
| 7 | 15.291／4.506 | 1.205／0.712 |

载荷和深度较旧 pressure 降低，但没有同时改善六 pad 承载或 Lift；较晚掉落也不能作为抓取成功。完整逐 pad 的首次承载、最长连续、最后承载、请求累计、stop code／tick、请求前后实际间距与载荷在 `new_geometry_acquisition_mechanism_v2.json` 中保留。该文件 SHA256 为 `0e65107ce1df4777ed9615f74079960e4981822e745da69bf4ad88323ff5f2b8`；按同一统计口径重放旧 pressure 的 `old_pressure_mechanism_baseline_v4.json` 为 `9e543f1fee4aeef651d208262b8d870bd3b48a9f05e087a91026dba22723e3c5`。精简比较文件 `geometry_acquisition_vs_pressure_mechanism_v1.json` 为 `f2b4b7dc3cc1be6582728e0a3aeae7bfb1cfe05706d84268f9db8410ee01b4c5`。原请求的 `new_geometry_acquisition_mechanism_v1.json` 同时保留，SHA256 `584f95e0fd2d07357dab3b942adf0c7b0469d3ba42bbaf541423704d3f71ee4f`。

几何请求在首秒内因 `stop_code=4`（局部预测准入失败）永久停止，而非耗尽接近预算。各 pad 累计请求仅约67–346 µm、0.083–0.35 s，无进展累计不超过约0.083 s，尚未达到1 mm／1 s／0.5 s限制。前60步有120个 active 样本和240个动作变化样本：其中120个变化来自未激活时冻结缺失手指的回退，不能把 changed 全部记作有效优化。该前缀最大关节预测误差为0.000321746 rad，低于0.005 rad门槛，未触发预测误差禁用；这仅解释前缀，不替代全回合结果。

## 下一单因素：几何约束求解

对环境4首次六次几何停止，独立按实际轨迹顺序重放 tick0–44，已发出动作误差为0，停止和预算一致。首次拒绝分别发生在 tick6（右食指）、9（左拇指）、20（右拇指）、28（左中指）、36（右中指）、44（左食指）。其中4次只有未来H2–H4不满足间距，2次首步也失败；未发现朝向或侧壁判据失败。tick6的未来H4间距反而变大，其余主要为预测越过侧面。tick44当前正间距仅约0.034 µm。

这6个实际状态均能在同一双腕动作、同一局部 Jacobian、原物理范围／`q±0.1 rad`／每步0.005 rad限制内构造全H4精确圆柱可行轨迹，最大 witness slew 为0.003413 rad。因而单次无约束 soft 目标解被拒不能证明该状态局部不可达。证据 `native02_first_geometry_rejections.json` SHA256 为 `dc8602ff6783e938b5919f5f43adea3956ee16d89bb2071f08c3efffb8170536`；完整输入 `native02_first_geometry_rejection_contexts.json` 为 `ef887d020dbe90c30bc3aa89813434fe70619dc1e7ce43d214fb6da74fe00fd2`，均在仓库外保留。

因此新 profile 选择 `pressure_geometry_constrained`，仅增加可行性约束求解，不实施重试策略、腕优化或就绪阶段切换。公开入口为 `ConstrainedAcquisitionTeacherReferenceMPC(env, source_sha, reference_actions, *, grasp_styles, cylinder_radius_m, cylinder_length_m, pad_atlas=None, trust_radius_rad=.001)`；圆柱参数来自原已验证配置（radius0.008 m、length0.5 m），不猜测任意物体参数。

新纯 planner [tabletop_geometric_constrained_mpc.py](../../../src/go2_pvcnn/control/dh116/tabletop_geometric_constrained_mpc.py) 与 adapter [tabletop_vision_teacher_constrained_mpc.py](../../../src/go2_pvcnn/control/dh116/tabletop_vision_teacher_constrained_mpc.py) 继承既有路径，未修改旧 acquisition/helper、Pressure、SurfaceGripMPC 或 runtime。planner SHA256 为 `0596d027d778d32fa99f8a845b02641aaa63841eacc94c7300d6265890a8851d`，adapter 为 `4984b40b42ea31eda28783c0e5dac3f9d780628eae85746ebc58f5c5bcd8ed4b`，本地和服务器逐字节一致。

首先仍由旧 `SurfaceGripMPC` 生成 soft 解。无几何目标时逐值走原 planner；soft 已满足全时域几何与压力保护约束时，轨迹、预测和成本逐值保留，不调用约束优化，也不标记修复。只对不可行 soft hand 使用 CPU SciPy SLSQP，每手H4×6个变量，复用原 response0.5 递推、点 Jacobian、原腕运动和目标函数（几何尺度1 mm、nominal权重0.02、smooth权重0.002）。

约束包括每个未来步的原 affine 关节范围、实际 `q±0.1 rad`、0.005 rad slew，非负圆柱侧面间距且较当前正间距改善、facing≥0.8、有限侧壁，以及过载 pad 的相对向内预测位移≤`1e-10 m`。数值裕度只收紧，不放宽旧 exact checker；输出 dtype/device 和实际首动作编码后再核对。无可行解时返回原 soft 结果，继续旧7轮目标剔除和永久 code4 回退，原速度／距离／时间／无进展预算不变。双腕逐值保持 Teacher，已有承载压力规律不变。

可行修复的成本可以高于被拒的不可行 soft 解；诊断明确区分 `baseline_soft_feasible`、`feasibility_repair`、成本变化和 solver 状态，不能称所有修复都降低原目标成本。完整H4指令、模型状态、几何／压力上下文、约束状态和编码后首步检查均保存，以便独立复算。这仍是特权 Teacher 的局部运动学 MPC，没有力、接触或物体动力学预测，也没有扩大 Actor 输入或改变24D动作契约。

## Constrained CPU 与独立6快照验证

新 adapter 测试实际先得到13个缺少模块的失败；纯 planner 初始16个预期失败，另保留2个边界回归和1个“已可行 soft 不应被替换”的真实失败记录。最终相关9个测试模块 **261 passed，19.38 s，exit0**，记录为 `constrained_full_green_v2.{log,xml}`；JUnit SHA256 `52c0f561f90d49226d9703330878acb88be55c1cb17d5f17b7da7dc50d4e89ee`。覆盖原范围／slew／lead、全H4几何与压力方向、不可行回退、dtype/device和行隔离、无几何／二指逐值保持，以及实际编码首动作。

仓库外6个真实快照测试 **6 passed，4.82 s，exit0**，为 `constrained_real_contexts_final_v3.{log,xml}`；逐 case 完整轨迹保存在 `constrained_context_checks_final_v3/tick_{6,9,20,28,36,44}.json`。较早外部测试错误要求每个有几何的 hand 都必须 repair，已改为“原 soft 经检查可行或真正 repair”，同时核对可行 soft 逐值保留；旧失败记录保留，不计为最终通过。

独立 NumPy oracle 在相同6个实际快照复算全H4和float32首动作编码，圆柱间距／朝向／侧壁、过载保护、原物理范围／lead／slew、原目标成本均通过；无几何和原 soft 已可行的路径逐值不变。审阅未留下P1/P2。该独立复验使用的是同6个快照，不能计为额外6次原生回合。证据 `constrained_six_native_contexts_independent_review.json` SHA256 为 `10f6cc9c5eb6568ebc3e16fd8fdefcc273647aa7430720d174a989f024f5b0e9`；最终冻结记录 `constrained_geometry_candidate_frozen_v1.json` 为 `44afae09ac16333fc8424b31d16b8c30252809179fe283210f35414412bc424c`。

factory／采集尾段集成另有最终 **177 passed，22.20 s**，JUnit 为 `constrained_factory_tail_final_02.xml`。这与上述局部测试和完整仓库测试的范围不同，不能互相替代。

最终完整 `full_cpu_final_05.xml` 为 **5613 passed、96个原有failed、28个原有errors、49 skipped、91 passing subtests**，307.13 s，仍非全仓库全绿。对 CPU04 的5742个节点无缺失或状态变化，新增44项全部通过（pure planner20、adapter13、factory2、tail9）；经 CPU04 对原 HEAD 的5670节点核对，本阶段新增合计116项全部通过。JUnit SHA256 为 `accb55cf4190e2813b512c9c100cb99763239ec43eab51ca29edc3f91d4e856d`，节点对照 `full_cpu_final_05_comparison.json` SHA256 为 `0b7b71318977ae9e3bbcc69ac58c115ab59eae7e50d7d5c048c662be70490a49`。

## 运行03最终原生结果

`geometry_constrained_pilot_03` 使用显式 `pressure_geometry_constrained`，监督器 `finished/exit0`，耗时2437.261 s，`frozen_runtime_unchanged=true`。独立核对冻结清单1307个文件，0处变化；manifest SHA256 为 `630a61d2210c86d6baaa5644d3b6112b8f06c097e6de46823d36929df6bceaf2`。待发布的12个 Python 文件与03实际冻结运行副本逐字节相同，证据 `publication_runtime_source_identity.json` SHA256 为 `bffb814656f4cb0cc141fb21e0ad39c5fe174ba6929d87380bd7431dd4b91a08`。

正式独立原生审计已执行一次并 exit0，`geometry_constrained_pilot_03_independent_audit.json` SHA256 为 `6a75f0649d567b2a15e4cea626fe93027cdc2d1d90b3f10ca0d1179e7509e97c`；对应 summary SHA256 为 `eedafcae0d7075a39691c79487e0e166cae9c67ead4289ec37cf110aea0f80b1`。冷重置精确相等；1800个原始帧、113个分块、10559个执行样本均有来源与物理证据。全执行数据 SHA256 为 `b1c45d0dc67dbc89ec01b81cdf97dfeadd31f6af88e0833f7315f755f4b89836`，collection report 为 `746ae9526cbaae78ffe790254c927bb37e3b815d3349b59cd91b12449370a0d4`。

二指4/4达到30 mm／5 s原生成功，最高31.627704 mm，合格监督行仍为4698行二指。三指0/4达到30 mm、0/4达到5 s保持、0/4成功；四个最低点净抬升峰值的 P50／P90／P95／max 为 **0.011349／0.017737／0.018728／0.019719 mm**。没有合格三指监督数据，没有 PPO 更新或新 Student 训练／成功结论。

| 三指环境 | 03最高 Lift (mm) | 首终止控制数 | 实际终止 | 终止所在阶段 | 采集截断 |
|---|---:|---:|---|---|---|
| 4 | 0.009584 | 1800 | 原 timeout，failure8 | 零参考尾段 | 否 |
| 5 | 0.019719 | 1800 | 原 timeout，failure8 | 零参考尾段 | 否 |
| 6 | 0.013113 | 1092 | object_dropped，failure7 | 来源前缀 | 否 |
| 7 | 0.008547 | 1169 | object_dropped，failure7 | 来源前缀 | 否 |

三指来源均为1272步。环境4、5各执行528个真实记录的尾段控制步，合计1056行；每行来源参考24维严格为0，实际 MPC 发出动作另行记录，没有重复来源最后动作。全运行9503行属于来源前缀，1056行属于尾段，8个首回合均完整、censored=0。请求600额外步后实际可用采集上限为 `min(1900,1272+600)=1872`，环境4、5在原1800步超时处终止，早于1872步采集上限，未用采集上限代替物理终止。环境4、5的旧 `reference_sequence_complete=false` 字段不表示这两个执行回合被截断；`recorded_execution_complete=true` 和实际 failure8 共同证明完整终止。

来源前缀和尾段均无三指成功。尾段不能作为与02同预算的改善证据，也不能冒称原 Teacher 来源成功；02四个三指回合已在来源结束前实际掉落。采集尾段功能仍默认0，仅显式实验启用，原物理 timeout／首终止规则未变。

原0.5 mm标准具体指 `SurfaceLiftSuccess`／`hold_window_step` 对物体点在两手坐标系中、相对固定 anchor 的 `drift_m` 上限，可称相对滑移／漂移，不等同于逐接触点物理 slip 测量。03三指均没有合格高位接触窗口，条件 slip／drift 为 **NA/null**，不能填0；实际 torque 仍未记录。本说明不修改阈值或成功判据。

### 480-Hz实际承载、再捕获和载荷

机制脚本沿用02的实际480-Hz有效载荷口径，排除整个终止控制周期，避免 reset 污染；原生首终止事件另外核对。完整结果与独立审计一致：**环境4、5各有一个物理子步的六 pad 同时承载且双侧对向，各2.083 ms；环境6、7为0**。不能把此前60-Hz前440步快照“六 pad 重叠为0”外推为480-Hz全程不存在；也不能把一个子步视为稳定抓取。

| 三指环境 | 左手三指最长连续 (ms) | 右手三指最长连续 (ms) | 六 pad＋双侧对向最长连续 (ms) | 双侧对向最长连续 (ms) | 双侧对向时间比例 |
|---|---:|---:|---:|---:|---:|
| 4 | 15120.833 | 110.417 | 2.083 | 175.000 | 0.840745% |
| 5 | 14114.583 | 16.667 | 2.083 | 6.250 | 0.048638% |
| 6 | 8154.167 | 316.667 | 0 | 437.500 | 2.406049% |
| 7 | 8997.917 | 283.333 | 0 | 460.417 | 2.440068% |

更严格的六 pad／对向／平衡／深度与被拒载荷联合就绪诊断四环境均为0。tick437 后，右拇指均未实际再捕获，其最后承载 tick 为431、424、426、432；右食指和中指出现间歇承载回归，中指分别有3、2、2、2段持续至少50 ms的再捕获，但没有恢复双侧对向或六 pad 承载。右食指最后承载 tick 为474、469、460、461；右中指为503、506、504、484。这些是实际观测关联，不证明几何请求单独造成了再捕获。环境4、5尾段没有双侧对向／六 pad 承载，最低点净 Lift 最大值分别为−0.118552、−0.137558 mm。

实际载荷与深度较02升高。下表使用各首回合完整终止前窗口的实测峰值，**不是合格 Hold 窗口的载荷**；`total hand` 来自 `raw.hand_total_loads_n`，与前文“两手有效 pad 载荷之和”不是同一指标。

| 三指环境 | 02／03单 pad 峰值 (N) | 03被拒 hand 峰值 (N) | 03 total hand 峰值 (N) | 02／03最大有效深度 (mm) |
|---|---:|---:|---:|---:|
| 4 | 3.494／9.820 | 5.798 | 19.109 | 0.783／1.083 |
| 5 | 3.484／8.851 | 5.895 | 19.152 | 0.647／1.190 |
| 6 | 3.722／11.029 | 5.881 | 18.362 | 0.662／1.026 |
| 7 | 3.591／10.771 | 5.849 | 16.161 | 0.712／0.923 |

03跨环境峰值为 pad **11.028534 N**、rejected hand **5.894920 N**、total hand **19.152383 N**、深度 **1.189953 mm**。这些是原生仿真测量，不是硬件校准压力；终止窗口的原始峰值另列于独立审计，不能混用范围。先前匹配的前440个60-Hz观察也已显示载荷增加，但完整回合时长不同，不能把更晚终止或额外尾段本身当作控制效果改善。

完整机制文件 `native03_complete_mechanism_v1.json` SHA256 为 `8ec7dd0067142121a1c0fcc6200d98f08e4d571d663eba93227d8c0ead94d9db`；精简摘要 `native03_mechanism_summary_v1.json` 为 `01f4737d765fd2f04db4fdc72f74f8db3192ecdcaf1de9bdf7b7d5c942aefb73`。外部分析脚本在 `mechanism_analysis_cpu_v1/` 归档，10项合成／完整IO测试通过，输入文件和 tensor 不变；该 CPU 分析没有新增原生回合。

### 求解器、永久停止与模型证书

完整独立审计核对5861行三指 geometry 诊断、524次 solver call、300个 hand repair、333个已可行 soft hand，以及45504个实际获准压力子步，执行约束问题数为0；所有已发出腕动作逐值等于来源。全H4／成本／实际编码首步均复算。跨GPU／CPU的 baseline 重算按 `1e-12 rad` 数值容差比较，不声称跨设备逐位一致。优化调用数、hand数、实际承载次数和成功回合数是不同单位。

累计请求没有耗尽1 mm距离或1 s尝试上限。各 pad 请求距离约0.1866–0.8944 mm，最大请求时间0.9333 s；这些是请求预算，不是实际物理位移。环境6、7右中指在 tick143 达到累计0.5 s无进展门槛并以 code3 停止。10个 pad 以 code4 停止：四个左中指在419–422、四个右食指在422–426、环境4右中指在422，以及环境6右拇指在441。其余 pad 的无停止码不代表仍可获得几何目标；方向、间距和整手保护仍会使目标无权重。

对前9个 code4 事件（419–426，8个 hand 状态）进一步只读分析发现，与02前6个有可行 witness 的状态不同，保存的**固定法向、局部点 Jacobian、原腕运输H4模型**中，间距改善与 facing≥0.8 两条必要条件发生冲突。把径向上界放松为切平面、删除非穿透下界等条件后，必要条件 LP 仍有正矛盾下界。独立审阅从实际输入重新按递推构造矩阵，核对数值证书，8/8成立，最小归一化下界约0.014196，额外扣除 `1e-7` 后仍为正。

这仅排除上述保存模型的H4可行解，是带残差核对的数值证书，不是区间算术形式证明；**不证明真实手指／物体在物理上不可达，也不是 qlead 或 slew 瓶颈的证据**。模型不包含随关节变化的未来 pad 法向、物体运动、接触或力动力学；该证书也不覆盖 tick441 的额外右拇指事件。提高优化器迭代数不能消除已证实的模型约束冲突，但不能据此放松物理或接触判据。证据归档于 `native03_facing_readonly_v1/`，独立审阅 `native03_facing_certificate_independent_review.json` SHA256 为 `fdcfccff61325152493518600005502416da6c54013e9a82f1af77c850f538ae`。

### CAD中心间距与实测接触的交叉审计

另按60-Hz动作前观察交叉统计 CAD 中央参考点 gap、实测载荷／有效性、深度和 helper reason，不能用这些观察数冒充480-Hz连续时长。剔除了无效几何、轴上零间距占位值、无效传感器和终止／reset 行；实际样本中几何不可信数为0。02／03分别有4407／5853行可与原生 `tick*8-1` 前帧对齐，载荷误差和有效性不匹配数均为0；tick0没有前物理帧时深度保持未知。

| 运行／所需 pad | 完整回合 missing 观察数 | 其中 inside_or_on_surface | 有实测承载的观察数 | 承载时 gap<0 的观察数 |
|---|---:|---:|---:|---:|
| 02 right thumb | 3817 | 437 | 594 | 23 |
| 03 right thumb | 5695 | 1256 | 162 | 63 |
| 02 right middle | 4326 | 886 | 85 | 85 |
| 03 right middle | 5126 | 1031 | 731 | 632 |

全回合时长不同，以上计数不能直接作为成功率改善。匹配 `[0,440)` 的观察中，right thumb 的 inside＋missing／missing 比例从02的437/1166（37.48%）到03的1256/1598（78.60%），right middle 从886/1675（52.90%）到950/1127（84.29%）。实际承载时仍可出现负中心间距，证明此 CAD 点不是实际接触谓词，更不是整个 patch／hull 的 signed distance。

03在 tick437 后共有4109个每-pad观察：右拇指全部 missing 且没有 inside reason，不能把后段右拇指失联归因于负 gap 门槛。右中指有4011个 missing，其中93个为 inside reason，发生在四环境各自437–457/461区间；这些93行实测载荷和深度均为0，gap约−0.543至−0.000245 mm。之后仍有98个右中指承载观察，其中心 gap 为正。负中心间距确实能与零承载同时出现并阻止该几何目标，但这些相关性既不解释全部后段失联，也不授权继续穿入或修改物理。

完整分布 `native02_pad_center_gap_complete_v1.json` SHA256 为 `0d5d990dd9c1922f9c4445f316490ca77113555b78286da626494da8cdc8d8cf`，`native03_pad_center_gap_complete_v1.json` 为 `be595c0df83809a1ff99bdaf4deff83cc23d27889870b4d9f396d0b9ffa7db8b`。脚本及11项外部回归记录位于 `pad_center_gap_readonly_v1/`。本轮实际证据支持区分“局部预测模型约束冲突”“CAD点参考不足”和“实际三指承载不稳定”，尚未获得满足原标准的三指 Teacher，更没有三指 Student 成功证据。

## 实际执行命令与审计入口

以下为可核验的运行记录。原生任务由 root 唯一监督器串行管理，监督器保留已有锁、冻结副本和来源检查；不是绕过 `flock` 直接启动训练脚本的命令。02结束后才启动03，两条记录分别执行，不并发运行：

```bash
/mnt/mydisk/xk/envs/dh116/bin/python /tmp/dh116_tripod_acquisition_pilot_02.py
```

```bash
/mnt/mydisk/xk/envs/dh116/bin/python /tmp/dh116_tripod_constrained_pilot_03.py
```

实际完整 argv 保存在本阶段输出目录的 `geometry_acquisition_pilot_02_argv.json`（SHA256 `69dcbdc51029c135e7a93ab7a97297fa0a17fb15c7f74c8d7ad38504199c91b5`）和 `geometry_constrained_pilot_03_argv.json`（SHA256 `bc7faad1edfe0fec9ec6a7a71288c07d9bef2897541a2979722a0f7373af8b79`）。03明确使用 `--vision-teacher-mpc-profile pressure_geometry_constrained`、`--vision-teacher-max-control-steps 1900`、`--vision-teacher-hold-tail-control-steps 600`；其余原生配置以完整 argv、来源清单和冷重置证明核对，避免手工省略参数造成不同实验。

CPU05实际命令如下，GPU对该测试进程隐藏；只列运行所需路径，不导出完整环境：

```bash
cd /mnt/mydisk/xk/projects/dh116-tripod-20261107
USD=/mnt/mydisk/xk/envs/dh116/lib/python3.10/site-packages/isaacsim/extscache/omni.usd.libs-1.0.1+d02c707b.lx64.r.cp310
CUDA_VISIBLE_DEVICES='' \
PYTHONPATH="$USD:src:.:vendor/DeformX" \
LD_LIBRARY_PATH="$USD/bin" \
PYTEST_DEBUG_TEMPROOT=/tmp \
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
OMP_NUM_THREADS=2 \
/mnt/mydisk/xk/envs/dh116/bin/python -m pytest \
  -p no:cacheprovider tests --continue-on-collection-errors -q --tb=short \
  --junitxml=/mnt/mydisk/xk/dh116/runs/tripod_contact_20261107T0400Z/full_cpu_final_05.xml \
  > /mnt/mydisk/xk/dh116/runs/tripod_contact_20261107T0400Z/full_cpu_final_05.log 2>&1
```

177项集成测试的实际 XML 范围如下，来自 `constrained_factory_tail_final_02.xml`，SHA256 `7d4150da0690bc21eb95de5c8f0842a37744d43660fa65dbd3f5ad03fc8a6307`；各项均为 passed，无需猜测未保存的 shell 包装细节。

| 测试文件 | XML case数 |
|---|---:|
| `tests/test_geometric_constrained_mpc.py` | 20 |
| `tests/test_teacher_geometric_acquisition.py` | 42 |
| `tests/test_vision_teacher_absolute_mpc.py` | 22 |
| `tests/test_vision_teacher_acquisition_mpc.py` | 28 |
| `tests/test_vision_teacher_collection.py` | 25 |
| `tests/test_vision_teacher_constrained_mpc.py` | 13 |
| `tests/test_vision_teacher_hold_tail.py` | 9 |
| `tests/test_vision_teacher_pressure_mpc.py` | 18 |

03正式全量只读审计已在监督器 finished/exit0、冻结文件不变后实际执行一次，exit0；其证据 SHA 见最终结果节。命令如下，它与前60步审核、局部机制分析分开，没有新增原生回合：

```bash
OMP_NUM_THREADS=2 /mnt/mydisk/xk/envs/dh116/bin/python \
  /tmp/audit_teacher_constrained_native_20261107.py \
  --root /mnt/mydisk/xk/dh116/runs/tripod_contact_20261107T0400Z \
  --mode geometry_constrained_pilot_03 \
  --code /mnt/mydisk/xk/dh116/runs/tripod_contact_20261107T0400Z/geometry_constrained_pilot_03_code \
  --manifest /mnt/mydisk/xk/dh116/runs/tripod_contact_20261107T0400Z/geometry_constrained_pilot_03_source_manifest.json \
  --output /mnt/mydisk/xk/dh116/runs/tripod_contact_20261107T0400Z/geometry_constrained_pilot_03_independent_audit.json
```
