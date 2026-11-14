# 右手 Pinch：实际接触、完整物理账本与自然退出

## 本次原生结果

执行HEAD `cbf2bfe5814938dd54a060adb18bfaa3377eb6b6`。GPU3，1环境，单Isaac，234步=2初始化+232测量，29个60Hz控制窗口，12个solver子步/480Hz步，0PPO。自然耗时140.042824秒，小于180秒。外层exit=0、GDB inferior exit=0、signal=null，未重跑、未发送SIGTERM/SIGKILL。app.close真实返回，单实例锁已核验。

配方SHA `ea4e8bc0068cd2cf5ec0457dca8f574d15e9b9cc26a5251cc687bf993e64b46c`。
调度器代码配置SHA `2fcb2478d3c04d8e52d47638a10ec0bbecc4f34247d308a058b662d45d38d873`。
回执SHA `3e12bb3204f46df748e1de7b35b77306e57be016e6a7f626249f0790d466c83d`。
物理账本SHA `79602654bc7f5b5f0be3e6a6cd46e76882fcb0f5140516e792dc2e903d22c3f2`。

限定执行范围 `right_owned_pinch_2_setup_232_guarded_contact_diagnostic_only` 通过。**不是抓取成功，也不是Teacher PPO准入。** 父调度器等worker满180秒时返回None，因为worker还在做退出后的CPU审计；后续只读取已完成同一回执，没有再次调用start_native。原生进程本身已在140.04秒自然退出；CPU审计不计物理步。

## 接触时序：同一实际episode，非几何预测

env_id=0，reset_id=1，episode_uid=`dual-pinch-6deded5da7e84077915b1b0d4b0bf5b7:one`。

| 指标 | 原生solver接触/载荷子步 | 首次SDK步/solver子步 | 连续观察时长 |
|---|---:|---|---:|
| 右食指 finger23 | 560 | 188 / 4 | 97.222 ms |
| 右拇指 finger14 | 530 | 190 / 10 | 92.014 ms |
| 右侧两指同时正载荷 | 530 | 190 / 10 | 92.014 ms |
| 右侧对向方向符号候选 | 528 | 191 / 0 | 91.667 ms |
| 双手所需指腹同时载荷 | 0 | 未发生 | 0 ms |

所有持续时间均是至本次有限窗口结束的观测值；不是5秒稳定Hold。对向方向符号不等同力闭合、稳定抓取或固定锚无滑移。左手本次命令为零，不能把另一历史运行的左手接触拼进这个episode。

桌面在全部2784个solver子步承托杆，114144条桌面pair。线缆节点平均World-Z相对Reset最高仍为−0.000089407mm（float32诊断值，发生在SDK3），没有Lift成功。手与桌面的接触传感来源仍NA，不能由几何无穿透推断“无外部支撑”。

## 实际控制与几何

原24D右手6/7/8/17接近动作，未改变幅度/关节限位/碰撞保护。所有232步原guard为空；29窗口完整，233份Reset/物理端点参数包有效。

- 右主动关节目标与实际位置绝对误差最大0.005605191rad，P95=0.004294574rad。合法SDK目标不等于零跟踪误差。
- 右被动耦合误差最大0.0000037023rad；finger12实际0.008912619～0.010057464rad，未越过原保护线。
- 右base_link实测Z相对首个已完成样本SDK3最低−0.498176mm；明确不是Reset-relative掌心位移。命令总接近−0.5mm。
- 全部source/cooked形状最小桌面净空：左2.117664mm，右2.205340mm；右侧非目标形状避杆保守下界最小仅0.003001mm，余量很小，不能直接推广新轨迹。
- 右所需指腹SDF最小−0.003378mm，保持原−0.3mm保护门槛。

四个曾关注的指腹，本次全过程最小桌面净空如下；各自穿透深度均0mm：

| 指腹 | 左source / cooked mm | 右source / cooked mm |
|---|---:|---:|
| finger14 | 2.992700 / 3.086673 | 3.280619 / 3.325185 |
| finger23 | 2.833199 / 3.172788 | 2.671139 / 2.925593 |
| finger43 | 55.877917 / 56.030032 | 47.524255 / 47.559293 |
| finger53 | 73.364324 / 73.510211 | 61.358155 / 61.610769 |

初态差异：使用已独立静态验证的右手root/q，初始root高于终点0.5mm，接近时下降0.5mm；左侧初态、杆、桌面、原物理参数保持。未修改原始碰撞网格、Mimic或硬限位。

## 生命周期与GDB

owned hand、Articulation view、composer、RopeBatch/runtime、episode和参数读取器弱引用已释放。asset simulation view在STOP后释放；回调清理回执保留。SimulationContext的Python对象在构造/调用栈中仍存活至app.close之后，不能声称所有Python/native别名在close前归零。

本次无SIGSEGV，保留正常finalize的C++栈、线程及动态库映射。此结果证明本配方执行自然退出，**不证明旧C5相机入口或所有Tensor View生命周期问题均已解决**。

## 新增只读分析与CPU验证

`scripts/audit_dh116_right_owned_evidence.py` 校验固定回执、每个原始文件SHA、完整动作/物理顺序；读取真实q/qd/SDK目标及source/cooked几何，沿2784个已记录solver子步复用ContactTimeline。未启动Isaac或推进物理。新增测试明确目标与实测误差、错误来源/索引/NaN/shape拒绝、空值NA和负位移。

33项CPU回归通过：
`bash scripts/python.sh -m pytest -q -p no:cacheprovider tests/test_dh116_right_owned_evidence.py tests/test_deformable_pair_source_basis.py`。
原生发布前后分别有37项集成和16项worker回归。CPU、原生诊断、Teacher成功三者不混淆。

只读分析结果：
`/mnt/mydisk/xk/projects/dh116-autonomous-3c80166/right-owned-native-evidence-20261114-t36T8e8d/run02/result.json`
SHA `644a90b2cc207d3f9087a58c4b3e0552c01c8cb178880bad2159eaa675e53bad`。
原始账本、接触子步文件、GDB、模型不上传GitHub。

## 剩余门槛与下一批

1. 左/右接触在各自独立运行已观察到，下一步必须同一自由杆同一episode双手接触验证；不能合并历史样本冒充双手。
2. 双手共同动作需新CPU路径/碰撞/作用域合同和独立冻结配方，不能重跑已消耗右侧预算。
3. R0稳定性、固定材料锚滑移仍NA，371D仍只有66维已有可靠公共字段；完整奖励、Reset与状态准入未通过，Teacher PPO继续blocked。
4. Tripod/Power分别预检与接触验证，不能套用Pinch姿态。WireSeg本轮0新增训练步；既有seg/depth各128步诊断历史不变，摄像路径不阻断无相机准备。
5. 528个保护文件、两个历史Checkpoint、30mm/5s/0.5mm标准不变。PR #2保持Draft。
