# P0 接触诊断、静态几何筛选与历史模型配对验证

本阶段按用户要求优先完成接触诊断和几何筛选。Tanh Gaussian PPO、默认 v2、3844D Visual Actor、371D Privileged Critic、24D 动作合同、原物理与执行器限制，以及 30mm / 5秒 / 0.5mm 原生标准均保留。PR #1 面向 `11-5`，保持 Draft。

原 R0 Adam 控制器在等待原生配对评估时被取消，没有启动任何 R0 更新；取消前状态已单独保留。原129/1439配对评估自然完成。本阶段没有启动长时间 PPO，没有覆盖任何历史模型或原始数据。

## 实现及证据边界

`tabletop_grasp_contact_diagnostics.py` 是纯 CPU、流式事件归约器；`tabletop_grasp_contact_native.py` 在原生终止 reset 前读取同一子步的 object-filtered PhysX 缓冲。每个环境、episode、左右手和五个 distal pad 都保留建立、持续、丢失与 censor 事件，并绑定绝对物理步、控制子步、仿真时间和 dt。未知观测、缺步、episode 结束及诊断截止不会伪造接触丢失。

原始位置、法向、signed normal scalar、separation、sensor/slot/body、count/start/capacity 与实际 body-link pose 独立保存。指腹通过 CAD 区域、表面距离和法向分别门控；加权接触位置和合力方向明确标为归约值，不能冒充单个原始法向。缓冲区容量信息不能证明上游 PhysX 从未发生 overflow，报告保留这一限制。

七种原因是可同时出现的 flags：`NO_CONTACT`、`ONE_SIDE_CONTACT`、`WRONG_NORMAL`、`CONTACT_LOSS`、`NON_OPPOSED`、`TORQUE_LIMITED`、`SLIP`。全部10个 pad 的事件均被记录；与抓法有关的原因和对向接触按 pinch/thumb-index 或 tripod/thumb-index-middle mask 统计。每手、双手分别给有效物理秒、有效样本比例和 coverage。力矩饱和使用原 implicit-actuator 模拟估计与原 effort limit，98% 仅为诊断阈值，不是新成功标准或硬件测量。NaN、缺失或不完整力矩向量保留原始逐关节诊断并使 torque coverage 无效，不能破坏独立接触记录。滑移沿用有效固定锚漂移；未建立锚时为 NA，不填0。

默认路径不新增传感读取、观测维度或控制项。`--grasp-contact-diagnostics` 只用于 inference-only evaluation；`--grasp-contact-probe` 限定2至16个平衡环境、最多180个控制步、冻结模型且无 training resume，不创建 PPO/Adam，也不能签发每抓法100次的原生评估证书。

这个短 probe 仅验证原始历史初态的诊断采集。优化候选到原生场景的 source/physics/reset 准入器尚未实现；所有几何结果的 `native_experiment_eligible` 固定为 False。不得把历史观察冒烟说成优化候选的闭环验证。

## 静态几何实现与真实源资源

`tabletop_centerline_grasp.py` 及 `generate_dh116_centerline_candidates.py` 沿显式中心线的弧长枚举左右抓位和局部方向，在原6主动角、affine mimic FK和有界腕姿态内优化。每手读取17个原 USD 碰撞 hull，并为 thumb/index/middle/ring/little 各自构建中央指腹 atlas。优化后检查全部可用 hull、整条加粗中心线、self/inter-hand、桌面及原支架障碍；没有分离证据记 `UNCERTIFIED`，不伪称测到碰撞或获得通过。

静态 LP 使用 ON ROD 力方向、显式质量、摩擦、12电机 effort、重力及补偿来源。非零冻结补偿不能跨优化构型直接认证，返回 `compensation_model_required`。`geometry_feasible`、`static_feasible` 与 `native_contact_verified` 分开；accepted 绝不回退为失败候选。几何与静态通过只能生成 `static_proposal_ready_for_native_verification`，不赋予原生准入。

真实冻结输入来自129/1439共同 readback 每种抓法的首个 trial。roots 是已去除 clone origin 的实际 `base_link`，q12 是实际主动编码器读回；没有将参考目标作为测量。原场景保留完整桌面和双支架，均列为8顶点障碍。物体仍是原0.5m、半径8mm、质量0.12kg刚性圆柱代理。原 max-combine 高摩擦系数为5；原模拟手部 `disable_gravity=True`，因此本场景手部重力补偿显式为0。这些是既有模拟设置，不是新修改，也不能代表真实线缆或硬件接触标定。COM为原均匀圆柱模型推导，不冒称独立实测。

刚性圆柱 seed 必须具有单位四元数，并与中心线中点、Y轴端点、COM及显式长度一致。弯曲中心线必须声明独立 proxy，不能直接送入当前原生刚体场景。冻结JSON和本地SHA仅验证来源/内部一致性，不签发原生物理认证。

冻结目录：`/mnt/mydisk/xk/dh116/runs/grasp_contact_geometry_20261106T1030Z`。`source_scene_export_manifest.json` SHA256=`56c2037e38923fe13d076a113d89674f1545ac9662b7399c63f665d804d79f1e`。原始输入、拒绝报告和后续独立输出保留在服务器，不上传运行日志或模型。

首轮实际 CAD 搜索分别检查二指和三指各18个候选（3个弧长位置、3种局部方向、每项最多60次几何优化），两组 accepted 均为0。主要拒因为距离/目标弧长、错误法向、中央指腹及完整线缆/自碰分离证据不足；没有放宽门槛或删除支架。它只是有限搜索的否定结果，不证明任何抓取位置都不可行，也不能当作原生失败/成功次数。

复审修复后的最终版02搜索扩大为每组24个候选（3弧长位置、4种方向、最多200次优化）。二指94.823秒、三指99.935秒，两组 `geometry_feasible_count=0`、`static_feasible_count=0`、`accepted_count=0`；全部24项均完成筛选。保留01及02原件，未将失败候选送入原生闭环。未覆盖原seed或改变物理门槛。

补充03仅扩大原限位内的姿态搜索：原USD为intrinsic XYZ串联、各轴±90°，prismatic实际float32范围±0.10000000149011612m。搜索rotvec范数上限1rad、world平移每轴仍≤0.02m；逐候选按 `R_seed.T @ R_candidate` 还原局部XYZ复核原authored限位，不能把优化边界当作修改执行器。独立CPU复算48候选、96手腕均通过，最大Euler二指0.962143rad、三指0.954474rad；每组270份受保护输入/源码/控制脚本SHA前后完全相同。

03两组各24候选仍全部拒绝（搜索86.667/93.085秒；含预检的控制器88.53/94.92秒）。下表是「涉及该原因的候选数」，原因可重叠：

|03拒因|二指 / 24|三指 / 24|
|---|---:|---:|
|指腹距离|22|24|
|目标弧长偏差|20|24|
|错误朝向|23|23|
|中央ellipse范围|21|24|
|整根线缆未取得分离证书|11|16|
|自碰未取得分离证书|14|22|
|缺少对向指腹|20|9|

两组所有03候选均因所需几何接触无效而未进入LP；因此不能将它们写成“LP证明无承托解”。桌面、原支架障碍与双手分离没有增加03拒因。该有限搜索没有证明所有可能姿态不可行。03输出SHA256二指=`bf50dd8b2f2be16a92e65864a091017684f4995648d943b9ea1f8f47af46861b`，三指=`a4b49812b20b708055713751ad1e7d346b6481c6f12281c14dd8928e3ea605b4`；独立控制脚本SHA256=`3e2ce96cb6b208afa0428af1a87e0d75accfda63b445ef9b1bfca7bbcc01319b`。未启动新的native实例或PPO。

## 实际129/1439相同初态配对评估

两者均是实际历史 v2 视觉模型，不是269/482历史 peak，也不是新版 Tanh Gaussian 的训练结果。每个模型二指和三指各100个冷初态，200个 first-terminal 完整记录，0 invalid。两模型的真实 reset readback 文件逐字节相同，相机和物理合同一致。

| 历史轮次 / 抓法 | N | Lift P50 (mm) | P90 | P95 | 最大净lift | 30mm率 | 最大合格Hold (s) | 原生成功 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|129 二指|100|0.000030|0.000916|0.003668|0.005627|0%|0|0/100|
|129 三指|100|0.009632|0.391500|0.501771|1.033019|0%|0|0/100|
|1439 二指|100|0.028282|0.086090|0.132241|0.346551|0%|0|0/100|
|1439 三指|100|0.000109|0.000217|0.000688|0.004369|0%|0|0/100|

|历史轮次 / 抓法|曾形成有效双手对向接触|有效对向物理时间占比|观测到丢失的trial|丢失事件数|模拟力矩饱和率|
|---|---:|---:|---:|---:|---:|
|129 二指|0/100|0%|0/100|0|6.913689%|
|129 三指|62/100|1.072823%|62/100|1248|15.902894%|
|1439 二指|23/100|0.288329%|23/100|568|3.422111%|
|1439 三指|0/100|0%|0/100|0|0.015650%|

历史接触定义是原先 all-five-pad 的 bilateral eligibility；丢失只计同trial连续有效物理样本的 `true→false`，不是新模块的逐指腹接触事件。零丢失但从未形成有效接触不能解释为稳定。分母按各组实际存活物理秒/有效joint-substep记录，不能只比较事件总数而忽略策略导致的episode长度差异。新style-masked逐pad指标与历史定义保持区分。

全部滑移为 NA、coverage=0、原因 no_anchor；所有Hold P50/P90/P95/最大值、5秒合格保持率和原生成功率均为0。不能将缺失滑移解释为小于0.5mm。原始固定初态扰动仅为局部小范围，不能代表长度、直径、质量、摩擦或弯曲形状的广泛泛化。

129原完整854757条重放已审计。本次只为新增接触丢失指标追加一次22.817秒窄扫描。1439唯一完整重放1082263条、70.608秒，精确复现原报告；没有把合成fixture或重复reset计入100次。1439 certificate SHA256=`4172eaeefc068a23c0b3565f9f3955c26e5e7729b8b35892fb6e82a004f856d3`，report SHA256=`15cd635c10945214f2d56237337626aaa3a58d27ac5514fb31855f883e409eb3`。

## 当前验证命令和限制

新隔离stage为 `/mnt/mydisk/xk/projects/dh116-contact-geometry-20261106`，使用已有 `/mnt/mydisk/xk/envs/dh116/bin/python`，无环境重装。CPU运行隐藏CUDA、禁自动pytest插件、OMP2。实际执行：

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:. PYTHONDONTWRITEBYTECODE=1 \
 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_NUM_THREADS=2 \
 /mnt/mydisk/xk/envs/dh116/bin/python -m pytest -p no:cacheprovider -q --tb=short \
 tests/test_grasp_contact_diagnostics.py tests/test_grasp_contact_native.py \
 tests/test_grasp_contact_probe_options.py tests/test_vision_evaluation.py \
 tests/test_vision_evaluation_options.py
```

结果：87项通过，8.16秒。包括原有评估路径；TDD逐次观察新功能与复审问题的RED，再得到GREEN。几何新30项及clearance/certificate/refinement、linear coupling、gravity feasibility、surface kinematics相关回归共90项通过，9.67秒；需将现有 `omni.usd.libs-1.0.1+d02c707b.lx64.r.cp310` 加入 PYTHONPATH、其bin加入LD_LIBRARY_PATH。实际USD在纯CPU加载，每手17hull、各5pad；没有把CPU加载或fixture计为原生试验。独立复审通过诊断/门禁修复及几何两项P1修复。

同基线完整CPU命令保留 `tests --continue-on-collection-errors -q --tb=short` 范围、原USD/DeformX路径、`PYTEST_DEBUG_TEMPROOT=/tmp`、CUDA隐藏、OMP2及禁自动插件，JUnit输出 `/tmp/dh116-contact-geometry-final.xml`。实际结果：5243 passed、96 failed、49 skipped、28 errors、91 subtests passed，250.59秒，exit1。旧5314个JUnit节点无缺失或状态变化，新增102节点全部通过；没有隐藏旧失败/错误。节点对照与JUnit哈希保存于独立artifact。不能称全仓库全绿。

## 原生历史观察冒烟

已在唯一Isaac实例中完成8环境、120个控制步，即每环境960个480Hz物理子步。模型是实际1439轮显式派生的v3 Tanh Gaussian checkpoint，再做inference-only批量投影；不是新的PPO训练，也不等同于上表实际v2历史模型。Actor输入维度3844、动作24，原源checkpoint SHA256=`002812709ac5e6b9a83ef64c66f4c60f205945e90edcb9afdc243ee5ddae4477` 在前后保持。

输出 `historical_observation_v3_8env_120steps_01`，只在原始冻结历史初态采集接触。每组4环境、3840条live物理样本、累计8环境秒，接触有效coverage均100%。二指样本中双手style-masked有效对向为296/3840（7.708333%），三指0/3840；二指左右手单独对向为452/2247条，三指左右均0。两组值只描述这2秒短窗和所选4个初态，不能与完整100-case历史模型结果直接比较或用于选模。

独立CPU审核7680条完整record、39708个原始接触点和76800个pad归约，逐点CAD门控、正压力、载荷、slot/body/count/start/capacity与实际body pose均一致。流式重放16441个事件及summary完全一致：建立1613、持续13215、丢失1601、结束截断12。丢失均有连续可信样本依据，截断未冒充丢失。所有动作24维有界，abs最大0.7229746；初始8×3844输入的CPU确定性动作与原生首动作最大差1.49e-7。680份冻结运行时文件和源模型SHA均匹配。独立审计JSON SHA256=`8961f08c3dfe6fee52c425c390aea28ef9f080bb79a38b135c162c7fd896e12d`。

下表为每组4环境合计的「接触环境秒 / 逐pad丢失事件数」，不是单一环境的连续保持时间；左右数值也不代表同一时刻的双手轨迹。二指不要求中指，但原始10pad仍全部保留。

|抓法|左拇指|左食指|左中指|右拇指|右食指|右中指|
|---|---:|---:|---:|---:|---:|---:|
|二指|6.025 / 46|1.648 / 85|不要求|5.058 / 361|6.202 / 227|不要求|
|三指|5.323 / 628|0.0417 / 5|0 / 0|0 / 0|4.835 / 182|1.758 / 67|

三指组左中指、右拇指未建立有效指腹接触，左食指有效接触很少，是该短窗中具体可定位的几何/接触缺口。这里的零丢失是未建立接触，不能解释为稳定。二指2548/3840条出现至少一个有效模拟电机估计达到诊断98% effort门槛；三指0条。该「任一电机的环境子步比例」与历史表「joint-substep饱和比例」分母不同，不能直接比较。

该短窗没有任何episode完整结束（completed0/expected8），原评估报告正确为 `not_completed`。没有PPO/Adam更新、训练模型/iterations/training_start输出或生产评估证书；它不计入每抓法100次，也不证明稳定5秒Hold或30mm Lift。原支架场景保持，报告明确 `support_bearing_not_attested`；不能据此宣称无支架保持或抬升。

接触报告SHA256=`3cb9d4d52de4fd410ab7d20d007b0d0927e204989e4adb5d0a379f1e98b70a12`，原评估报告SHA256=`c7bbdc2408cc93934e9aedb55be103706165a06e3ff20548f0971b10a8e1b594`。独立运行时680文件清单SHA256=`8c141800023490f1785b03511d656c34ebbbb9bc034a9db069742e900d7c78fe`；仅诊断运行时模块被执行，几何搜索与其输出独立。

## 仍未完成

尚无通过静态全部门槛的真实候选或优化候选闭环抓取结果，native candidate source/scene/reset准入器仍缺。不能声称抓取或Lift退化已解决。Grasp/Hold/Lift物理判据课程、R0-R5单因素消融、Adam保留/重置对照、独立验证选模和P2原生泛化本阶段均未执行。R4/R5继续等待独立因素有效证据。后续应先用逐pad真实接触证据定位几何缺口，再验证新候选，保持单Isaac实例和独立目录。
