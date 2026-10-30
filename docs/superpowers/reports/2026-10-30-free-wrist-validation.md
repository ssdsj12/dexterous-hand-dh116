# 自由姿态控制与圆杆提升尝试：实现及验证记录

日期：2026-10-30。当前阶段：控制与反馈已实现，CPU及实际Isaac验证完成；**尚未成功抓起圆杆**。

## 自由度口径与实现

两个DH116本体合计12个主动关节、10个从动关节，共22个手指自由度。策略只对12个主动关节发指令，从动关节沿既有联动模型运动。新增模式的26维命令另含双手外部手腕的12个运动指令（XYZ平移与XYZ旋转），以及2个法向目标力参数；目标力不是机械自由度，外部虚拟手腕不是DH116本体关节。

`--free-wrist-grasp` 是显式选择的新模式，261维观测、26维命令。手指按当前命令积分，不再受复位抓姿附近的狭窄包络限制；位置、速度和力矩仍受原实际配置限幅。手腕通过三个平移和三个转动关节驱动，正常运行不写根位姿。复位抓姿仅为初始种子，手指、抓点及手掌朝向可变，无close/lift准入或抓姿锁存。

新增四指几何/实测承力特征，以及接近、部分/四指接触、实际离桌高度、法向力跟踪、逐指平衡和动作变化反馈。所有新奖励率只乘一次dt；新模式关闭旧lift和旧grasp_shaping项，保留现有有符号向上力奖励、向下力惩罚。FK参考点接近物体不等于真实接触。2–12N是每手法向目标范围，不保证实际瞬时负载不超限；过载释放及失败隔离仍生效。

普通有限联动偏差保持诊断和有界纠偏，不作为手腕运动门槛。新增零手腕四元数检查，坏状态隔离复位；坏复位在下一物理步之前停止。旧20维及手动入口保留。

外部转动驱动候选为40Nm/rad、4Nms/rad、2Nm、1rad/s，未完成硬件标定。原手指配置0.2Nm、0.3rad/s、圆杆质量/重力及材料参数未改；现有extreme_grip摩擦5仍属仿真假设，不能据此宣称硬件抓取能力。

## 参考实现

已核对 [DexterousHands lift任务](https://github.com/PKU-MARL/DexterousHands/blob/dc41afd134cebfe4d80a5b861f0b8c15d734320a/bidexhands/tasks/shadow_hand_lift_underarm.py)。借鉴独立手指/基座平移旋转控制、手物距离反馈和实际物体目标距离；未复制ShadowHand关节数、8cm阈值或驱动参数。该任务配置actionsMovingAverage=1，不能将其误报为实际启用了平滑。

## CPU与原生驱动验证

部署前保存原件并核对统一LF的SHA-256。分工实现后完成规格审查与质量审查，修复了零手腕四元数恢复、首触点平衡代价、旧合同字段覆盖及最终configuration持久化问题。

最终定向回归33文件：**668 passed in 12.64s**。不是全仓测试。精确文件列表与日志保存在 `runs/deployment/free_wrist_20261030/regression_files.json`、`regression.log`；可复现命令：

```bash
cd /root/hand
DH116_NUM_THREADS=6 bash scripts/python.sh -c 'import json,sys,pytest; sys.exit(pytest.main(["-q", *json.load(open("runs/deployment/free_wrist_20261030/regression_files.json"))]))'
```

先红后绿覆盖了26维数据流、原实际关节范围、六轴驱动与读回、选中环境复位、几何和奖励、无效状态隔离。另把旧噪声测试改用当前RSL-RL3.1.3的公开`policy.act`接口，生产噪声实现未改。

原生Isaac空手验证2环境、100控制步/800物理步，无复位。左roll实测+0.029606rad、右pitch实测−0.029613rad（目标±0.041667rad）；只复位环境0时环境1保持隔离。SDK计数实测12主动、22手指关节、12外部手腕驱动。随后同一Isaac实例运行实际PPO入口1次更新、32条transition，观测/损失有限，成功0。

命令为 `DH116_NUM_THREADS=6 bash scripts/python.sh deploy/free_wrist_smoke.py`，原始报告在 `runs/free_wrist_smoke_20261030a/{wrist_smoke,free_wrist_smoke}.json`。空手驱动测试暂将圆杆移开，随后reset恢复；下面所有抬杆试验均使用原圆杆物理状态。

## 实际抬杆尝试

各试验串行执行，仅一个Isaac实例。10mm进展判据为实际物体底面高于桌面10mm、双手实测接触、速度<0.03m/s并持续0.3s；原30mm/0.5s正式成功判据未降低。

| 试验 | 环境/更新/样本 | 最大实际离桌高度 | 结果 |
|---|---|---:|---|
| PPO A，反馈v1 | 64 / 4 / 16384 | 0.009081mm | 四指最小承力项稀疏；未抬起 |
| PPO B，反馈v2 | 64 / 8 / 32768 | 0.019672mm | 有部分接触；未抬起 |
| PPO C2，反馈v3及可训练初始速度倾向 | 64 / 8 / 32768 | 0.133572mm | 左食指失载、右手瞬时过载；未抬起 |
| 自由位姿接触搜索A | 4 / 0 / 0 | 0.010472mm | 360控制步；未抬起 |
| 自由位姿接触搜索B | 4 / 0 / 0 | 0.000425mm | 360控制步；未抬起 |

所有10mm保持事件和正式成功数均为0。C2最高点左拇指/食指约2.996/0N，右约11.231/12.407N，右手合计23.638N；该瞬时峰值不满足目标力或稳定抓取要求，不能将0.13mm微小运动称为成功。

C2仅给可训练actor偏置设置初始闭合与向上速度倾向，全部网络/噪声参数继续训练，无固定关节姿态、阶段门槛或动作覆盖。一次C驱动脚本因全局函数名碰撞在优化前失败，捕获实际生产函数后修正并先做CPU检查；只有C2计入上述更新数。

接触搜索使用实际SDK雅可比、承力触点（无载时才使用FK参考点），每手9个搜索变量为3平移+3转动+3个拇指/食指主动量，不是新增9个手指关节。B将目标接触径向高度分为0/−0.5/−1/−2mm，纳入既有桌面下降边界；中央有限差分雅可比校验最大误差0.00014847m/rad。搜索不做PPO优化，也不证明PPO已学会抓取。

实际运行的诊断入口：

```bash
DH116_NUM_THREADS=6 bash scripts/python.sh deploy/free_wrist_trial.py
DH116_NUM_THREADS=6 bash scripts/python.sh deploy/free_wrist_trial_b.py
DH116_NUM_THREADS=6 bash scripts/python.sh deploy/free_wrist_trial_c.py
DH116_NUM_THREADS=6 bash scripts/python.sh deploy/free_wrist_contact_search.py
DH116_NUM_THREADS=6 bash scripts/python.sh deploy/free_wrist_contact_search_b.py
```

诊断脚本与原始结果保留在服务器deploy/runs及本机部署目录，运行产物不进入Git。A/B的中间反馈版本有单独源码快照，当前源码是v3，不能把重跑当前文件称为精确复现A/B。

现有证据确认驱动可以运动、雅可比映射正确；观察到失载、侧向推杆及过载。尚不能据此断言唯一根因为扭矩、摩擦或控制积分，未继续盲目增加力矩。

## 新参考：DexGrasp-Anything（仅调研，未集成）

用户追加要求后核对论文、项目主页及代码，固定到 `4DVLab/DexGrasp-Anything@86353d710ca71d0f547f0d8b15cc8ca959c78163`。

- [几何损失](https://github.com/4DVLab/DexGrasp-Anything/blob/86353d710ca71d0f547f0d8b15cc8ca959c78163/utils/handmodel.py#L513)：SPF让附近指腹内侧点靠近表面；ERF惩罚手物穿透；SRF惩罚自穿透。这些force名称是几何优化损失，不是N单位的接触力伺服。代码的2cm阈值不适合直接移到直径16mm圆杆。
- [采样修正](https://github.com/4DVLab/DexGrasp-Anything/blob/86353d710ca71d0f547f0d8b15cc8ca959c78163/models/optimizer/grasp_with_object.py)：生成多样抓姿并沿几何损失梯度修正；可参考为DH116原生主动空间的多种候选和教师样本，不直接套用ShadowHand角度或权重。
- [评估代码](https://github.com/4DVLab/DexGrasp-Anything/blob/86353d710ca71d0f547f0d8b15cc8ca959c78163/envs/tasks/grasp_test_force_shadowhand.py#L289)将物体重力关闭，再检查六方向外力下位移；同文件L583明确disable_gravity=True。这是抓姿稳定性测试，不是桌面重力lift验收。[项目主页](https://dexgraspanything.github.io/)也明确区分手动全局升降与IK运动规划。

建议下一步先将单个FK参考点扩为符合DH116碰撞/指腹几何的多个表面点，使用带符号距离区分靠近与穿透；在12主动+10从动联动约束下搜索多种抓姿。筛选必须保留本项目重力、桌面、实测承力和真实提升条件；筛选是离线验证，不重设在线close/lift运动门槛。对当前单一圆杆，暂不优先迁移完整扩散模型、LLM和百万级数据环境。

## 保存与未完成项

备份位于 `runs/deployment/free_wrist_20261030/source_before.tar`、`source_env_before.tar`，另有trial_a_source.tar和trial_b_source.tar。最终交付回执单独记录本地提交、服务器Git HEAD和统一LF的逐文件哈希，避免将工作树部署当作服务器Git提交。

新模式控制和反馈验证完成；稳定双手抓握、合格10mm提升及PPO学会抓取仍未完成。没有启动长训练，也没有把DexGrasp-Anything安装进hand环境。TensorBoard本机16006可访问。
