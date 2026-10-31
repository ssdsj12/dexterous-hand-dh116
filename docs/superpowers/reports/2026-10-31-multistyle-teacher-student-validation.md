# 二指、三指原生teacher与student训练

## 已通过的二指示范

用户要求补齐每手二指，以形成多抓法teacher。AM从AA约6.979 mm的同一二指初态继续验证，仍只有每手拇指、食指承力，保留原0.2 Nm/.3 rad/s、物理从动联动、质量、材料和端支撑。

AL复测表明，等到实际离地3 mm再保留手指目标太晚：上升开始后食指失去有效指腹接触，圆杆没有真正拾起。AL在11.483 s仿真、261.24 s墙钟时有序结束并保存，reset为0，没有导出成功teacher。

AM新增显式、仅教师使用的抬升前指令保持：原生控制周期均值确认每手有效载荷至少2 N、两个选中指腹各至少0.5 N、持续0.2 s且无reset后，保留已有手指和手掌朝向指令，再让双腕整体向上。这里选择有负载的示范起点，不是PPO的lift准入门，也不要求精确3 N或等力；最终成功仍按实际物体高度、防滑、有效指腹和连续5 s检查。

AM第二参数例通过独立原始数据复核：

| 指标 | 实际结果 |
|---|---:|
| 最大真实净升（相对初始支撑杆底面） | 31.5364 mm |
| 连续保持窗口 | 14.5833–19.6000 s，持续5.0167 s |
| 窗口最小真实净升 | 30.0112 mm |
| 最大三点相对双腕漂移 | 0.003673 mm |
| 最大原生有效指腹深度 | 0.06736 mm |
| 左/右手平均总法向载荷 | 1.1961 / 1.5553 N |
| 其它三指最大控制周期有效载荷 | 0 N |
| 普通碰撞最大载荷 | 0.000272 N |
| 保持窗口reset/adapter异常 | 0 / 0 |
| 4.0333 s保留后全部12个手指命令的最大跨度 | 0 rad |
| 双手真实法向+摩擦向上合力均值 / 杆重 | 1.177128 / 1.177200 N |
| 竖直冲量残差 | 3.22e-7 N·s |

稳定保持时左拇指/食指约0.5704/0.6257 N，右拇指/食指约0.7886/0.7667 N。当前满足实际lift，不能称为每手3 N精确力控达标。第一参数例也抬升31.56 mm，停止时已保持3.70 s，未将未满5 s的案例导出为合格示范。

成功原件位于`/root/hand/runs/native_pinch2_prelift_20261031_am/`，包含`independent_pinch2_success.json`、`native_pose_hold_audit.json`、`successful_pinch2_teacher_env1.pt`及完整物理/控制轨迹。新示范明确标记`grasp_style=pinch_thumb_index`。独立审计重新算出控制周期均值、对向面和全部原始子步的高度/深度/状态、固定锚点防滑窗口；其它手指不能混入二指资格。

## 已有三指与student验证

三指AK原生成功示范保留。通过普通TensorEnv重新采集时，两个fresh环境都实际成功，获得2542组真实351维观测/24维动作；这两个回合属于同一三指抓法，不算两种新抓姿。静态初态仍在原支架上，没有空中reset或物体绑定。

三指student行为克隆运行2000步，训练集MSE由0.0326415降至0.00221480，仅说明离线拟合。随后64环境、32步rollout、2次实际PPO更新完成4096条transition，模型、optimizer、TensorBoard保存正常，示范辅助MSE约0.003669、权重0.0995，GPU占用约6.16 GiB。成功数0，未宣称student已经会lift，未启动三指专用长训练。

PPO rollout只来自student新采样，示范通过独立的衰减辅助损失使用；没有把教师动作伪装成当前策略的PPO likelihood数据。teacher原件、BC actor和PPO checkpoint分别保存。

## 多抓法数据与验证范围

二指AN已在同一普通TensorEnv完成2个fresh实际成功回合，2349组351/24样本；保持窗口只有每手拇指、食指承力。采集器按真实控制周期均值判别二指/三指，并检查第三指不能混入二指资格。launcher使用二指独立预抓姿，未把二指动作放进三指初态。

二指、三指初始化不同，合并时显式保留完整初始化族、来源文件SHA和资格证明；只允许两个已声明的初始化SHA字段变化，其余物理/观察/控制合同严格相同。两个模式的12主动/10从动手指结构、351/24接口及原限制不变，student仍可探索其它手指组合。

AO混合库共4891组样本、4个实际成功回合；按抓法样本数反比加权，使二指和三指总采样概率各50%。混合BC运行10000步，训练集MSE从0.0283594降至0.000648296，不作为闭环成功证据。

AP真实跨抓法PPO验证：64环境、每种初态2次更新，总4次、8192条transition，自动退出0。二指→三指切换只记录`surface_reset_family_switch`，完整policy/optimizer保留；rollout重新从当前student采样，示范仅作独立衰减辅助步。没有物体空中reset、绑定或外力。

AQ已从AP第4次更新续训，64环境、每种初态各追加64次更新，最终预算第132次。每个原生训练块使用一种已验证初态，每次辅助更新都均衡采样两类示范。独立tmux会话`dh116_multistyle_20261031`中运行，输出`/root/hand/runs/surface_multistyle_ppo_20261031_aq/`；启动后已实际更新、损失有限，GPU约6.18 GiB。当前早期成功数0，不声称student已独立学会lift。

局部指腹MPC是独立的inference执行层，4步时域、12个手指目标滚动优化，student保留双腕意图；真实腕命令限幅、失触锚点清理、NaN保护和分环境fallback已修复并CPU验证。模型不预测完整物体/摩擦动力学，尚未证明实际提升成功率改善，不将模型代价下降或teacher成功等同于student成功。

最终主线程集成命令：

```bash
DH116_NUM_THREADS=2 bash scripts/python.sh -m pytest -q \
 tests/test_surface_reset_resume.py tests/test_surface_imitation.py \
 tests/test_surface_demo_library.py tests/test_surface_demonstrations.py \
 tests/test_surface_mpc_runtime.py tests/test_surface_mpc.py \
 tests/test_dh116_tensor_training.py tests/test_dh116_terminal_monitor.py \
 tests/test_surface_policy_entry.py tests/test_pose_carry.py \
 tests/test_native_teacher_reference.py
```

357项通过，7.76 s。审查确认实际库两种style的权重各为1（归一化后各50%）、模板SHA与族成员一致，实际跨style验证也已通过。没有提交或推送GitHub。
