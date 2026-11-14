# 杆轴Reset最大桌面余量：左手161点闭合几何通过，右手仍阻断

## 实际修改

基线ee8c21fc2b627280712d23f194f0d84d15e66ae4。
- 新增 scripts/audit_dh116_pinch_rod_axis_margin.py：extreme_indices、select_margin、sampled_depth_gate、独立90秒入口。
- 现有 audit_dh116_pinch_rod_axis_clearance.py 的 audit_side/run 增加显式可选角度选择器与全目标点SDF检查；默认仍是原最小姿态改动模式，没有改变旧运行默认值。
- 新增12项测试；覆盖实际顶点约束索引、退化点集、已知解析最优值、原约束全量复核、故障注入回退和未知/超深度拒绝。
- 没有修改控制器、24D、371D、PPO、USD、Mimic、执行器、材料或成功标准。

把每个形状/路径的约束写成a cosθ+b sinθ−c后，对相同c选取2D凸包实际顶点索引。左2334178行压至5374条、右2317595行压至3215条用于二分；每个候选角度始终复查**全量**原始约束。子集不可行仍是真正原约束的必要条件；子集可行本身不授予通过。发生全量不符时回退原始约束求交，而不是删除困难顶点。

## 结果与证据

独立输出：
/mnt/mydisk/xk/projects/dh116-autonomous-3c80166/evidence-rod-axis-margin-20261113-Pz4lNR/
包含RED、冻结SHA、完整progress.jsonl、result.json、outer.exit和pytest.log。
result SHA256：d1fc5b21cd2ed0d3e9d4c17e5cc56eff1259a13dc6558942c052c668f9d1c03b。

644/644几何状态；25.588秒/90秒；每侧24次余量二分；全量回退0次；0新物理、0PPO。来源沿用上包原左候选/独立右候选与冻结的参数任务实际杆节点，SHA均重新核验；不是重新原生采集。

| 指标 | 左手 | 右手 |
|---|---:|---:|
| 新固定杆轴角rad | -0.018274929 | -0.093715949 |
| 连续桌面净空最小mm | 2.111213 | 3.075814 |
| 超2.1mm门槛的余量µm | 11.213 | 975.814 |
| 非目标杆间距下界mm | +0.025735 | -0.166184 |
| 全初态沉降包络下界mm | +0.535408 | +0.955233 |
| 161点中拇指最小SDF gap mm | -0.208257 | +0.580459 |
| 161点中食指最小SDF gap mm | -0.250922 | +0.025566 |
| 原0.3mm指腹深度界（161目标点） | 通过 | 通过 |
| 全部本次静态门槛 | 通过 | 阻断 |
| 原生/Teacher资格 | 否 | 否 |

“深度不过大”并不等于接触建立。右手拇指仍有约0.580mm间隙、非目标source finger13下界失败、预测法向仍不对向，故不能仅凭更大桌面余量晋级。负的保守间距下界本身也不是实测穿透。

左侧四个重点指腹的连续桌面净空，source与cooked取更低：
finger14 2.986263mm、finger23 2.824066mm、finger43 55.877937mm、finger53 73.364362mm。
全手限制项仍为source finger13。旧碰撞/失败账本没有覆盖。

左Reset base平移相对原候选[-2.059962,约0,+1.213479]mm；q与闭合角增量不变。完整新WXYZ姿态：
[0.49536949549910153,0.24306903145211275,0.4183242614490973,-0.5295182142666321,-0.3849048284415139,0.7556583559095579,0.020956692161450814]。
右姿态变化独立记录在原始结果中，不复制左侧，不用于默认运行。

二分左额外余量数值区间[11.212049548,11.212484907]µm，最终全量检查11.212351614µm（另含原1nm额外安全余量）。这是本冻结浮点静态模型的数值区间，**不是物理模型误差界**。非目标源碰撞、实际跟踪、杆变形、native shape附着身份、自碰撞及安装可达性仍需原生证据；当前native_ready始终false。

## 测试、运行、未解决问题

RED：模块不存在，测试先失败，保留red.log。
```bash
DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 bash scripts/python.sh -m pytest -q tests/test_dh116_pinch_rod_axis_margin.py tests/test_dh116_pinch_rod_axis_clearance.py tests/test_dh116_pinch_closure_margin.py tests/test_dh116_pinch_dynamic_fk.py tests/test_dh116_measured_closure_geometry.py
```
**98 passed / 0 failed，37.55秒**。
```bash
DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 bash scripts/python.sh -m scripts.audit_dh116_pinch_rod_axis_margin --output <新的独立目录>
```
本配置CPU批次已用完，不原样重跑。所有新原生配方仍需实际冻结SHA、资源锁及预算账本；用户同范围持续授权不允许复用已耗尽配方。

R0只做源码检查，未改奖励：原stable同时影响instability终止与hold_progress，不以“没有异常”填true，也不把原刚性stable（包含Lift/contact）当数值稳定。当前仍blocked这两个来源；训练reward仍None，Teacher0/3。公共371D来源仍66/371。WireSeg本轮新增0训练步，既有128seg/128depth不变。Tripod/Power维持各自碰撞/闭合阻断；相机与下载不阻断无相机准备。

下一包优先：以本左候选冻结新的实际Reset/短接触/退出诊断，升级最初1.9mm固定手桌面检查至真实运行时2.1mm并逐物理步保留原保护。不能只更改配置SHA或把旧v4成功复制为新成功。右手另行构造，保留所有失败源。
528保护文件与两历史Checkpoint由发布保护器核对，PR2保持Draft。
