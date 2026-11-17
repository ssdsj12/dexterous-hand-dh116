# DH116 双目 RGB Teacher–Student：集成实施与验证

基线为 `11-10@6cee7418a5e49f56b7f0090fc5eaf20b143c0b1b`，独立开发分支为 `codex/dh116-stereo-privileged-dagger`。完整读取用户930行规范后，按后续授权从Phase0继续实现Phase1～7；分辨率补充要求覆盖原规范中的224×96建议。原分支、旧Teacher/Student/MPC/PPO和Checkpoint不覆盖。

这里区分**实现、CPU软件验证、原生测量和物理任务资格**。实现BC/DAgger/PPO代码不等于已训练可靠抓取策略；640×480默认精度配置也不等于已选择最终训练分辨率。

## 各阶段实际交付

|阶段|已实现内容|尚需原生证据或接口|
|---|---|---|
|0|371行字段来源/单位/坐标/归一化/可观测性、24行动作、奖励公式/系数/dt/重算风险及源码SHA审计|真实硬件反馈能力不能由仿真来源推定|
|1|独立371D privileged Teacher、Tanh Gaussian PPO、R0奖励闭合、断点保存恢复和expert dataset准入|新Teacher双抓法完整成功、力学来源资格；R1/R2测量工厂与独立geometry资格|
|2|同步左右RGB、calibration读回、显式rectification、浮点视差/LR检查、深度/坐标变换、128点可见线缆提取；已执行10次独立原生相机配置采集，核验K/T、时钟、wire-ID及开销|所有实测深度精度仍未取得资格；尚无最终分辨率、Actor延迟或完整训练吞吐结论|
|3|共享GN-ResNet18、双目cross-attention、四帧时序Transformer、多状态head及逐维mask/SO(3)损失|持出状态估计精度、可见几何对应与force监督来源|
|4|pre-action同步Teacher采集、独立BC、初态/source-group分组holdout、最佳权重保存|本次采集自己的完整成功回合，不能借历史Teacher报告授信|
|5|Student实际访问状态上实时查询Teacher、整24D owner切换、append-only聚合和独立Adam|正式Teacher及Stereo资格、原生DAgger闭环和独立提升|
|6|独立asymmetric Student PPO、完整优化断点、Actor-only导出和raw双RGB独立推理|原生短更新/恢复、最终Student任务能力与真实机器人驱动验收|
|7|100/抓法冷初态runner、逐480Hz回放、Wilson统计、heldout模型选择及公共晋级门槛|实际100独立初态/抓法、完整训练来源登记与全部物理条件|

补充落地：`tabletop_privileged_shaping.py` 实现R1四类几何势能改善，以及R2双手承载分配/对向指腹均衡/接触恢复/法向与切向超载/不必要执行力目标变化，原support deficit只计一次。候选配置默认关闭。承载目标是实际抓取点/质心/重力在适用静力模型下导出的显式份额，80/20合法分配不受罚；缺目标保持NA，不能隐含50/50。26项新增因子测试及148项相关回归通过；它们只证明纯算术，真实measurement factory、逐项source/tick接线及独立geometry资格尚未完成。现有geometry资格门槛仍过强地要求完整成功episode/force，不能把这个接口缺口简写成只缺数据。

原native监督全False的集成缺口已修复：只对与同刻实际SDK q/qd、原Task低高界和归一化逐维核对通过的`0:24`开放state监督。source masks为`[B,371]`，不向其它proprio或force/geometry广播；范围明确是`simulation_SDK_encoder_readback_only`，硬件资格False。actual `TensorTask.observe`内核CPU正例验证24/371 coverage及非零state梯度，异常/错位逐维failclosed。其余状态不补零或误标可观测。

world→左右掌的转换也接入真实数据路径：native runtime与独立raw RGB runner均用同一帧RGB/stereo worldpoints及当前测量FK，保留左右掌局部点、无效NaN/mask和身份。它们是派生传感几何，不能当GT标签，也不改变Actor已声明的world-frame输入。相关runtime/geometry组合119项及raw runner10项CPU通过；未冒充真实手部坐标精度验收。

三份Phase0审计文档：

- [371D逐维合同](../../stereo_privileged_state_contract.md)
- [24D动作合同](../../stereo_action_contract.md)
- [奖励审计](../../privileged_teacher_reward_audit.md)

各阶段报告保留在本目录`2026-11-10-stereo-phase0-audit.md`、`phase1-integration.md`、`phase2.md`～`phase7.md`，另有cold-trials、bootstrap和resolution-v2报告。不要将不同组合的测试数量相加为独立用例数。

## 关键合同与保护

- 新Actor只接收版本化`StereoHistoryBatch`：左右RGB、由该RGB计算的深度/点云、实际本体反馈、previous issued action、时间和有效性mask。Teacher/Critic/GT/任务phase/success字段不能进入Actor。
- 保留旧3844D RGB-D路径；新Stereo输入有独立版本，371D Critic和canonical24D动作保持。动作语义SHA为`498cebedfe520cc2f6593b32776581c3da826ba95e4db441b7938d29176c55d9`。
- Checkpoint绑定calibration、实际matcher/point处理参数及实现源码SHA。history-only fixture包不能当raw RGB部署包；预处理改变不能静默warm-start。
- 每次实验独占新目录。源模型只读；BC、DAgger、PPO与Actor-only包分开，PPO不混入伪造Teacher transition。
- 实际terminal在reset前保存；global physics/policy clock、episode-local clock、UID和trial身份分开。手掌/物体/SDK目标不混同。
- 单Isaac实例锁和原480Hz物理/60Hz控制不变。30mm净Lift、5秒正式Hold及0.5mm固定锚滑移标准不变；不将CPU/Teacher/MPC成功算Student成功。

## 分辨率实施与测量顺序

旧224×96文件原SHA为`f5c53f5c9d3042ee0ce97453a3420dda3ecb7417469d7362be20604b0171d494`，保留为低分辨率/整数视差基线。新建320×240、640×360及主精度640×480配置，并允许合理自定义。重建相机与图像resize采用不同K计算，T与80mm基线保留；左右尺寸/物理时间严格一致。

采用显式亚像素匹配、逐视差plane流式计算和独立图像处理batch；env数、rollout、minibatch及图像batch分别可调，不隐藏缩图、不降物理频率。分辨率实验对旧224显式使用同一subpixel设置并保存resolved config，避免把算法变化冒充尺寸收益。

按照用户要求，先修复完整软件路径、CPU回归和服务器兼容，再执行独立短相机测量；该顺序已实际执行，最终结果见下文和独立原生报告。深度reference与instance-ID只用于验证，永不输入Actor。缺实际Actor包时Actor吞吐为NA；全图有效覆盖或RGB颜色候选不能冒充wire真值完整率。最终尺寸必须依实测精度/覆盖/训练资源决定。

## 服务器首次CPU兼容测试：失败及定位

通过现有`/mnt/mydisk/xk/dh116/scripts/python.sh`与解释器`/mnt/mydisk/xk/envs/dh116/bin/python`，在独立`/tmp/dh116-stereo-20261110T111700Z/snapshot`验证1165个冻结文件，未修改原项目或安装依赖。实际Python3.10.12、Torch2.7.0+cu128、NumPy1.26.4、OpenCV4.11.0、SciPy1.15.3；CUDA对CPU测试显式隐藏，没有导入Isaac。

首次结果：458 passed、6 failed、1 skipped，exit1。3项子进程继承了原wrapper的旧源码PYTHONPATH；2项由于快照遗漏371D人读文档；另1项是Python3.10的AST缺Python3.12的空`type_params`字段。逐24个原AST符号实测：未适配24个hash不同，仅补空字段后0个不同；原文件字节SHA、行范围、实际非空结构检查都保留。修改私有打包/入口及审计适配后，以新冻结目录重跑，不覆盖失败材料。

首次私有归档SHA为`c2dc823ff4c57a0ddd96ae88e81bd8f9adf4f794e5f7e28d8a14216a6be2aa3f`。原始日志、归档、pytest模型及SDK源快照均保留为私有artifact，不提交Git。

第二份冻结归档为1172个文件，SHA `10a3cb564eac9ca7a3c901b30a1b48cb8af8033b51d0678d1f0dede75f429abd`，使用独立`/tmp/dh116-stereo-20261110T112000Z/snapshot`，不覆盖首轮失败快照。其source-manifest SHA为`0b3b448f14ed8e435cf2abc153c94b38ae9ddb0bcee7f808c538dfae51bafdb8`。

## 尚未满足的晋级条件

原生Bridge的force/support来源仍未具完整ON_OBJECT、partner、逐指切向与external/authored-force/attachment独立审计；这些值不补零。独立几何资格和R1/R2真实测量工厂不能由CPU因子算术代替。深度辅助decoder/qualified监督、持出状态精度、100独立初态/抓法与最终Student成功均必须单独报告。

原生训练来源清单只证明已登记artifact的内容/父链，不自动证明从未漏登记外部训练。缺完整清单或重叠初态时禁止正式heldout晋级。默认M0/旧路径不改，不启动长时间训练，PR保持Draft，不合并。

## 最终冻结回归（2026-11-10）

本轮最终源代码在Windows CPU为 **564 passed、1 skipped，75.44s，exit0**；服务器现有Python3.10.12/Torch2.7.0+cu128/NumPy1.26.4/OpenCV4.11.0环境为 **565 passed、1 skipped，67.68s，exit0**。两者不是相加的独立用例数；本机缺可选OpenCV而服务器执行了相应数值路径，服务器缺依赖负例测试skip。服务器测试最后明确输出 `native_modules_imported=[]`，没有安装或升级依赖。

本机验证设置 `PYTHONPATH=src;artifacts/stage7_lifecycle_cpu_deps_v01`、`CUDA_VISIBLE_DEVICES=`、`PYTHONDONTWRITEBYTECODE=1`、`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`，运行现有Python的 `-m pytest -q -p no:cacheprovider`，选择全部 `tests/test_dh116_stereo_*.py` 及下列用例：

```text
test_dh116_privileged_teacher.py  test_dh116_privileged_cli.py
test_dh116_privileged_shaping.py  test_stereo_privileged_state_contract_audit.py
test_stereo_action_contract_audit.py  test_privileged_teacher_reward_audit.py
test_stereo_phase0_source_freeze.py  test_surface_success.py
test_surface_reward_ablation.py  test_surface_feedback.py  test_squashed_gaussian.py
```

服务器命令使用原wrapper及新冻结入口（仅CPU）：

```bash
DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=4 \
CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 PYTHONUTF8=1 \
/mnt/mydisk/xk/dh116/scripts/python.sh \
/tmp/dh116-stereo-20261110T120000Z/frozen_entry.py pytest
```

该冻结源码清单1197文件（含实际资产副本）SHA为 `20e7a6757e22b357ae07d0b33e34cc06a9830688f2662b94172710ec62742384`；归档SHA为 `7429e6689b949650febd659768b9842264558ee18a52a2c7f573e8e3af77d6d9`。日志在服务器同目录 `final_server_cpu.log`；Windows私有日志在 `artifacts/stereo-phase0-20261110/full-integration-04.log`。这些日志、归档、原生模型/图像和SDK源码均不上传Git。

## 原生接线修复与失败材料

CPU完整路径通过后才启动相机实验。原launcher预检发现新配置误带需要旧demo的imitation参数；仅修正新增配置，保留原稳定Hold reward条件。随后补齐原vision guard要求的真实三指reset副本和旧路由只读camera配置；没有用二指reset冒充三指，也没有更改旧guard。新reset JSON来自现有mixed训练实际初始化，其SHA为 `c36e8a06c9ff227f7cfb32477dd83de3c4ab425092e1397894445d47a5f1053b`，不是新的成功示范或Checkpoint。

首次Kit启动因私有冻结包漏资产而失败；在新的独立目录复制并逐SHA核验已有14份资产后再运行，旧失败目录保留。没有下载替换物理资产。SDK float32 K读回的舍入误差使用独立数值容差；T和物理容差未放宽。原生instance-ID会重编号，不能硬编码ID0或Wire数值。v2解析实际Wire根/mesh清单、逐env ROI和未知ID对应的真实参考深度有效性；有限深度的未知ID、缺Wire清单继续NA。旧v1数据不追溯升级资格。

SDK CameraView源码SHA为 `58557bbcb900574a4b9858c13e9e479b96fd5dfe09d872b07d5c010cca64d99e`，annotator定义SHA为 `756dbfb4f262e623606bcbff2ddb315be723cf371e7a72983493d7e9977ddd56`。参考是光轴Z，不是射线距离；由独立验证render product同物理时刻取得，Actor不接收参考。

## 分辨率实测结论

实际执行 **10次完成的原生相机配置采集**：旧rig四分辨率、front v3四分辨率、roll20的640×480 block5/9。每次2环境、4计量帧、2预热帧，seed116，保持480Hz/60Hz；只有零24D动作推进物理，未训练Teacher/Student或执行MPC。40帧×2环境观测不等于100个独立初态/抓法。所有producer exit0且公共重读验证通过；**10次accuracy qualification均为false，selected_resolution=null**。

旧rig实际画面中手和线缆被裁切；front v3根据实际SDK掌位和线缆位置重新构图，旧配置保留。front中线缆近乎平行基线，纹理弱，不能靠提高分辨率解决匹配歧义。独立roll20物理相机rig旋转两相机和基线而非仅旋转像素，保持80mm基线、K、物理初态及控制合同；block5/9只改单个匹配窗口参数。

640×480 roll20原生block5：wire Z-MAE/RMSE/P95为 **7.350/16.533/34.384mm**，有效覆盖43.61%，≤5mm正确3D点完整率32.98%；block9为 **7.978/15.116/21.341mm**，有效覆盖93.87%，正确完整率46.05%。共同6784有效像素上，block5→9 MAE为6.716→7.973mm；窗口扩大改善总体覆盖，但不能宣称精度全面改善。

block9采集+深度约8.71双目对/s（2环境），排除Actor、物理step、优化与训练；PyTorch峰值allocated/reserved约133.48/176.00MiB，选中设备总占用约9.48GiB（含Kit/renderer）。验证专用参考render product也消耗资源，不能把该总占用直接外推1024环境或正式训练。完整训练吞吐和Actor延迟仍NA。

在完全相同原生RGB上额外做一次离线SGBM9，正确3D点完整率81.96%、wire MAE6.980mm，但RMSE17.762mm、P9546.881mm，global误差及有效覆盖也未达原门槛。此项是离线再处理，不是新的原生算法资格或已选分辨率；坏尾部误差与局部改善同时保留，不降低门槛。完整10行表和artifact SHA见 [原生分辨率最终报告](2026-11-10-stereo-native-resolution-final.md) 及 [机器可读实测摘要](../../contracts/stereo_native_resolution_results.json)。

## 本次交付边界与后续顺序

Phase0～7的隔离软件路径和CPU测试已落地，不能因此称整个抓取工程验收完成。[逐文件清单](../../stereo_phase0_7_file_inventory.md) 与 [readiness账本](../../contracts/stereo_phase0_7_readiness.json) 区分实现和晋级。原有528份生产源码/脚本/配置raw SHA不变；原224配置未覆盖；历史Checkpoint未改。

后续先修复并重验深度离群/遮挡及可见几何精度，同时接通已审计的force/support测量工厂和独立几何资格；之后才能开展新Teacher双抓法独立试验及每种至少100初态的验收。取得Teacher与Stereo资格后再准入真实BC/DAgger，最后短Student PPO保存恢复、独立持出Student评估和部署/训练吞吐。当前无Teacher/Student原生成功统计，不将“未测”写成0/100，不选最终训练分辨率、不启动长PPO、不合并PR。
