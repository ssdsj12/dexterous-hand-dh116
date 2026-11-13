# 右 Pinch 独立 Reset 求解：非线性几何仍阻断

基线6709e72e170456ae05d30275a33d4a6dfd9d9833。新增代码和有界CPU诊断，不运行Isaac/PPO，不重复已消耗原生预算。Teacher资格仍0/3。

## 实际修改

scripts/solve_dh116_right_pinch_reset_direction.py：decode、reset_transform、local_pass、Geometry.measure/path、run。复用只读SDK234实际右手23link/q/qd、右USD/FK和锁定SDF，source17+cooked5；所有源和原生ledger SHA校验。

独立Reset变量：拇指展开初值±0.05rad；原两指累计闭合各在原0.1rad上限内；绕杆world-Y固定Reset角±0.1rad；固定掌位X/Z各±1mm、Y±6.25mm。Y半宽来自原12.5mm节点间距。它们不是在线手腕动作，不改变原24D速度、控制频率、关节硬限位或恢复预算。

几何使用T_actual × inv(FK(q_actual)) × FK(q_candidate)，然后应用统一Reset刚体变换。原source/cooked顶点不改，全部41杆节点不重采样。初始检查实际杆及任务Reset杆，末态检查实际杆；每个候选完整FK、全部形状和指腹SDF/法向重新计算。

每局部方向先两尺度差分，误差>0.2即停止；线性max-min/L1仅给候选，四个回溯点均实际非线性检查。最终要求原2.1mm桌面净空，非目标形状净空和指腹SDF -0.3..-0.1mm、法向dot<-0.2。优化预留更严格table0.1mm、非目标0.1mm；不放宽门槛。

全部输出exclusive、flush/fsync、源SHA。每批wall120s，无无限搜索。CPU程序exit0只表示诊断完成，不表示几何或Teacher通过。

## 有界批次与失败证据

|版本及明确变更|实际端点评估/上限|耗时s|结果SHA|结果|
|---|---:|---:|---|---|
|V1，4变量|17/42|5.966226|fb1c9efc44506192bc24c50cb051d79240d6e878e149cb315a84eb3053dd16ae|两尺度Jacobian差0.342444；主要来自指腹SDF法向单元切换，停止|
|V2，固定X/Z Reset与更小差分|58/58|7.072549|df98e69777da46f2f6460ffd6f1fc9fe110b2850e1a13c8e7a23398b4164db01|局部方向可计算，近端源形状/指腹约束仍冲突|
|V3，沿杆固定Y Reset|99/132|7.656418|8008acd225b32f09282dcf985dc93f0fe131d7c02d021df38e0916c9c7d4aaa4|最大违约降至0.260521mm，下一大步所有回溯均失败|
|V4，同域缩小局部信赖半径|132/132|7.752563|bd311cc064544f0b6c4ee295ac49d7b5e72f883dacd22c9f36ea6daa8787b865|最大违约0.148557mm，未通过，预算结束|

输出根目录均在/mnt/mydisk/xk/projects/dh116-autonomous-3c80166：
- right-reset-direction-20261113-8vgsz2mm/run01
- right-reset-direction-v2-20261113-sag30yrm/run01
- right-reset-direction-v3-20261113-2xfw3hqj/run01
- right-reset-direction-v4-20261113-379rl4yl/run01

V4 source/保存本次冻结脚本与测试；历史CPU结果和代码指纹保留。总306端点评估，新增原生步0、PPO0、全路径采样0（端点尚未通过）。不得从局部失败推断全局不可达。

## 节点与指腹的几何诊断

保持原41实际节点；只在CPU沿两个实测端点间的直线采401点以定位采样误差，不输入求解器，不输出视觉重建点，也不当实际接触。右拇指原节点gap1.871493mm，线采样gap0.813298mm，最小点沿Y相差约-4.999757mm；食指2.451136→2.095436mm，Y约+2.500266mm。

这证明离散节点与指腹SDF位置关系需要审查，不证明接触力、真实曲线或重采样方案合格。诊断SHA da6e24fc1ae9ba2cc0f8cb538459dddd9273d81f65fa43c9e4b41baee358034b。

## V4候选（没有执行）

初始thumb-spread +0.033675rad；thumb/index累计闭合0.021984586/0.030808246rad；固定world-Y Reset -0.0475rad；掌位平移[-0.006415,+2.522241,-0.749363]mm。原其他初始主动关节保留。

初始所有源/cooked形状桌面最小2.452127mm（source finger22）；初始杆净空下界最小0.846239mm（source finger14）。均仅候选几何，非新实际物理。
末态指腹gap拇指+0.048557mm、食指-0.014313mm，未达到要求的-0.1mm接触裕量。非目标source finger13距离下界-0.045660mm，不能证明净空；负下界本身不能报告为穿透深度。因此没有完整路径或原生配方。

原234步实测左手四指腹无穿透证据仍保留；本包不修改它们、不替换右手实际测量。

## 测试与保护

先记录模块缺失、6D/7D Reset与信赖半径接口RED，再修正实现。LP二次最小移动若使用t+容差可产生微小负约束，已改为额外正裕量，不放宽判据。

13新增+77相关=90 passed、0 failed、6.73s。日志/mnt/mydisk/xk/projects/dh116-autonomous-3c80166/right-reset-full-20261113.log，SHA34ea111e787d1a2b3d0c9e359573454cae6eaba6a7aa671f0e64a1def6ceb33a。

```sh
DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 bash scripts/python.sh -m pytest -q tests/test_dh116_right_pinch_reset_direction.py tests/test_dh116_right_pinch_measured_direction.py tests/test_dh116_polyline_support_certificate.py tests/test_dh116_pinch_closure_margin.py tests/test_dh116_pinch_rod_axis_clearance.py tests/test_dh116_pinch_segment_geometry.py
```

发布器另检查528保护文件与两个历史Checkpoint。只发布源码/测试/计划/报告/readiness，原始大数据不上传。PR2保持Draft。

## 下一步与独立任务

本包诊断执行完成，右手原生接触资格blocked。不能只继续增加闭合量；下一几何工作需要新的实测依据和独立Reset候选。Tripod/Power保持各自阻断，不复制此姿态。
无相机Teacher的状态来源与R0测量继续独立开发；正式PPO仍需真实Reset、动作、稳定/滑移和奖励资格。WireSeg本包训练步0，既有128seg/128depth诊断未增加。相机本包未运行。
