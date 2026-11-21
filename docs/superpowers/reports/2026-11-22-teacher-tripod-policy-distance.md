# Tripod policy-owned distance source

基线 d9ba234557af226fdd6dd1322bdca78dbf40f4b5，Ubuntu工作树11-17。新增独立Tripod来源读取器，复用已验证的SDK捕获、SDF与几何证书。没有创建第二套PPO或脚本抓取控制器。

## 实际代码

- tabletop_native_teacher_policy_tripod_distance.py：
  - PolicyOwnedTripodGeometry 固定3目标指垫/手：finger14_link、finger23_link、finger33_link。完整核验原始左右USD/SDF绑定、实际17 mesh顺序、每手7路COM/World/body身份、外部table及双支架载荷与同钟pair buffer。
  - 通过现有prepare_initial_state核验SHA绑定的独立Tripod任务配置、真实重力、无固定杆节点及原table支持几何；不以该配置的 authored hand pose 代替SDK测量。
  - 全原17 source＋5 cooked/手的桌面、非目标、双手与手腕守卫保留。选定pad使用原0.15mm floor的完整杆凸组下界，支架与非目标原0.25mm保护保留。源/熟几何使用原rotated_points的原始四元数运算，不擅自归一化。
  - positive_pad_mean 仅接受3个完整、梯度有效的SDF样本；不是把Pinch二指均值改名。任一中指无覆盖即unknown，不补零、不退回二指。
  - PolicyOwnedTripodApproach.capture 绑定真实history owner类型、已记录row hash、实际issued24D、SDK targets hash、token与同钟endpoint；拒绝篡改及Pinch重标。距离来源保留单位、坐标系、固定41材料ID、scope及未知uncertainty。geometry不推断contact。
- tabletop_native_teacher_policy_pinch_distance.py：仅将固定的required pads/count/version提取为可覆写类常量。默认仍2垫、5路、原Pinch版本与行为；没有更改NativePinchSources、PPO、GAE、KL或Actor动作合同。
- tests/test_native_teacher_policy_tripod_distance.py：覆盖3-pad均值、真实硬件SDF无中指覆盖、丢反力通道、错误body ID、丢支撑及history/抓法重标。

## 测试与范围

新测试先在缺失Tripod读取器时失败；日志保留。初次实现迭代还暴露CPU夹具的来源frame/USD、pose cache、完整rod_boundary及中指SDF覆盖不足，均修正夹具本身，未放松生产校验。
最终来源/几何相关7文件：96 passed，0 failed，281.33s（含新增6项）。
独立owner/factory/update/resume相关6文件：70 passed，0 failed，92.83s。两组不重叠，共166项所选CPU回归；这不是全仓或原生验证。
上包已复现的旧封存R0 replay来源SHA失败仍未解决，本轮未删除该测试或重标旧数据。
CPU夹具使用SDK-shaped doubles和锁定真实硬件SDF，额外通道为显式CPU合成数据。可观测中指案例将合成getter位置放入真实SDF格点以验证数据合同，不是物理可达抓取，不是新原生数据。不得以其证明7路实际耦合、接触成功或Teacher资格。
每次物理endpoint的几何检查不等于完整九帧/连续控制区间证书；本包不开放NativePreflight的Tripod入口。原运行时连续/安全守卫仍是后继接线与原生验收的必要条件。
无新Isaac实例、无原生PPO更新；CPU更新/恢复单元测试只用临时fixture模型。两台固定RGB配置、旧模型、受保护文件与物理参数不变。

## 审查与下一步

依据已完成的Antigravity Teacher-Tripod-Smoke R1来源拆分方案实施，Codex将其建议的8路纠正为实际7路。外部账户quota已耗尽，本补丁未获得新的Antigravity代码复审；独立源码核验与上述CPU测试已完成。
尚缺：Tripod同钟Reset距离基线、独立17 mesh零步clear/add及current/target/solver读回、fresh supported owner、真实7路首帧交换与原连续守卫、source factory/preflight。不得把SDK240静置快照当SDK2 fresh Reset，也不得把第一policy动作伪称stationary。
这些通过后复用既有66＋真实rod＋context Teacher，执行独立update1/resume2有界接线。Pinch已完成的旧原生两次更新保留；Power仍按掌面＋五指全握与实际载荷独立验收。R2继续关闭。

## 命令

复用Registry现有CPU环境，在工作树根目录执行：

```bash
/mnt/mydisk/xk/envs/dh116/bin/python -m pytest -q tests/test_native_teacher_policy_tripod_distance.py tests/test_native_teacher_policy_pinch_distance.py tests/test_native_teacher_supported_reset.py tests/test_dual_hand_reset.py tests/test_dh116_flexible_live_geometry.py tests/test_dh116_measured_closure_geometry.py tests/test_dh116_continuous_envelope_refinement.py
/mnt/mydisk/xk/envs/dh116/bin/python -m pytest -q tests/test_native_teacher_session.py tests/test_native_teacher_actual_factories.py tests/test_native_teacher_actual_provider.py tests/test_native_teacher_actual_update.py tests/test_native_teacher_actual_resume_update.py tests/test_native_teacher_actual_inputs.py
```

私有运行日志位于 ref902/tripod-distance-*.log，SHA见readiness；不上传日志、原始数组或模型。

Registry命令：`/mnt/mydisk/xk/envs/dh116/bin/python -m pytest -q tests/test_dh116_work_packages.py`；60 passed，0 failed，17.13s。
