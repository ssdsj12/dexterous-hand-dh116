# 原投影 Pinch：同钟 SDK、跟踪、实际杆几何分解

本包只读分析已经完成的原生区间，不新增物理步、PPO 或成功资格。原始保护、关节范围、指腹椭圆、30mm/5s/0.5mm 均不变。任务源码基线 b3ac14ea66eb8e6d4cf9a73c04691c6a2b53249f。

## 实际修改

- `scripts/audit_dh116_projected_pinch_tracking.py`：`load_source` 核验冻结回执/原始账本/独立退出审计 SHA；`validate_boundary` 要求关节、掌位、杆、动作和接触诊断同身份；`named_sdk_active` 从真实 DOF 名称与 SDK indices 读取主动关节目标；`Geometry.compare` 分别使用左右实际 USD/FK/SDF，并保留原始逐子步接触记录。
- `tests/test_projected_pinch_tracking_decomposition.py`：先失败，再验证真实目标映射、完整真实区间、连续身份、非有限值、位姿形状、终止拒绝和几何差分解释。

连续分析同一 env0、episode `dual-pinch-555cbae972d144ba9b8a9458579dfc26:one` 的 SDK211–234，共24个480Hz边界；各边界含24D实际请求、SDK与实测主动关节、11个耦合关节残差、左右四个指腹的五层几何结果。没有拼接不同环境的峰值。

## 核实的具体偏差

SDK234，时间0.4875s。下表是原椭圆归一化平方半径，≤1才可能满足区域条件，区域本身仍不代表有效对向承托。

| 几何层 | 左拇指 | 左食指 | 右拇指 | 右食指 |
|---|---:|---:|---:|---:|
| 原计划＋名义直杆 | 0.999886123 | 0.991600293 | 0.824109068 | 0.993227530 |
| 实际SDK目标＋名义杆 | 1.028018685 | 1.019030490 | 0.824043133 | 0.993248712 |
| 实测主动关节FK＋名义杆 | 1.009281778 | 1.011072424 | 0.880506795 | 0.989390291 |
| 实测指腹link＋名义杆 | 1.009263936 | 1.011019637 | 0.879817379 | 0.989360685 |
| 实测指腹link＋实际杆 | 0.987841891 | 1.125072285 | 0.909390852 | 0.997298616 |

左手 SDK Z 目标比未投影计划高0.064200465mm，实测根位置只比实际SDK高0.001170835mm。不能将两者合并称为执行跟踪误差。左食指在24个真实边界上的最终区域指标为1.123203–1.127114，始终在区域外；当步原始接触也没有任何通过原区域条件的左食指记录。

左侧实际主动关节相对SDK最大偏差0.001923084rad；右侧0.005666852rad。左、右被动耦合最大残差分别0.000006907和0.000083694rad。它们被记录但不能据此声称是唯一失败原因。

在SDK234的**既定分解顺序**下，左食指从计划0.991600到SDK1.019030，跟踪读回后下降至1.011072；实际杆形状再提高至1.125072。因此“关节跟踪独自导致越界”不受该分解支持。原计划仅勉强满足名义几何，原安全投影及实际杆变形都改变了接触位置。中间层是未执行的CPU几何替换，差分有顺序依赖，不是随机对照原生因果实验。

## 证据和测试

- 原生回执 SHA256：`04ebfe2460b6366960d87cda6f36200f503072e4fdfa2abffd25bc8dfcf775c3`。
- 物理账本 SHA256：`9cec6ed43c06f23cf23b327a22009c7f1b3a98574de27883ba89c60b0bf255d4`。
- 24边界结果 SHA256：`964bda6cf767913aa994a943ecd198cde7f5fc108c70ee276790634ee9e51045`。
- 先失败：3 failed、6 errors，缺失待实现分析模块；日志保留 `projected-pinch-20261114/red-decomposition.log`。
- 实现后9 CPU passed、0 failed，46.36s，日志 SHA256 `9e0e698f873aed65694f8259876f737f17eb7badcf7615e11387541ceda096f5`。
- 集成再运行本包与旧真实跟踪回放回归；实际结果写入Registry/Readiness。
- 本包新原生0、PPO0、WireSeg更新0。引用的234步自然退出只资格化短诊断，不资格化抓取。

Ubuntu命令（原始结果位于私有证据目录，不上传）：

```bash
DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python CUDA_VISIBLE_DEVICES='' \
  bash scripts/python.sh -m pytest -q -p no:cacheprovider tests/test_projected_pinch_tracking_decomposition.py
DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python CUDA_VISIBLE_DEVICES='' \
  bash scripts/python.sh scripts/audit_dh116_projected_pinch_tracking.py --output NEW_PRIVATE_OUTPUT.json
```

## 下一工作包与阻断

以本次实际运动后的左侧相关几何重新评估候选；保留原安全投影、有效区域、执行器和非目标碰撞约束。不得复用“保持不动左手”的历史误差模型冒充本次闭合。任何新的目标仍需独立完整路径和新冻结原生诊断，不能直接将CPU预测当作真实恢复接触。R0稳定性来源、完整训练奖励资格和三种抓法独立验收仍未完成。
