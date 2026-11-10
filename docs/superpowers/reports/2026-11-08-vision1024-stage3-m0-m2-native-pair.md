# 视觉 Student 第三阶段 M0/M2 原生配对诊断

## 实验边界

两次实验均从同一最终视觉 Checkpoint、同一 1024 环境配置、同一 reset 配置和同一 PPO 更新起点开始，输出目录完全独立。M0 保持原始观测/动作路径；M2 使用显式 actor-side `mirror_observation -> quaternion canonicalize`，再用 `mirror_action24` 把 actor 动作逆变换回物理坐标。371D Critic保持原输入，24D动作合同和物理限制不变。M2没有写入旧模型合同，也没有接入默认部署路径。

## 原生结果

| 项目 | M0 原始路径 | M2 独立镜像路径 |
|---|---:|---:|
| 输出目录 | `dh116_gpu3_stage2_dualdiag_1update_20261108T053021Z` | `dh116_gpu3_m2_native_counterfactual_1update_20261108T060501Z` |
| 60Hz policy记录 | 32768 | 32768 |
| 480Hz physics记录 | 262144 | 262144 |
| 原生 successes | 0 | 0 |
| 二指净Lift | 0.000074 mm | 0.003487 mm |
| 三指净Lift | 0.007928 mm | 0.014168 mm |
| 右掌Z动作均值 | -0.289228 | -0.115422 |
| 右掌实际峰值（二指/三指） | 0.172883/0.121742 mm | 0.190794/0.188023 mm |
| 失败 | 无 | `joint_limit` 2次 |

M2使右掌Z动作偏置减弱，且短回放中两种抓法的净Lift数值高于M0；但两者都远低于30mm，原生成功均为0，且M2出现2次joint_limit。这个结果只能标记为一次独立反事实短回放，不能宣称M2已经改善真实抓取，也不能将单次差异解释为因果。

## 诊断合同核对

M2输出的 `vision_mirror_preprocessing.json` 明确记录：

- actor观测：`mirror_observation_then_quaternion_canonicalize`
- 物理动作：`mirror_action24_inverse`
- `inverse_is_involution=true`
- `checkpoint_contract_unchanged=true`

M2没有重复镜像，也没有遗漏动作逆变换。M0默认路径仍为identity。M2的物理投影前后差值约 `3.67e-10 m`，投影比例约 `3.05e-5`，SDK目标误差仍小于 `6e-16 m`。

## 结论

M2短原生实验验证了独立预处理和动作逆变换能够进入真实RGB-D/TensorWorld路径并完成一个PPO更新，但尚未达到成功标准。后续如要纳入新模型合同，必须使用同初态多次重复评估证明接触、净Lift和失败率的可重复改善；本次结果不足以修改默认部署路径或旧Checkpoint。
