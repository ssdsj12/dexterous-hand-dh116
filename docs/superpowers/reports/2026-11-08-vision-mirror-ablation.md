# 视觉 Student M0/M1/M2 镜像等变性离线消融

## 数据与边界

本实验只做 CPU 确定性推理，不启动 Isaac、不更新 PPO、不修改奖励、不拆分 Actor。输入来自最终视觉 Checkpoint 的独立 `initial_visual_actor_inputs.pt`，共 1024 个观测样本。保留 3844D Actor、371D Critic 合同和 24D 动作合同；本实验只调用 Actor。

该来源 Checkpoint 的 Actor quaternion preprocessing contract 为 `raw`，因此下面结果是输入镜像诊断，不是 Tanh Gaussian latent 分布的训练结论。

来源：

- Checkpoint：`/mnt/mydisk/xk/dh116/runs/dh116_gpu3_vision1024_ppo1024_20261107T131048Z/vision_student_ppo.pt`
- 观测：`/mnt/mydisk/xk/dh116/runs/dh116_gpu3_vision1024_actiondiag_final_1update_20261108T042804Z/initial_visual_actor_inputs.pt`
- 结果：`/mnt/mydisk/xk/dh116/runs/dh116_gpu3_actiondiag_fixedbaseline_1update_20261108T045950Z/mirror_ablation_m0_m2.json`

镜像变换交换左右手通道，沿 `diag(-1,1,1)` 反射掌心/局部点云 X，主动关节块交换，腕部 quaternion 使用 `(w,x,y,z) -> (w,x,-y,-z)`，24D 动作使用同一手交换和 intrinsic XYZ 旋转符号合同。

## 结果

| 模式 | 处理 | 动作 L2 均值 | L2 P95 | 方向一致率 |
|---|---|---:|---:|---:|
| M0 | 原始输入 | 0.754994 | 0.881472 | 0.562500 |
| M1 | 仅 quaternion hemisphere 规范化 | 0.806492 | 0.905909 | 0.604167 |
| M2 | quaternion + 完整镜像 paired canonical frame（离线） | 0.296581 | 0.345446 | 0.833333 |

M2 的 `offline_only=true`，它把镜像样本映回同一个 canonical frame，用于判断输入坐标不一致是否能解释动作差异；它没有接入部署路径，也没有改写旧 Checkpoint。M2 的误差下降是当前离线证据，不能单独证明原生抓取成功或允许策略修改。

同一批1024个观测的镜像双应用回环中，四帧历史、左右局部点云、validity、12D主动关节、12D关节速度、掌心XYZ、腕部quaternion和previous action 的 `max_abs` 与 `mean_abs` 全部为0，说明该离线坐标变换本身可逆。

## 结论与限制

M0 的左右动作镜像误差较大；M1 仅统一 quaternion 符号，未消除误差；M2 在完整 paired 镜像归一化后误差明显下降，说明局部坐标/相对几何不一致值得继续核对。该结果仍不能区分“坐标合同残差”和“网络本身学习不对称”，也没有使用第513轮模型（该 Checkpoint不存在）。下一步应在短时独立原生验证中只记录诊断，不能静默改变现有模型推理。
