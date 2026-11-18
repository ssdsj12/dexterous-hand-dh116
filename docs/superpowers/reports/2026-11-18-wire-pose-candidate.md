# DH116 线缆局部姿态候选与反力接管

预抓取姿态可以先提供接近目标和手指预形状。接触后由同一物理步的线缆反作用力、对向接触和固定锚滑移约束继续动作。三种抓法的 PPO 可以独立推进。先验姿态验证不依赖全部 Teacher 收敛。

## 代码审查

Antigravity 三轮只读复审了 Ubuntu `11-17` 的 `cf754dc96cfceaef66c15c9667db5a5165660621`。独立核对的关键源码中，`tabletop_teacher_force_runtime.py` SHA-256 为 `c2b6627daef367bccc1c9ef9596ae993be0c6fd4b939a073a61dd1a017fd3a5c`，`tabletop_centerline_grasp.py` SHA-256 为 `b16691cd12a69790b78ad15c37ad39a06429c3596eab72a35b56ede4f3b3d962`。

`FlexibleR0ControlStep` 结算已经发生的物理事件。复查确认真正的动作挂点位于诊断运行时的 `ForceContactHook.before_substep`，可投影现有 24D 指令。记稳前它固定两腕目标。`after_substep` 校验物理时钟、载荷和双侧对向接触，仍把固定锚滑移记为未知，并且不授最终保持成功。旧固定臂 `TabletopGraspSequence` 使用另一原生时步合同。

## 姿态适配

新增 `scripts/prepare_dh116_wire_pose_candidate.py`。输入为已经映射到 DH116 的参考姿态和带实际材料 ID 的目标 41 节点几何。每手选择独立内部站位，以材料 ID 正向、局部切向和投影的世界上方向建立右手坐标。输出保留各手相对于参考线缆的根姿偏移，保持原手指 q、wxyz、关节限位和被动联动检查。拒绝 ID 重复、零长区间、折返和高度近似竖直的局部框架。线缆数据只按 ID 还原顺序，节点坐标不变。输出记录两个输入 SHA 和候选抓法；场景碰撞、原生接触和 Teacher 准入待独立核验。

```bash
PYTHONPATH=src:. /mnt/mydisk/xk/envs/dh116/bin/python -B -m scripts.prepare_dh116_wire_pose_candidate \
  --reference reference_seed.json --target wire41.json --output new_candidate.json \
  --grasp pinch --left-node 30 --right-node 10
```

`reference_seed.json` 使用现有 `nodes`、`roots`、`joints` 合同，且线缆节点按 0–40 排列。`wire41.json` 用 `material_node_ids` 与 `nodes_w_m` 显式配对。示例材料 ID 用于演示接口，实际选择须检查腕部可达性、手指与杆相对距离、双手和场景净空。[OakInk 官方数据说明](https://github.com/oakink/OakInk/blob/main/docs/datasets.md) 中的人手抓姿可用于上游 DH116 拟合。此次使用仓库现有三种 DH116 参考姿态。

## 验证

CPU 回归 147 项通过，其中 19 项新测例。Registry 另有 55 项回归。命令以 `PYTHONPATH=src:.` 运行当前 Ubuntu 解释器，测试文件见 readiness。

在现有 Pinch、Tripod、Power 参考姿态上分别做了 CPU CLI 验证。目标线缆是在各自参考节点上加小幅平移和弯曲的测试几何。临时选取靠近参考腕部的内部材料 ID，三种输出均通过 `seed_arrays`，手指关节和目标杆节点逐项相同。

| 抓法 | 候选材料 ID 左／右 | 根位移左／右 |
|---|---:|---:|
| Pinch | 39 / 1 | 3.87 / 7.31 mm |
| Tripod | 36 / 8 | 2.64 / 5.60 mm |
| Power | 33 / 8 | 1.88 / 5.60 mm |

三姿态几何 CLI 日志 SHA-256 为 `e419615a107323d31c34a0d2e6ec607f8a42f77d42ae0ad3e1b294f0e6325798`。尺寸为输入姿态的位移量。

## 静态净空对照

对参考和候选根姿分别计算了两手各 17 个源 USD 凸包、各 5 个已锁定历史 cooked 指腹的桌面盒体 SAT，并用连续线段包络下界检查线缆间隙。第一种凸包距离求解器在左 Pinch 弯杆候选上未收敛，保留该失败。独立保守算法随后完成 24 个对照。候选 Pinch 两手最低桌面间距约 2.11 mm，对线缆的最小间隙下界约 0.354 mm。候选 Tripod 的右手、Power 的两手均有源凸包与桌面相交，涉及 `finger32_link` 和 `finger33_link`，需要先独立修改预抓取姿态。对照范围仅含静态桌面和测试线缆，仍需核对双手互碰、支架、轨迹和现场 PhysX 读回。

完整联合回归 202 项通过。CPU 回归、初次凸包求解失败和替代静态检查的日志 SHA 分别为 `9ce8d94e4edbaf9dc91fad8a06cbd49f34631ba27326c9dc7e12f884fb25cec6`、`016655a59cb70e921c54f8384a98a9a470a44eaaa5f13cd04246198e1f2f0264`、`0e0fcf9d88960b4657bd01e4969fa08954317c17c349958a0b30415d59edf15e`。

## 原生后续

先筛一组 Pinch 候选，比较默认初态和重定向初态的净空及可达性。经有界原生运行验证闭合、腕部固定时的连续反力、仿真时钟和自然退出。后续以同一根杆上的独立双手固定锚检查滑移并持续许可抬升，接触丢失时制动。Tripod 需各手食指和中指都与拇指对向。Power 需真实掌面或近端反力加上五指包覆。`ForceContactHook` 原接触诊断和 PPO 运行时未改动。本轮原生运行、实体接触证明和训练更新均为零。
