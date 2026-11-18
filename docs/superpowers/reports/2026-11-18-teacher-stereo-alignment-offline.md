# Teacher 双 RGB 离线对齐审计

基线提交为 `bf8a4452877015cedff0282880dbaa8a0f9d8afd`。本工作包采用 `next_steps.md` 的重投影和连续时序核验思路。现有碰撞校验、控制、Teacher 输入定义与训练准入沿用原版本。

## 可执行核验

- `audit_frame` 核验标定 SHA、固定 env 0 的相机内外参、左右 RGB PNG 的 SHA 与实际尺寸，以及左右渲染帧和 SDK 物理时刻的身份。以原始世界坐标材料节点复算左右图像像素、Optical-Z 和双掌局部坐标。
- 独立人工标注 JSON 需要 `receipt_sha256`、`source` 为 `manual_rgb_annotation`、以及 `observations`。每条 observation 需要 `side`、`node_index`、`rgb_sha256` 和 `pixel_uv`。只有能够从图像独立定位并明确对应材料节点的标注可用于误差统计。轮廓表面、不可辨认的节点和被遮挡点应留空。工具核验绑定关系，独立标注过程仍依赖采集者。报告逐点残差、均值、RMSE、P95、最大值，以及已标注点是否全部严格小于 2 px。缺标注时该项为空。
- `audit_sequence` 检查按采集顺序传入的单环境同一 episode 片段。相邻 60 Hz policy tick 应跨 8 个 480 Hz SDK 物理步，左右 RGB 共享渲染 epoch，渲染 epoch 随帧递增。模拟时钟用于计算帧间距与抖动。墙钟延迟需要独立单调计时字段，当前回执无该字段。

```bash
PYTHONPATH=src /mnt/mydisk/xk/envs/dh116/bin/python -B scripts/audit_dh116_teacher_stereo_alignment.py /path/to/teacher_stereo_alignment.json --annotations /path/to/annotations.json
```

## 验证与剩余工作

- Ubuntu CPU 联合回归 **599 通过，0 失败**，含 8 项新增测试。单独运行 Registry 回归 55 项通过。新增用例覆盖双掌坐标、双图几何、2 px 边界、损坏或不匹配的图像/标定/数据、陈旧渲染 epoch 和缺失策略步。
- 真实 RGB 回执、人工标注、连续轨迹及关联动作计数均为 **0**。现有首次采集合同仍是 67D 原始 Teacher 的第 2 SDK 步、零动作快照。该格式缺少生成完整 state SHA 的附加状态字段。离线结果明确记录完整状态哈希未复验，完整 episode 和真实动作匹配均为零。
- 下一步先取得获准的原生首帧证据，再进行人工重投影复核。连续采集、动作绑定与完整回合验证需在首帧结果之后单独实施和验证。
