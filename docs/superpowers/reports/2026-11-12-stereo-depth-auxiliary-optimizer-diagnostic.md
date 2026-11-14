# 双目深度辅助头的有界监督优化诊断

复用 `StereoDepthAuxiliary`、`native_depth_targets()` 和 `depth_auxiliary_loss()`。CPU 诊断脚本新增 `--optimizer-steps 0..8`，默认 0 保留原前向和梯度检查。显式启用后，用同一 SHA 绑定的归档双目帧做固定次数的 SGD 更新，学习率 0.01。优化参数为共享编码器和有监督的输出头，无参考的输出头不进入 optimizer。模型输入仍为左右 RGB 和双视图有效性。光轴 Z 只进入损失，传统匹配器的预测与有效掩码不决定哪些参考点可监督。

## 同帧诊断结果

Ubuntu CPU 对已有 Roll20 VGA 原生归档帧 21 执行两次更新，二指、三指各一个环境。模型不保存。两个环境的左图各有 132 个参考标签，右图均无合格原生标签。原始网格为 15×20。

| 项目 | 实际结果 |
|---|---:|
| 同帧 Smooth L1（米，初始/更新 1/更新 2） | 0.137148 / 0.266535 / 0.131038 |
| 左输出头参数最大绝对变化 | 0.00192118 |
| 右输出头参数最大绝对变化 | 0 |
| 共享骨干首参数最大绝对变化 | 0.000386177 |
| Isaac 物理步、模型保存数 | 0、0 |

第一次更新的损失上升也保留在记录中。这是同帧优化接线诊断，持出性能、泛化和部署资格仍待独立双侧参考数据。输出 SHA256 为 `dfac116445b1a29969376444dcda23ba43e6c7ea84ec7ddab8b2c07545153f56`，存于 Ubuntu 私有目录 `/tmp/dh116-depth-aux-fit-replay-3SOF8z`。

```text
python scripts/diagnose_dh116_stereo_depth_auxiliary.py --proof /tmp/dh116-stereo-20261110T120000Z/native_roll20_block5/640x480/native_stereo_proof_completed.json --proof-sha256 fda6bb67d0d9a141baa2352a7b9b8a4cc4cd7f3e90a7aae8ec411f1269de4172 --optimizer-steps 2 --output <new_private_output.json>
python -m pytest -q -p no:cacheprovider tests/test_dh116_stereo_depth_auxiliary.py tests/test_dh116_stereo_estimator.py tests/test_dh116_stereo_wire_segmentation.py tests/test_dh116_stereo_confidence_audit.py
```

两个新增测试先复现缺少预算入口的失败。最终 Ubuntu 67 项 CPU 测试通过，退出码 0，日志 SHA256 `f3f6d91dd2b9802e6f23319069ed8fbd594593caf86532217183d752333cc86e`。测试覆盖预算过限时拒绝读取数据、传统匹配器全无效时独立参考仍能监督、默认零更新，以及缺右标签时右输出头冻结。下一步在按物理初态分组的独立双侧参考数据上做持出评估。
